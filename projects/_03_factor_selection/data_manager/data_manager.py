"""
数据管理器 - 单因子测试终极作战手册
第二阶段：数据加载与股票池构建

实现配置驱动的数据加载和动态股票池构建功能
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from data.local_data_load import load_suspend_d_df
from projects._03_factor_selection.config_manager.function_load.load_config_file import _load_file
from projects._03_factor_selection.config_manager.factor_info_config import FACTOR_FILL_CONFIG_FOR_STRATEGY, \
    FILL_STRATEGY_FFILL_UNLIMITED, \
    FILL_STRATEGY_CONDITIONAL_ZERO, FILL_STRATEGY_FFILL_LIMIT_5, FILL_STRATEGY_NONE, FILL_STRATEGY_FFILL_LIMIT_65
from projects._03_factor_selection.data_manager.suspend_state import (
    _aggregate_suspend_events_to_eod_states,
    _build_suspend_eod_matrix,
)
from projects._03_factor_selection.data_manager.stock_history import apply_history_days_filter
from projects._03_factor_selection.data_manager.entry_pool import (
    apply_open_buy_filter,
    build_open_tradeable_mask,
)
from projects._03_factor_selection.utils.IndustryMap import PointInTimeIndustryMap
from quant_lib.data_loader import DataLoader
from projects._03_factor_selection.utils.component_loader import IndexComponentLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.constant_config import get_market_data_path, permanent__day
from quant_lib.config.logger_config import setup_logger, log_warning

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logger(__name__)


# 执行 check_field_level_completeness 对应逻辑。
def check_field_level_completeness(raw_df: Dict[str, pd.DataFrame]):
    dfs = raw_df.copy()
    logger.info("原始字段缺失率体检报告:")
    for item_name, df in dfs.items():
        # missing_rate_daily = df.isna().mean(axis=1)

        # logger.info(f"{item_name}因子缺失率最高的10天 between {first_date} and {end_date}")
        # logger.info(f"{missing_rate_daily.sort_values(ascending=False).head(10)}")  # 其实也不需要太看重，只能说是辅助日志，如果总缺失率高 可以看看整个辅助排查而已！

        # 计算每只股票（每一列）的缺失率(相当于看这股票 在这一段时间的完整率！---》推导：最后一天才上市！，那么缺失率可能高达99.99% 所以不需要看重这个！)  注释掉
        #
        # logger.info(f"{item_name}（不是很重要）因子缺失率最高的10只股票 between {first_date} and {end_date}")
        # logger.info(f"{missing_rate_per_stock.sort_values(ascending=False).head(10)}")

        # 计算整个DataFrame的缺失率
        total_cells = df.size
        df_all_cells = df.isna().sum().sum()
        global_na_ratio = df_all_cells / total_cells
        tip = _get_nan_comment(item_name, global_na_ratio)
        if tip:
            logger.info(f'\t{tip}')


# 执行 _get_nan_comment 对应逻辑。
def _get_nan_comment(field: str, rate: float):
    logger.info(f"field：{field}在原始raw_df 确实占比为：{rate}")
    if field in ['delist_date']:
        # f"{field} in 白名单，这类因子缺失率很高很正常"
        return None
    if rate >= 0.4:
        raise ValueError(f'field:{field}缺失率超过50% 必须检查')
    """根据字段名称和缺失率，提供专家诊断意见"""
    if field in ['pe_ttm', 'pe', 'pb',
                 'pb_ttm', 'amount'] and rate <= 0.4:  # 亲测 很正常，有的垃圾股票 price earning 为负。那么tushare给我的数据就算nan，合理！
        # " (正常现象: 主要代表公司亏损)"
        return None

    if field in ['dv_ttm', 'dv_ratio']:
        # " (正常现象: 主要代表公司不分红, 后续应填充为0)"
        return None
    if field in ['industry']:  # 亲测 industry 可以直接放行，不需要care 多少缺失率！因为也就300个，而且全是退市的，
        # return "正常现象：不需要care 多少缺失率"
        return None
    if field in ['circ_mv', 'total_mv',
                 'turnover_rate',
                 'close_raw', 'open_raw', 'high_raw', 'low_raw', 'vol_raw', 'adj_factor',
                 'close_hfq', 'open_hfq', 'high_hfq', 'low_hfq',
                 'pre_close', 'amount'] and rate < 0.25:  # 亲测 一大段时间，可能有的股票最后一个月才上市，导致前面空缺，有缺失 那很正常！
        # "正常现象：不需要care 多少缺失率"
        return None
    if field in ['list_date'] and rate <= 0.01:
        # "正常现象：不需要care 多少缺失率"
        return None
    if field in ['beta'] and rate <= 0.25:
        # return "正常"
        return None
    if field in ['ps_ttm'] and rate <= 0.25:
        # return "正常"
        return None

    raise ValueError(f"(🚨 警告: 此字段{field}缺失ratio:{rate}!) 请自行配置通过ratio 或则是缺失率太高！")


class DataManager:
    """
    数据管理器 - 负责数据加载和股票池构建
    
    按照配置文件的要求，实现：
    1. 原始数据加载
    2. 动态股票池构建
    3. 数据质量检查
    4. 数据对齐和预处理
    """

    # 执行 __init__ 对应逻辑。
    def __init__(self, config: dict):
        """
        初始化数据管理器
        
        Args:
            config: 已解析的研究配置
        """
        self.st_matrix = None  # 注意 后续用此字段，需要注意前视偏差
        self._tradeable_matrix_by_suspend_resume = None
        self.config = config
        self._pit_map = None
        self.research_start_date = self.config['research_window']['start_date']
        self.research_end_date = self.config['research_window']['end_date']
        self.buffer_start_date = None
        self.data_loader = DataLoader()
        self.stock_pools_dict = None
        self._entry_pools = {}
        self._existence_matrix = None
        self.component_loader = None

    def prepare(self) -> None:
        """准备本次研究共用的数据与股票池。"""
        self._prepare_dates()#预热天数
        self._prepare_stock_pool()

    def _prepare_dates(self) -> None:
        # 日历只在准备研究日期时读取，不再扫描字段来源。
        self.data_loader.trade_cal = self.data_loader._load_trade_cal()
        self.trading_dates = self.data_loader.get_trading_dates(
            self.research_start_date, self.research_end_date
        )
        self.buffer_start_date = self._resolve_buffer_start_date()
        self._prebuffer_trading_dates = self.data_loader.get_trading_dates(
            self.buffer_start_date, self.research_end_date
        )

    @property
    def pit_map(self):
        """行业因子或行业预处理首次使用时加载，同次运行共享。"""
        if self._pit_map is None:
            self._pit_map = PointInTimeIndustryMap()
        return self._pit_map

    def get_preprocessing_industry_map(self):
        preprocessing = self.config["preprocessing"]
        neutralization = preprocessing["neutralization"]
        needs_industry = any(
            preprocessing.get(step, {}).get("by_industry") is not None
            for step in ("winsorization", "standardization")
        ) or (
            neutralization["enable"] and "industry" in neutralization["factors"]
        )
        return self.pit_map if needs_industry else None

    # 执行 _resolve_buffer_start_date 对应逻辑。
    def _resolve_buffer_start_date(self) -> str:
        factor_days = [
            self._get_factor_preheat_trading_days(name, set())
            for name in self.get_experiments_factor_names()
        ]
        preheat_days = max(factor_days)
        first_date = self.trading_dates[0]
        trade_cal = self.data_loader.trade_cal
        prior_dates = trade_cal.loc[
            (trade_cal['is_open'] == 1) & (trade_cal['cal_date'] < first_date),
            'cal_date',
        ]
        if len(prior_dates) < preheat_days:
            raise ValueError(
                f"交易日历不足以满足预热期: first_date={first_date.date()}, "
                f"required_days={preheat_days}, available_days={len(prior_dates)}"
            )
        buffer_start = pd.Timestamp(prior_dates.iloc[-int(preheat_days)])
        logger.info(
            f"预热期已解析: factor_days={max(factor_days)}, "
            f"selected_days={preheat_days}, "
            f"buffer_start={buffer_start.date()}"
        )
        return buffer_start.strftime('%Y%m%d')
    #取因子预热天数
    def _get_factor_preheat_trading_days(
            self, factor_name: str, ancestors: set[str]
    ) -> int:

        definition = self.get_factor_definition(factor_name).iloc[0]

        if definition['action'] == 'composite':
            children = definition['cal_require_base_fields']
            return max(
                self._get_factor_preheat_trading_days(
                    child, ancestors | {factor_name}
                )
                for child in children
            )

        return  definition['preheat_trading_days']

    def get_raw_field(self, field_name: str) -> pd.DataFrame:
        """直接读取研究窗口内的字段，不缓存；收盘价和成交额保留原始网格，其余对齐收盘价。"""
        df = self.data_loader.read_field(
            field_name, self.buffer_start_date, self.research_end_date,
        )
        if field_name in ('close_raw', 'amount'):
            return df
        close = self.data_loader.read_field(
            'close_raw', self.buffer_start_date, self.research_end_date,
        )
        return df.reindex(index=close.index, columns=close.columns)

    def _prepare_stock_pool(self) -> None:
        """构建本次股票池，数据由各过滤步骤按需读取。"""
        pool_name = self.config['stock_pool_name']
        pool_config = self.config['stock_pool_profiles'][pool_name]
        self.stock_pools_dict = {
            pool_name: self.create_stock_pool(pool_config, pool_name)
        }
        self._entry_pools.clear()

    def get_entry_pool(self, pool_name: str) -> pd.DataFrame:
        """T 行对应 T 日信号及 T+1 开盘买入资格，同次研究的所有因子共用。"""
        # 首次构建并缓存买入资格；同轮各因子复用，避免评价样本口径发生变化。
        if pool_name not in self._entry_pools:
            pool = self.stock_pools_dict[pool_name]
            # 从停复牌事件得到各交易日 09:30 可交易状态，再结合原始开盘价和涨停价。
            tradeable = build_open_tradeable_mask(
                load_suspend_d_df(), pool.index, list(pool.columns)
            )
            if self.config['stock_pool_profiles'][pool_name]['filters']['remove_st']:
                # 自然日先后移一天，再对齐交易日，周末公告可阻止周一买入。
                st_before_open = self.st_matrix.shift(1, freq='D').reindex(
                    index=pool.index, columns=pool.columns,
                )
                tradeable = tradeable & ~st_before_open
            prices = [
                self.get_raw_field(field).reindex(index=pool.index, columns=pool.columns)
                for field in ('open_raw', 'up_limit')
            ]
            # 将次日可开盘买入条件移到信号日，与 T 日基础池取交集；最后一日无次日资格。
            entry_pool = apply_open_buy_filter(pool, *prices, tradeable)
            logger.info(
                f'{pool_name} 开盘买入过滤剔除 {(pool & ~entry_pool).sum().sum()} 个股票日样本'
            )
            self._entry_pools[pool_name] = entry_pool
        return self._entry_pools[pool_name]

    # institutional_profile   = stock_pool_profiles['institutional_profile']#为“基本面派”和“趋势派”因子，提供一个高市值、高流动性的环境
    # microstructure_profile = stock_pool_profiles['microstructure_profile']#用于 微观（量价/情绪）因子
    # product_universe =self.product_universe (microstructure_profile,trading_dates)

    # 执行 _check_data_quality 对应逻辑。
    def _check_data_quality(self):
        """检查数据质量"""
        print("  检查数据完整性和质量...")

        for field_name, df in [("close_raw", self.get_raw_field('close_raw'))]:
            # 检查数据形状
            print(f"  {field_name}: {df.shape}")

            # 检查缺失值比例
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            print(f"    缺失值比例: {missing_ratio:.2%}")

            # 检查异常值
            if field_name in ['close_raw', 'total_mv', 'pb', 'pe_ttm']:
                negative_ratio = (df <= 0).sum().sum() / df.notna().sum().sum()
                print(f"  极值(>99%分位) 占比: {((df > df.quantile(0.99)).sum().sum()) / (df.shape[0] * df.shape[1])}")

                if negative_ratio > 0:
                    print(f"    警告: {field_name} 存在 {negative_ratio:.2%} 的非正值")

    # ok 这支股票在这一天是否已上市且未退市_df
    def build_existence_matrix(self) -> pd.DataFrame:
        """
        根据每日更新的上市/退市日期面板，构建每日“存在性”矩阵。
        """
        logger.info("    正在构建股票“存在性”矩阵..")
        # 1. 获取作为输入的上市和退市日期面板
        list_date_panel = self.get_raw_field('list_date')
        delist_date_panel = self.get_raw_field('delist_date')

        # 2. 【核心】向量化构建布尔掩码 (Boolean Masks)

        # a. 创建一个“基准日期”矩阵，用于比较
        #    该矩阵的每个单元格[date, stock]的值，就是该单元格的日期'date'
        #    这允许我们将每个单元格的“当前日期”与它的上市/退市日期进行比较
        dates_matrix = pd.DataFrame(
            data=np.tile(list_date_panel.index.values, (len(list_date_panel.columns), 1)).T,
            index=list_date_panel.index,
            columns=list_date_panel.columns
        )

        # b. 构建“是否已上市”的掩码 (after_listing_mask)
        #    直接比较两个相同形状的DataFrame
        #    如果 当前日期 >= 上市日期, 则为True
        after_listing_mask = (dates_matrix >= list_date_panel)

        # c. 构建“是否未退市”的掩码 (before_delisting_mask)
        #    同样，先用一个遥远的未来日期填充NaT（未退市的情况）
        future_date = pd.Timestamp(permanent__day)
        delist_dates_filled = delist_date_panel.fillna(future_date)

        #    如果 当前日期 < 退市日期, 则为True
        delist_not_null_count = delist_date_panel.notna().sum().sum()
        if delist_not_null_count == 0:
            raise ValueError('严重数据异常：delist_date_df全为空')
        # 原有逻辑：使用退市日期
        before_delisting_mask = (dates_matrix < delist_dates_filled)

        # 4. 合并掩码，得到最终的“存在性”矩阵
        #    一个股票当天“存在”，当且仅当它“已上市” AND “未退市”
        existence_matrix = after_listing_mask & before_delisting_mask

        # === 🔍 调试输出 - 统计存在性矩阵 ===
        total_cells = existence_matrix.size
        true_cells = existence_matrix.sum().sum()
        false_cells = total_cells - true_cells
        print(f"存在性矩阵统计: 总单元格={total_cells}, True={true_cells}, False={false_cells}")
        print(f"False比例: {false_cells / total_cells:.1%} (这些是'不存在'的股票-日期对)")
        logger.info("    股票“存在性”矩阵构建完毕。")
        # 缓存起来，因为它在一次回测中是不变的
        self._existence_matrix = existence_matrix

    # 执行 build_tradeable_matrix_by_suspend_resume 对应逻辑。
    def build_tradeable_matrix_by_suspend_resume(
            self,
    ) -> pd.DataFrame:
        """
        根据完整停复牌历史，构建每日收盘后的可交易状态矩阵。

        T 行用于 T 日收盘后的选股决策。
        """
        if self._tradeable_matrix_by_suspend_resume is not None:
            logger.info(
                "self._tradeable_matrix_by_suspend_resume 之前以及被初始化，无需再次加载（这是全量数据，一次加载即可")
            return self._tradeable_matrix_by_suspend_resume

        ts_codes = list(set(self.get_stock_codes()))
        trading_dates = self.data_loader.get_trading_dates(start_date=self.research_start_date,
                                                           end_date=self.research_end_date)

        logger.info("正在重建每日收盘后的可交易状态矩阵...")
        # 完整历史用于确定研究期首日的既有状态；同日多事件先在独立模块内确定性聚合，
        # 再把离散的日终状态变化传播到研究期交易日。
        daily_states = _aggregate_suspend_events_to_eod_states(load_suspend_d_df())
        tradeable_matrix = _build_suspend_eod_matrix(daily_states, trading_dates, ts_codes)

        logger.info("每日收盘后的可交易状态矩阵重建完毕。")
        self._tradeable_matrix_by_suspend_resume = tradeable_matrix.astype(bool)
        return self._tradeable_matrix_by_suspend_resume

    def build_st_period_from_namechange(
            self,
    ) -> pd.DataFrame:
        """按自然日重建公告收盘后已知的 ST 状态，供信号池及次日买入过滤使用。"""
        logger.info("正在根据名称变更历史，重建每日‘已知风险’状态st矩阵...")
        # 多取前一自然日，供研究首日读取开盘前状态。
        ts_codes = list(set(self.get_stock_codes()))
        calendar_dates = pd.date_range(
            pd.Timestamp(self.research_start_date) - pd.Timedelta(days=1),
            self.research_end_date,
        )
        namechange_df = self.get_namechange_data()

        # --- 1. 准备工作 ---
        namechange_df = namechange_df.copy()
        namechange_df['ann_date'] = pd.to_datetime(namechange_df['ann_date'])
        # 2020 年起，同一股票同一公告日不得出现相反的 ST 状态。
        recent = namechange_df.loc[namechange_df['ann_date'] >= pd.Timestamp('2020-01-01')]
        states = recent['name'].str.upper().str.contains('ST', regex=False)
        conflicts = states.groupby([recent['ts_code'], recent['ann_date']]).nunique()
        conflicts = conflicts[conflicts > 1]
        if not conflicts.empty:
            raise ValueError(f"同一股票同一公告日存在冲突的 ST 状态：{conflicts.index.tolist()}")

        # 【关键】必须用 np.nan 初始化，作为“未知状态”
        st_matrix = pd.DataFrame(pd.NA, index=calendar_dates, columns=ts_codes, dtype='boolean')

        # --- 2. “打点”：一个循环处理所有历史事件 ---
        for ts_code, group in namechange_df.groupby('ts_code'):
            group_sorted = group.sort_values(by='ann_date')
            for _, row in group_sorted.iterrows():
                ann_date = row['ann_date']

                # 研究期前的公告依次覆盖首日，期内公告写入对应自然日。
                ann_date_loc = calendar_dates.searchsorted(ann_date,
                                                          side='left')

                # 只处理那些能影响到我们回测周期的事件
                if ann_date_loc < len(calendar_dates):
                    name_upper = row['name'].upper()
                    is_risk_event = 'ST' in name_upper
                    st_matrix.loc[calendar_dates[ann_date_loc], ts_code] = is_risk_event

        # --- 3. “传播”与“收尾” ---
        st_matrix = st_matrix.ffill(inplace=False)
        st_matrix = st_matrix.fillna(False, inplace=False)

        logger.info("每日‘已知风险’状态矩阵重建完毕。")
        self.st_matrix = st_matrix.astype(bool)
        return self.st_matrix

    # 使用截至 T 日收盘可观察到的有效交易日数。
    def _filter_by_history_days(self, stock_pool_df: pd.DataFrame, history_days: int) -> pd.DataFrame:
        filtered_pool = apply_history_days_filter(
            stock_pool_df,
            self.get_raw_field('close_raw') if history_days else None,
            history_days,
        )
        self.show_stock_nums_for_per_day(f'_filter_by_history_days',filtered_pool)
        return filtered_pool

    # ok 已经处理前视偏差
    def _filter_st_stocks(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame:
        self.build_st_period_from_namechange()
        if self.st_matrix is None:
            raise ValueError("    警告: 未能构建ST状态矩阵，无法过滤ST股票。")
        st_mask = self.st_matrix
        # 对齐两个DataFrame的索引和列，确保万无一失
        # join='left' 表示以stock_pool_df的形状为准
        aligned_universe, aligned_st_status = stock_pool_df.align(st_mask, join='left',
                                                                  fill_value=False)  # 至少做 行列 保持一致的对齐。 下面才做赋值！ #fill_value=False ：st_Df只能对应一部分的股票池_Df.股票池_Df剩余的行列 用false填充！

        # 将ST的股票从universe中剔除
        # aligned_st_status为True的地方，在universe中就应该为False
        aligned_universe[aligned_st_status] = False
        self.show_stock_nums_for_per_day(f'_filter_st_stocks',aligned_universe)

        return aligned_universe

    # ok
    #
    def _filter_by_existence(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.0-优化版】基于预先构建好的“存在性”矩阵，进行最高效的过滤。
        此过滤器同时处理了“未上市”和“已退市”两种情况，是存在性检验的唯一入口。
        """
        logger.info("    应用统一的存在性过滤 (上市 & 退市)...")

        # 1. 获取或构建权威的存在性矩阵 (应该已被缓存)
        #    这个矩阵已经包含了所有上市/退市的完整信息。
        if self._existence_matrix is None:
            self.build_existence_matrix()

        existence_matrix = self._existence_matrix

        existence_mask = existence_matrix

        # 3. 安全对齐并应用过滤器
        #    fill_value=False 表示，如果一个股票在您的基础池中，
        #    但不在我们的存在性矩阵的考虑范围内，我们默认它不存在。
        aligned_pool, aligned_existence_mask = stock_pool_df.align(
            existence_mask,
            join='left',
            axis=None,
            fill_value=False
        )

        filtered_pool = aligned_pool & aligned_existence_mask

        # 4. 统计日志
        original_count = stock_pool_df.sum().sum()
        filtered_count = filtered_pool.sum().sum()
        delisted_removed_count = original_count - filtered_count
        logger.info(
            f"      existence上市退市股票过滤(: 在整个回测期间，共移除了 {delisted_removed_count:.0f} 个'已退市'的股票次（股票累计非existence天数）")
        self.show_stock_nums_for_per_day('by_统一存在性_filter', filtered_pool)

        return filtered_pool

    # 适配停经历复牌事件的可交易股票池 ok
    def _filter_tradeable_matrix_by_suspend_resume(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame:
        self.build_tradeable_matrix_by_suspend_resume()
        if self._tradeable_matrix_by_suspend_resume is None:
            raise ValueError("警告: 未能构建 _tradeable_matrix_by_suspend_resume 状态矩阵。")

        # T 日收盘状态用于 T 日信号；次日实际开盘资格由 entry_pool 处理。
        tradeable_mask = self._tradeable_matrix_by_suspend_resume

        # 以股票池为左侧边界，避免停复牌数据增加或删除股票池的股票集合；
        # 未出现停复牌状态的单元格没有不可交易证据，延续原逻辑填 True。
        aligned_universe, aligned_tradeable_mask = stock_pool_df.align(
            tradeable_mask,
            join='left',
            fill_value=True
        )

        # 停复牌过滤只能从既有股票池中剔除，不得把其他过滤器已排除的股票重新加入。
        final_pool = aligned_universe & aligned_tradeable_mask

        return final_pool

    # ok
    def _filter_by_liquidity(self, stock_pool_df: pd.DataFrame, min_percentile: float) -> pd.DataFrame:
        """按流动性过滤 """
        turnover_df = self.get_raw_field('turnover_rate')
        turnover_df = turnover_df.reindex(index=stock_pool_df.index, columns=stock_pool_df.columns)
        missing = stock_pool_df & turnover_df.isna()
        if missing.to_numpy().any():
            row, col = np.argwhere(missing.to_numpy())[0]
            raise ValueError(f"股票池内 turnover_rate 缺失：日期={missing.index[row]}，股票={missing.columns[col]}")
        # 【关键】股票池构建的时间逻辑：
        # - 构建 T 日收盘后的信号股票池。
        # T 日收盘后使用当日换手率。

        # 1. 【确定样本】只保留 stock_pool_df 中为 True 的换手率数据
        # “只对当前股票池计算”
        valid_turnover = turnover_df.where(stock_pool_df)

        # 2. 【计算标准】沿行（axis=1）一次性计算出每日的分位数阈值
        thresholds = valid_turnover.quantile(min_percentile, axis=1)

        # 3. 【应用标准】将原始换手率与每日阈值进行比较，生成过滤掩码
        low_liquidity_mask = turnover_df.lt(thresholds, axis=0)

        # 4. 将需要剔除的股票在 stock_pool_df 中设为 False
        stock_pool_df[low_liquidity_mask] = False
        self.show_stock_nums_for_per_day(f'_filter_by_liquidity',stock_pool_df)

        return stock_pool_df

    # ok
    def _filter_by_market_cap(self,
                              stock_pool_df: pd.DataFrame,
                              min_percentile: float) -> pd.DataFrame:
        """
        按市值过滤 -
        Args:
            stock_pool_df: 动态股票池
            min_percentile: 市值最低百分位阈值
        """
        mv_df = self.get_raw_field('circ_mv')
        mv_df = mv_df.reindex(index=stock_pool_df.index, columns=stock_pool_df.columns)
        missing = stock_pool_df & mv_df.isna()
        if missing.to_numpy().any():
            row, col = np.argwhere(missing.to_numpy())[0]
            raise ValueError(f"股票池内 circ_mv 缺失：日期={missing.index[row]}，股票={missing.columns[col]}")
        # T 日收盘后使用当日市值。

        # 1. 【屏蔽】只保留在当前股票池(stock_pool_df)中的股票市值，其余设为NaN
        valid_mv = mv_df.where(stock_pool_df)

        # 2. 【计算标准】向量化计算每日的市值分位数阈值
        # axis=1 确保了我们是按行（每日）计算分位数
        thresholds = valid_mv.quantile(min_percentile, axis=1)

        # 3. 【生成掩码】将原始市值与每日阈值进行比较
        # .lt() 是“小于”操作，axis=0 确保了 thresholds 这个Series能按行正确地广播
        mv_mask = mv_df.lt(thresholds, axis=0)

        # 4. 【应用过滤】将所有市值小于当日阈值的股票，在股票池中标记为False
        # 这是一个跨越整个DataFrame的布尔运算，极其高效
        stock_pool_df[mv_mask] = False
        self.show_stock_nums_for_per_day(f'_filter_by_market_cap',stock_pool_df)

        return stock_pool_df

    # ok 这个属于感知未来，用不得！ todo 用的时候 必须考虑 ：open_df = self.raw_dfs['open'] 要不要是后复权的
    ##
    #
    #         open_df = self.raw_dfs['open']
    #         high_df = self.raw_dfs['high']
    #         low_df = self.raw_dfs['low']
    #         pre_close_df = self.raw_dfs['pre_close']  # T日的pre_close就是T-1日的close#
    # def _filter_next_day_limit_up(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame:#这个实现铁不对！ 要基于复权数据进行才行 todo
    #     """
    #      剔除在T日开盘即一字涨停的股票。
    #     这是为了模拟真实交易约束，因为这类股票在开盘时无法买入。
    #     Args:
    #         stock_pool_df: 动态股票池DataFrame (T-1日决策，用于T日)
    #     Returns:
    #         过滤后的动态股票池DataFrame
    #     """
    #     logger.info("    应用次日涨停股票过滤...")
    #
    #     # --- 1. 数据准备与验证 ---
    #     required_data = ['open_raw', 'high_raw', 'low_raw', 'pre_close']
    #     for data_key in required_data:
    #         if data_key not in self.raw_dfs:
    #             raise RuntimeError(f"缺少行情数据 '{data_key}'，无法过滤次日涨停股票")
    #
    #     open_df = self.raw_dfs['open_raw']
    #     high_df = self.raw_dfs['high_raw']
    #     low_df = self.raw_dfs['low_raw']
    #     pre_close_df = self.raw_dfs['close_raw)'].shift(1)  # T日的pre_close就是T-1日的close
    #
    #     # --- 2. 向量化计算每日涨停价 ---
    #     # a) 创建一个与pre_close_df形状相同的、默认值为1.1的涨跌幅限制矩阵
    #     limit_rate = pd.DataFrame(1.1, index=pre_close_df.index, columns=pre_close_df.columns)
    #
    #     # b) 识别科创板(688开头)和创业板(300开头)的股票，将其涨跌幅限制设为1.2
    #     star_market_stocks = [col for col in limit_rate.columns if str(col).startswith('688')]
    #     chinext_stocks = [col for col in limit_rate.columns if str(col).startswith('300')]
    #     limit_rate[star_market_stocks] = 1.2
    #     limit_rate[chinext_stocks] = 1.2
    #
    #     # c) 计算理论涨停价 (这里不需要shift，因为pre_close已经是T-1日的信息)
    #     limit_up_price = (pre_close_df * limit_rate).round(2)
    #
    #     # --- 3. 生成“开盘即涨停”的布尔掩码 (Mask) ---
    #     # 条件1: T日的开盘价、最高价、最低价三者相等 (一字板的特征)
    #     is_one_word_board = (open_df == high_df) & (open_df == low_df)
    #
    #     # 条件2: T日的开盘价大于或等于理论涨停价
    #     is_at_limit_price = open_df >= limit_up_price
    #
    #     # 最终的掩码：两个条件同时满足
    #     limit_up_mask = is_one_word_board & is_at_limit_price
    #
    #     # --- 4. 应用过滤 ---
    #     # 将在T日开盘即涨停的股票，在T日的universe中剔除
    #     # 这个操作是“未来”的，但它是良性的，因为它模拟的是“无法交易”的现实
    #     # 它不需要.shift(1)，因为我们是拿T日的状态，来过滤T日的池子
    #     stock_pool_df[limit_up_mask] = False
    #
    #     self.show_stock_nums_for_per_day('过滤次日涨停股后--final', stock_pool_df)
    #     return stock_pool_df

    # def _filter_next_day_suspended(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame: #todo 实盘的动态股票池 可能会用到 这个实现铁不对！ 要基于复权数据进行才行
    #     """
    #       剔除次日停牌股票 -
    #
    #       Args:
    #           stock_pool_df: 动态股票池DataFrame
    #
    #       Returns:
    #           过滤后的动态股票池DataFrame
    #       """
    #     if 'close' not in self.raw_dfs:
    #         raise RuntimeError(" 缺少价格数据，无法过滤次日停牌股票")
    #
    #     close_df = self.raw_dfs['close']
    #
    #     # 1. 创建一个代表“当日有价格”的布尔矩阵
    #     today_has_price = close_df.notna()
    #
    #     # 2. 创建一个代表“次日有价格”的布尔矩阵
    #     #    shift(-1) 将 T+1 日的数据，移动到 T 日的行。这就在一瞬间完成了所有“next_date”的查找
    #     #    fill_value=True 优雅地处理了最后一天，我们假设最后一天之后不会停牌
    #     tomorrow_has_price = close_df.notna().shift(-1, fill_value=True)
    #
    #     # 3. 计算出所有“次日停牌”的掩码 (Mask) （为什么要剔除！质疑自己：明天的事情我为什么要管？ 答：你不怕明天停牌卖不出去？  !!!!糟糕！，明天的事情你今天无法感知啊，这个函数必须删除
    #     #    次日停牌 = 今日有价 & 明日无价
    #     next_day_suspended_mask = today_has_price & (~tomorrow_has_price)
    #
    #     # 4. 一次性从股票池中剔除所有被标记的股票
    #     #    这个布尔运算会自动按索引对齐，应用到整个DataFrame
    #     stock_pool_df[next_day_suspended_mask] = False
    #
    #     return stock_pool_df

    # 执行 _load_dynamic_index_components 对应逻辑。
    def _load_dynamic_index_components(self, index_code: str,
                                       start_date: str, end_date: str) -> pd.DataFrame:
        """加载动态指数成分股数据"""
        # print(f"    加载 {index_code} 动态成分股数据...")

        index_file_name = index_code.replace('.', '_')
        index_data_path = get_market_data_path('index_weights') / index_file_name

        if not index_data_path.exists():
            raise ValueError(f"未找到指数 {index_code} 的成分股数据，请先运行downloader下载")

        # 直接读取分区数据，pandas会自动合并所有year=*分区
        components_df = pd.read_parquet(index_data_path)
        components_df['trade_date'] = pd.to_datetime(components_df['trade_date'])

        # 时间范围过滤
        # 大坑啊 ，start_date必须提前6个月！！！  两条数据时间跨度间隔（新老数据间隔最长可达6个月！）。后面逐日填充成分股信息：原理就是取上次数据进行填充的！
        extended_start_date = pd.Timestamp(start_date) - pd.DateOffset(months=12)
        mask = (components_df['trade_date'] >= extended_start_date) & \
               (components_df['trade_date'] <= pd.Timestamp(end_date))
        components_df = components_df[mask]

        # print(f"    成功加载符合当前回测时间段： {len(components_df)} 条成分股记录")
        return components_df

    # ok 已经解决前视偏差 在于：available_components = components_df[components_df['trade_date'] < date]
    def _build_dynamic_index_universe(self, stock_pool_df, index_code: str) -> pd.DataFrame:
        """
        【最终版】根据指定指数代码，构建动态股票池。
        该函数会自动处理简单指数和复合指数（如中证800）。
        """
        print(f"  > 正在基于指数 '{index_code}' 进行股票池过滤...")
        index_stock_pool_df = stock_pool_df.copy()

        # --- 定义指数构成规则，方便未来扩展 ---
        index_composition_rules = {
            '000906': ['000300', '000905'],  # 中证800 = 沪深300 + 中证500
            '000300': ['000300'],  # 沪深300
            '000905': ['000905'],  # 中证500
            # ... 未来可以轻松扩展更多指数，例如国证2000等
        }

        if index_code not in index_composition_rules:
            raise ValueError(f"指数 '{index_code}' 的构成规则未定义，请在 index_composition_rules 中添加。")

        # 获取构建该指数所需要的基础指数代码列表
        component_source_codes = index_composition_rules[index_code]
        self.component_loader = IndexComponentLoader(index_codes=component_source_codes)

        # --- 逐日应用过滤 ---
        for date in index_stock_pool_df.index:
            current_date_ts = pd.to_datetime(date)

            # 2. 从加载器高效获取成分股集合 (内部有缓存，速度飞快)
            daily_components = self.component_loader.get_members_on_date(current_date_ts, component_source_codes)
            # print(f"基础数据每天目标指数内的股票数量{len(daily_components)}")
            if not daily_components:
                index_stock_pool_df.loc[date, :] = False
                continue

            # 3. 应用过滤 (布尔掩码逻辑)
            current_mask = index_stock_pool_df.loc[date]
            index_mask = index_stock_pool_df.columns.isin(daily_components)
            index_stock_pool_df.loc[date, :] = current_mask & index_mask
            # print(f"对齐后每天目标指数内的股票数量{ index_stock_pool_df.loc[date, :].sum()}")

        return index_stock_pool_df

    # 执行 get_universe 对应逻辑。
    def get_universe(self) -> pd.DataFrame:
        """获取股票池"""
        return self.stock_pool_df

    # 执行 get_stock_codes 对应逻辑。
    def get_stock_codes(self) -> pd.DataFrame:
        return self.get_raw_field('close_raw').columns.tolist()

    # 执行 get_namechange_data 对应逻辑。
    def get_namechange_data(self) -> pd.DataFrame:
        """获取name改变的数据"""
        namechange_path = get_market_data_path('namechange.parquet')
        namechange_df = pd.read_parquet(namechange_path)
        namechange_df['ann_date'] = pd.to_datetime(namechange_df['ann_date'])
        namechange_df = namechange_df.sort_values(by=['ts_code', 'ann_date'], inplace=False)
        return namechange_df



    # 执行 save_data_summary 对应逻辑。
    def save_data_summary(self, output_dir: str):
        """保存数据摘要"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存股票池统计
        universe_stats = {
            'daily_count': self.stock_pool_df.sum(axis=1),
            'stock_coverage': self.stock_pool_df.sum(axis=0)
        }

        summary_path = os.path.join(output_dir, 'data_summary.xlsx')
        with pd.ExcelWriter(summary_path) as writer:
            # 每日股票数统计
            universe_stats['daily_count'].to_frame('stock_count').to_excel(
                writer, sheet_name='daily_stock_count'
            )

            # 股票覆盖统计
            universe_stats['stock_coverage'].to_frame('coverage_days').to_excel(
                writer, sheet_name='stock_coverage'
            )

            # 数据质量报告
            quality_report = []
            for field_name, df in [("close_raw", self.get_raw_field('close_raw'))]:
                quality_report.append({
                    'field': field_name,
                    'shape': f"{df.shape[0]}x{df.shape[1]}",
                    'missing_ratio': f"{df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%}",
                    'valid_ratio': f"{df.notna().sum().sum() / (df.shape[0] * df.shape[1]):.2%}"
                })

            pd.DataFrame(quality_report).to_excel(
                writer, sheet_name='data_quality', index=False
            )

        print(f"数据摘要已保存到: {summary_path}")

    # 执行 show_stock_nums_for_per_day 对应逻辑。
    def show_stock_nums_for_per_day(self, describe_text, pool_df:None, simplePrint :bool=True):
        daily_count = pool_df.sum(axis=1)
        logger.info(f"    {describe_text}动态股票池:")
        logger.info(f"      平均每日股票数: {daily_count.mean():.0f} --- 最少每日股票数: {daily_count.min():.0f} --- 最多每日股票数: {daily_count.max():.0f}")

        if simplePrint:
           return
        total_cells = pool_df.size
        valid_cells = (pool_df != False).sum().sum()
        coverage = valid_cells / total_cells if total_cells > 0 else 0
        logger.info(f"  {describe_text}: 后形状 {pool_df.shape}, 为true状态股票覆盖度 {coverage:.1%}")

    # 执行 get_cal_require_base_fields_for_composite 对应逻辑。
    def get_cal_require_base_fields_for_composite(self, name):
        factor_config = self.get_factor_definition(name)
        if factor_config.empty:
            raise ValueError(f"factor_definition 中不存在因子: {name}")
        action = factor_config['action'].iloc[0]
        if action != 'composite':
            raise ValueError(f"因子 {name} 不是等权复合因子，实际 action={action!r}")
        return factor_config['cal_require_base_fields'].iloc[0]
    # ok #ok
    def create_stock_pool(self, stock_pool_config_profile, pool_name):
        """按原有顺序过滤，返回每日可参与研究的股票掩码。"""
        logger.info(f"  构建{pool_name}动态股票池...")
        # T 日有正成交额的股票进入信号日候选池。
        amount = self.get_raw_field('amount')
        pool = amount.reindex(self.trading_dates).gt(0)
        index_config = stock_pool_config_profile.get('index_filter', {})
        if index_config.get('enable', False):
            pool = self._build_dynamic_index_universe(pool, index_config['index_code'])
            pool = pool.loc[:, pool.any(axis=0)]
        filters = stock_pool_config_profile['filters']
        if 'history_days' not in filters:
            raise ValueError("股票池 filters 缺少必填字段 history_days。")
        pool = self._filter_by_history_days(pool, filters['history_days'])
        if filters['remove_st']:
            pool = self._filter_st_stocks(pool)
        # 分位数依赖此前过滤后的股票池，流动性和市值过滤不可交换。
        if filters.get('min_liquidity_percentile', 0) > 0:
            pool = self._filter_by_liquidity(pool, filters['min_liquidity_percentile'])
        if filters.get('min_market_cap_percentile', 0) > 0:
            pool = self._filter_by_market_cap(pool, filters['min_market_cap_percentile'])
        self.show_stock_nums_for_per_day(f'{pool_name}最终股票池', pool,false)
        return pool

    def get_which_field_of_factor_definition_by_factor_name(self, factor_name, which_field):
        cur_factor_definition = self.get_factor_definition(factor_name)
        return cur_factor_definition[which_field]

    # 执行 get_factor_definition_df 对应逻辑。
    def get_factor_definition_df(self):
        return pd.DataFrame(self.config['factor_definition'])

    # 执行 is_composite_factor 对应逻辑。
    def is_composite_factor(self, factor_name):
        factor_config = self.get_factor_definition(factor_name)
        if factor_config.empty:
            raise ValueError(f"factor_definition 中不存在因子: {factor_name}")
        return factor_config['action'].iloc[0] == 'composite'

    # 执行 get_pool_profiles 对应逻辑。
    def get_pool_profiles(self):
        return self.config['stock_pool_profiles']

    # 执行 get_pool_profile_by_pool_name 对应逻辑。
    def get_pool_profile_by_pool_name(self, pool_name):
        return self.get_pool_profiles()[pool_name]

    # 执行 get_stock_pool_storage_name_by_name 对应逻辑。
    def get_stock_pool_storage_name_by_name(self, name):
        """结果存储使用指数代码；未启用指数过滤时使用股票池名称。"""
        profile = self.get_pool_profile_by_pool_name(name)
        index_filter = profile['index_filter']
        if not index_filter.get('enable', False):
            return name
        return self.get_stock_pool_index_code_by_name(name)

    # 执行 get_stock_pool_index_code_by_name 对应逻辑。
    def get_stock_pool_index_code_by_name(self, name):
        index_filter = self.get_pool_profile_by_pool_name(name)['index_filter']
        index_code = index_filter.get('index_code')
        if not isinstance(index_code, str) or not index_code:
            raise ValueError(
                f"股票池 {name!r} 未配置非空 index_code，无法执行指数过滤或 Beta 计算。"
            )
        return index_code

    # 执行 get_factor_definition 对应逻辑。
    def get_factor_definition(self, factor_name):
        all_df = self.get_factor_definition_df()
        return all_df[all_df['name'] == factor_name]

    # 执行 get_target_factors_for_evaluation 对应逻辑。
    def get_target_factors_for_evaluation(self):
        return self.get_experiments_factor_names()

    # 执行 get_experiments_factor_names 对应逻辑。
    def get_experiments_factor_names(self):
        return [experiment['factor_name'] for experiment in self.config['experiments']]

    # 执行 get_need_product_pool_name 对应逻辑。
    def get_need_product_pool_name(self):
        self.get_experiments_factor_names()
        pass


# 执行 fill_self 对应逻辑。
def fill_self(factor_name, df, _existence_matrix):
    # 步骤2: 根据配置字典，应用填充策略
    # =================================================================
    strategy = FACTOR_FILL_CONFIG_FOR_STRATEGY.get(factor_name)
    df = df.copy(deep=True)

    if strategy is None:
        raise KeyError(f"因子 '{factor_name}' 的填充策略未在 FACTOR_FILL_CONFIG 中定义！请添加。")

    # logger.info(f"  > 正在对因子 '{factor_name}' 应用 '{strategy}' 填充策略...")
    if strategy == FILL_STRATEGY_FFILL_UNLIMITED:
        # 前向填充：适用于价格、市值、估值、行业等
        # 这些值在股票不交易时，应保持其最后一个已知值
        return df.ffill()

    elif strategy == FILL_STRATEGY_CONDITIONAL_ZERO:
        # 填充为0：适用于成交量、换手率等交易行为数据
        # 不交易的日子，这些指标的真实值就是0
        if _existence_matrix is not None:
            return df.where(_existence_matrix,
                            0)  # _existence_matrix为false（意味着无法交易（可能是停牌停牌导致的 将原值以及nan统统写为0 /无容置疑：但凡非交易的，这类数据（交易行为类 (换手率, 成交量, 振幅)） 缺失可以直接填0
        return df  # 不填充~
    elif strategy == FILL_STRATEGY_FFILL_LIMIT_5:
        return df.ffill(limit=5)
    elif strategy == FILL_STRATEGY_FFILL_LIMIT_65:
        return df.ffill(limit=65)

    elif strategy == FILL_STRATEGY_NONE:
        # 不填充：适用于计算出的技术因子
        # 如果因子因为数据不足而无法计算，就不应凭空创造它的值
        return df

    raise RuntimeError(f"此因子{factor_name}没有指明频率，无法进行填充")


# 对于 是先 fill 还是先where 的考量 ：还是别先ffill了：极端例子：停牌了99天的，100。 若先ffill那么 这100天都是借来的数据！  如果先where。那么直接统统nan了。在ffill也是nan，更具真实
# 跟stock——pool对齐，这是铁的防线！，因为市场环境：1000只股票。可能就50能交易的，。我们不跟可交易股票池进行对齐，那么后面的ic、分组，用上无相关的950的股票池做计算，那有什么用，所以一定要对齐过滤！！
def fill_and_align_by_stock_pool(factor_name=None, df=None,
                                 stock_pool_df: pd.DataFrame = None,
                                 _existence_matrix: pd.DataFrame = None):  # 这个只是用于填充pct_chg这类数据的决策判断
    # 当前执行路径仅按股票池对齐和过滤，下面的 fill_self 调用仍被注释，不会填充缺失值。
    if stock_pool_df is None or stock_pool_df.empty:
        raise ValueError("stock_pool_df 必须传入且不能为空的 DataFrame")
    # 定义不同类型数据的填充策略

    # df = fill_self(factor_name, df, _existence_matrix)
    # 步骤1: 对齐到修剪后的股票池 对齐到主模板（stock_pool_df的形状）
    return my_align(df, stock_pool_df)


# 执行 my_align 对应逻辑。
def my_align(df, stock_pool_df):
    # 步骤1: 对齐到修剪后的股票池 对齐到主模板（stock_pool_df的形状）
    aligned_df = df.reindex(index=stock_pool_df.index, columns=stock_pool_df.columns)
    aligned_df = aligned_df.sort_index()
    aligned_df = aligned_df.where(stock_pool_df)
    return aligned_df


# 执行 create_data_manager 对应逻辑。
def create_data_manager(config_path: str) -> DataManager:
    """
    创建数据管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DataManager实例
    """
    return DataManager(_load_file(config_path))

# if __name__ == '__main__':
#     # dataManager_temp = DataManager(
#     #     "../factory/config.yaml",
#     #     need_data_deal=False
#     # )
#     #
#     # calculate_rolling_beta(
#     #     dataManager_temp.config_manager['research_window']['start_date'],
#     #     dataManager_temp.config_manager['research_window']['end_date'],
#     #     dataManager_temp.get_pool_of_factor_name_of_stock_codes('beta')
#     # )
