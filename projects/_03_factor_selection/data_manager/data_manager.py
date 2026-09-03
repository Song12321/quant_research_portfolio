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
from projects._03_factor_selection.config_manager.base_config import experiments_yaml_path, config_yaml_path
from projects._03_factor_selection.config_manager.function_load.debug_temp_fast_config import IS_DEBUG_TEMP
from projects._03_factor_selection.config_manager.function_load.load_config_file import _load_file
from projects._03_factor_selection.config_manager.factor_info_config import FACTOR_FILL_CONFIG_FOR_STRATEGY, \
    FILL_STRATEGY_FFILL_UNLIMITED, \
    FILL_STRATEGY_CONDITIONAL_ZERO, FILL_STRATEGY_FFILL_LIMIT_5, FILL_STRATEGY_NONE, FILL_STRATEGY_FFILL_LIMIT_65
from projects._03_factor_selection.data_manager.suspend_state import (
    _aggregate_suspend_events_to_eod_states,
    _build_suspend_eod_matrix,
)
from projects._03_factor_selection.data_manager.stock_history import apply_history_days_filter
from projects._03_factor_selection.utils.IndustryMap import PointInTimeIndustryMap
from quant_lib.data_loader import DataLoader
from projects._03_factor_selection.utils.component_loader import IndexComponentLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.constant_config import LOCAL_PARQUET_DATA_DIR, permanent__day
from quant_lib.config.logger_config import setup_logger, log_warning

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logger(__name__)


def check_field_level_completeness(raw_df: Dict[str, pd.DataFrame]):
    dfs = raw_df.copy()
    logger.info("原始字段缺失率体检报告:")
    for item_name, df in dfs.items():
        # missing_rate_daily = df.isna().mean(axis=1)

        # logger.info(f"{item_name}因子缺失率最高的10天 between {first_date} and {end_date}")
        # logger.info(f"{missing_rate_daily.sort_values(ascending=False).head(10)}")  # 其实也不需要太看重，只能说是辅助日志，如果总缺失率高 可以看看整个辅助排查而已！

        # 计算每只股票（每一列）的缺失率(相当于看这股票 在这一段时间的完整率！---》推导：最后一天才上市！，那么缺失率可能高达99.99% 所以不需要看重这个！)  注释掉
        missing_rate_per_stock = df.isna().mean(axis=0)
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
                 'close_raw', 'open_raw', 'high_raw', 'low_raw', 'vol_raw',
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

    RESIDENT_RAW_FIELDS = ('close_hfq', 'circ_mv', 'turnover_rate', 'list_date')

    def __init__(self, config_path: str=config_yaml_path, experiments_config_path: str=experiments_yaml_path, need_data_deal: bool = True):
        """
        初始化数据管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.st_matrix = None  # 注意 后续用此字段，需要注意前视偏差
        self._tradeable_matrix_by_suspend_resume = None
        self.config = _load_file(config_path)
        self.experiments_config = _load_file(experiments_config_path)
        self.research_start_date = self.config['research_window']['start_date']
        self.research_end_date = self.config['research_window']['end_date']
        self.buffer_start_date = None
        if need_data_deal:
            configured_data_root = Path(self.config['data_root']).resolve()
            if configured_data_root != LOCAL_PARQUET_DATA_DIR.resolve():
                raise ValueError(
                    "data_root 与当前仍由专项加载器使用的数据目录不一致: "
                    f"configured={configured_data_root}, expected={LOCAL_PARQUET_DATA_DIR.resolve()}"
                )
            self.data_loader = DataLoader(data_path=configured_data_root)
            self.raw_dfs = {}
            self.temporary_raw_dfs = {}
            self.stock_pools_dict = None
            self.trading_dates = self.data_loader.get_trading_dates(self.research_start_date, self.research_end_date)
            self.buffer_start_date = self._resolve_buffer_start_date()
            self._prebuffer_trading_dates = self.data_loader.get_trading_dates(self.buffer_start_date,
                                                                               self.research_end_date)
            self._existence_matrix = None
            self.pit_map = None

            self.component_loader = IndexComponentLoader()

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
        buffer_start = pd.Timestamp(prior_dates.iloc[-preheat_days])
        logger.info(
            f"预热期已解析: factor_days={max(factor_days)}, "
            f"selected_days={preheat_days}, "
            f"buffer_start={buffer_start.date()}"
        )
        return buffer_start.strftime('%Y%m%d')

    def _get_factor_preheat_trading_days(self, factor_name: str, ancestors: set[str]) -> int:
        if factor_name in ancestors:
            raise ValueError(f"复合因子依赖存在循环: factor={factor_name}")
        definitions = self.get_factor_definition(factor_name)
        if len(definitions) != 1:
            raise ValueError(f"因子预热期配置必须唯一: factor={factor_name}, count={len(definitions)}")
        definition = definitions.iloc[0]
        if definition['action'] == 'composite':
            children = definition['cal_require_base_fields']
            if not isinstance(children, list) or not children:
                raise ValueError(f"复合因子缺少非空子因子列表: factor={factor_name}")
            next_ancestors = ancestors | {factor_name}
            return max(self._get_factor_preheat_trading_days(name, next_ancestors) for name in children)
        if 'preheat_trading_days' not in definition.index:
            raise ValueError(f"基础因子缺少 preheat_trading_days: factor={factor_name}")
        preheat_days = definition['preheat_trading_days']
        if isinstance(preheat_days, bool) or not isinstance(preheat_days, (int, np.integer)) or preheat_days <= 0:
            raise ValueError(
                f"基础因子 preheat_trading_days 必须为正整数: "
                f"factor={factor_name}, actual={preheat_days!r}"
            )
        return int(preheat_days)

    def prepare_basic_data(self) -> Dict[str, pd.DataFrame]:
        """
        优化的两阶段数据处理流水线（只加载一次数据）
        Returns:
            处理后的数据字典
        """

        self.clear_temporary_raw_fields()
        self.raw_dfs = self.data_loader.get_raw_dfs_by_require_fields(fields=list(self.RESIDENT_RAW_FIELDS),
                                                                      buffer_start_date=self.buffer_start_date,
                                                                      end_date=self.research_end_date)
        # 加载辅助数据，
        self.pit_map = PointInTimeIndustryMap()  # 它能自动加载数据

        check_field_level_completeness(self.raw_dfs)
        logger.info(f"raw_dfs加载完成，共加载 {len(self.raw_dfs)} 个字段")

        # === 第一阶段：基于已加载数据构建权威股票池 ===
        logger.info("第一阶段：构建两个权威股票池（各种过滤！）")
        if (IS_DEBUG_TEMP):
            log_warning("debug 用于快速测试数据！ 不加载繁琐的股票池")
            return None
        self._build_stock_pools_from_loaded_data(self.research_start_date, self.research_end_date)
        # 强行检查一下数据！完整率！ 不应该在这里检查！，太晚了， 已经被stock_pool_df 动了手脚了（低市值的会被置为nan，

    def can_load_raw_field(self, field_name: str) -> bool:
        """判断字段是否已就绪或有明确的本地数据源。"""
        if not isinstance(field_name, str) or not field_name:
            raise TypeError(f"field_name 必须是非空字符串，实际为 {field_name!r}")
        return (
            field_name in self.raw_dfs or
            field_name in self.temporary_raw_dfs or
            field_name in self.data_loader.field_map
        )

    def get_raw_field(self, field_name: str) -> pd.DataFrame:
        """按当前研究窗口临时加载原始字段，并对齐到常驻价格网格。"""
        if not self.can_load_raw_field(field_name):
            raise ValueError(f"无法加载原始字段: field={field_name}, 原因=未找到数据源")
        if field_name in self.temporary_raw_dfs:
            return self.temporary_raw_dfs[field_name]

        loaded = self.data_loader.get_raw_dfs_by_require_fields(
            fields=[field_name],
            buffer_start_date=self.buffer_start_date,
            end_date=self.research_end_date,
        )
        if set(loaded) != {field_name}:
            raise RuntimeError(
                f"原始字段加载结果违反契约: field={field_name}, loaded={sorted(loaded)}"
            )
        check_field_level_completeness(loaded)
        base_df = self.raw_dfs.get('close_hfq')
        if base_df is None:
            raise RuntimeError("临时原始字段对齐失败: 常驻字段 close_hfq 未加载")
        aligned_df = loaded[field_name].reindex(index=base_df.index, columns=base_df.columns)
        self.temporary_raw_dfs[field_name] = aligned_df
        return aligned_df

    def clear_temporary_raw_fields(self) -> None:
        """释放当前因子临时读取的原始宽表。"""
        cleared_count = len(self.temporary_raw_dfs)
        self.temporary_raw_dfs.clear()
        if cleared_count:
            logger.info(f"临时原始字段已清理，释放了 {cleared_count} 个宽表")

    # ok
    def _build_stock_pools_from_loaded_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        第一阶段：基于已加载的数据构建权威股票池

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            权威股票池DataFrame
        """
        # print("1. 验证股票池构建所需数据...")

        # 验证必需字段是否已加载
        required_fields_for_universe = ['close_hfq', 'circ_mv', 'turnover_rate', 'list_date']
        missing_fields = [field for field in required_fields_for_universe if field not in self.raw_dfs]

        if missing_fields:
            raise ValueError(f"构建股票池缺少必需字段: {missing_fields}")

        self.build_diff_stock_pools()

    def build_diff_stock_pools(self):
        stock_pool_df_dict = {}
        stock_pool_profiles = self.config['stock_pool_profiles']
        # 按需生成动态股票池
        experiments_pool_names = self.get_experiments_pool_names()
        for pool_name in experiments_pool_names:
            pool_config = stock_pool_profiles[pool_name]
            product_universe = self.create_stock_pool(pool_config, pool_name)
            stock_pool_df_dict[pool_name] = product_universe
        self.stock_pools_dict = stock_pool_df_dict

    # institutional_profile   = stock_pool_profiles['institutional_profile']#为“基本面派”和“趋势派”因子，提供一个高市值、高流动性的环境
    # microstructure_profile = stock_pool_profiles['microstructure_profile']#用于 微观（量价/情绪）因子
    # product_universe =self.product_universe (microstructure_profile,trading_dates)

    def _check_data_quality(self):
        """检查数据质量"""
        print("  检查数据完整性和质量...")

        for field_name, df in self.raw_dfs.items():
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
        list_date_panel = self.raw_dfs.get('list_date')
        delist_date_panel = self.raw_dfs.get('delist_date')

        if list_date_panel is None or delist_date_panel is None:
            raise ValueError("缺少'list_date'或'delist_date'面板数据，无法构建存在性矩阵。")

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

    def build_tradeable_matrix_by_suspend_resume(
            self,
    ) -> pd.DataFrame:
        """
        根据完整停复牌历史，构建每日收盘后的可交易状态矩阵。

        该矩阵随后 shift(1)，仅用于 T-1 日终状态对 T 日开盘决策的约束。
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

    # ok
    def build_st_period_from_namechange(
            self,
    ) -> pd.DataFrame:
        """
         根据namechange历史，重建每日“已知风险”状态矩阵。
         此版本通过searchsorted隐式处理初始状态，逻辑最简且结果正确。
         """
        if self.st_matrix is not None:
            logger.info("self.st_matrix 之前已经被初始化，无需再次加载（这是全量数据，一次加载即可")
            return self.st_matrix
        logger.info("正在根据名称变更历史，重建每日‘已知风险’状态st矩阵...")
        # 数据准备 获取所有股票和交易日期
        ts_codes = list(set(self.get_stock_codes()))
        trading_dates = self.data_loader.get_trading_dates(start_date=self.research_start_date,
                                                           end_date=self.research_end_date)
        namechange_df = self.get_namechange_data()

        # --- 1. 准备工作 ---
        if not trading_dates._is_monotonic_increasing:
            trading_dates = trading_dates.sort_values(ascending=True)

        # 【关键】必须按“生效日”排序，以确保状态的正确延续和覆盖
        namechange_df['start_date'] = pd.to_datetime(namechange_df['start_date'])
        namechange_df = namechange_df.sort_values(by=['ts_code', 'start_date'], inplace=False)

        # 【关键】必须用 np.nan 初始化，作为“未知状态”
        st_matrix = pd.DataFrame(np.nan, index=trading_dates, columns=ts_codes)

        # --- 2. “打点”：一个循环处理所有历史事件 ---
        for ts_code, group in namechange_df.groupby('ts_code'):
            group_sorted = group.sort_values(by='start_date')
            for _, row in group_sorted.iterrows():
                start_date = row['start_date']

                # 发生在回测期前的日期，会被自动映射到位置 0  or 发生在回测期内的日期，会被映射到它对应的正确位置
                start_date_loc = trading_dates.searchsorted(start_date,
                                                            side='left')  # 遍历trading_dates找到首个>=start_date的下标！ 如果是rigths ：则首个>的下标

                # 只处理那些能影响到我们回测周期的事件
                if start_date_loc < len(trading_dates):
                    name_upper = row['name'].upper()
                    is_risk_event = 'ST' in name_upper or name_upper.startswith('S')
                    # 使用.iloc进行赋值
                    start_trade_date = pd.DatetimeIndex(trading_dates)[start_date_loc]
                    st_matrix.loc[start_trade_date, ts_code] = is_risk_event

        # --- 3. “传播”与“收尾” ---
        st_matrix = st_matrix.ffill(inplace=False)
        st_matrix = st_matrix.fillna(False, inplace=False)

        logger.info("每日‘已知风险’状态矩阵重建完毕。")
        self.st_matrix = st_matrix.astype(bool)
        return self.st_matrix

    # 使用截至 T-1 收盘可观察到的有效交易日数，避免把 T 日收盘价用于 T 日选股。
    def _filter_by_history_days(self, stock_pool_df: pd.DataFrame, history_days: int) -> pd.DataFrame:
        filtered_pool = apply_history_days_filter(
            stock_pool_df,
            self.raw_dfs.get('close_hfq'),
            history_days,
        )
        if history_days:
            self.show_stock_nums_for_per_day(f"有效交易日少于{history_days}天的过滤", filtered_pool)
        return filtered_pool

    # ok 已经处理前视偏差
    def _filter_st_stocks(self, stock_pool_df: pd.DataFrame) -> pd.DataFrame:
        if self.st_matrix is None:
            raise ValueError("    警告: 未能构建ST状态矩阵，无法过滤ST股票。")
        # 【核心】将“历史真相”矩阵整体向前（未来）移动一天。 (因为st_matrix 是以据生效start_Day日计算的。t下单，只能用t-1的数据跑，t单日的st无法感知！
        # 这确保了我们在T日做决策时，看到的是T-1日的真实状态 。
        st_mask_shifted = self.st_matrix.shift(1, fill_value=False)
        # 对齐两个DataFrame的索引和列，确保万无一失
        # join='left' 表示以stock_pool_df的形状为准
        aligned_universe, aligned_st_status = stock_pool_df.align(st_mask_shifted, join='left',
                                                                  fill_value=False)  # 至少做 行列 保持一致的对齐。 下面才做赋值！ #fill_value=False ：st_Df只能对应一部分的股票池_Df.股票池_Df剩余的行列 用false填充！

        # 将ST的股票从universe中剔除
        # aligned_st_status为True的地方，在universe中就应该为False
        aligned_universe[aligned_st_status] = False

        # 统计过滤效果
        original_count = stock_pool_df.sum(axis=1).mean()
        filtered_count = aligned_universe.sum(axis=1).mean()
        st_filtered_count = original_count - filtered_count
        print(f"      ST股票过滤: 平均每日剔除 {st_filtered_count:.0f} 只ST股票")
        self.show_stock_nums_for_per_day(f'by_ST状态(判定来自于name的变化历史)_filter', aligned_universe)

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

        # 2. 【核心】应用T-1原则
        #    将整个“存在性”状态矩阵向前移动一天。
        #    这样在T日决策时，使用的就是T-1日该股票是否存在的信息。
        existence_mask_shifted = existence_matrix.shift(1, fill_value=False)

        # 3. 安全对齐并应用过滤器
        #    fill_value=False 表示，如果一个股票在您的基础池中，
        #    但不在我们的存在性矩阵的考虑范围内，我们默认它不存在。
        aligned_pool, aligned_existence_mask = stock_pool_df.align(
            existence_mask_shifted,
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
        if self._tradeable_matrix_by_suspend_resume is None:
            raise ValueError("警告: 未能构建 _tradeable_matrix_by_suspend_resume 状态矩阵。")

        # 矩阵 T 行表示 T 日收盘后的状态，不能用于 T 日开盘前的选股决策。
        # shift(1) 后，T 日股票池只读取 T-1 日终状态；首行没有更早研究期行，
        # 延续既有约定填 True。该掩码不是 T 日盘中任意时刻的实际成交能力。
        tradeable_mask_shifted = self._tradeable_matrix_by_suspend_resume.shift(1, fill_value=True)

        # 以股票池为左侧边界，避免停复牌数据增加或删除股票池的股票集合；
        # 未出现停复牌状态的单元格没有不可交易证据，延续原逻辑填 True。
        aligned_universe, aligned_tradeable_mask = stock_pool_df.align(
            tradeable_mask_shifted,
            join='left',
            fill_value=True
        )

        # 统计过滤前的数量
        pre_filter_count = aligned_universe.sum().sum()

        # 停复牌过滤只能从既有股票池中剔除，不得把其他过滤器已排除的股票重新加入。
        final_pool = aligned_universe & aligned_tradeable_mask

        # 统计过滤后的数量
        post_filter_count = final_pool.sum().sum()
        filtered_out_count = pre_filter_count - post_filter_count
        logger.info(f"      停牌股票过滤: 共剔除 {filtered_out_count:.0f} 个停牌的股票-日期对。")
        self.show_stock_nums_for_per_day('过滤停牌股后', final_pool)
        return final_pool

    # ok
    def _filter_by_liquidity(self, stock_pool_df: pd.DataFrame, min_percentile: float) -> pd.DataFrame:
        """按流动性过滤 """
        if 'turnover_rate' not in self.raw_dfs:
            raise RuntimeError("缺少换手率数据，无法进行流动性过滤")

        turnover_df = self.raw_dfs['turnover_rate']
        # 【关键】股票池构建的时间逻辑：
        # - 我们要构建T日的股票池（决定T日哪些股票可交易）
        # - 但判断依据必须基于T-1及更早的信息
        # - 因此这里需要shift(1)来获取T-1的换手率用于T日的决策
        turnover_df = turnover_df.shift(1)

        # 1. 【确定样本】只保留 stock_pool_df 中为 True 的换手率数据
        # “只对当前股票池计算”
        valid_turnover = turnover_df.where(stock_pool_df)

        # 2. 【计算标准】沿行（axis=1）一次性计算出每日的分位数阈值
        thresholds = valid_turnover.quantile(min_percentile, axis=1)

        # 3. 【应用标准】将原始换手率与每日阈值进行比较，生成过滤掩码
        low_liquidity_mask = turnover_df.lt(thresholds, axis=0)

        # 4. 将需要剔除的股票在 stock_pool_df 中设为 False
        stock_pool_df[low_liquidity_mask] = False
        self.show_stock_nums_for_per_day(f'by_剔除流动性低的_filter', stock_pool_df)

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

        Returns:
            过滤后的动态股票池
        """
        if 'circ_mv' not in self.raw_dfs:
            raise RuntimeError("缺少市值数据，无法进行市值过滤")

        mv_df = self.raw_dfs['circ_mv']
        # 【关键】同样的逻辑：用T-1的市值数据来决定T日的股票池
        mv_df = mv_df.shift(1)

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
        self.show_stock_nums_for_per_day(f'by_剔除市值低的_filter', stock_pool_df)

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

    def _load_dynamic_index_components(self, index_code: str,
                                       start_date: str, end_date: str) -> pd.DataFrame:
        """加载动态指数成分股数据"""
        # print(f"    加载 {index_code} 动态成分股数据...")

        index_file_name = index_code.replace('.', '_')
        index_data_path = LOCAL_PARQUET_DATA_DIR / 'index_weights' / index_file_name

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

        # --- 逐日应用过滤 ---
        for date in index_stock_pool_df.index:
            current_date_ts = pd.to_datetime(date)

            # 1. 【安全港原则】获取 T-1 日收盘后的成分股列表，作为 T 日的股票池
            #    我们直接查询 T-1 日的成分股即可。 （我倒要看看你昨天在不在。
            prev_date = current_date_ts - pd.Timedelta(days=1)

            # 2. 从加载器高效获取成分股集合 (内部有缓存，速度飞快)
            daily_components = self.component_loader.get_members_on_date(prev_date, component_source_codes)
            # print(f"基础数据每天目标指数内的股票数量{len(daily_components)}")
            if not daily_components:  # 如果当天（T-1）获取不到成分股，则当天股票池为空
                index_stock_pool_df.loc[date, :] = False
                continue

            # 3. 应用过滤 (布尔掩码逻辑)
            current_mask = index_stock_pool_df.loc[date]
            index_mask = index_stock_pool_df.columns.isin(daily_components)
            index_stock_pool_df.loc[date, :] = current_mask & index_mask
            # print(f"对齐后每天目标指数内的股票数量{ index_stock_pool_df.loc[date, :].sum()}")

        # 函数结尾的日志打印 ()
        self.show_stock_nums_for_per_day(f'by_成分股指数_filter{index_code}', index_stock_pool_df)
        return index_stock_pool_df

    def get_universe(self) -> pd.DataFrame:
        """获取股票池"""
        return self.stock_pool_df

    def get_stock_codes(self) -> pd.DataFrame:
        first_df = next(iter(self.raw_dfs.values()))  # 取第一个 DataFrame
        return first_df.columns.tolist()

    def get_namechange_data(self) -> pd.DataFrame:
        """获取name改变的数据"""
        namechange_path = LOCAL_PARQUET_DATA_DIR / 'namechange.parquet'

        return pd.read_parquet(namechange_path)

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
            for field_name, df in self.raw_dfs.items():
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

    def show_stock_nums_for_per_day(self, describe_text, pool_df):
        daily_count = pool_df.sum(axis=1)
        logger.info(f"    {describe_text}动态股票池:")
        logger.info(f"      平均每日股票数: {daily_count.mean():.0f}")
        logger.info(f"      最少每日股票数: {daily_count.min():.0f}")
        logger.info(f"      最多每日股票数: {daily_count.max():.0f}")
        # 统计过滤后的覆盖度
        total_cells = pool_df.size
        valid_cells = (pool_df != False).sum().sum()
        coverage = valid_cells / total_cells if total_cells > 0 else 0
        logger.info(f"  {describe_text}: 后形状 {pool_df.shape}, 为true状态股票覆盖度 {coverage:.1%}")

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
        """
                构建动态股票池
                Returns:
                    股票池DataFrame，True表示该股票在该日期可用
                """
        logger.info(f"  构建{pool_name}动态股票池...")
        # 第一步：基础股票池 - 有价格数据的股票
        if 'close_hfq' not in self.raw_dfs:
            raise ValueError("缺少价格数据，无法构建股票池")

        # 【简化修复】价格数据的连续性已经隐含处理了退市股票，无需重复过滤

        # 基于T-1日的价格数据构建股票池
        close_raw_shifted = self.raw_dfs['close_hfq'].shift(1)  # 使用T-1日的收盘价信息
        final_stock_pool_df = close_raw_shifted.notna()  # T-1日有收盘价的股票，T日可以考虑交易
        final_stock_pool_df = final_stock_pool_df.reindex(self.trading_dates)
        self.show_stock_nums_for_per_day('根据收盘价notna生成的', final_stock_pool_df)

        # 注释掉存在性过滤，因为它与价格过滤重复
        # if stock_pool_config_profile.get('remove_not_existence', True):
        #     final_stock_pool_df = self._filter_by_existence(final_stock_pool_df)
        # 第二步：各种过滤！
        # --基础过滤 指数成分股过滤（如果启用）
        index_config = stock_pool_config_profile.get('index_filter', {})
        if index_config.get('enable', False):
            # print(f"    应用指数过滤: {index_config['index_code']}")
            final_stock_pool_df = self._build_dynamic_index_universe(final_stock_pool_df, index_config['index_code'])
            # ✅ 在这里进行列修剪是合理的！ 因为中证800成分股是基于外部规则，不是基于未来数据表现
            valid_stocks = final_stock_pool_df.columns[final_stock_pool_df.any(axis=0)]
            final_stock_pool_df = final_stock_pool_df[valid_stocks]
            logger.info(
                f"根据指数裁剪股票池：形状：{final_stock_pool_df.shape} 二：为true状态股票覆盖度:{(final_stock_pool_df != False).sum().sum() / final_stock_pool_df.size}")
        # 其他各种指标过滤条件
        universe_filters = stock_pool_config_profile['filters']

        # --普适性 过滤 （通用过滤）
        if 'history_days' not in universe_filters:
            raise ValueError("股票池 filters 缺少必填字段 history_days。")
        final_stock_pool_df = self._filter_by_history_days(
            final_stock_pool_df,
            universe_filters['history_days'],
        )
        if universe_filters['remove_st']:
            # 构建ST矩阵
            self.build_st_period_from_namechange()
            final_stock_pool_df = self._filter_st_stocks(final_stock_pool_df)  # 剔除ST股票
        if universe_filters['adapt_tradeable_matrix_by_suspend_resume']:
            # 基于停复牌事件构建的可交易的池子
            self.build_tradeable_matrix_by_suspend_resume()
            final_stock_pool_df = self._filter_tradeable_matrix_by_suspend_resume(final_stock_pool_df)

        # 2. 流动性过滤
        if universe_filters.get('min_liquidity_percentile', 0) > 0:
            # print("    应用流动性过滤...")
            final_stock_pool_df = self._filter_by_liquidity(
                final_stock_pool_df,
                universe_filters['min_liquidity_percentile']
            )

        # 3. 市值过滤
        if universe_filters.get('min_market_cap_percentile', 0) > 0:
            # print("    应用市值过滤...")
            final_stock_pool_df = self._filter_by_market_cap(
                final_stock_pool_df,
                universe_filters['min_market_cap_percentile']
            )
        # #自定义。
        # stock_codes =  ['002913.SZ', '300677.SZ', '300827.SZ', '603072.SH', '000625.SZ', '603181.SH', '300279.SZ', '601857.SH', '002635.SZ', '300808.SZ', '300399.SZ', '600488.SH', '603399.SH', '600566.SH', '300396.SZ', '300192.SZ', '600803.SH', '688179.SH', '002195.SZ', '688590.SH', '300779.SZ', '002931.SZ', '002554.SZ', '000909.SZ', '300626.SZ', '603538.SH', '300863.SZ', '002197.SZ', '603331.SH', '600289.SH', '002991.SZ', '688196.SH', '300246.SZ', '600393.SH', '002095.SZ', '601121.SH', '301280.SZ', '000886.SZ', '600857.SH', '301139.SZ', '600527.SH', '600691.SH', '002228.SZ', '002735.SZ', '605066.SH', '000912.SZ', '002172.SZ', '300322.SZ', '300955.SZ', '003037.SZ', '600439.SH', '002629.SZ', '688356.SH', '603599.SH', '601996.SH', '601169.SH', '002043.SZ', '601688.SH', '688080.SH', '688010.SH', '301329.SZ', '300602.SZ', '300757.SZ', '600552.SH', '300057.SZ', '688543.SH', '301237.SZ', '600328.SH', '300009.SZ', '002637.SZ', '301223.SZ', '000796.SZ', '000717.SZ', '000695.SZ', '002466.SZ', '002343.SZ', '603359.SH', '301165.SZ', '300612.SZ', '002438.SZ', '300768.SZ', '301538.SZ', '300491.SZ', '301125.SZ', '600082.SH', '000635.SZ', '600990.SH', '300469.SZ', '600197.SH', '603357.SH', '603986.SH', '603668.SH', '002418.SZ', '688101.SH', '688576.SH', '301207.SZ', '605555.SH', '001209.SZ', '002601.SZ', '688362.SH', '600497.SH', '000671.SZ', '002958.SZ', '605177.SH', '301201.SZ', '600665.SH', '000005.SZ', '000065.SZ', '300694.SZ', '603616.SH', '002504.SZ', '300400.SZ', '600519.SH', '688200.SH', '688031.SH', '002201.SZ', '688141.SH', '000858.SZ', '603766.SH', '300652.SZ', '600613.SH', '601077.SH', '002283.SZ', '600338.SH', '301502.SZ', '600880.SH', '600587.SH', '300700.SZ', '300443.SZ', '688788.SH', '600789.SH', '600510.SH', '002565.SZ', '300247.SZ', '688220.SH', '300219.SZ', '600499.SH', '002251.SZ', '603009.SH', '300375.SZ', '002777.SZ', '000755.SZ', '301519.SZ', '002430.SZ', '000985.SZ', '002021.SZ', '002603.SZ', '600309.SH', '300984.SZ', '600392.SH', '603580.SH', '301072.SZ', '001965.SZ', '600985.SH', '688800.SH', '605339.SH', '688077.SH', '300329.SZ', '600301.SH', '605178.SH', '603016.SH', '601615.SH', '688176.SH', '600851.SH', '300775.SZ', '002897.SZ', '605337.SH', '600216.SH', '301607.SZ', '603027.SH', '605389.SH', '603168.SH', '300703.SZ', '301392.SZ', '300112.SZ', '600602.SH', '301628.SZ', '688353.SH', '688601.SH', '000670.SZ', '300490.SZ', '603908.SH', '000088.SZ', '688658.SH', '688271.SH', '300695.SZ', '688597.SH', '002258.SZ', '001267.SZ', '300817.SZ', '600706.SH', '600255.SH', '001286.SZ', '603706.SH', '001382.SZ', '603086.SH', '301268.SZ', '301589.SZ', '000998.SZ', '688568.SH', '002417.SZ', '600820.SH', '002281.SZ', '600405.SH', '600626.SH', '605208.SH', '300176.SZ', '600770.SH', '600993.SH', '603585.SH', '600928.SH', '000507.SZ', '688778.SH', '301269.SZ', '603768.SH', '600152.SH', '001228.SZ', '002948.SZ', '600156.SH', '688505.SH', '603218.SH', '001328.SZ', '002515.SZ', '301068.SZ', '600764.SH', '603210.SH', '600708.SH', '601211.SH', '600173.SH', '603797.SH', '600030.SH', '603801.SH', '002427.SZ', '688663.SH', '600111.SH', '300957.SZ', '600548.SH', '600874.SH', '000802.SZ', '300358.SZ', '301216.SZ', '600235.SH', '600127.SH', '300152.SZ', '301608.SZ', '300847.SZ', '603350.SH', '300377.SZ', '600769.SH', '601366.SH', '300618.SZ', '601908.SH', '002059.SZ', '002762.SZ', '601021.SH', '301590.SZ', '688582.SH', '688078.SH', '300525.SZ', '301328.SZ', '002780.SZ', '600351.SH', '603456.SH', '688388.SH', '003015.SZ', '000767.SZ', '002091.SZ', '002087.SZ', '600358.SH', '688379.SH', '600210.SH', '688613.SH', '601696.SH', '002694.SZ', '603815.SH', '300836.SZ', '600020.SH', '000411.SZ', '002275.SZ', '301251.SZ', '688981.SH', '603358.SH', '688657.SH', '600685.SH', '601665.SH', '600280.SH', '600166.SH', '002380.SZ', '603958.SH', '300518.SZ', '001322.SZ', '688047.SH', '600654.SH', '300094.SZ', '301172.SZ', '600800.SH', '601330.SH', '688090.SH', '300669.SZ', '300238.SZ', '688698.SH', '300046.SZ', '600833.SH', '603102.SH', '300674.SZ', '000547.SZ', '600753.SH', '600861.SH', '001258.SZ', '600089.SH', '300551.SZ', '301622.SZ', '605180.SH', '002080.SZ', '301359.SZ', '688233.SH', '300239.SZ', '002177.SZ', '002524.SZ', '002312.SZ', '600348.SH', '301298.SZ', '300316.SZ', '000589.SZ', '688520.SH', '000550.SZ', '002837.SZ', '300222.SZ', '300965.SZ', '300307.SZ', '001335.SZ', '300580.SZ', '000526.SZ', '600318.SH', '002187.SZ', '688625.SH', '603819.SH', '000725.SZ', '688620.SH', '600570.SH', '300975.SZ', '002247.SZ', '000895.SZ', '002600.SZ', '600885.SH', '000800.SZ', '301369.SZ', '003035.SZ', '300162.SZ', '002221.SZ', '300008.SZ', '600202.SH', '300536.SZ', '600593.SH', '301332.SZ', '603345.SH', '301189.SZ', '000828.SZ', '688668.SH', '300203.SZ', '603272.SH', '300135.SZ', '600619.SH', '601107.SH', '301021.SZ', '300868.SZ', '688607.SH', '600895.SH', '002339.SZ', '301349.SZ', '603722.SH', '002778.SZ', '002632.SZ', '002772.SZ', '600467.SH', '002462.SZ', '002759.SZ', '603458.SH', '002006.SZ', '688066.SH', '601689.SH', '301092.SZ', '300287.SZ', '603887.SH', '688281.SH', '002463.SZ', '301235.SZ', '300861.SZ', '000029.SZ', '002563.SZ', '002056.SZ', '002206.SZ', '000933.SZ', '603360.SH', '688114.SH', '688185.SH', '300180.SZ', '688636.SH', '600078.SH', '600598.SH', '300858.SZ', '603186.SH', '300006.SZ', '688184.SH', '600353.SH', '000975.SZ', '002723.SZ', '003005.SZ', '300980.SZ', '600141.SH', '001218.SZ', '002114.SZ', '002011.SZ', '601187.SH', '000757.SZ', '300260.SZ', '603279.SH', '002333.SZ', '603685.SH', '000584.SZ', '600862.SH', '002985.SZ', '002648.SZ', '688593.SH', '002299.SZ', '300630.SZ', '603353.SH', '605080.SH', '000572.SZ', '688088.SH', '688571.SH', '600276.SH', '002009.SZ', '300245.SZ', '601899.SH', '002741.SZ', '000927.SZ', '300128.SZ', '688529.SH', '002660.SZ', '002511.SZ', '301469.SZ', '600223.SH', '600077.SH', '000610.SZ', '600137.SH', '300179.SZ', '603211.SH', '600037.SH', '600343.SH', '002042.SZ', '300382.SZ', '688345.SH', '301273.SZ', '300572.SZ', '300850.SZ', '002444.SZ', '002949.SZ', '002373.SZ', '688681.SH', '300821.SZ', '300672.SZ', '600110.SH', '605117.SH', '603662.SH', '301086.SZ', '000038.SZ', '600643.SH', '605167.SH', '002922.SZ', '301108.SZ', '603155.SH', '301082.SZ', '600360.SH', '002658.SZ', '300963.SZ', '300170.SZ', '600967.SH', '002799.SZ', '301176.SZ', '002875.SZ', '605168.SH', '300075.SZ', '603115.SH', '000416.SZ', '603777.SH', '002906.SZ', '301376.SZ', '002301.SZ', '000156.SZ', '603106.SH', '601777.SH', '300153.SZ', '002261.SZ', '603786.SH', '603096.SH', '688086.SH', '605488.SH', '603711.SH', '301206.SZ', '603229.SH', '600495.SH', '603193.SH', '300633.SZ', '603176.SH', '300599.SZ', '600470.SH', '603816.SH', '300642.SZ', '300464.SZ', '300447.SZ', '300902.SZ', '300825.SZ', '600182.SH', '603829.SH', '301001.SZ', '002371.SZ', '603703.SH', '000830.SZ', '300986.SZ', '600877.SH', '603325.SH', '688053.SH', '002468.SZ', '002727.SZ', '002532.SZ', '000719.SZ', '600525.SH', '600636.SH', '002920.SZ', '601033.SH', '000876.SZ', '002740.SZ', '300767.SZ', '605128.SH', '601868.SH', '300993.SZ', '605081.SH', '002779.SZ', '603568.SH', '002823.SZ', '300835.SZ', '688158.SH', '002547.SZ', '603238.SH', '002755.SZ', '688669.SH', '600984.SH', '301069.SZ', '600791.SH', '600690.SH', '688348.SH', '300648.SZ', '603062.SH', '300267.SZ', '688565.SH', '601388.SH', '603919.SH', '301229.SZ', '688787.SH', '300916.SZ', '688380.SH', '002542.SZ', '002437.SZ', '603997.SH', '600106.SH', '601020.SH', '300689.SZ', '300852.SZ', '300099.SZ', '001380.SZ', '688132.SH', '000902.SZ', '600938.SH', '002661.SZ', '300353.SZ', '300425.SZ', '688318.SH', '603729.SH', '002797.SZ', '300608.SZ', '601200.SH', '300440.SZ', '300889.SZ', '603206.SH', '002227.SZ', '301380.SZ', '300496.SZ', '002995.SZ', '688011.SH', '002128.SZ', '002826.SZ', '003036.SZ', '688148.SH', '600051.SH', '001914.SZ', '600526.SH', '605318.SH', '002272.SZ', '000756.SZ', '600916.SH', '600117.SH', '002670.SZ', '603813.SH', '600178.SH', '002910.SZ', '301030.SZ', '300678.SZ', '300452.SZ', '000606.SZ', '002937.SZ', '002174.SZ', '600835.SH', '002833.SZ', '688150.SH', '002364.SZ', '300782.SZ', '605369.SH', '688113.SH', '688679.SH', '002057.SZ', '002518.SZ', '002549.SZ', '603335.SH', '300294.SZ', '300465.SZ', '002120.SZ', '002712.SZ', '000766.SZ', '002863.SZ', '000573.SZ', '600331.SH', '688443.SH', '300751.SZ', '300159.SZ', '603004.SH', '300423.SZ', '601086.SH', '301043.SZ', '605055.SH', '600697.SH', '301581.SZ', '002890.SZ', '600894.SH', '603536.SH', '002159.SZ', '601108.SH', '301310.SZ', '000739.SZ', '300478.SZ', '603113.SH', '300032.SZ', '300997.SZ', '600460.SH', '300433.SZ', '300091.SZ', '603867.SH', '601089.SH', '601872.SH', '600420.SH', '603709.SH', '301678.SZ', '688050.SH', '300204.SZ', '300296.SZ', '600603.SH', '300001.SZ', '603297.SH', '002141.SZ', '000025.SZ', '002663.SZ', '002282.SZ', '300126.SZ', '002167.SZ', '000791.SZ', '688378.SH', '002375.SZ', '000918.SZ', '002617.SZ', '603682.SH', '002346.SZ', '300883.SZ', '002472.SZ', '003028.SZ', '688737.SH', '603386.SH', '600583.SH', '300698.SZ', '603789.SH', '300793.SZ', '600113.SH', '002049.SZ', '001356.SZ', '002829.SZ', '600292.SH', '600052.SH', '600629.SH', '603817.SH', '600035.SH', '600728.SH', '600372.SH', '300745.SZ', '600989.SH', '001208.SZ', '600009.SH', '301178.SZ', '688072.SH', '003020.SZ', '600696.SH', '300726.SZ', '002003.SZ', '688522.SH', '301070.SZ', '301102.SZ', '600580.SH', '001202.SZ', '688275.SH', '000913.SZ', '603603.SH', '300783.SZ', '002709.SZ', '300383.SZ', '300301.SZ', '301067.SZ', '300969.SZ', '603153.SH', '688357.SH', '600805.SH', '600765.SH', '300378.SZ', '600268.SH', '688338.SH', '301338.SZ', '301559.SZ', '002839.SZ', '300065.SZ', '300182.SZ', '300350.SZ', '000713.SZ', '301155.SZ', '688583.SH', '603277.SH', '000506.SZ', '002595.SZ', '002366.SZ', '600059.SH', '688312.SH', '600231.SH', '002100.SZ', '301383.SZ', '002085.SZ', '000758.SZ', '300197.SZ', '603719.SH', '002699.SZ', '002706.SZ', '688029.SH', '603613.SH', '301371.SZ', '600751.SH', '300778.SZ', '002880.SZ', '002574.SZ', '002901.SZ', '301399.SZ', '000980.SZ', '688469.SH', '000905.SZ', '603219.SH', '688225.SH', '300482.SZ', '000797.SZ', '002232.SZ', '600663.SH', '002812.SZ', '603871.SH', '600316.SH', '688339.SH', '301215.SZ', '603377.SH', '603808.SH', '002384.SZ', '603779.SH', '300605.SZ', '603612.SH', '300014.SZ', '300763.SZ', '002060.SZ', '688566.SH', '603216.SH', '601111.SH', '600810.SH', '002526.SZ', '300769.SZ', '300624.SZ', '603605.SH', '301307.SZ', '002019.SZ', '600815.SH', '000576.SZ', '300435.SZ', '000688.SZ', '002223.SZ', '605589.SH', '300194.SZ', '002217.SZ', '600227.SH', '000982.SZ', '603336.SH', '000751.SZ', '300205.SZ', '601699.SH', '002512.SZ', '600562.SH', '000806.SZ', '300797.SZ', '300471.SZ', '603390.SH', '300485.SZ', '600213.SH', '601010.SH', '688376.SH', '002970.SZ', '603379.SH', '002842.SZ', '002065.SZ', '300228.SZ', '301413.SZ', '688758.SH', '688696.SH', '300084.SZ', '600897.SH', '002726.SZ', '002086.SZ', '002887.SZ', '601058.SH', '300363.SZ', '300079.SZ', '301196.SZ', '688371.SH', '600858.SH', '600741.SH', '301098.SZ', '603466.SH', '601866.SH', '300453.SZ', '002165.SZ', '002651.SZ', '001229.SZ', '301192.SZ', '301033.SZ', '300049.SZ', '688301.SH', '002728.SZ', '688701.SH', '002222.SZ', '300603.SZ', '000681.SZ', '600846.SH', '603444.SH', '300266.SZ', '605060.SH', '000428.SZ', '600973.SH', '688646.SH', '688165.SH', '300097.SZ', '002411.SZ', '000592.SZ', '300495.SZ', '301221.SZ', '301391.SZ', '600783.SH', '300795.SZ', '001389.SZ', '688526.SH', '688381.SH', '603215.SH', '300328.SZ', '301295.SZ', '688588.SH', '000953.SZ', '002030.SZ', '600664.SH', '002835.SZ', '003010.SZ', '603733.SH', '301321.SZ', '600735.SH', '002815.SZ', '600060.SH', '688563.SH', '002270.SZ', '300831.SZ', '002356.SZ', '002255.SZ', '002131.SZ', '301616.SZ', '001331.SZ', '600161.SH', '600575.SH', '603551.SH', '603826.SH', '600073.SH', '002106.SZ', '603189.SH', '688183.SH', '600713.SH', '002168.SZ', '603579.SH', '300866.SZ', '002758.SZ', '002082.SZ', '600649.SH', '603056.SH', '002035.SZ', '688231.SH', '603822.SH', '000678.SZ', '002599.SZ', '300163.SZ', '603075.SH', '600408.SH', '603638.SH', '603282.SH', '688223.SH', '688767.SH', '601989.SH', '603083.SH', '003022.SZ', '002273.SZ', '300407.SZ', '601789.SH', '000860.SZ', '300346.SZ', '002656.SZ', '688798.SH', '600428.SH', '688325.SH', '300947.SZ', '603267.SH', '688392.SH', '605598.SH', '002678.SZ', '300753.SZ', '300973.SZ', '300333.SZ', '301262.SZ', '300878.SZ', '301035.SZ', '000410.SZ', '300374.SZ', '603799.SH', '688299.SH', '000733.SZ', '002530.SZ', '000078.SZ', '688065.SH', '300015.SZ', '300543.SZ', '603107.SH', '000027.SZ', '600403.SH', '002317.SZ', '301636.SZ', '301188.SZ', '000060.SZ', '600512.SH', '600320.SH', '600189.SH', '601333.SH', '603070.SH', '002668.SZ', '301056.SZ', '300675.SZ', '688125.SH', '300498.SZ', '601100.SH', '605222.SH', '300718.SZ', '600177.SH', '600740.SH', '002529.SZ', '603699.SH', '603191.SH', '603187.SH', '600483.SH', '600884.SH', '688039.SH', '603915.SH', '000426.SZ', '688146.SH', '301586.SZ', '301234.SZ', '300041.SZ', '000560.SZ', '600312.SH', '301113.SZ', '002968.SZ', '600725.SH', '600118.SH', '300616.SZ', '603633.SH', '002527.SZ', '301486.SZ', '300113.SZ', '600084.SH', '603321.SH', '601326.SH', '603661.SH', '300291.SZ', '000035.SZ', '600476.SH', '300644.SZ', '601155.SH', '603609.SH', '300515.SZ', '000708.SZ', '601236.SH', '600872.SH', '000831.SZ', '300235.SZ', '002026.SZ', '603566.SH', '600798.SH', '603230.SH', '600018.SH', '002813.SZ', '300061.SZ', '600818.SH', '601117.SH', '603381.SH', '000008.SZ', '002638.SZ', '600273.SH', '688247.SH', '301047.SZ', '002616.SZ', '600501.SH', '688552.SH', '688589.SH', '301149.SZ', '300655.SZ', '603977.SH', '000582.SZ', '300288.SZ', '600119.SH', '002286.SZ', '605020.SH', '600061.SH', '001288.SZ', '605008.SH', '603725.SH', '002927.SZ', '002406.SZ', '300905.SZ', '603003.SH', '301128.SZ', '603209.SH', '002145.SZ', '002973.SZ', '301267.SZ', '301061.SZ', '603926.SH', '301312.SZ', '300771.SZ', '000838.SZ', '301551.SZ', '000513.SZ', '001373.SZ', '688262.SH', '301117.SZ', '603657.SH', '603033.SH', '300976.SZ', '002579.SZ', '600176.SH', '688365.SH', '301239.SZ', '300249.SZ', '301577.SZ', '300826.SZ', '300426.SZ', '603392.SH', '002420.SZ', '002442.SZ', '000656.SZ', '600155.SH', '300877.SZ', '002175.SZ', '688358.SH', '000816.SZ', '300503.SZ', '601882.SH', '002674.SZ', '688739.SH', '300243.SZ', '300037.SZ', '600456.SH', '002459.SZ', '000753.SZ', '300402.SZ', '688157.SH', '300506.SZ', '688057.SH', '000558.SZ', '000777.SZ', '688009.SH', '603790.SH', '600865.SH', '688719.SH', '000520.SZ', '300786.SZ', '603928.SH', '301010.SZ', '688085.SH', '300590.SZ', '603477.SH', '603976.SH', '688652.SH', '603183.SH', '301306.SZ', '002001.SZ', '603688.SH', '688202.SH', '002575.SZ', '688621.SH', '600781.SH', '605500.SH', '688058.SH', '601666.SH', '300737.SZ', '001379.SZ', '002749.SZ', '002429.SZ', '603877.SH', '002878.SZ', '002426.SZ', '600612.SH', '002401.SZ', '001323.SZ', '300311.SZ', '300067.SZ', '300541.SZ', '301512.SZ', '300093.SZ', '688728.SH', '301389.SZ', '000679.SZ', '002676.SZ', '600387.SH', '603337.SH', '002767.SZ', '600986.SH', '600359.SH', '301057.SZ', '300756.SZ', '688006.SH', '000630.SZ', '300282.SZ', '600196.SH', '688013.SH', '300254.SZ', '688361.SH', '300854.SZ', '600645.SH', '301327.SZ', '600172.SH', '603097.SH', '300473.SZ', '300290.SZ', '603237.SH', '002393.SZ', '002182.SZ', '300638.SZ', '300166.SZ', '301305.SZ', '301387.SZ', '301059.SZ', '600876.SH', '002372.SZ', '301439.SZ', '605189.SH', '000761.SZ', '300645.SZ', '600295.SH', '688368.SH', '301079.SZ', '688206.SH', '000540.SZ', '600398.SH', '000961.SZ', '688819.SH', '603886.SH', '688671.SH', '600628.SH', '600368.SH', '603776.SH', '002341.SZ', '603991.SH', '300722.SZ', '603006.SH', '300716.SZ', '300470.SZ', '600908.SH', '000935.SZ', '688288.SH', '600300.SH', '601992.SH', '002181.SZ', '600577.SH', '688549.SH', '688222.SH', '300208.SZ', '002435.SZ', '300002.SZ', '603163.SH', '002548.SZ', '000911.SZ', '300123.SZ', '000692.SZ', '002736.SZ', '603992.SH', '688193.SH', '688147.SH', '300339.SZ', '600050.SH', '605286.SH', '688038.SH', '300598.SZ', '002254.SZ', '002811.SZ', '301060.SZ', '002158.SZ', '301626.SZ', '300314.SZ', '002885.SZ', '600054.SH', '688502.SH', '002240.SZ', '002148.SZ', '000722.SZ', '002279.SZ', '000718.SZ', '605016.SH', '688120.SH', '600522.SH', '000993.SZ', '000333.SZ', '600585.SH', '603179.SH', '688216.SH', '002941.SZ', '600565.SH', '600732.SH', '300376.SZ', '603380.SH', '300732.SZ', '603188.SH', '688328.SH', '300967.SZ', '300048.SZ', '603269.SH', '300864.SZ', '000565.SZ', '603408.SH', '300867.SZ', '600498.SH', '300890.SZ', '601139.SH', '600435.SH', '000567.SZ', '002194.SZ', '002876.SZ', '688201.SH', '601133.SH', '300688.SZ', '301358.SZ', '600246.SH', '002848.SZ', '688260.SH', '688718.SH', '600759.SH', '002153.SZ', '301368.SZ', '603055.SH', '000551.SZ', '601368.SH', '600032.SH', '001212.SZ', '001696.SZ', '000407.SZ', '002646.SZ', '002425.SZ', '301129.SZ', '605006.SH', '000532.SZ', '002514.SZ', '300566.SZ', '000010.SZ', '002622.SZ', '301005.SZ', '600538.SH', '600883.SH', '002169.SZ', '688131.SH', '600022.SH', '688685.SH', '301591.SZ', '000026.SZ', '001287.SZ', '002912.SZ', '300935.SZ', '603037.SH', '603095.SH', '600125.SH', '601633.SH', '001269.SZ', '600039.SH', '300876.SZ', '002803.SZ', '301630.SZ', '002846.SZ', '000810.SZ', '300554.SZ', '000826.SZ', '688209.SH', '600892.SH', '002620.SZ', '600493.SH', '000712.SZ', '001207.SZ', '000989.SZ', '603759.SH', '300120.SZ', '601616.SH', '600192.SH', '688323.SH', '300305.SZ', '300723.SZ', '002630.SZ', '688545.SH', '600229.SH', '688151.SH', '600777.SH', '002313.SZ', '002415.SZ', '601878.SH', '301101.SZ', '000019.SZ', '000020.SZ', '002717.SZ', '301323.SZ', '605011.SH', '301479.SZ', '301528.SZ', '002775.SZ', '603656.SH', '300022.SZ', '300132.SZ', '688111.SH', '300919.SZ', '600995.SH', '300962.SZ', '601825.SH', '688162.SH', '301662.SZ', '603019.SH', '600610.SH', '688799.SH', '603233.SH', '301319.SZ', '301372.SZ', '002551.SZ', '600382.SH', '600661.SH', '002010.SZ', '002959.SZ', '002335.SZ', '300085.SZ', '600779.SH', '603131.SH', '603948.SH', '603889.SH', '688387.SH', '688538.SH', '600845.SH', '601678.SH', '301058.SZ', '300622.SZ', '603053.SH', '301119.SZ', '300588.SZ', '002215.SZ', '300139.SZ', '600250.SH', '688015.SH', '300699.SZ', '688569.SH', '300149.SZ', '301565.SZ', '301520.SZ', '002594.SZ', '300832.SZ', '003023.SZ', '603839.SH', '600048.SH', '002349.SZ', '605507.SH', '600866.SH', '601579.SH', '002403.SZ', '688037.SH', '300345.SZ', '601788.SH', '600479.SH', '600567.SH', '600981.SH', '600836.SH', '603836.SH', '603057.SH', '000429.SZ', '300575.SZ', '688393.SH', '000778.SZ', '603488.SH', '600987.SH', '688697.SH', '300559.SZ', '688349.SH', '688210.SH', '603029.SH', '300268.SZ', '001395.SZ', '002136.SZ', '603363.SH', '002262.SZ', '002024.SZ', '600839.SH', '002665.SZ', '603727.SH', '600622.SH', '002747.SZ', '000949.SZ', '600138.SH', '300190.SZ', '600926.SH', '300514.SZ', '300380.SZ', '688670.SH', '600792.SH', '001227.SZ', '603899.SH', '002460.SZ', '688316.SH', '002491.SZ', '688181.SH', '300824.SZ', '688332.SH', '601158.SH', '002714.SZ', '002535.SZ', '300557.SZ', '002611.SZ', '688660.SH', '301252.SZ', '603165.SH', '688197.SH', '600809.SH', '000926.SZ', '300210.SZ', '301361.SZ', '002229.SZ', '600004.SH', '600939.SH', '600933.SH', '603626.SH', '600838.SH', '688693.SH', '001318.SZ', '000938.SZ', '001319.SZ', '688596.SH', '300454.SZ', '300414.SZ', '002872.SZ', '002916.SZ', '688616.SH', '301408.SZ', '600336.SH', '601990.SH', '601966.SH', '605100.SH', '002110.SZ', '300016.SZ', '301548.SZ', '301316.SZ', '688500.SH', '000978.SZ', '301370.SZ', '688322.SH', '300510.SZ', '603569.SH', '300719.SZ', '000683.SZ', '603136.SH', '300460.SZ', '300118.SZ', '605122.SH', '000852.SZ', '300509.SZ', '600977.SH', '002550.SZ', '688455.SH', '688107.SH', '002955.SZ', '300033.SZ', '603367.SH', '601956.SH', '603109.SH', '601988.SH', '002496.SZ', '300174.SZ', '301169.SZ', '002783.SZ', '300505.SZ', '002481.SZ', '301232.SZ', '002987.SZ', '001225.SZ', '002338.SZ', '301111.SZ', '605108.SH', '000976.SZ', '002445.SZ', '300979.SZ', '300830.SZ', '601519.SH', '002843.SZ', '300839.SZ', '688061.SH', '300119.SZ', '600961.SH', '301051.SZ', '600327.SH', '300413.SZ', '600811.SH', '600710.SH', '000937.SZ', '300711.SZ', '002347.SZ', '002499.SZ', '002235.SZ', '603949.SH', '000629.SZ', '603600.SH', '002088.SZ', '688283.SH', '001390.SZ', '603118.SH', '300475.SZ', '600863.SH', '600416.SH', '600330.SH', '002344.SZ', '000600.SZ', '300218.SZ', '300592.SZ', '603150.SH', '300894.SZ', '300755.SZ', '300733.SZ', '002517.SZ', '600550.SH', '002124.SZ', '601116.SH', '300511.SZ', '688128.SH', '603978.SH', '600313.SH', '002990.SZ', '603555.SH', '300289.SZ', '300822.SZ', '600860.SH', '000955.SZ', '300772.SZ', '688005.SH', '002845.SZ', '603283.SH', '002950.SZ', '603988.SH', '000623.SZ', '301397.SZ', '002183.SZ', '300533.SZ', '300548.SZ', '002264.SZ', '000032.SZ', '300770.SZ', '002938.SZ', '688239.SH', '300938.SZ', '600179.SH', '300027.SZ', '300526.SZ', '600581.SH', '603101.SH', '002370.SZ', '301313.SZ', '601369.SH', '605086.SH', '002374.SZ', '002534.SZ', '301118.SZ', '000521.SZ', '688385.SH', '601528.SH', '600006.SH', '300391.SZ', '601319.SH', '603879.SH', '600081.SH', '002457.SZ', '002037.SZ', '300739.SZ', '000822.SZ', '688334.SH', '688311.SH', '603087.SH', '603707.SH', '688251.SH', '003021.SZ', '002465.SZ', '600829.SH', '688523.SH', '600760.SH', '688278.SH', '603301.SH', '300628.SZ', '000887.SZ', '688272.SH', '300903.SZ', '603061.SH', '003030.SZ', '000034.SZ', '603610.SH', '603897.SH', '002125.SZ', '300662.SZ', '603511.SH', '000829.SZ', '300410.SZ', '688551.SH', '300317.SZ', '002662.SZ', '301011.SZ', '600506.SH', '688425.SH', '002219.SZ', '300921.SZ', '600841.SH', '301595.SZ', '603099.SH', '688163.SH', '000899.SZ', '601995.SH', '002440.SZ', '688530.SH', '600893.SH', '001324.SZ', '300946.SZ', '002487.SZ', '603848.SH', '002936.SZ', '000595.SZ', '002278.SZ', '301498.SZ', '600929.SH', '603212.SH', '002984.SZ', '600869.SH', '603260.SH', '601038.SH', '301557.SZ', '002424.SZ', '605268.SH', '300278.SZ', '003031.SZ', '688244.SH', '603028.SH', '601928.SH', '601900.SH', '600717.SH', '688403.SH', '002596.SZ', '000559.SZ', '002926.SZ', '605138.SH', '601566.SH', '301288.SZ', '688496.SH', '002288.SZ', '603578.SH', '300563.SZ', '600293.SH', '601166.SH', '301065.SZ', '300762.SZ', '600109.SH', '002568.SZ', '600917.SH', '688337.SH', '301282.SZ', '002943.SZ', '002540.SZ', '688155.SH', '600729.SH', '300720.SZ', '002919.SZ', '003007.SZ', '300394.SZ', '300534.SZ', '300617.SZ', '002857.SZ', '300643.SZ', '002889.SZ', '603286.SH', '600703.SH', '001336.SZ', '000868.SZ', '600855.SH', '605259.SH', '600767.SH', '300438.SZ', '688105.SH', '688656.SH', '688553.SH', '600180.SH', '600684.SH', '600787.SH', '600230.SH', '000880.SZ', '002400.SZ', '688002.SH', '301036.SZ', '000669.SZ', '000721.SZ', '300587.SZ', '002591.SZ', '301161.SZ', '601827.SH', '002807.SZ', '603755.SH', '002046.SZ', '002139.SZ', '600148.SH', '603639.SH', '600410.SH', '600071.SH', '301363.SZ', '688603.SH', '600455.SH', '603679.SH', '603393.SH', '605068.SH', '300458.SZ', '605056.SH', '002702.SZ', '002040.SZ', '000735.SZ', '002331.SZ', '605069.SH', '300537.SZ', '000607.SZ', '688293.SH', '600576.SH', '600889.SH', '600221.SH', '000921.SZ', '603160.SH', '300788.SZ', '688373.SH', '300913.SZ', '688691.SH', '603787.SH', '603882.SH', '300129.SZ', '603693.SH', '003029.SZ', '000401.SZ', '000488.SZ', '000639.SZ', '002609.SZ', '688153.SH', '300569.SZ', '300869.SZ', '002079.SZ', '300087.SZ', '688280.SH', '300860.SZ', '300564.SZ', '301077.SZ', '688191.SH', '301501.SZ', '002348.SZ', '002033.SZ', '000931.SZ', '002039.SZ', '002961.SZ', '002244.SZ', '000301.SZ', '600837.SH', '300074.SZ', '600094.SH', '600517.SH', '300195.SZ', '600302.SH', '605198.SH', '002325.SZ', '301382.SZ', '300293.SZ', '001213.SZ', '688228.SH', '301529.SZ', '002730.SZ', '002108.SZ', '002824.SZ', '300207.SZ', '603311.SH', '300497.SZ', '002639.SZ', '605088.SH', '300742.SZ', '300978.SZ', '000510.SZ', '301550.SZ', '603683.SH', '002492.SZ', '688449.SH', '601233.SH', '300749.SZ', '002332.SZ', '688081.SH', '603843.SH', '301013.SZ', '300746.SZ', '600721.SH', '300138.SZ', '601921.SH', '000803.SZ', '603883.SH', '002975.SZ', '688557.SH', '002242.SZ', '002122.SZ', '301182.SZ', '300611.SZ', '601918.SH', '603567.SH', '601238.SH', '603023.SH', '301110.SZ', '300910.SZ', '688249.SH', '300964.SZ', '603507.SH', '688170.SH', '603214.SH', '600489.SH', '600572.SH', '601002.SH', '300692.SZ', '600715.SH', '301558.SZ', '300430.SZ', '600716.SH', '000009.SZ', '000070.SZ', '000859.SZ', '600031.SH', '001316.SZ', '688028.SH', '603022.SH', '601668.SH', '688411.SH', '603960.SH', '600258.SH', '603589.SH', '002686.SZ', '603598.SH', '002368.SZ', '300388.SZ', '600909.SH', '603319.SH', '603351.SH', '603927.SH', '300619.SZ', '301133.SZ', '301536.SZ', '002113.SZ', '600532.SH', '688049.SH', '002909.SZ', '605028.SH', '300911.SZ', '300690.SZ', '301611.SZ', '300519.SZ', '603007.SH', '300666.SZ', '600790.SH', '000525.SZ', '600518.SH', '600702.SH', '000690.SZ', '688786.SH', '002655.SZ', '301489.SZ', '000837.SZ', '300092.SZ', '605196.SH', '603049.SH', '002606.SZ', '600831.SH', '688217.SH', '600193.SH', '603322.SH', '300445.SZ', '002329.SZ', '300007.SZ', '001330.SZ', '002397.SZ', '603863.SH', '000408.SZ', '688382.SH', '002816.SZ', '300145.SZ', '002608.SZ', '000523.SZ', '300368.SZ', '600556.SH', '688682.SH', '601198.SH', '688475.SH', '003032.SZ', '688008.SH', '300750.SZ', '601515.SH', '603938.SH', '000534.SZ', '600611.SH', '000890.SZ', '603060.SH', '002409.SZ', '300355.SZ', '301171.SZ', '605179.SH', '002265.SZ', '300076.SZ', '601028.SH', '300077.SZ', '002298.SZ', '600540.SH', '300500.SZ', '601838.SH', '688398.SH', '002882.SZ', '002399.SZ', '601916.SH', '002808.SZ', '002721.SZ', '300508.SZ', '000417.SZ', '601136.SH', '688135.SH', '000612.SZ', '600168.SH', '000851.SZ', '002781.SZ', '300257.SZ', '688229.SH', '000089.SZ', '688519.SH', '601229.SH', '600017.SH', '688017.SH', '688187.SH', '301509.SZ', '002173.SZ', '002821.SZ', '688766.SH', '605058.SH', '300670.SZ', '002058.SZ', '301205.SZ', '002873.SZ', '688360.SH', '002166.SZ', '605588.SH', '600873.SH', '603607.SH', '301459.SZ', '002156.SZ', '688456.SH', '603718.SH', '002853.SZ', '603833.SH', '301357.SZ', '601968.SH', '600660.SH', '002576.SZ', '002031.SZ', '001332.SZ', '300381.SZ', '600530.SH', '601179.SH', '301148.SZ', '002014.SZ', '301019.SZ', '600378.SH', '600681.SH', '300489.SZ', '603203.SH', '300252.SZ', '002792.SZ', '603205.SH', '601106.SH', '601808.SH', '002486.SZ', '300697.SZ', '603689.SH', '002692.SZ', '603315.SH', '002969.SZ', '301418.SZ', '688683.SH', '688600.SH', '300990.SZ', '603660.SH', '300412.SZ', '603506.SH', '600085.SH', '301296.SZ', '300717.SZ', '601818.SH', '600477.SH', '300125.SZ', '300925.SZ', '603045.SH', '688699.SH', '603000.SH', '601727.SH', '603773.SH', '601658.SH', '002198.SZ', '600397.SH', '600854.SH', '300221.SZ', '002451.SZ', '300284.SZ', '688273.SH', '688609.SH', '301099.SZ', '000505.SZ', '688515.SH', '300651.SZ', '601019.SH', '603630.SH', '603878.SH', '600319.SH', '002598.SZ', '000090.SZ', '600425.SH', '002582.SZ', '603398.SH', '301448.SZ', '605366.SH', '301039.SZ', '003009.SZ', '300161.SZ', '000546.SZ', '300063.SZ', '000726.SZ', '688062.SH', '002623.SZ', '301335.SZ', '603518.SH', '688285.SH', '301603.SZ', '600339.SH', '603890.SH', '300250.SZ', '000404.SZ', '300492.SZ', '002687.SZ', '300549.SZ', '603030.SH', '301093.SZ', '300791.SZ', '300406.SZ', '002327.SZ', '300321.SZ', '600242.SH', '301339.SZ', '300610.SZ', '300915.SZ', '603738.SH', '605199.SH', '600618.SH', '603666.SH', '601958.SH', '002613.SZ', '002297.SZ', '003039.SZ', '601399.SH', '688190.SH', '300309.SZ', '300522.SZ', '300417.SZ', '300747.SZ', '002160.SZ', '600395.SH', '688261.SH', '605266.SH', '000627.SZ', '000807.SZ', '001376.SZ', '603079.SH', '688680.SH', '300136.SZ', '300147.SZ', '002036.SZ', '688776.SH', '601398.SH', '603207.SH', '301366.SZ', '600707.SH', '601702.SH', '300741.SZ', '301180.SZ', '000729.SZ', '300855.SZ', '600714.SH', '301217.SZ', '301024.SZ', '300601.SZ', '603202.SH', '603920.SH', '600170.SH', '000795.SZ', '300810.SZ', '601375.SH', '603999.SH', '300281.SZ', '603323.SH', '000801.SZ', '688007.SH', '601608.SH', '002805.SZ', '688138.SH', '601011.SH', '300896.SZ', '688226.SH', '300871.SZ', '603355.SH', '600143.SH', '002774.SZ', '688118.SH', '603993.SH', '300862.SZ', '002765.SZ', '688303.SH', '300637.SZ', '002184.SZ', '000413.SZ', '688103.SH', '688012.SH', '002422.SZ', '000518.SZ', '600850.SH', '300437.SZ', '600558.SH', '603950.SH', '600298.SH', '300343.SZ', '601228.SH', '002441.SZ', '002089.SZ', '688321.SH', '603395.SH', '300738.SZ', '003013.SZ', '600269.SH', '688692.SH', '002751.SZ', '002386.SZ', '688615.SH', '300780.SZ', '301160.SZ', '688627.SH', '688087.SH', '603998.SH', '600609.SH', '300232.SZ', '603803.SH', '300857.SZ', '600623.SH', '000593.SZ', '002488.SZ', '301130.SZ', '600423.SH', '300189.SZ', '300897.SZ', '601901.SH', '603860.SH', '000596.SZ', '000423.SZ', '603496.SH', '000566.SZ', '688633.SH', '001268.SZ', '688199.SH', '000848.SZ', '002641.SZ', '002581.SZ', '002069.SZ', '002745.SZ', '002693.SZ', '600232.SH', '600307.SH', '000973.SZ', '688535.SH', '300040.SZ', '002395.SZ', '000812.SZ', '002077.SZ', '002210.SZ', '300820.SZ', '300558.SZ', '300968.SZ', '603257.SH', '300740.SZ', '300560.SZ', '603982.SH', '600778.SH', '002642.SZ', '002505.SZ', '002132.SZ', '001236.SZ', '001299.SZ', '603299.SH', '600389.SH', '601005.SH', '300736.SZ', '000789.SZ', '600237.SH', '300512.SZ', '600749.SH', '300244.SZ', '603167.SH', '600083.SH', '300088.SZ', '002178.SZ', '688048.SH', '002214.SZ', '605098.SH', '603916.SH', '603880.SH', '301270.SZ', '600203.SH', '002850.SZ', '002334.SZ', '000968.SZ', '600563.SH', '001226.SZ', '300838.SZ', '300840.SZ', '688287.SH', '002672.SZ', '003008.SZ', '300275.SZ', '603290.SH', '002605.SZ', '301162.SZ', '002817.SZ', '688389.SH', '600848.SH', '603013.SH', '002521.SZ', '688305.SH', '002718.SZ', '601390.SH', '600158.SH', '605376.SH', '688677.SH', '301533.SZ', '600621.SH', '601606.SH', '603602.SH', '001216.SZ', '688559.SH', '301122.SZ', '300950.SZ', '600638.SH', '301127.SZ', '002787.SZ', '301618.SZ', '688046.SH', '301348.SZ', '000977.SZ', '603499.SH', '002849.SZ', '300680.SZ', '688192.SH', '601113.SH', '603177.SH', '301031.SZ', '600136.SH', '600238.SH', '300256.SZ', '600941.SH', '600859.SH', '600095.SH', '002932.SZ', '300851.SZ', '002610.SZ', '002304.SZ', '688083.SH', '688721.SH', '600208.SH', '002249.SZ', '002980.SZ', '600252.SH', '603077.SH', '301552.SZ', '600259.SH', '000709.SZ', '000686.SZ', '002146.SZ', '300184.SZ', '002898.SZ', '301190.SZ', '603903.SH', '300183.SZ', '002068.SZ', '300528.SZ', '301049.SZ', '002029.SZ', '601890.SH', '000415.SZ', '300885.SZ', '300815.SZ', '688169.SH', '001301.SZ', '300036.SZ', '300102.SZ', '301257.SZ', '002531.SZ', '600262.SH', '688079.SH', '002161.SZ', '002142.SZ', '600072.SH', '603196.SH', '603166.SH', '300142.SZ', '002773.SZ', '603929.SH', '688528.SH', '601799.SH', '301195.SZ', '002135.SZ', '603270.SH', '002828.SZ', '688115.SH', '600463.SH', '300106.SZ', '002719.SZ', '603788.SH', '605499.SH', '003038.SZ', '600108.SH', '603306.SH', '600539.SH', '603190.SH', '001368.SZ', '300595.SZ', '688578.SH', '600676.SH', '601137.SH', '603197.SH', '301026.SZ', '600727.SH', '600668.SH', '300959.SZ', '600278.SH', '300802.SZ', '002012.SZ', '600396.SH', '000622.SZ', '688506.SH', '688108.SH', '002992.SZ', '300881.SZ', '600606.SH', '300354.SZ', '601007.SH', '000069.SZ', '301159.SZ', '002800.SZ', '688320.SH', '000995.SZ', '002789.SZ', '002454.SZ', '688336.SH', '002104.SZ', '603078.SH', '002045.SZ', '600671.SH', '688188.SH', '300686.SZ', '301199.SZ', '301105.SZ', '002827.SZ', '301309.SZ', '002094.SZ', '001201.SZ', '002412.SZ', '600705.SH', '002396.SZ', '301038.SZ', '688291.SH', '688175.SH', '688060.SH', '000818.SZ', '688651.SH', '605116.SH', '002899.SZ', '688750.SH', '603730.SH', '300384.SZ', '605398.SH', '000811.SZ', '603001.SH', '600653.SH', '300927.SZ', '301580.SZ', '605003.SH', '000663.SZ', '300846.SZ', '301212.SZ', '300684.SZ', '301322.SZ', '002915.SZ', '600468.SH', '600935.SH', '000100.SZ', '301278.SZ', '300806.SZ', '300676.SZ', '000929.SZ', '002458.SZ', '603696.SH', '603739.SH', '300664.SZ', '300171.SZ', '300269.SZ', '600187.SH', '000856.SZ', '000996.SZ', '300140.SZ', '002647.SZ', '001206.SZ', '000056.SZ', '603439.SH', '601015.SH', '300069.SZ', '002292.SZ', '301213.SZ', '002494.SZ', '300813.SZ', '603517.SH', '300987.SZ', '601156.SH', '603368.SH', '688269.SH', '300983.SZ', '000545.SZ', '002971.SZ', '300336.SZ', '603330.SH', '300623.SZ', '300251.SZ', '603396.SH', '002062.SZ', '300899.SZ', '603278.SH', '000543.SZ', '603296.SH', '688112.SH', '600557.SH', '000737.SZ', '688351.SH', '001215.SZ', '300792.SZ', '002497.SZ', '300906.SZ', '000037.SZ', '002886.SZ', '001378.SZ', '688027.SH', '301292.SZ', '002830.SZ', '000068.SZ', '603235.SH', '300586.SZ', '001359.SZ', '688213.SH', '300898.SZ', '688619.SH', '300977.SZ', '600163.SH', '603068.SH', '000007.SZ', '301123.SZ', '300348.SZ', '688067.SH', '688536.SH', '688097.SH', '002928.SZ', '603069.SH', '300552.SZ', '300259.SZ', '000999.SZ', '600354.SH', '002921.SZ', '300215.SZ', '600400.SH', '688472.SH', '000553.SZ', '603076.SH', '601595.SH', '000617.SZ', '301330.SZ', '603170.SH', '301246.SZ', '000972.SZ', '688100.SH', '688106.SH', '000667.SZ', '301078.SZ', '601985.SH', '603828.SH', '600055.SH', '688667.SH', '600015.SH', '000599.SZ', '002171.SZ', '002133.SZ', '000603.SZ', '301297.SZ', '600475.SH', '603587.SH', '300115.SZ', '601101.SH', '301373.SZ', '001388.SZ', '002196.SZ', '603637.SH', '688498.SH', '300292.SZ', '301377.SZ', '300455.SZ', '002588.SZ', '000752.SZ', '002405.SZ', '600010.SH', '600191.SH', '688212.SH', '603031.SH', '603895.SH', '603042.SH', '301395.SZ', '603588.SH', '688416.SH', '301535.SZ', '002653.SZ', '000915.SZ', '600596.SH', '000552.SZ', '688326.SH', '600693.SH', '000768.SZ', '002737.SZ', '603158.SH', '002806.SZ', '301325.SZ', '600641.SH', '300653.SZ', '300416.SZ', '603085.SH', '603806.SH', '002180.SZ', '300805.SZ', '300671.SZ', '301231.SZ', '300411.SZ', '000839.SZ', '300255.SZ', '300880.SZ', '603185.SH', '000965.SZ', '603936.SH', '002253.SZ', '300237.SZ', '601939.SH', '603608.SH', '301356.SZ', '603132.SH', '605151.SH', '300597.SZ', '603313.SH', '003002.SZ', '003011.SZ', '603876.SH', '301601.SZ', '000048.SZ', '002248.SZ', '300277.SZ', '603869.SH', '300729.SZ', '605336.SH', '301103.SZ', '300078.SZ', '002903.SZ', '301115.SZ', '000519.SZ', '300299.SZ', '600007.SH', '000420.SZ', '603601.SH', '002753.SZ', '603989.SH', '603018.SH', '000597.SZ', '301317.SZ', '605358.SH', '301183.SZ', '000923.SZ', '002523.SZ', '002192.SZ', '688227.SH', '605090.SH', '601377.SH', '688099.SH', '000155.SZ', '301126.SZ', '688177.SH', '600222.SH', '002840.SZ', '300696.SZ', '000049.SZ', '002179.SZ', '301055.SZ', '002493.SZ', '688126.SH', '603032.SH', '300330.SZ', '605007.SH', '002144.SZ', '603255.SH', '603329.SH', '301283.SZ', '600513.SH', '002715.SZ', '300540.SZ', '000504.SZ', '301511.SZ', '600184.SH', '002947.SZ', '002469.SZ', '603026.SH', '300274.SZ', '300971.SZ', '300581.SZ', '002649.SZ', '002028.SZ', '301186.SZ', '600543.SH', '001211.SZ', '301555.SZ', '002121.SZ', '301208.SZ', '688292.SH', '600968.SH', '002428.SZ', '000967.SZ', '300585.SZ', '002074.SZ', '300665.SZ', '300212.SZ', '601061.SH', '603117.SH', '688690.SH', '300055.SZ', '000957.SZ', '002836.SZ', '300056.SZ', '300366.SZ', '600486.SH', '301003.SZ', '600185.SH', '002390.SZ', '688686.SH', '688560.SH', '000970.SZ', '002791.SZ', '603979.SH', '603648.SH', '002367.SZ', '603690.SH', '300206.SZ', '300693.SZ', '001255.SZ', '301167.SZ', '002703.SZ', '002038.SZ', '688109.SH', '601208.SH', '688375.SH', '603856.SH', '002287.SZ', '002105.SZ', '301135.SZ', '600761.SH', '300086.SZ', '300922.SZ', '603810.SH', '002997.SZ', '603912.SH', '688391.SH', '601186.SH', '688265.SH', '002502.SZ', '301063.SZ', '300562.SZ', '000046.SZ', '688095.SH', '688550.SH', '688517.SH', '301336.SZ', '600379.SH', '688458.SH', '300441.SZ', '300568.SZ', '300448.SZ', '002696.SZ', '002025.SZ', '300369.SZ', '688252.SH', '002096.SZ', '300932.SZ', '600624.SH', '603012.SH', '688236.SH', '600505.SH', '300082.SZ', '002507.SZ', '002225.SZ', '300211.SZ', '603226.SH', '000736.SZ', '300337.SZ', '300017.SZ', '301168.SZ', '603312.SH', '688478.SH', '601318.SH', '300409.SZ', '603516.SH', '301087.SZ', '000788.SZ', '000668.SZ', '300570.SZ', '000571.SZ', '688661.SH', '688030.SH', '301029.SZ', '002320.SZ', '603328.SH', '002883.SZ', '000930.SZ', '002355.SZ', '688329.SH', '002495.SZ', '300342.SZ', '002825.SZ', '002234.SZ', '002556.SZ', '002267.SZ', '600679.SH', '600830.SH', '300169.SZ', '300390.SZ', '600515.SH', '300121.SZ', '300196.SZ', '002421.SZ', '688369.SH', '300573.SZ', '603298.SH', '600745.SH', '301331.SZ', '600719.SH', '002838.SZ', '300389.SZ', '002044.SZ', '603811.SH', '603310.SH', '601999.SH', '603615.SH', '600900.SH', '600999.SH', '603090.SH', '002988.SZ', '300800.SZ', '300895.SZ', '002536.SZ', '300071.SZ', '000878.SZ', '300231.SZ', '600016.SH', '600297.SH', '603344.SH', '688459.SH', '600586.SH', '600376.SH', '605289.SH', '600523.SH', '688575.SH', '600748.SH', '600211.SH', '600864.SH', '001279.SZ', '603156.SH', '300631.SZ', '300283.SZ', '688215.SH', '605303.SH', '002946.SZ', '003816.SZ', '300707.SZ', '301345.SZ', '300198.SZ', '300607.SZ', '605089.SH', '002583.SZ', '000888.SZ', '301606.SZ', '002905.SZ', '301136.SZ', '002129.SZ', '688219.SH', '300951.SZ', '300635.SZ', '000657.SZ', '600998.SH', '002354.SZ', '603917.SH', '603199.SH', '301556.SZ', '603700.SH', '002322.SZ', '300731.SZ', '002752.SZ', '601816.SH', '300173.SZ', '603073.SH', '688558.SH', '002470.SZ', '605577.SH', '301107.SZ', '300953.SZ', '688152.SH', '002391.SZ', '002361.SZ', '603171.SH', '600217.SH', '600901.SH', '300338.SZ', '688314.SH', '603036.SH', '002358.SZ', '300424.SZ', '300467.SZ', '002066.SZ', '002952.SZ', '600763.SH', '002376.SZ', '000402.SZ', '002414.SZ', '300474.SZ', '300994.SZ', '600823.SH', '688359.SH', '301175.SZ', '002453.SZ', '000702.SZ', '300272.SZ', '301266.SZ', '600704.SH', '601163.SH', '688313.SH', '601898.SH', '688096.SH', '603618.SH', '603098.SH', '601098.SH', '600249.SH', '001337.SZ', '603291.SH', '002516.SZ', '603527.SH', '688237.SH', '688579.SH', '002137.SZ', '002917.SZ', '603273.SH', '688246.SH', '000620.SZ', '688308.SH', '601607.SH', '301037.SZ', '600329.SH', '002117.SZ', '001282.SZ', '300721.SZ', '001277.SZ', '600726.SH', '300949.SZ', '600657.SH', '300550.SZ', '000530.SZ', '300227.SZ', '605567.SH', '688295.SH', '601858.SH', '002965.SZ', '605050.SH', '002557.SZ', '301209.SZ', '301152.SZ', '603348.SH', '688172.SH', '600642.SH', '002152.SZ', '001230.SZ', '003016.SZ', '000063.SZ', '600159.SH', '301488.SZ', '600738.SH', '300583.SZ', '002482.SZ', '301318.SZ', '002016.SZ', '600578.SH', '600794.SH', '688577.SH', '603968.SH', '688330.SH', '603161.SH', '600736.SH', '688793.SH', '300439.SZ', '300578.SZ', '002884.SZ', '600096.SH', '600355.SH', '000150.SZ', '688055.SH', '688276.SH', '603505.SH', '603011.SH', '688234.SH', '600021.SH', '300127.SZ', '603082.SH', '301202.SZ', '002360.SZ', '600673.SH', '600511.SH', '300334.SZ', '601012.SH', '300331.SZ', '002277.SZ', '603697.SH', '600322.SH', '601677.SH', '688189.SH', '300062.SZ', '002061.SZ', '300681.SZ', '300730.SZ', '000906.SZ', '603373.SH', '300887.SZ', '603712.SH', '600599.SH', '002963.SZ', '301179.SZ', '601929.SH', '300262.SZ', '003018.SZ', '002659.SZ', '002064.SZ', '002063.SZ', '688399.SH', '600481.SH', '603339.SH', '300477.SZ', '300154.SZ', '300418.SZ', '603809.SH', '300614.SZ', '600120.SH', '600123.SH', '300436.SZ', '601798.SH', '300058.SZ', '600323.SH', '002801.SZ', '002004.SZ', '603063.SH', '301285.SZ', '688707.SH', '300133.SZ', '300160.SZ', '600332.SH', '002761.SZ', '002587.SZ', '300357.SZ', '000615.SZ', '002436.SZ', '600521.SH', '688648.SH', '688266.SH', '001222.SZ', '601700.SH', '603721.SH', '300776.SZ', '002490.SZ', '002552.SZ', '600918.SH', '603332.SH', '600559.SH', '300948.SZ', '002716.SZ', '000898.SZ', '600409.SH', '300191.SZ', '002966.SZ', '300107.SZ', '000601.SZ', '300760.SZ', '300451.SZ', '002891.SZ', '605133.SH', '600962.SH', '600114.SH', '688709.SH', '002757.SZ', '300687.SZ', '300995.SZ', '300395.SZ', '688521.SH', '000655.SZ', '002795.SZ', '603303.SH', '300701.SZ', '600033.SH', '688717.SH', '000883.SZ', '688789.SH', '600688.SH', '300546.SZ', '002362.SZ', '600150.SH', '001308.SZ', '600448.SH', '688035.SH', '688166.SH', '300224.SZ', '300379.SZ', '002123.SZ', '002541.SZ', '600635.SH', '300996.SZ', '603126.SH', '002508.SZ', '300748.SZ', '000652.SZ', '002112.SZ', '002766.SZ', '600438.SH', '688708.SH', '000990.SZ', '300943.SZ', '300501.SZ', '001326.SZ', '002977.SZ', '002983.SZ', '301510.SZ', '001283.SZ', '300051.SZ', '300462.SZ', '300668.SZ', '300531.SZ', '300481.SZ', '300814.SZ', '688716.SH', '301181.SZ', '688139.SH', '688662.SH', '001400.SZ', '603596.SH', '600547.SH', '603391.SH', '688655.SH', '600816.SH', '603901.SH', '603758.SH', '002410.SZ', '002636.SZ', '603486.SH', '688501.SH', '688432.SH', '603116.SH', '002328.SZ', '002239.SZ', '688509.SH', '601599.SH', '600683.SH', '600737.SH', '601698.SH', '002652.SZ', '001358.SZ', '600698.SH', '603223.SH', '600490.SH', '601728.SH', '688396.SH', '002340.SZ', '688098.SH', '600346.SH', '300939.SZ', '601567.SH', '601800.SH', '600734.SH', '603123.SH', '000590.SZ', '300809.SZ', '300539.SZ', '000958.SZ', '688205.SH', '688186.SH', '603611.SH', '000628.SZ', '300026.SZ', '300960.SZ', '605305.SH', '601059.SH', '603389.SH', '300318.SZ', '000790.SZ', '688221.SH', '688539.SH', '002336.SZ', '002302.SZ', '002569.SZ', '601218.SH', '600797.SH', '002456.SZ', '600960.SH', '000819.SZ', '603583.SH', '688277.SH', '002034.SZ', '688623.SH', '688561.SH', '600644.SH', '000619.SZ', '600689.SH', '002892.SZ', '600958.SH', '300030.SZ', '301596.SZ', '300073.SZ', '603015.SH', '002126.SZ', '688127.SH', '000672.SZ', '000422.SZ', '600064.SH', '603922.SH', '688004.SH', '301362.SZ', '300606.SZ', '300777.SZ', '000554.SZ', '301263.SZ', '603375.SH', '300667.SZ', '600212.SH', '600169.SH', '603093.SH', '300761.SZ', '605001.SH', '600795.SH', '600617.SH', '688510.SH', '603258.SH', '601628.SH', '300013.SZ', '300392.SZ', '600225.SH', '002407.SZ', '003000.SZ', '300679.SZ', '002204.SZ', '601018.SH', '601555.SH', '688377.SH', '000537.SZ', '603559.SH', '688726.SH', '000892.SZ', '605111.SH', '603017.SH', '002733.SZ', '300517.SZ', '601965.SH', '301007.SZ', '300988.SZ', '000430.SZ', '301592.SZ', '002573.SZ', '688513.SH', '688208.SH', '300819.SZ', '301045.SZ', '603508.SH', '603220.SH', '301419.SZ', '301265.SZ', '603173.SH', '300105.SZ', '601975.SH', '301261.SZ', '002584.SZ', '002790.SZ', '605123.SH', '600796.SH', '301173.SZ', '600936.SH', '000823.SZ', '688121.SH', '000707.SZ', '603151.SH', '600219.SH', '002939.SZ', '688173.SH', '603178.SH', '301500.SZ', '601718.SH', '002315.SZ', '601669.SH', '600808.SH', '003006.SZ', '301226.SZ', '002290.SZ', '301279.SZ', '600487.SH', '301381.SZ', '600509.SH', '301393.SZ', '002798.SZ', '301000.SZ', '002207.SZ', '300151.SZ', '301227.SZ', '300341.SZ', '300499.SZ', '300520.SZ', '688772.SH', '002879.SZ', '603383.SH', '600828.SH', '688171.SH', '002439.SZ', '603388.SH', '600785.SH', '000002.SZ', '605033.SH', '600875.SH', '002925.SZ', '600100.SH', '603577.SH', '301218.SZ', '688168.SH', '301150.SZ', '300765.SZ', '000920.SZ', '002017.SZ', '600462.SH', '603382.SH', '600183.SH', '300904.SZ', '601598.SH', '688137.SH', '605277.SH', '603162.SH', '002865.SZ', '300144.SZ', '300538.SZ', '688248.SH', '300371.SZ', '605338.SH', '300012.SZ', '600266.SH', '688507.SH', '300044.SZ', '603985.SH', '600271.SH', '301286.SZ', '002861.SZ', '300629.SZ', '002592.SZ', '600531.SH', '603103.SH', '301185.SZ', '600731.SH', '600568.SH', '688370.SH', '300167.SZ', '301085.SZ', '688256.SH', '600571.SH', '300029.SZ', '301516.SZ', '300790.SZ', '605333.SH', '600011.SH', '002151.SZ', '000727.SZ', '002303.SZ', '600592.SH', '600251.SH', '000701.SZ', '002202.SZ', '603159.SH', '301515.SZ', '300900.SZ', '002786.SZ', '688468.SH', '600151.SH', '300486.SZ', '600308.SH', '002072.SZ', '300909.SZ', '603669.SH', '600528.SH', '688352.SH', '300579.SZ', '300349.SZ', '002402.SZ', '002032.SZ', '300641.SZ', '301121.SZ', '301291.SZ', '601118.SH', '002896.SZ', '300373.SZ', '300875.SZ', '002316.SZ', '002471.SZ', '688653.SH', '600804.SH', '603201.SH', '600248.SH', '300709.SZ', '600712.SH', '301006.SZ', '001231.SZ', '000951.SZ', '000715.SZ', '002140.SZ', '002578.SZ', '603043.SH', '600584.SH', '605255.SH', '603239.SH', '601288.SH', '300043.SZ', '600377.SH', '300421.SZ', '600739.SH', '300532.SZ', '000813.SZ', '605288.SH', '002293.SZ', '300100.SZ', '603200.SH', '000799.SZ', '300141.SZ', '000605.SZ', '000798.SZ', '300945.SZ', '688617.SH', '000901.SZ', '002701.SZ', '301062.SZ', '688093.SH', '300705.SZ', '600608.SH', '002820.SZ', '601877.SH', '002309.SZ', '301508.SZ', '300547.SZ', '301258.SZ', '600369.SH', '002443.SZ', '688238.SH', '301100.SZ', '002150.SZ', '301568.SZ', '688383.SH', '603855.SH', '301200.SZ', '300530.SZ', '600579.SH', '300759.SZ', '600601.SH', '300217.SZ', '002081.SZ', '600426.SH', '002294.SZ', '300920.SZ', '688307.SH', '600757.SH', '002474.SZ', '300117.SZ', '600305.SH', '000952.SZ', '603535.SH', '601339.SH', '300035.SZ', '600433.SH', '605296.SH', '688180.SH', '600569.SH', '300200.SZ', '301095.SZ', '300263.SZ', '300634.SZ', '300199.SZ', '601500.SH', '300109.SZ', '002543.SZ', '688400.SH', '301633.SZ', '000966.SZ', '600000.SH', '603127.SH', '301302.SZ', '300823.SZ', '000969.SZ', '301097.SZ', '301116.SZ', '600724.SH', '600963.SH', '300933.SZ', '688003.SH', '300708.SZ', '688395.SH', '600058.SH', '301281.SZ', '300640.SZ', '000680.SZ', '301219.SZ', '688092.SH', '002993.SZ', '688503.SH', '600190.SH', '600549.SH', '000885.SZ', '300803.SZ', '688159.SH', '003017.SZ']
        # if '003017.SZ' not in stock_codes:
        #     stock_codes.append('003017.SZ')
        return final_stock_pool_df

    def get_which_field_of_factor_definition_by_factor_name(self, factor_name, which_field):
        cur_factor_definition = self.get_factor_definition(factor_name)
        return cur_factor_definition[which_field]

    def get_factor_definition_df(self):
        return pd.DataFrame(self.config['factor_definition'])

    def is_composite_factor(self, factor_name):
        factor_config = self.get_factor_definition(factor_name)
        if factor_config.empty:
            raise ValueError(f"factor_definition 中不存在因子: {factor_name}")
        return factor_config['action'].iloc[0] == 'composite'

    def get_pool_profiles(self):
        return self.config['stock_pool_profiles']

    def get_pool_profile_by_pool_name(self, pool_name):
        return self.get_pool_profiles()[pool_name]

    def get_stock_pool_index_code_by_name(self, name):
        return self.get_pool_profile_by_pool_name(name)['index_filter']['index_code']

    def get_factor_definition(self, factor_name):
        all_df = self.get_factor_definition_df()
        return all_df[all_df['name'] == factor_name]

    def get_target_factors_for_evaluation(self):
        namelists = list(self.experiments_config.keys())
        return namelists

    def get_experiments_factor_names(self):
        return list(pd.DataFrame(self.get_experiments_df())['factor_name'].unique())

    def get_experiments_pool_names(self):
        return list(pd.DataFrame(self.get_experiments_df())['stock_pool_name'].unique())

    def get_experiments_df(self):
        df = pd.DataFrame(self.experiments_config)
        return df.drop_duplicates(inplace=False)

    def get_need_product_pool_name(self):
        self.get_experiments_factor_names()
        pass


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
            existence_mask_shifted = _existence_matrix.shift(1, fill_value=False)
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
    if stock_pool_df is None or stock_pool_df.empty:
        raise ValueError("stock_pool_df 必须传入且不能为空的 DataFrame")
    # 定义不同类型数据的填充策略

    # df = fill_self(factor_name, df, _existence_matrix)
    # 步骤1: 对齐到修剪后的股票池 对齐到主模板（stock_pool_df的形状）
    return my_align(df, stock_pool_df)


def my_align(df, stock_pool_df):
    # 步骤1: 对齐到修剪后的股票池 对齐到主模板（stock_pool_df的形状）
    aligned_df = df.reindex(index=stock_pool_df.index, columns=stock_pool_df.columns)
    aligned_df = aligned_df.sort_index()
    aligned_df = aligned_df.where(stock_pool_df)
    return aligned_df


def create_data_manager(config_path: str) -> DataManager:
    """
    创建数据管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        DataManager实例
    """
    return DataManager(config_path)

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
