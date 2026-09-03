"""
因子预处理流水线 - 单因子测试终极作战手册
第三阶段：因子预处理

实现完整的因子预处理流水线：
1. 去极值 (Winsorization)
2. 中性化 (Neutralization) 
3. 标准化 (Standardization)
"""
from projects._03_factor_selection.utils.IndustryMap import PointInTimeIndustryMap

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    from sklearn.linear_model import LinearRegression
    HAS_STATSMODELS = False
import pandas as pd
import numpy as np
from typing import Dict
import warnings
import sys
from pathlib import Path

from projects._03_factor_selection.factor_manager.classifier.factor_classifier import FactorClassifier
from quant_lib.config.constant_config import permanent__day

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from quant_lib.config.logger_config import setup_logger, log_warning, log_flow_start
from projects._03_factor_selection.utils.data.residualization_rules import \
    need_residualization_in_neutral_processing as need_residualization_in_neutral_proceessing, \
    get_residualization_config

warnings.filterwarnings('ignore')

# 配置日志
logger = setup_logger(__name__)


class FactorProcessor:
    """
    因子预处理器 - 专业级因子预处理流水线
    
    按照华泰证券标准实现：
    1. 去极值：中位数绝对偏差法(MAD) / 分位数法
    2. 中性化：行业中性化 + 市值中性化
    3. 标准化：Z-Score标准化 / 排序标准化
    """

    def __init__(self, config: Dict):
        """
        初始化因子预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.winsorization_exclusions = []

    # ok
    def process_factor(self,
                       factor_df_shifted: pd.DataFrame,
                       target_factor_name: str,
                       neutral_dfs,
                       style_category: str,
                       need_standardize: bool = True, #标准化
                       pit_map:PointInTimeIndustryMap = None
                       ):
        """
        完整的因子预处理流水线
        
        Args:
            factor_data: 原始因子数据
            auxiliary_df_dict: 辅助数据（市值、行业等）
            
        Returns:
            预处理后的因子数据
        """
        # print("\n" + "=" * 30 + " 【进入 process_factor 调试模式】 " + "=" * 30)
        # print("--- 目标因子 (target_factor_df) 的最后5行 ---")
        # print(factor_df_shifted.tail())
        #
        # print("\n--- 中性化风格因子 (neutral_dfs) 的最后5行 ---")
        # for name, df in neutral_dfs.items():
        #     print(f"  > 风格因子: {name}")
        #     print(df.tail())
        #
        # print("=" * 80 + "\n")
        log_flow_start(f"{target_factor_name}因子进入因子预处理...")

        # 【调试输出】添加详细的数据统计
        print(f"\n=== 🔍 调试 {target_factor_name} 预处理流程 ===")
        print(f"输入数据形状: {factor_df_shifted.shape}")
        print(f"输入非空值数量: {factor_df_shifted.notna().sum().sum()}")
        print(f"输入非空值比例: {factor_df_shifted.notna().sum().sum() / (factor_df_shifted.shape[0] * factor_df_shifted.shape[1]):.3f}")

        # 检查每日股票数量
        daily_counts = factor_df_shifted.notna().sum(axis=1)
        print(f"每日有效股票数统计: 均值={daily_counts.mean():.1f}, 最小={daily_counts.min()}, 最大={daily_counts.max()}")

        processed_target_factor_df = factor_df_shifted.copy()

        if pit_map is None:
            pit_map = PointInTimeIndustryMap()
        # 步骤1：去极值
        # print("2. 去极值处理...")
        processed_target_factor_df = self.winsorize_robust(processed_target_factor_df,pit_map)
        # 步骤2：中性化
        if self.preprocessing_config.get('neutralization', {}).get('enable', False):
            processed_target_factor_df = self._neutralize(processed_target_factor_df, target_factor_name,
                                                          neutral_dfs, style_category)
            #用于测试验证中性化有无作用
            # verify_neutralization_effectiveness(processed_target_factor_df,processed_target_factor_df_E,neutral_dfs=neutral_dfs,test_dates=processed_target_factor_df.index[:100])
        else:
            logger.info("2. 跳过中性化处理...")
        if  need_standardize:
            # 步骤3：标准化
            processed_target_factor_df = self._standardize_robust(processed_target_factor_df,pit_map)
        else:
            logger.info("2. 跳过标准化处理...")

        # 统计处理结果
        from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
        FactorManager._validate_data_quality(processed_target_factor_df ,target_factor_name,'预处理完之后：')

        return processed_target_factor_df

    # # ok#ok
    # def winsorize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     去极值处理
    #
    #     Args:
    #         factor_data: 因子数据
    #
    #     Returns:
    #         去极值后的因子数据
    #     """
    #     winsorization_config = self.preprocessing_config.get('winsorization', {})
    #     method = winsorization_config.get('method', 'mad')
    #
    #     processed_factor = factor_data.copy()
    #
    #     if method == 'mad':
    #         # 中位数绝对偏差法 (Median Absolute Deviation)
    #         threshold = winsorization_config.get('mad_threshold', 5)
    #         # print(f"  使用MAD方法，阈值倍数: {threshold}")
    #
    #         # 向量化计算每日的中位数和MAD
    #         median = factor_data.median(axis=1)
    #         mad = (factor_data.sub(median, axis=0)).abs().median(axis=1)
    #
    #         # 向量化计算每日的上下边界
    #         upper_bound = median + threshold * mad
    #         lower_bound = median - threshold * mad
    #
    #         # 向量化clip，axis=0确保按行广播边界
    #         return factor_data.clip(lower_bound, upper_bound, axis=0)
    #     elif method == 'quantile':
    #         # 分位数法
    #         quantile_range = winsorization_config.get('quantile_range', [0.01, 0.99])
    #         print(f"  使用分位数方法，范围: {quantile_range}")
    #         # 向量化计算每日的分位数边界
    #         bounds = factor_data.quantile(q=quantile_range, axis=1).T  # .T转置是为了方便后续clip
    #         lower_bound = bounds.iloc[:, 0]
    #         upper_bound = bounds.iloc[:, 1]
    #         return factor_data.clip(lower_bound, upper_bound, axis=0)
    #
    #     return processed_factor

    def _winsorize_mad_series(self, series: pd.Series, threshold: float, min_samples: int = 10) -> pd.Series:
        """
        MAD去极值
        - 新增 min_samples 参数，用于处理小样本组
        - 【修复】正确处理NaN值和有效样本数检查
        """
        # 1. 【修复】先去除NaN，然后检查有效样本数
        valid_series = series.dropna()
        if len(valid_series) < min_samples:
            return series  # 有效样本太少，不处理，直接返回原序列

        # 2. 计算中位数和MAD (基于有效值)
        median = valid_series.median()
        mad = (valid_series - median).abs().median()

        # 3. 处理零MAD问题（所有值都相同）
        if mad == 0 or pd.isna(mad):
            return series

        # 4. 【新增】参数验证
        if threshold <= 0:
            logger.warning(f"MAD阈值无效: {threshold}，使用默认值5.0")
            threshold = 5.0

        # 5. 计算边界并clip
        const = 1.4826  # 正态分布下的MAD调整常数
        upper_bound = median + threshold * const * mad
        lower_bound = median - threshold * const * mad

        # 6. 【修复】只对有效值进行clip，保持NaN不变
        result = series.copy()
        mask = ~series.isna()
        result[mask] = series[mask].clip(lower_bound, upper_bound)

        return result

    def _record_winsorization_exclusions(self, date, stock_codes, reason: str) -> None:
        # 被排除值在结果中保持 NaN；同时保留逐股票记录，避免数据被静默丢弃。
        codes = sorted(str(code) for code in stock_codes)
        if not codes:
            return
        date_text = pd.Timestamp(date).strftime('%Y-%m-%d')
        self.winsorization_exclusions.extend(
            {'date': date_text, 'ts_code': code, 'reason': reason} for code in codes
        )
        log_warning(f"去极值明确排除: date={date_text}, reason={reason}, ts_codes={codes}")

    @staticmethod
    def _validate_winsorization_config(config: dict) -> None:
        # MAD 与分位数参数互斥，提前失败可防止配置存在但计算时被静默忽略。
        if 'method' not in config:
            raise ValueError("winsorization 配置缺少必填字段 method")
        method = config['method']
        if method == 'mad':
            if 'mad_threshold' not in config:
                raise ValueError("MAD 去极值配置缺少必填字段 mad_threshold")
            if 'quantile_range' in config:
                raise ValueError("method=mad 时不得配置 quantile_range；该字段不会参与 MAD 计算")
            threshold = config['mad_threshold']
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or threshold <= 0:
                raise ValueError(f"mad_threshold 必须是有限正数，实际值: {threshold!r}")
        elif method == 'quantile':
            if 'mad_threshold' in config:
                raise ValueError("method=quantile 时不得配置 mad_threshold")
            quantile_range = config.get('quantile_range')
            if not isinstance(quantile_range, (list, tuple)) or len(quantile_range) != 2:
                raise ValueError(f"quantile_range 必须包含两个分位点，实际值: {quantile_range!r}")
            lower_q, upper_q = quantile_range
            if not (isinstance(lower_q, (int, float)) and isinstance(upper_q, (int, float)) and 0 <= lower_q < upper_q <= 1):
                raise ValueError(f"quantile_range 必须满足 0 <= lower < upper <= 1，实际值: {quantile_range!r}")
        else:
            raise ValueError(f"不支持的 winsorization.method: {method!r}")

        industry_config = config.get('by_industry')
        if industry_config is None:
            return
        if method != 'mad':
            raise ValueError("分行业去极值当前只支持 method=mad")
        required = {'primary_level', 'fallback_level', 'min_samples'}
        missing = sorted(required - industry_config.keys())
        if missing:
            raise ValueError(f"winsorization.by_industry 缺少必填字段: {missing}")
        min_samples = industry_config['min_samples']
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError(f"winsorization.by_industry.min_samples 必须是正整数，实际值: {min_samples!r}")

    def _winsorize_quantile_series(self, series: pd.Series, quantile_range: list, min_samples: int = 10) -> pd.Series:
        """
        【辅助函数】对单个Series进行分位数去极值。
        【修复】正确处理NaN值和参数验证
        """
        # 1. 【修复】先去除NaN，然后检查有效样本数
        valid_series = series.dropna()
        if len(valid_series) < min_samples:
            return series

        # 2. 【新增】参数验证
        if not isinstance(quantile_range, (list, tuple)) or len(quantile_range) != 2:
            logger.warning(f"分位数范围无效: {quantile_range}，使用默认值[0.01, 0.99]")
            quantile_range = [0.01, 0.99]

        lower_q, upper_q = min(quantile_range), max(quantile_range)
        if not (0 <= lower_q < upper_q <= 1):
            logger.warning(f"分位数范围超出[0,1]: {quantile_range}，使用默认值[0.01, 0.99]")
            lower_q, upper_q = 0.01, 0.99

        # 3. 计算分位数 (基于有效值)
        lower_bound = valid_series.quantile(lower_q)
        upper_bound = valid_series.quantile(upper_q)

        # 4. 【修复】只对有效值进行clip，保持NaN不变
        result = series.copy()
        mask = ~series.isna()
        result[mask] = series[mask].clip(lower_bound, upper_bound)

        return result

    @staticmethod
    def _build_industry_mad_stats(
            data: pd.DataFrame,
            primary_col: str,
            fallback_col: str,
    ) -> pd.DataFrame:
        """计算 L2 与 L1 的中位数、有效样本数和 MAD。"""
        def mad_func(series: pd.Series) -> float:
            return (series - series.median()).abs().median()

        # count 只统计当日因子有效且行业映射完整的股票，与实际参与计算的样本一致。
        primary_stats = data.groupby(primary_col)['factor'].agg(['median', 'count', mad_func])
        primary_stats = primary_stats.rename(columns={
            'median': 'primary_median', 'count': 'primary_count', 'mad_func': 'primary_mad'
        })
        fallback_stats = data.groupby(fallback_col)['factor'].agg(['median', 'count', mad_func])
        fallback_stats = fallback_stats.rename(columns={
            'median': 'fallback_median', 'count': 'fallback_count', 'mad_func': 'fallback_mad'
        })
        data = data.merge(primary_stats, on=primary_col, how='left')
        return data.merge(fallback_stats, on=fallback_col, how='left')
        # =========================================================================
        # 【核心修改】新的辅助函数，处理单个截面日的回溯逻辑
        # =========================================================================
    #ok
    def _winsorize_cross_section_fallback(
            self,
            daily_factor_series: pd.Series,
            daily_industry_map: pd.DataFrame,
            winsorization_config: dict,
            date: pd.Timestamp,
    ) -> pd.Series:
        """对单个截面日执行 L2→L1 的 MAD 去极值。"""
        industry_config = winsorization_config['by_industry']
        primary_col = industry_config['primary_level']
        fallback_col = industry_config['fallback_level']
        min_samples = industry_config['min_samples']
        threshold = winsorization_config['mad_threshold']

        df = daily_factor_series.to_frame(name='factor')
        df.index.name = 'ts_code'
        merged_df = df.join(daily_industry_map, how='left')
        # 不用其他日期或其他字段猜测行业；T-1 映射缺失的股票当日明确排除。
        missing_industry = merged_df[[primary_col, fallback_col]].isna().any(axis=1)
        self._record_winsorization_exclusions(
            date, merged_df.index[missing_industry], 'missing_industry'
        )
        merged_df = merged_df.loc[~missing_industry].reset_index()
        if merged_df.empty:
            return pd.Series(index=daily_factor_series.index, dtype=float)

        merged_df = self._build_industry_mad_stats(merged_df, primary_col, fallback_col)

        # L2 样本不足时统一改用所属 L1 的统计量，避免小截面的中位数和 MAD 不稳定。
        use_fallback = merged_df['primary_count'] < min_samples
        merged_df['final_median'] = np.where(use_fallback, merged_df['fallback_median'], merged_df['primary_median'])
        merged_df['final_mad'] = np.where(use_fallback, merged_df['fallback_mad'], merged_df['primary_mad'])
        # L1 仍不足时已没有满足门槛的行业统计量，因此排除而不是继续用小样本计算。
        merged_df['insufficient_l1'] = use_fallback & (merged_df['fallback_count'] < min_samples)
        self._record_winsorization_exclusions(
            date, merged_df.loc[merged_df['insufficient_l1'], 'ts_code'], 'insufficient_l1_samples'
        )
        merged_df.set_index('ts_code', inplace=True)

        winsorized_factor = merged_df['factor'].copy()
        winsorized_factor.loc[merged_df['insufficient_l1']] = np.nan
        # MAD=0 时不存在可靠的离散尺度；保持该组原值，禁止用极小数伪造边界。
        can_winsorize = (~merged_df['insufficient_l1']) & merged_df['final_mad'].notna() & merged_df['final_mad'].ne(0)
        const = 1.4826
        upper = merged_df['final_median'] + threshold * const * merged_df['final_mad']
        lower = merged_df['final_median'] - threshold * const * merged_df['final_mad']
        winsorized_factor.loc[can_winsorize] = merged_df.loc[can_winsorize, 'factor'].clip(
            lower=lower.loc[can_winsorize], upper=upper.loc[can_winsorize]
        )
        return winsorized_factor.reindex(daily_factor_series.index)

        # =========================================================================
        # 【核心修改】重构后的 winsorize_robust 函数
        # =========================================================================

    def winsorize_robust(self, factor_data: pd.DataFrame,pit_industry_map: PointInTimeIndustryMap = None) -> pd.DataFrame:
        """
        去极值处理函数。
        支持全市场或分行业（带向上回溯功能）的MAD和分位数法。

        Args:
            factor_data (pd.DataFrame): 因子数据 (index=date, columns=stock)。
            industry_map (pd.DataFrame, optional): 行业分类数据 (index=stock, columns=['l1_code', 'l2_code',...])
                                                   如果提供此参数，则执行分行业去极值。
        Returns:
            pd.DataFrame: 去极值后的因子数据。
        """
        winsorization_config = self.preprocessing_config.get('winsorization', {})
        self.winsorization_exclusions = []
        self._validate_winsorization_config(winsorization_config)
        # method 和阈值位于 winsorization 外层，分行业函数必须接收完整配置才能真正生效。
        method = winsorization_config['method']
        industry_config = winsorization_config.get('by_industry')

        # --- 路径一：全市场去极值 (逻辑基本不变) ---
        if pit_industry_map is None or industry_config is None:
            logger.info("  执行全市场去极值...")
            if method == 'mad':
                params = {'threshold': winsorization_config['mad_threshold'],
                          'min_samples': 1}  # 全市场不需min_samples
                return factor_data.apply(self._winsorize_mad_series, axis=1, **params)
            elif method == 'quantile':
                params = {'quantile_range': winsorization_config['quantile_range'], 'min_samples': 1}
                return factor_data.apply(self._winsorize_quantile_series, axis=1, **params)

        # --- 路径二：分行业去极值 (采用回溯逻辑) ---
        else:
            logger.info(
                f"  执行分行业去极值 (主行业: {industry_config['primary_level']}, 当样本不足会自动回溯至: {industry_config['fallback_level']})...")
            #  为了高效获取前一交易日，提前创建交易日序列
            trading_dates_series = pd.Series(factor_data.index, index=factor_data.index)

            # 按天循环，在截面日上执行矢量化操作
            processed_data = {}
            for date in factor_data.index:
                # 获取当天的因子和行业数据
                daily_factor_series = factor_data.loc[date].dropna()

                # 如果当天没有有效因子值，则跳过
                if daily_factor_series.empty:
                    processed_data[date] = pd.Series(dtype=float)
                    log_warning(f"去极值过程中，发现当天{date}所有股票因子值都为空")
                    continue
                    # 在循环内部，为每一天获取正确的历史地图
                    #  获取 T-1 的日期
                prev_trading_date = trading_dates_series.shift(1).loc[date]
                # 处理回测第一天的边界情况
                if pd.isna(prev_trading_date):
                    # 没有 T-1 行业映射时不能沿用未经行业处理的原值，首日统一排除。
                    self._record_winsorization_exclusions(
                        date, daily_factor_series.index, 'missing_previous_trading_day'
                    )
                    processed_data[date] = pd.Series(
                        np.nan, index=daily_factor_series.index, dtype=float
                    )
                    log_warning(
                        f"去极值跳过 {date}：缺少前一交易日行业映射，"
                        "不能输出未经行业 MAD 处理的因子值。"
                    )
                    continue

                # 使用 T-1 的日期查询行业地图
                daily_industry_map = pit_industry_map.get_map_for_date(prev_trading_date)

                processed_data[date] = self._winsorize_cross_section_fallback(
                    daily_factor_series=daily_factor_series,
                    daily_industry_map=daily_industry_map,
                    winsorization_config=winsorization_config,
                    date=date,
                )

            # 将处理后的数据合并回DataFrame
            result_df = pd.DataFrame.from_dict(processed_data, orient='index')
            # 保持原始的索引和列顺序
            return result_df.reindex(index=factor_data.index, columns=factor_data.columns)

    # ok
    #考虑 传入的行业如果是二级行业那么行业变量多达130个！，我又不做全A，中证800才800，平均一个行业才5只股票 来进行中性化，有点不具参照！，必须用一级行业
    ##
    # 单测通过：
    # === 🔬 中性化效果验证报告 ===
    # 1️⃣ 基础统计检查
    #    原始因子: 均值=0.005922, 标准差=0.112081
    #    中性化后: 均值=0.000000, 标准差=0.082274
    #    数据覆盖: 原始76.26% → 中性化后76.26%
    # 2️⃣ 与风格因子相关性检查
    #    vs circ_mv     :  0.1737 →  0.1499 (降低了 0.0239)
    #    vs pct_chg_beta:  0.0266 →  0.0368 (降低了 0.0102)
    # 3️⃣ 行业中性化检查
    #    2019-07-11: 行业效应降低 0.0305
    #    2019-07-12: 行业效应降低 0.0308
    # 4️⃣ 截面相关性保持检查
    #    截面相关性保持度: 0.8222 (>0.5为良好)
    # 5️⃣ 具体日期验证
    #    📅 2019-07-10 详细检查:
    #       有效股票数: 799 → 799
    #       因子均值: 0.004928 → 0.000000
    #       因子标准差: 0.069131 → 0.068919
    #       极端值数量: 40 → 40
    #    📅 2019-07-11 详细检查:
    #       有效股票数: 788 → 788
    #       因子均值: 0.005155 → 0.000000
    #       因子标准差: 0.068800 → 0.061785
    #       极端值数量: 40 → 40
    #    📅 2019-07-12 详细检查:
    #       有效股票数: 787 → 787
    #       因子均值: 0.002713 → 0.000000
    #       因子标准差: 0.072949 → 0.065334
    #       极端值数量: 40 → 40#
    ##
    #  1. 中性化完全生效 ✅
    #     - 因子均值精确归零
    #     - 风格暴露有效降低
    #     - 截面信息良好保持
    #   2. 唯一的改进空间：
    #     - 市值中性化还可以更彻底（0.1499还能再降低）
    #     - 可能是因为样本不足导致的，可以考虑：
    #         - 降低最小样本数要求（从30→20）
    #       - 或使用一级行业而非二级行业
    #   3. 对因子选择的意义：
    #     - 这个中性化后的因子是纯净的alpha信号
    #     - 可以安全用于IC测试和分层回测
    #     - 风险调整后的表现会更稳定#


    ##
    # 大白话，中性化就是帮我们：剔除多余变量，剩下的值 （暴露的值：都是各凭本事出来的）#
    def _neutralize(self,
                    factor_data: pd.DataFrame,
                    target_factor_name: str,
                    # auxiliary_dfs: Dict[str, pd.DataFrame], # 在新架构下，beta也应在neutral_dfs中
                    neutral_dfs: Dict[str, pd.DataFrame],
                    style_category: str
                    ) -> pd.DataFrame:
        """
        【V2.0-重构版】根据因子所属的“门派”，自动选择最合适的中性化方案。
        此版本修复了潜在bug，并优化了数据构建流程，提升了运行效率和代码清晰度。
        """
        neutralization_config = self.preprocessing_config.get('neutralization', {})
        if not neutralization_config.get('enable', False):
            return factor_data

        logger.info(f"  > 正在对 '{target_factor_name}' 因子 '{target_factor_name}' 进行中性化处理...")
        processed_factor = factor_data.copy()

        # --- 阶段一：因子残差化 ---
        if need_residualization_in_neutral_proceessing(target_factor_name, style_category):
            # 获取该因子的定制化残差配置
            resid_config = get_residualization_config(target_factor_name)
            window = resid_config.get('window', 20)
            min_periods = resid_config.get('min_periods', max(1, int(window * 0.5)))
            factor_mean = processed_factor.rolling(window=window, min_periods=min_periods).mean()
            processed_factor = processed_factor - factor_mean
            logger.info(f"{target_factor_name} 进行因子残差化 for 中性化")


        # --- 阶段二：确定中性化因子列表 ---
        factors_to_neutralize = self.get_regression_need_neutral_factor_list(target_factor_name)
        if not factors_to_neutralize:
            logger.info(f"    > '{target_factor_name}' 因子无需中性化。")
            return processed_factor

        # logger.info(f"    > {target_factor_name} 将对以下风格进行中性化: {factors_to_neutralize}")

        skipped_days_count = 0
        total_days = len(processed_factor.index)

        # --- 阶段三：逐日截面回归中性化 ---
        for date in processed_factor.index:
            y_series = processed_factor.loc[date].dropna()
            if y_series.empty:
                skipped_days_count += 1
                log_warning(f"{target_factor_name}中性化跳过这一天{date} y_series.empty ")
                continue

            # --- a) 【效率优化】构建回归自变量矩阵 X ---
            X_df_parts = []  # 使用一个列表来收集所有自变量 Series

            # --- 市值因子 ---
            if 'market_cap' in factors_to_neutralize:
                # 【命名统一】从 neutral_dfs 中寻找规模因子，名字可以是 'log_circ_mv', 'log_circ_mv' 等
                market_cap_key = 'log_circ_mv'
                if market_cap_key not in neutral_dfs:
                    raise ValueError(f"neutral_dfs 中缺少市值因子 '{market_cap_key}'。")
                mv_series = neutral_dfs[market_cap_key].loc[date].rename('log_circ_mv')
                X_df_parts.append(mv_series)

            # --- 行业因子 ---
            if 'industry' in factors_to_neutralize:
                industry_dummy_keys = [k for k in neutral_dfs.keys() if k.startswith('industry_')]
                if not industry_dummy_keys:
                    raise ValueError("neutral_dfs 中未发现行业哑变量。")

                # 【效率优化】一次性从 neutral_dfs 中提取当天的所有行业哑变量
                daily_dummies_df = pd.concat(
                    [neutral_dfs[key].loc[date].rename(key) for key in industry_dummy_keys],
                    axis=1
                )
                X_df_parts.append(daily_dummies_df)

            # --- Beta 因子 ---
            if 'pct_chg_beta' in factors_to_neutralize:
                if 'pct_chg_beta' not in neutral_dfs:  # 建议将beta也统一放入neutral_dfs
                    raise ValueError("neutral_dfs 中缺少 'pct_chg_beta' 数据。")

                beta_series = neutral_dfs['pct_chg_beta'].loc[date].rename('pct_chg_beta')
                X_df_parts.append(beta_series)

            if not X_df_parts:
                log_warning(f"{target_factor_name}中性化跳过这一天{date} :not X_df_parts")
                continue

            # --- b) 【流程优化】将所有部分一次性合并，然后与 y 对齐 ---
            X_df = pd.concat(X_df_parts, axis=1)
            # 使用 join='inner' 可以一步到位地完成对齐和筛选
            combined_df = pd.concat([y_series.rename('factor'), X_df], axis=1, join='inner').dropna()

            # --- c) 样本量检查 (逻辑不变，但更健壮) ---
            num_predictors = X_df.shape[1]
            MIN_SAMPLES_ABSOLUTE = 30   # 绝对最小样本数（从150降到30）
            MIN_SAMPLES_RELATIVE_FACTOR = 1.5  # 相对最小样本倍数（从3降到1.5）

            # 同时满足相对和绝对两个条件
            is_sample_insufficient = (len(combined_df) < MIN_SAMPLES_ABSOLUTE) or \
                                     (len(combined_df) < MIN_SAMPLES_RELATIVE_FACTOR * num_predictors)

            if is_sample_insufficient:
                log_warning(
                    f"  警告: 日期 {date.date()} 清理后样本数为: "
                    f"({len(combined_df)}), 未满足 样本数 >绝对最小样本数:( {MIN_SAMPLES_ABSOLUTE}) 或未满足-- 样本数 > 相对最小样本倍数:({MIN_SAMPLES_RELATIVE_FACTOR})*自变量个数:({num_predictors}) 的条件，"
                    f"跳过中性化。"
                )
                # 【重要决策】当样本不足时，是保留原始因子值，还是设为空值？
                # 设为空值(NaN)是更保守、更诚实的选择。它承认了当天无法生成一个可靠的中性化信号。
                # 如果保留原始因子值（即：直接continue），意味着在这一天你交易的是一个未被中性化的、有风险暴露的因子。

                ##
                # 如果直接continue ()：这意味着在样本不足的日期，processed_factor 中保留的是原始因子值。这会使你的“纯净因子”在某些天突然变回“原始因子”，导致风险暴露不一致。#
                processed_factor.loc[date] = np.nan  #
                skipped_days_count += 1
                log_warning(f"{target_factor_name}中性化跳过这一天{date} 当天样本不够（经过残差化的前几天都是nan很正常！但是不会超过10天")
                continue
            # --- d) 执行回归并计算残差 ---
            y_clean = combined_df['factor']
            # 使用 sm.add_constant 添加截距项，是 statsmodels 的标准做法
            X_clean = sm.add_constant(combined_df.drop(columns=['factor']))
            # print(f"{date}:相关性：{pd.concat([y_clean, X_clean], axis=1).corr()}") 正常
            try:
                model = sm.OLS(y_clean, X_clean).fit()
                residuals = model.resid
                # self.neutral_gression_diagnostics(model,date,y_clean,X_clean,residuals)

                # 将中性化后的残差更新回 processed_factor
                processed_factor.loc[date, residuals.index] = residuals

            except Exception as e:
                logger.error(f"  错误: 日期 {date.date()} 中性化回归失败: {e}。该日因子数据将标记为NaN。")
                processed_factor.loc[date] = np.nan
                skipped_days_count += 1
        # 循环结束后，执行“熔断检查” ===
        # 从配置中获取最大跳过比例，如果未配置，则默认为10%
        max_skip_ratio = neutralization_config.get('max_skip_ratio', 0.20)  # 从0.10放宽到0.20

        actual_skip_ratio = skipped_days_count / total_days

        if actual_skip_ratio > max_skip_ratio:
            # 当实际跳过比例超过阈值时，直接抛出异常，中断程序
            raise ValueError(
                f"因子 '{target_factor_name}' 中性化失败：处理的 {total_days} 天中，"
                f"有 {skipped_days_count} 天 ({actual_skip_ratio:.2%}) 因样本不足被跳过，"
                f"超过了 {max_skip_ratio:.0%} 的容忍上限。"
                f"请检查上游因子数据质量或股票池设置。"
            )

        logger.info(f"  > 中性化完成。在 {total_days} 天中，共跳过了 {skipped_days_count} 天。")

        # 🚨 深度诊断：检查中性化前后的因子分布变化
        # self.show_neutral_result(factor_data,processed_factor)
        return processed_factor
    # # ok
    # def _standardize(self, factor_data: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     标准化处理
    #
    #     Args:
    #         factor_data: 因子数据
    #
    #     Returns:
    #         标准化后的因子数据
    #     """
    #     standardization_config = self.preprocessing_config.get('standardization', {})
    #     method = standardization_config.get('method', 'zscore')
    #
    #     processed_factor = factor_data.copy()
    #
    #     if method == 'zscore':
    #         # print("  使用Z-Score标准化 (健壮版)")
    #         mean = processed_factor.mean(axis=1)
    #         std = processed_factor.std(axis=1)
    #
    #         # 识别std=0的安全隐患
    #         std_is_zero_mask = (std == 0)
    #
    #         # 先进行标准化（会产生inf）
    #         processed_factor = processed_factor.sub(mean, axis=0).div(std, axis=0)
    #
    #         # 将std=0的行，结果安全地设为0
    #         processed_factor[std_is_zero_mask] = 0.0
    #
    #         return processed_factor
    #
    #     elif method == 'rank':
    #         print("  使用排序标准化 (健壮版)")
    #
    #         # 识别只有一个有效值的边界情况
    #         valid_counts = processed_factor.notna().sum(axis=1)
    #         single_value_mask = (valid_counts == 1)
    #
    #         # 正常计算排名
    #         ranks = processed_factor.rank(axis=1, pct=True)
    #         processed_factor = 2 * ranks - 1
    #
    #         # 将只有一个有效值的行，结果安全地设为0
    #         processed_factor[single_value_mask] = 0.0
    #         return processed_factor
    #
    #     raise RuntimeError("请指定标准化方式")

    # 你的辅助函数稍作调整，专注于计算本身
    def _zscore_series(self, s: pd.Series) -> pd.Series:
        """【辅助函数】对单个Series进行Z-Score标准化"""
        if s.count() < 2:
            result = s.copy()
            result.loc[result.notna()] = 0.0
            return result
        std_val = s.std()
        if not np.isfinite(std_val) or std_val == 0:
            result = s.copy()
            result.loc[result.notna()] = 0.0
            return result
        mean_val = s.mean()
        return (s - mean_val) / std_val

    def _rank_series(self, s: pd.Series) -> pd.Series:
        """【辅助函数】对单个Series进行排序标准化 (转换为[-1, 1]区间)"""
        return s.rank(pct=True, na_option='keep') * 2 - 1

        # =========================================================================
        # 【新增核心】处理截面标准化回溯的辅助函数
        # =========================================================================

    def _standardize_cross_section_fallback(
            self,
            daily_factor_series: pd.Series,
            daily_industry_map: pd.DataFrame,
            config: dict
    ) -> pd.Series:
        """对单个截面日的因子数据执行“向上回溯”Z-Score标准化。"""
        primary_col = config['primary_level']
        fallback_col = config['fallback_level']
        min_samples = config.get('min_samples', 3)  # 标准化至少需要2个点，设为3更稳健

        # 1. 数据整合
        df = daily_factor_series.to_frame(name='factor')
        merged_df = df.join(daily_industry_map, how='left')

        # 赶紧 先将索引ts_code重置为一列，以防在merge(merged_df.merge(primary_stats, on=primary_col, how='left'))中丢失
        merged_df=merged_df.reset_index(inplace=False)

        merged_df=merged_df.dropna(subset=['factor', primary_col, fallback_col], inplace=False)
        if merged_df.empty:
            return pd.Series(index=daily_factor_series.index, dtype=float)

        # 2. 计算各级别行业的统计数据 (mean, std, count)
        primary_stats = merged_df.groupby(primary_col)['factor'].agg(['mean', 'std', 'count'])
        primary_stats=primary_stats.rename(columns={'mean': 'primary_mean', 'std': 'primary_std', 'count': 'primary_count'},
                             inplace=False)

        fallback_stats = merged_df.groupby(fallback_col)['factor'].agg(['mean', 'std'])
        fallback_stats=fallback_stats.rename(columns={'mean': 'fallback_mean', 'std': 'fallback_std'}, inplace=False)

        # 3. 将统计数据映射回每只股票
        merged_df = merged_df.merge(primary_stats, on=primary_col, how='left')
        merged_df = merged_df.merge(fallback_stats, on=fallback_col, how='left')

        # 4. 核心回溯逻辑
        use_fallback = merged_df['primary_count'] < min_samples
        merged_df['final_mean'] = np.where(use_fallback, merged_df['fallback_mean'], merged_df['primary_mean'])
        merged_df['final_std'] = np.where(use_fallback, merged_df['fallback_std'], merged_df['primary_std'])

        # 修复标准化处理：更合理地处理标准差为0的情况
        merged_df['final_std'] = merged_df['final_std'].fillna(0)
        merged_df=merged_df.set_index('ts_code', inplace=False)

        # 对于标准差为0的情况，直接设为0，不进行标准化（在设置索引后创建mask）
        zero_std_mask = merged_df['final_std'] < 1e-6

        # 5. 原来方案 Z-Score标准化 ：弊端 万一就是有 天生就是std =1的呢
        # standardized_factor = (merged_df['factor'] - merged_df['final_mean']) / merged_df['final_std']

        # 对于标准差为0导致std被设为1的组，其(factor-mean)可能不为0，需要手动设为0
        # standardized_factor.loc[merged_df['final_std'] == 1.0] = 0
        # 5. 执行Z-Score标准化
        standardized_factor = pd.Series(index=merged_df.index, dtype=float)

        # 对于有效标准差的股票进行标准化
        valid_std_mask = ~zero_std_mask
        if valid_std_mask.any():
            # 确保索引操作返回Series而不是scalar
            factor_values = merged_df.loc[valid_std_mask, 'factor']
            mean_values = merged_df.loc[valid_std_mask, 'final_mean']
            std_values = merged_df.loc[valid_std_mask, 'final_std']

            # 计算标准化值
            standardized_values = (factor_values - mean_values) / std_values
            standardized_factor.loc[valid_std_mask] = standardized_values

        # 对于标准差为0的股票，设为0
        if zero_std_mask.any():
            standardized_factor.loc[zero_std_mask] = 0

        return standardized_factor.reindex(daily_factor_series.index)

        # =========================================================================
        # 【核心升级】重构后的 standardiize_robust 函数
        # =========================================================================

    def _standardize_robust(self, factor_data: pd.DataFrame,
                           pit_industry_map: PointInTimeIndustryMap = None) -> pd.DataFrame:
        """
        因子标准化函数。
        支持全市场或分行业（带向上回溯功能）的Z-Score和排序标准化。
        """
        config = self.preprocessing_config.get('standardization', {})
        method = config.get('method', 'zscore')
        industry_config = config.get('by_industry')

        # --- 路径一：全市场标准化 ---
        if pit_industry_map is None or industry_config is None:
            print("  执行全市场标准化...")
            if method == 'zscore':
                return factor_data.apply(self._zscore_series, axis=1)
            elif method == 'rank':
                return factor_data.apply(self._rank_series, axis=1)
            return factor_data

        # --- 路径二：分行业标准化 ---
        else:
            logger.info(
                f"  执行分行业标准化 (主行业: {industry_config['primary_level']}, 回溯至: {industry_config['fallback_level']})...")
            trading_dates_series = pd.Series(factor_data.index, index=factor_data.index)

            # Rank法通常在全市场进行才有意义，分行业Rank后不同行业的序无法直接比较。
            # 这里我们约定，分行业标准化主要针对Z-Score。
            if method == 'rank':
                print("    警告：分行业Rank标准化逻辑复杂且不常用，将执行全市场Rank标准化。")
                return factor_data.apply(self._rank_series, axis=1)

            processed_data = {}
            for date in factor_data.index:
                daily_factor_series = factor_data.loc[date].dropna()
                if daily_factor_series.empty:
                    processed_data[date] = pd.Series(dtype=float)
                    log_warning(f"标准化过程中，发现当天{date}所有股票因子值都为空( 如果是 微观结构因子&连续&10条以内 那:正常 因为微观结构因子 根据时间序列残差化(滑动取的均值)")
                    continue

                # 在循环内部，为每一天获取正确的历史地图
                #  获取 T-1 的日期
                prev_trading_date = trading_dates_series.shift(1).loc[date]
                # 处理回测第一天的边界情况
                if pd.isna(prev_trading_date):
                    # 与分行业去极值保持同一首日口径，不输出未经行业标准化的原值。
                    processed_data[date] = pd.Series(index=daily_factor_series.index, dtype=float)
                    continue

                # 使用 T-1 的日期查询行业地图
                daily_industry_map = pit_industry_map.get_map_for_date(prev_trading_date)
                processed_data[date] = self._standardize_cross_section_fallback(
                    daily_factor_series=daily_factor_series,
                    daily_industry_map=daily_industry_map,
                    config=industry_config
                )

            result_df = pd.DataFrame.from_dict(processed_data, orient='index')
            return result_df.reindex(index=factor_data.index, columns=factor_data.columns)

    def _print_processing_stats(self,
                                original_factor: pd.DataFrame,
                                processed_factor: pd.DataFrame
                                ):
        """打印处理统计信息"""
        logger.info("因子预处理统计:")

        # 原始因子统计
        orig_valid = original_factor.notna().sum().sum()
        orig_total = original_factor.shape[0] * original_factor.shape[1]

        # 处理后因子统计
        proc_valid = processed_factor.notna().sum().sum()

        # 分布统计
        all_values = processed_factor.values.flatten()
        all_values = all_values[~np.isnan(all_values)]

        if len(all_values) > 0:
            logger.info(
                f"  处理后分布: 均值={all_values.mean():.3f}, 标准差={all_values.std():.3f} （z标准化，均值一定是0）")
            logger.info(f"  分位数: 1%={np.percentile(all_values, 1):.3f}, "
                        f"99%={np.percentile(all_values, 99):.3f}")



    def get_configured_neutralization_factors(self) -> list[str]:
        """读取并校验 YAML 中定义的中性化变量。"""
        neutralization_config = self.preprocessing_config.get('neutralization', {})
        factors = neutralization_config.get('factors')
        allowed_factors = {'market_cap', 'industry', 'pct_chg_beta'}

        if not isinstance(factors, list) or not factors:
            raise ValueError(
                "preprocessing.neutralization.factors 必须是非空列表，"
                "例如 ['market_cap', 'industry']。"
            )
        if any(not isinstance(factor, str) for factor in factors):
            raise ValueError("preprocessing.neutralization.factors 只能包含字符串。")
        if len(factors) != len(set(factors)):
            raise ValueError("preprocessing.neutralization.factors 不允许重复变量。")

        unknown_factors = set(factors) - allowed_factors
        if unknown_factors:
            raise ValueError(
                "preprocessing.neutralization.factors 包含不支持的变量: "
                f"{sorted(unknown_factors)}；允许值为 {sorted(allowed_factors)}。"
            )
        return factors

    def get_regression_need_neutral_factor_list(self, target_factor_name: str) -> list[str]:
        """
        根据 YAML 策略定义和目标因子名称，取得实际回归变量。

        YAML 是中性化策略的唯一来源；只动态排除目标因子本身，防止
        自身回归。不会再按 style_category 静默替换配置。
        """
        base_neutralization_list = self.get_configured_neutralization_factors()
        final_list = []
        for risk_factor in base_neutralization_list:
            # 检查市值
            if risk_factor == 'market_cap' and FactorClassifier.is_size_factor(target_factor_name):
                logger.info(f"  - 目标是市值因子，已从中性化列表中移除 'market_cap'")
                continue  # 跳过，不加入final_list

            # 检查行业
            if risk_factor == 'industry' and FactorClassifier.is_industry_factor(target_factor_name):
                logger.info(f"  - 目标是行业因子，已从中性化列表中移除 'industry'")
                continue

            # 检查Beta
            if risk_factor == 'pct_chg_beta' and FactorClassifier.is_beta_factor(target_factor_name):
                logger.info(f"  - 目标是Beta因子，已从中性化列表中移除 'pct_chg_beta'")
                continue

            final_list.append(risk_factor)

        # #临时的 记得删除
        # if target_factor_name in ['bm_ratio', 'ep_ratio', 'sp_ratio','beta']:
        #     if 'market_cap' in final_list:
        #         final_list.remove('market_cap')
        logger.info(f"最终用于回归的中性化目标因子为: {final_list}\n")
        return final_list

    def neutral_gression_diagnostics(self,model,date,y_clean,X_clean,residuals):
        # 🚨 详细的回归诊断
        r_squared = model.rsquared

        # 🔍 特别检查波动率因子的回归情况
        logger.info(f"  📊 波动率因子回归详情 - 日期 {date.date()}:")
        logger.info(f"    R² = {r_squared:.4f}")
        logger.info(f"    样本数 = {len(y_clean)}")
        logger.info(f"    自变量数 = {X_clean.shape[1]}")
        logger.info(f"    因子值统计: 均值={y_clean.mean():.6f}, 标准差={y_clean.std():.6f}")
        logger.info(f"    残差统计: 均值={residuals.mean():.6f}, 标准差={residuals.std():.6f}")

        # 检查回归系数
        coefficients = model.params
        logger.info(f"    回归系数: {dict(zip(X_clean.columns, coefficients))}")

        # 检查是否存在异常大的系数
        max_coef = coefficients.abs().max()
        if max_coef > 10:
            logger.warning(f"    🚨 发现异常大的回归系数: {max_coef:.4f}")

        # 检查残差与原因子值的关系
        residual_factor_corr = np.corrcoef(y_clean, residuals)[0, 1]
        logger.info(f"    残差与原因子值的相关性: {residual_factor_corr:.6f}")

        # 如果R²过高，发出警告
        if r_squared > 0.8:
            logger.warning(f"    ⚠️ R²={r_squared:.4f} 过高，可能存在过度拟合！")
        pass

    def show_neutral_result(self,factor_data,processed_factor):
        logger.info("🔬 波动率因子深度诊断 - 中性化前后对比:")

        # 原始因子统计
        orig_flat = factor_data.stack().dropna()
        proc_flat = processed_factor.stack().dropna()

        logger.info(f"  原始因子: 样本数={len(orig_flat)}, 均值={orig_flat.mean():.6f}, 标准差={orig_flat.std():.6f}")
        logger.info(f"  原始因子: 最小值={orig_flat.min():.6f}, 最大值={orig_flat.max():.6f}")
        logger.info(f"  原始因子: 25%分位={orig_flat.quantile(0.25):.6f}, 75%分位={orig_flat.quantile(0.75):.6f}")

        logger.info(f"  中性化后: 样本数={len(proc_flat)}, 均值={proc_flat.mean():.6f}, 标准差={proc_flat.std():.6f}")
        logger.info(f"  中性化后: 最小值={proc_flat.min():.6f}, 最大值={proc_flat.max():.6f}")
        logger.info(f"  中性化后: 25%分位={proc_flat.quantile(0.25):.6f}, 75%分位={proc_flat.quantile(0.75):.6f}")

        # 检查是否存在大量相同值
        unique_orig = len(orig_flat.unique())
        unique_proc = len(proc_flat.unique())
        logger.info(f"  唯一值数量: 原始={unique_orig}, 中性化后={unique_proc}")

        # 检查最常见的值
        most_common_orig = orig_flat.value_counts().head(3)
        most_common_proc = proc_flat.value_counts().head(3)
        logger.info(f"  原始因子最常见值: {most_common_orig.to_dict()}")
        logger.info(f"  中性化后最常见值: {most_common_proc.to_dict()}")


# 模拟一个更真实的、包含历史变更的行业隶属关系数据
def mock_full_historical_industry_data():
    """
    模拟 index_member_all 的全量历史返回
    - S3 在 2023-02-01 从 L2_A1 变更到 L2_A2
    - S6 早期存在，但在 2023-01-15 被剔除
    """
    data = [
        # S1, S2, S4, S5 保持不变
        ['L1_A', 'L2_A1', 'S1', '20200101', None],
        ['L1_A', 'L2_A1', 'S2', '20200101', None],
        ['L1_B', 'L2_B1', 'S4', '20200101', None],
        ['L1_B', 'L2_B1', 'S5', '20200101', None],
        # S3 的变更历史
        ['L1_A', 'L2_A1', 'S3', '20200101', '20230131'], # 旧的隶属关系，在31日结束
        ['L1_A', 'L2_A2', 'S3', '20230201', None],       # 新的隶属关系，从2月1日开始
        # S6 的历史
        ['L1_C', 'L2_C1', 'S6', '20200101', '20230115'],
    ]
    columns = ['l1_code', 'l2_code', 'ts_code', 'in_date', 'out_date']
    df = pd.DataFrame(data, columns=columns)
    df['in_date'] = pd.to_datetime(df['in_date'])
    # out_date 为 None 的表示至今有效，为了便于比较，我们用一个未来的日期代替
    df['out_date'] = pd.to_datetime(df['out_date']).fillna(pd.Timestamp(permanent__day))
    return df




if __name__ == '__main__':
    # 1. 获取全量历史行业数据
    raw_industry_df = mock_full_historical_industry_data()

    # 2. 一次性构建PIT查询引擎
    pit_map_engine = PointInTimeIndustryMap(raw_industry_df)

    # 3. 准备因子数据，日期跨越S3的行业变更日
    factor_data_df = pd.DataFrame({
        'S1': [0.01, 0.02, 0.03],
        'S2': [0.03, 0.04, 0.05],
        'S3': [5.00, 0.01, 6.00],  # S3在 2023-01-31 和 2023-02-01 的极端值
        'S4': [-4.0, 0.05, 0.06],
        'S5': [0.06, 0.07, 0.08]
    }, index=pd.to_datetime(['2023-01-31', '2023-02-01', '2023-02-02']))

    # 4. 准备配置和QuantDeveloper实例
    app_config = {
        'preprocessing': {'winsorization': {
            'method': 'mad', 'mad_threshold': 3.0,
            'by_industry': {'primary_level': 'l2_code', 'fallback_level': 'l1_code', 'min_samples': 2}
        }}}
    developer = FactorProcessor(config=app_config)

    # 5. 执行去极值
    winsorized_df = developer.winsorize_robust(factor_data = factor_data_df, pit_industry_map=pit_map_engine)

    print("\n--- 原始因子数据 ---\n", factor_data_df)
    print("\n--- 去极值后因子数据 ---\n", winsorized_df)

    # 6. 验证 S3 在不同日期的处理逻辑
    print("\n--- 验证S3的行业归属和处理逻辑 ---")
    s3_val_before = winsorized_df.loc['2023-01-31', 'S3']
    s3_val_after = winsorized_df.loc['2023-02-01', 'S3']

    # 在2023-01-31，S3属于L2_A1，该组有S1,S2,S3三只股票，样本足够，使用组内数据
    print(f"2023-01-31, S3(5.0) 属于 L2_A1, 组员[S1,S2,S3], 因子[0.01,0.03,5.0], 处理后值为: {s3_val_before:.4f}")

    # 在2023-02-01，S3变更到L2_A2，该组只有它自己，样本不足，回溯到L1_A
    # L1_A 当天有 S1,S2,S3，因子值为 [0.02, 0.04, 0.01]，用这组的统计量来处理S3
    print(
        f"2023-02-01, S3(0.01) 属于 L2_A2(小样本), 回溯至 L1_A, 组员[S1,S2,S3], 因子[0.02,0.04,0.01], 处理后值为: {s3_val_after:.4f}")
