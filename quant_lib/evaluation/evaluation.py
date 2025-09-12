import math

from pandas import Series
from prompt_toolkit.key_binding.bindings.named_commands import self_insert
from scipy.stats._mstats_basic import winsorize

from quant_lib import logger
from quant_lib.config.logger_config import log_warning
from quant_lib.utils.dataFrame_utils import align_dataframes

"""
评估模块

提供因子评价和策略评估功能，包括IC分析、分层回测、业绩归因等。
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy import stats
from scipy.stats import ttest_1samp, spearmanr

# 尝试导入statsmodels，如果没有安装则使用简化版本
try:
    import statsmodels.api as sm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("警告: statsmodels未安装，将使用简化版本的回归分析")


# 辅助函数：安全的、可在apply中使用的截面去极值函数
def safe_winsorize_series(series: pd.Series, limits: list = [0.025, 0.025]) -> pd.Series:
    """
    对一个Series（DataFrame的一行或一列）进行去极值处理，并安全处理全是NaN的情况。
    """
    # 如果剔除NaN后为空，直接返回原序列
    if series.dropna().empty:
        return series

    # 对非NaN值进行winsorize
    winsorized_values = winsorize(series.dropna(), limits=limits)

    # 将处理后的值放回原序列的索引位置
    return pd.Series(winsorized_values, index=series.dropna().index).reindex(series.index)


def calculate_forward_returns_tradable_o2o(period: int,
                                           open_df: pd.DataFrame,
                                           winsorize_limits: list = [0.025, 0.025]) -> pd.DataFrame:
    """
    周三收盘算出来的因子！t-1
    周四：t这一日：实际上是t-1的因子数据  （所以需要t日开盘价参与收益率计算
    计算从 T日开盘价 到 T+period日收盘价 的未来收益率。
    包含了生存偏差过滤和截面去极值处理。
    """
    open_prices = open_df.copy(deep=True)

    # 1. 定义起点和终点价格 (逻辑核心)
    start_price = open_prices
    end_price = open_prices.shift(-period)

    # 2. 创建“未来存续”掩码
    survived_mask = start_price.notna() & end_price.notna()

    # 3. 计算原始收益率
    forward_returns_raw = (end_price / start_price) - 1

    # 4. 应用掩码
    forward_returns_masked = forward_returns_raw.where(survived_mask)

    # 5. 在截面 (axis=1) 上进行去极值
    forward_returns_winsorized = forward_returns_masked.apply(
        safe_winsorize_series,
        axis=1,
        limits=winsorize_limits
    )

    return forward_returns_winsorized
# #ok
# # C2C 仅用于学术对比或历史统计，不适合实盘回测
# def calculate_forward_returns_c2c(period: int,
#                                   close_df: pd.DataFrame,
#                                   winsorize_limits: list = [0.025, 0.025]) -> pd.DataFrame:
#     """
#     【生产级 C2C】计算从 T日收盘价 到 T+period日收盘价 的未来收益率。
#     包含了生存偏差过滤和截面去极值处理。
#     """
#     prices = close_df.copy(deep=True)
#
#     # 1. 定义起点和终点价格 (逻辑核心)
#     start_price = prices
#     end_price = prices.shift(-period)
#
#     # 2. 创建“未来存续”掩码 (处理退市等情况)
#     # 确保在持有期的起点和终点，股票价格都存在
#     survived_mask = start_price.notna() & end_price.notna()
#
#     # 3. 计算原始收益率
#     forward_returns_raw = (end_price / start_price) - 1
#
#     # 4. 应用掩码，过滤掉无效收益
#     forward_returns_masked = forward_returns_raw.where(survived_mask)
#
#     # 5. 在截面 (axis=1) 上对每日的收益率进行去极值处理
#     # 这是至关重要的一步，可以大幅提高回测结果的稳定性
#     forward_returns_winsorized = forward_returns_masked.apply(
#         safe_winsorize_series,
#         axis=1,
#         limits=winsorize_limits
#     )
#
#     return forward_returns_winsorized
#最新注释： 无法贴近实际，因为往往很难做到第二天一大早就能顺利买入

# 我觉得这个更能说明因子的潜力，在运动过程中（真的过程（交易过程）中， 来看因子 跟此段收益率的协同关系
# def calcu_forward_returns_open_close(period: int,
#                                      close_df: pd.DataFrame,
#                                      open_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     计算从T日开盘价到T+period-1日收盘价的未来收益率 (Open-to-Close)。
#     这是一种更贴近实盘的、更严格的收益计算方式。
#
#     Args:
#         period (int): 持有周期。
#         close_df (pd.DataFrame): 收盘价矩阵。
#         open_df (pd.DataFrame): 开盘价矩阵。
#
#     Returns:
#         pd.DataFrame: O2C未来收益率矩阵。
#     """
#     # 1. 定义起点和终点价格
#     # 起点是 T 日的开盘价，它本身不需要 shift
#     close_df = close_df.copy(deep=True)
#     open_df = open_df.copy(deep=True)
#     start_price = open_df
#     # 终点是 T+period-1 日的收盘价
#     end_price = close_df.shift(-(period - 1))
#
#     # 2. 创建“未来存续”掩码
#     survived_mask = start_price.notna() & end_price.notna()
#
#     # 3. 计算原始收益率，并应用掩码过滤
#     forward_returns_raw = end_price / start_price - 1
#     forward_returns = forward_returns_raw.where(survived_mask)
#     def safe_winsorize(x, limits=[0.025, 0.025]):
#         if x.dropna().empty:#就是会有这一行收益率全空的啊，比如最后这一天，你依赖后一天的收盘价，来做计算，显然拿不到，那就是nan咯
#             return x
#         return pd.Series(
#             winsorize(x.dropna(), limits=limits),
#             index=x.dropna().index
#         ).reindex(x.index)
#
#     winsorized_returns = forward_returns.apply(safe_winsorize, axis=1)
#     return winsorized_returns
# ok
##
#
#
# 你的策略是在 T-1日收盘后 做出决策。
#
# 真实的交易执行，最早只能在 T日开盘时 发生。
#
# 所以，一个基于 T-1 决策的收益，它的衡量起点也必须是 T日。
#
# 如果用这个代码，却匹配了一个从 T-1日 就已经开始计算的收益！ ，在T-1日决策的瞬间，就已经“偷看”到了T-1日到未来的收益，它没有模拟“持有”这个真实世界中的、需要时间流逝的过程。
#
# 所以，这个对齐方式虽然看起来 T-1 对上了 T-1，但它违反了真实世界“决策”与“执行”之间的时间差，是一个隐蔽的未来函数。#
##
# 太坑了，！！！！害我排查两天！这个经常搞出极度异常的单调性！（尤其是volatility相关因子！，而切换成o2c 骤降！瞬间恢复正常
# #
# def calcu_forward_returns_close_close(period, price_df):
#     # 1. 定义起点和终点价格 (严格遵循T-1原则)
#     start_price = price_df.shift(1) #问题所在！
#     end_price = price_df.shift(1 - period)
#
#     # 2. 创建“未来存续”掩码 (确保在持有期首尾股价都存在)
#     survived_mask = start_price.notna() & end_price.notna()
#
#     # 3. 计算原始收益率，并应用掩码过滤
#     forward_returns_raw = end_price / start_price - 1
#     forward_returns = forward_returns_raw.where(survived_mask)
#
#     # clip 操作应该在所有计算和过滤完成后进行
#     return forward_returns.clip(-0.15, 0.15)
# if __name__ == '__main__':
    # # 1. 构造一个简单的价格DataFrame
    # price_data = {'price': [100, 110, 121, 133.1, 146.41]}  # 每天上涨10%
    # dates = pd.to_datetime(['2025-08-01', '2025-08-02', '2025-08-03', '2025-08-04', '2025-08-05'])
    # price_df = pd.DataFrame(price_data, index=dates)
    #
    # # 2. 构造一个简单的因子DataFrame
    # factor_data = {'factor': [1, 2, 3, 4, 5]}  # 因子值每天递增
    # factor_df = pd.DataFrame(factor_data, index=dates)
    # # 计算2日收益率
    # buggy_returns = calcu_forward_returns_close_close(2,price_df)
    #
    # print("--- 你的函数计算出的收益率 ---")
    # print(buggy_returns)
    #
    # # 模拟你的主函数中的合并步骤
    # print("\n--- 模拟合并因子和收益 ---")
    # merged_df = pd.concat([factor_df, buggy_returns.rename(columns={'price': 'return'})], axis=1)
    # print(merged_df)
    # # 2. 构造一个简单的因子DataFrame
    # factor_data = {'factor': [1, 2, 3, 4, 5]}  # 因子值每天递增
    # factor_df = pd.DataFrame(factor_data, index=dates)

import alphalens as al
import pandas as pd

from data.local_data_load import load_income_df, load_price_hfq
from alphalens.utils import  get_clean_factor_and_forward_returns
def alphalens_ic(factor_wide_df, price_df):
    alphalens_factor = prepare_alphalens_factor_data(factor_wide_df)
    ret = get_clean_factor_and_forward_returns(factor=alphalens_factor,prices=price_df)

    ic_cs = al.performance.factor_information_coefficient(ret)
    return ic_cs

    # ... 后续可以基于 factor_data 进行更多分析 ...
def prepare_alphalens_factor_data( factor_wide_df: pd.DataFrame) -> pd.Series:
    """
    将“宽表”因子数据 (index=日期, columns=股票)
    转换为 Alphalens 所需的“长表”格式 (MultiIndex Series)。
    """
    # 确保日期索引是 datetime 类型
    factor_wide_df = factor_wide_df.copy()
    factor_wide_df.index = pd.to_datetime(factor_wide_df.index)

    # 转成长表 Series
    factor_long = factor_wide_df.stack()
    factor_long.index.names = ['date', 'asset']

    return factor_long

# ok ok
def calculate_ic(
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        forward_periods: List[int],
        method: str = 'spearman',
        returns_calculator: Callable[[int, pd.DataFrame], pd.DataFrame] = calculate_forward_returns_tradable_o2o,
        min_stocks: int = 20
) -> Tuple[Dict[str, Series], Dict[str, pd.DataFrame]]:
    """
    向量化计算因子IC值及相关统计指标。
    Args:
        factor_df: 因子值DataFrame
        forward_returns: 未来收益率DataFrame
        method: 相关系数计算方法, 'pearson'或'spearman'
        min_stocks: 每个日期至少需要的【有效配对】股票数量

    Returns:
        一个元组 (ic_series, stats_dict):
        - ic_series (pd.Series): IC时间序列，索引为满足条件的有效日期。
        - stats_dict (Dict): 包含IC均值、ICIR、t值、p值等核心统计指标的字典。
    """
    # local_df=pd.read_parquet(r"D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\workspace\result\000906\log_circ_mv\o2o\20250101_20250710\processed_factor.parquet")
    # close_hfq=pd.read_parquet(r"D:\lqs\codeAbout\py\Quantitative\import_file\quant_research_portfolio\workspace\result\000906\close_hfq\20250101_20250710\close_hfq.parquet")
    # close_hfq_local=calculate_forward_returns_c2c(period=forward_periods[0],close_df=close_hfq)
    logger.info(f"\t向量化计算 {method.capitalize()} 类型IC (生产级版本)...")
    stats_periods_dict = {}
    ic_series_periods_dict = {}
    if factor_df.empty or price_df.empty:
        raise ValueError("输入的因子或价格数据为空，无法计算IC。")
    for period in forward_periods:
        ic_series_cleaned = calculate_non_overlapping_ic_series(factor_df, returns_calculator,period,min_stocks)

        # 修正胜率计算和添加更多统计指标
        ic_mean = ic_series_cleaned.mean()
        ic_std = ic_series_cleaned.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_t_stat, ic_p_value = stats.ttest_1samp(ic_series_cleaned, 0)
        ic_new_t_stat, ic_new_p_value = _calculate_newey_west_tstat(ic_series_cleaned)

        # 胜率！。（表示正确出现的次数/总次数）
        ic_win_rate = ((ic_series_cleaned * ic_mean) > 0).mean()  # 这个就是计算胜率，简化版！ 计算的是IC值与IC均值同向的比例
        # 方向性检查
        if abs(ic_mean) > 1e-10 and np.sign(ic_t_stat) != np.sign(ic_mean):
            raise ValueError("严重错误：t统计量与IC均值方向不一致！")
        dayStr = f'{period}d'
        ic_series_periods_dict[dayStr] = ic_series_cleaned
        stats_periods_dict[dayStr] = {
            # 'ic_series': ic_series,
            'ic_mean': ic_mean,  # >=0.02 及格 。超过0.04良好 超过0.06 超级好
            'ic_std': ic_std,  # 标准差，波动情况
            'ic_ir': ic_ir,  # 稳定性。>0.3才行 >0.5非常稳定优秀！
            'ic_win_rate': ic_win_rate,  # 胜率，在均值决定的方向上，正确出现的次数 >0.55才行
            'ic_abs_mean': ic_series_cleaned.abs().mean(),  # 不是很重要，这个值大的话，才有研究意义，能说明 在方向上有效果，而不是趋于0， 个人推荐>0.03
            'ic_t_stat': ic_t_stat,  # 大于2才有意义
            'ic_p_value': ic_p_value,  # <0.05 说明因子真的有效果
             'ic_new_t_stat': ic_new_t_stat,  # 大于2才有意义
            'ic_new_p_value': ic_new_p_value,  # <0.05 说明因子真的有效果
            'ic_significant': ic_new_p_value < 0.05,

            'ic_Valid Days': len(ic_series_cleaned),
            'ic_Total Days': len(ic_series_cleaned),
            'ic_Coverage Rate': len(ic_series_cleaned) / len(ic_series_cleaned)
        }
    return ic_series_periods_dict, stats_periods_dict

def calculate_non_overlapping_ic_series(
        factor_df: pd.DataFrame,
        returns_calculator: Callable,
        rebalance_period: int,
        min_stocks:int ,
        method: str = 'spearman'
) -> pd.Series:
    """
    【V3 核心函数】计算并返回一个贯穿全历史的、干净的、非重叠的IC序列。
    这是所有后续分析的唯一数据源。

    Args:
        factor_df (pd.DataFrame): 完整的因子值DataFrame。
        returns_calculator (Callable): 收益率计算函数。
        rebalance_period (int): 调仓周期，也即采样间隔。
        method (str): IC计算方法。

    Returns:
        pd.Series: 一个干净的、非重叠的IC时间序列（例如，月度IC序列）。
    """
    print(f"--- 正在生成周期为 {rebalance_period}d 的非重叠IC序列 ('黄金序列')... ---")

    # 1. 计算与调仓周期匹配的远期收益
    forward_returns = returns_calculator(period=rebalance_period)

    # 2. 对齐数据
    aligned_factor, aligned_returns = factor_df.align(forward_returns, join='inner', axis=0)

    # 3. 确定非重叠的采样日期 （“调仓日”)
    all_dates = aligned_factor.index.sort_values().unique()
    non_overlapping_dates = get_non_overlapping_dates(rebalance_period, all_dates)

    # 4. 【效率核心】只在这些非重叠的“调仓日”进行IC计算
    factor_sampled = aligned_factor.loc[non_overlapping_dates]
    returns_sampled = aligned_returns.loc[non_overlapping_dates]

    # ---计算有效配对数并筛选日期 ---
    paired_valid_counts = (factor_sampled.notna() & returns_sampled.notna()).sum(axis=1)
    valid_dates = paired_valid_counts[paired_valid_counts >= min_stocks].index
    logger.info(f"calculate_ic 满足最小股票数量要求的日期数量:{len(valid_dates)}")

    if valid_dates.empty:
        raise ValueError(f"没有任何日期满足最小股票数量({min_stocks})要求，无法计算IC。")

    ic_series = factor_sampled.loc[valid_dates].corrwith(
        returns_sampled.loc[valid_dates],
        axis=1,
        method=method.lower()
    ).rename("IC")
    print(f"--- '黄金IC序列' 生成完毕，共 {len(ic_series)} 个独立观测点。 ---")
    if len(ic_series) < 2:  # t检验至少需要2个样本
        raise ValueError(f"有效IC值数量过少({len(ic_series)})，无法计算统计指标。")
    return ic_series

def calculate_ic_decay(factor_df: pd.DataFrame,
                       returns_calculator,
                       price_df: pd.DataFrame,
                       periods: List[int] = [1, 5, 10, 20, 60],
                       method: str = 'pearson',
                       use_vectorized: bool = True) -> pd.DataFrame:
    """
    计算因子IC衰减

    Args:
        factor_df: 因子值DataFrame
        price_df: 价格DataFrame
        periods: 未来时间周期列表
        method: 相关系数计算方法
        use_vectorized: 是否使用向量化计算方法

    Returns:
        不同时间周期的IC均值DataFrame
    """
    logger.info("计算IC衰减...")

    results = {
        'period': periods,
        'IC_Mean': [],
        'IC_IR': []
    }

    for period in periods:
        # 计算未来收益率
        forward_returns = returns_calculator(period = period)
        # 计算IC
        ic,_ = calculate_ic(factor_df, forward_returns, method)

        # 存储结果
        results['IC_Mean'].append(ic.mean())
        results['IC_IR'].append(ic.mean() / ic.std() if ic.std() > 0 else 0)

    # 创建结果DataFrame
    result_df = pd.DataFrame(results).set_index('period')

    return result_df


# ok

def quantile_stats_result(
        results: Dict[int, pd.DataFrame],
        n_quantiles: int
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
   【修正版】计算并汇总分层回测的关键性能指标。
   此版本通过对每日收益序列进行“非重叠采样”，确保所有统计指标都在统计学上无偏和可靠。
   """
    quantile_stats_periods_dict = {}
    quantile_returns_periods_dict = {}

    for period, daily_result in results.items():
        if daily_result.empty:
            continue

        # --- 核心修正 1：创建非重叠的收益序列 ---
        # daily_result 是每日调仓的模拟收益，其收益计算周期为 period 天，因此存在重叠
        # 我们需要以 period 为步长进行采样，得到一个干净、无偏的独立样本序列

        non_overlapping_dates = get_non_overlapping_dates(period, daily_result.index.sort_values())
        # clean_result 是我们用于所有统计计算的【干净】DataFrame
        clean_result = daily_result.loc[non_overlapping_dates]
        clean_tmb_series = clean_result['TopMinusBottom'].dropna()

        if clean_tmb_series.empty or len(clean_tmb_series) < 2:
            continue

        # --- 核心修正 2：在【干净的非重叠序列】上计算所有统计指标 ---

        # 这是每个非重叠周期的平均收益 (例如，每 20 天的平均收益)
        tmb_mean_period_return = clean_tmb_series.mean()
        tmb_std_period_return = clean_tmb_series.std()

        # 正确的年化收益率计算 (基于复利)
        # (1 + 周期平均收益) ^ (每年周期数) - 1
        periods_per_year = 252 / period if period > 0 else 0
        tmb_annual_return = (1 + tmb_mean_period_return) ** periods_per_year - 1 if periods_per_year > 0 else 0

        # 正确的夏普比率计算
        # (周期平均收益 / 周期标准差) * sqrt(每年周期数)
        tmb_sharpe = (tmb_mean_period_return / tmb_std_period_return) * np.sqrt(periods_per_year) \
            if tmb_std_period_return > 0 and periods_per_year > 0 else 0

        tmb_win_rate = (clean_tmb_series > 0).mean()

        # 最大回撤也必须在干净的序列上计算累计净值
        max_drawdown, mdd_start, mdd_end = calculate_max_drawdown_robust(clean_tmb_series)

        # --- 单调性检验（这部分逻辑不变，但在干净的数据上计算更可靠） ---
        mean_returns = clean_result.mean()  # 使用干净结果的均值
        quantile_means = [mean_returns.get(f'Q{i + 1}', np.nan) for i in range(n_quantiles)]
        monotonicity_spearman = np.nan

        if not any(np.isnan(q) for q in quantile_means):
            #如果 分位数收益的方差极小，导致微小差异被放大 所以总单调性很大，很正常
            monotonicity_spearman, _ = spearmanr(np.arange(1, n_quantiles + 1), quantile_means)

        # --- 存储结果 ---
        # 保留原始的每日收益数据，用于画图
        quantile_returns_periods_dict[f'{period}d'] = daily_result
        # 统计数据全部基于干净的、非重叠的序列
        quantile_stats_periods_dict[f'{period}d'] = {
            'mean_returns': mean_returns,
            'tmb_mean_period_return': tmb_mean_period_return, # 特定周期的平均收益 (例如，5日平均收益)
            'tmb_annual_return': tmb_annual_return,# 年化后的多空组合收益率
            'tmb_sharpe': tmb_sharpe, # * 周期调整后的夏普比率
            'tmb_win_rate': tmb_win_rate,
            'tmb_max_drawdown': max_drawdown,
            'mdd_start_date': mdd_start,  # 最大回撤开始日期
            'mdd_end_date': mdd_end,  # 最大回撤结束日期
            'quantile_means': quantile_means,
            'monotonicity_spearman': monotonicity_spearman,
            # 新增一个指标，告诉你统计是基于多少个独立样本
            'non_overlapping_samples': len(clean_tmb_series)
        }

    return quantile_returns_periods_dict, quantile_stats_periods_dict

def calculate_quantile_returns(
        factor_df: pd.DataFrame,
        returns_calculator: Callable,
        price_df: pd.DataFrame,
        n_quantiles: int = 5,
        forward_periods: List[int] = [1, 5, 20]
) -> Dict[int, pd.DataFrame]:
    """
   计算因子分位数的未来收益率。
    该版本采用向量化实现，并使用rank()进行稳健分组，

    Args:
        factor_df (pd.DataFrame): 因子值DataFrame (index=date, columns=stock)
        price_df: pd.DataFrame: 价格DataFrame (index=date, columns=stock)
        n_quantiles (int): 要划分的分位数数量，默认为5
        forward_periods (List[int]): 未来时间周期列表，如[1, 5, 20]

    Returns:
        Dict[int, pd.DataFrame]: 一个字典，键是未来时间周期(period)，
                                 值是对应的分位数收益DataFrame。
                                 每个DataFrame的index是日期，columns是Q1, Q2... TopMinusBottom。
    """
    logger.info("  > core test 正在计算因子分组表现...")
    # ####
    # factor_df.to_csv('/tests/workspace/mem_momentum_12_1.csv')
    # return_df= returns_calculator(period=1)
    # return_df.to_csv('D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\tests\\workspace\\mem_forward_return_o2c.csv')
    # ###
    # factor_df = pd.read_csv('D:\\lqs\\codeAbout\\py\\Quantitative\\quant_research_portfolio\\tests\\workspace\\local_volatility.csv', index_col=[0], parse_dates=True)

    results = {}
    for period in forward_periods:
        logger.info(f"  > 正在处理向前看 {period} 周期...")

        # 1. 计算未来收益率
        forward_returns = returns_calculator(period=period)

        # 2. 数据转换与对齐：从“宽表”到“长表”
        # 有效域掩码：显式定义分析样本
        # 单一事实来源 - 明确定义所有有效的(date, stock)坐标点
        valid_mask = factor_df.notna() & forward_returns.notna()

        # 应用掩码，确保因子和收益具有完全相同的NaN分布
        final_factor = factor_df.where(valid_mask)
        final_returns = forward_returns.where(valid_mask)

        # 数据转换：从"宽表"到"长表"（现在是安全的）
        factor_long = final_factor.stack().rename('factor')
        returns_long = final_returns.stack().rename('return')

        # 合并数据（不再需要dropna，因为已经完全对齐）
        merged_df = pd.concat([factor_long, returns_long], axis=1)

        if merged_df.empty:
            log_warning(
                f"  > 在周期 {period}，因子和收益数据没有重叠，无法计算。")  # 考虑 要不要直接报错 不能，因为forward_returns有nan很正常
            # 创建一个空的DataFrame以保持输出结构一致性
            empty_cols = [f'Q{i + 1}' for i in range(n_quantiles)] + ['TopMinusBottom']
            results[period] = pd.DataFrame(columns=empty_cols, dtype='float64')
            continue

        # 4. 稳健的分组：使用rank()进行等数量分组 (我们坚持的稳健方法)
        # 按日期(level=0)分(因为是多重索引，这里取第一个索引：时间)组，对每个截面内的因子值进行排名
        merged_df['rank'] = merged_df.groupby(level=0)['factor'].rank(method='first')

        # 因为rank列是唯一的，所以不需要担心duplicates问题。
        # 【改进】更严格的分组样本要求，确保统计稳定性
        MIN_SAMPLES_FOR_GROUPING = max(50, n_quantiles * 10)  # 总样本至少50个，或每组至少10个
        merged_df['quantile'] = merged_df.groupby(level=0)['rank'].transform(
            lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop') + 1
            if len(x) >= MIN_SAMPLES_FOR_GROUPING else np.nan
        )
        # 5. 计算各分位数的平均收益 （时间+组别 为一个group。进行求收益率平均） 今天q1组收益平均结果
        daily_quantile_returns = merged_df.groupby([merged_df.index.get_level_values(0), 'quantile'])['return'].mean()

        # 6. 数据转换：从“长表”恢复到“宽表”
        quantile_returns_wide = daily_quantile_returns.unstack()
        # 保持NaN不填充：当某分组全部股票停牌/退市导致收益率缺失时，
        # 应该保持NaN状态，这样更真实反映该分组在该日期无法交易的情况
        # pandas的后续统计函数(mean, cumprod等)都能正确处理NaN
        # quantile_returns_wide= quantile_returns_wide.fillna(0, inplace=False)

        # 改个列名
        quantile_returns_wide.columns = [f'Q{int(col)}' for col in quantile_returns_wide.columns]

        # 7. 计算多空组合收益
        top_q_col = f'Q{n_quantiles}'
        bottom_q_col = 'Q1'

        if top_q_col in quantile_returns_wide.columns and bottom_q_col in quantile_returns_wide.columns:
            quantile_returns_wide['TopMinusBottom'] = quantile_returns_wide[top_q_col] - quantile_returns_wide[
                bottom_q_col]
        else:
            # 确保即使在极端情况下，列也存在，值为NaN，保持DataFrame结构一致
            quantile_returns_wide['TopMinusBottom'] = np.nan

        # 8. 存储结果
        results[period] = quantile_returns_wide.sort_index(axis=1)
    return  results
#
# def calculate_turnover(positions_df: pd.DataFrame) -> pd.Series:
#     """
#     计算换手率
#
#     Args:
#         positions_df: 持仓DataFrame，index为日期，columns为股票代码
#
#     Returns:
#         换手率序列
#     """
#     logger.info("计算换手率...")
#
#     turnover = pd.Series(index=positions_df.index[1:])
#
#     for i in range(1, len(positions_df)):
#         prev_pos = positions_df.iloc[i - 1]
#         curr_pos = positions_df.iloc[i]
#
#         # 计算持仓变化
#         pos_change = abs(curr_pos - prev_pos).sum() / 2
#
#         turnover.iloc[i - 1] = pos_change
#
#     logger.info(f"换手率计算完成: 平均换手率={turnover.mean():.4f}")
#     return turnover

# ok
def calculate_top_quantile_turnover_dict(        factor_df: pd.DataFrame,
        n_quantiles: int = 5,
        forward_periods: List[int] = [1, 5, 20],
                                                 target_quantile: int = 5  # 默认为做多头部（第5组）

                                                 ) -> Dict[str, pd.Series]:
        ret = {}
        for period in forward_periods:
            ret[f'{period}d'] = calculate_top_quantile_turnover(factor_df, period, n_quantiles, 5)
        return ret

def calculate_top_quantile_turnover(
        factor_df: pd.DataFrame,
        rebalance_period: int,
        n_quantiles: int = 5,
        target_quantile: int = 5  # 默认为做多头部（第5组）
) -> pd.Series:
    """
    【V2 专业版】计算头部组合的实际换手率。
    此函数精确模拟策略在调仓日的买卖行为，用于估算真实交易成本。

    Args:
        factor_df (pd.DataFrame): 完整的因子值DataFrame。
        rebalance_period (int): 调仓周期/采样间隔。
        n_quantiles (int): 分位数总数。
        target_quantile (int): 目标持仓组合的编号 (例如 5 代表做多Top组, 1 代表做空Bottom组)。

    Returns:
        pd.Series: 一个以调仓日为索引的、真实的策略换手率时间序列。
    """
    if factor_df.empty:
        return pd.Series(dtype=float)

    # --- 1. 确定非重叠的调仓日 ---
    all_dates = factor_df.index.sort_values().unique()
    non_overlapping_dates = get_non_overlapping_dates(rebalance_period, all_dates)

    if len(non_overlapping_dates) < 2:
        return pd.Series(dtype=float)  # 至少需要两个调仓日才能计算换手率

    # --- 2. 计算每个调仓日的分位归属 ---
    # 只在调仓日进行计算，提高效率
    factor_on_rebalance_days = factor_df.loc[non_overlapping_dates]
    quantile_groups = np.ceil(
        factor_on_rebalance_days.rank(axis=1, pct=True, method='first') * n_quantiles
    )

    turnover_list = []

    # --- 3. 逐期计算组合换手 ---
    # 从第二个调仓日开始，与前一个调仓日对比
    for i in range(1, len(quantile_groups)):
        prev_date = quantile_groups.index[i - 1]
        curr_date = quantile_groups.index[i]

        # a. 获取前后两个时期的【目标组合】成分股（使用集合set，便于计算交集）
        prev_series = quantile_groups.loc[prev_date].dropna()
        curr_series = quantile_groups.loc[curr_date].dropna()

        prev_portfolio = set(prev_series[prev_series == target_quantile].index)
        curr_portfolio = set(curr_series[curr_series == target_quantile].index)

        # b. 计算交集和换手率
        # 换手率 = 1 - 留存比例 = 1 - (交集股票数 / 当前组合股票数)
        common_stocks_count = len(prev_portfolio.intersection(curr_portfolio))
        current_portfolio_size = len(curr_portfolio)

        if current_portfolio_size == 0:
            turnover = np.nan  # 避免除以零
        else:
            turnover = 1.0 - (common_stocks_count / current_portfolio_size)

        turnover_list.append({'date': curr_date, 'turnover': turnover})

    # --- 4. 转换为 Pandas Series ---
    if not turnover_list:
        return pd.Series(dtype=float)

    turnover_df = pd.DataFrame(turnover_list).set_index('date')
    return turnover_df['turnover']

def get_non_overlapping_dates(period, all_dates):
    non_overlapping_dates = []
    i = 0
    while i < len(all_dates):
        non_overlapping_dates.append(all_dates[i])
        i += period  # 以 period 为步长进行采样
    return non_overlapping_dates
#ok
def calculate_max_drawdown_robust(
        returns: pd.Series
) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    【健壮版】计算最大回撤及其开始和结束日期。
    此版本修复了在单调上涨行情下的边界Bug，并增强了对异常输入的处理。

    Args:
        returns: 收益率序列 (pd.Series)

    Returns:
        一个元组 (max_drawdown, start_date, end_date):
        - max_drawdown (float): 最大回撤值 (一个负数或0)。
        - start_date (pd.Timestamp or None): 最大回撤开始的日期（前期高点）。
        - end_date (pd.Timestamp or None): 最大回撤结束的日期（最低点）。
    """
    # --- 1. 输入验证与数据清洗 (提升健壮性) ---
    if returns is None or returns.empty:
        return 0.0, None, None

    # 填充NaN值，最常见的处理是视为当天无收益
    returns_cleaned = returns.fillna(0)

    # --- 2. 核心计算 (与原版逻辑相同) ---
    # 计算累计净值曲线 (通常从1开始)
    cumulative_returns = (1 + returns_cleaned).cumprod()

    # 计算历史最高点 (High-Water Mark)
    running_max = cumulative_returns.cummax()

    # 计算回撤序列 (当前值距离历史最高点的百分比)
    drawdown = (cumulative_returns / running_max) - 1

    # --- 3. 结果提取 (修复Bug) ---
    max_drawdown = drawdown.min()

    # 如果没有回撤 (策略一直盈利), 直接返回
    if max_drawdown == 0:
        return 0.0, None, None

    end_date = drawdown.idxmin()

    # 【核心Bug修复】
    # 在最大回撤结束点之前的序列中寻找高点
    peak_series = cumulative_returns.loc[:end_date]
    start_date = peak_series.idxmax()

    return max_drawdown, start_date, end_date


def fama_macbeth_regression(
        factor_df: pd.DataFrame, # <-- 接收原始 T 日因子
        returns_calculator: Callable,
        price_df: pd.DataFrame,
        forward_returns_period: int = 20,
        circ_mv_df_shifted: pd.DataFrame = None, # <-- 接收原始 T 日权重
        neutral_factors: Dict[str, pd.DataFrame] = None # 因为factor_df以及除杂过，现在不需要再次进行除杂了
) -> Tuple[Series,Series, Dict[str, Any]]:
    """
    对单个因子进行Fama-MacBeth回归检验。
    此版本逻辑结构清晰，代码健壮，并使用Newey-West标准误修正t检验，符合学术界和业界的严格标准。
    return:Series 表示纯因子带来的收益，纯收益
    """
    # # 初始化logger
    # from quant_lib.config_manager.logger_config import setup_logger
    # logger = setup_logger(__name__)
    logger.info(f"开始Fama-MacBeth回归分析 (前向收益期: {forward_returns_period}天)")

    # --- 0. 前置检查 ---
    if not HAS_STATSMODELS:
        raise ValueError("statsmodels未安装，无法执行Fama-MacBeth回归")
    if factor_df.empty:
        raise ValueError("输入的因子数据为空")
    factor_std = factor_df.stack().std()
    if factor_std < 1e-6 or np.isnan(factor_std):
        raise ValueError("因子值在所有截面上几乎无变化或全为NaN，无法进行回归。")
    # 【新增检查】如果传入了不应有的中性化因子，发出警告
    if neutral_factors:
        raise ValueError(
            "警告：已向本函数传入了预处理后的因子，但 neutral_factors 参数不为空。回归将继续，但请确认这是否是预期行为。")

    # --- 1. 数据准备 (已简化) ---
    logger.info("\t步骤1: 准备和对齐数据...")
    try:
        # 步骤A: 计算目标结果 (Y变量)
        forward_returns = returns_calculator(period=forward_returns_period)

        # 步骤B: 直接构建对齐字典。由于 neutral_factors 为空，流程大大简化。
        all_dfs_to_align = {
            'factor': factor_df,
            'returns': forward_returns
        }
        if circ_mv_df_shifted is not None:
            all_dfs_to_align['weights'] = circ_mv_df_shifted

        aligned_dfs = align_dataframes(all_dfs_to_align)

        # 步骤C: 从对齐结果中分离出 Y 和 X 的“原材料”
        aligned_returns = aligned_dfs['returns']
        aligned_factor = aligned_dfs['factor']
        aligned_weights = aligned_dfs.get('weights')  # 使用 .get() 安全获取


    except Exception as e:
        logger.error(f"数据准备或对齐失败: {e}")
        return pd.Series(dtype=float), {'error': f'数据准备失败: {e}'}

    # --- 2. 逐日截面回归 ---
    factor_returns = []
    factor_t_stats = [] # <--- 【新增】用于存储每日t值的列表
    valid_dates = []
    total_dates_to_run = len(aligned_factor.index)

    for date in aligned_factor.index:
        # a) 准备当天数据
        y_series = aligned_returns.loc[date].rename('returns')

        # 【简化】X变量的构建变得非常简单，只包含目标因子
        x_df = pd.DataFrame({'factor': aligned_factor.loc[date]})

        all_data_for_date = [y_series, x_df]
        if aligned_weights is not None:
            weights_series = np.sqrt(aligned_weights.loc[date]).rename('weights')
            all_data_for_date.append(weights_series)

        # b) 有效域掩码：显式定义当日有效样本
        # 先合并所有数据
        combined_df = pd.concat(all_data_for_date, axis=1, join='outer')
        # 显式定义有效掩码：所有变量都不为NaN
        valid_mask = combined_df.notna().all(axis=1)
        # 应用掩码
        combined_df = combined_df[valid_mask]

        # 样本量检查现在更简单
        if len(combined_df) < 10:  # 对于单变量回归，可以设置一个较小的绝对值门槛
            continue

        # c) 执行模型
        try:
            y_final = combined_df['returns']
            X_final = sm.add_constant(combined_df[['factor']])  # 只对因子列回归

            if aligned_weights is not None:
                w_final = combined_df['weights']
                if (w_final <= 0).any() or w_final.isna().any():
                    continue
                model = sm.WLS(y_final, X_final, weights=w_final).fit()
            else:
                model = sm.OLS(y_final, X_final).fit()

            if 'factor' not in model.params.index:
                continue

            factor_return = model.params['factor']
            # 【新增】提取因子对应的t值
            t_stat_daily = model.tvalues['factor']
            if np.isnan(factor_return) or np.isinf(factor_return):
                continue

            factor_returns.append(factor_return)
            factor_t_stats.append(t_stat_daily) # <--- 【新增】存入每日t值
            valid_dates.append(date)
        except (np.linalg.LinAlgError, ValueError):
            raise ValueError("失败")

    # --- 3. 分析与报告 ---
    # logger.info("\t步骤3: 分析回归结果并生成报告...")
    num_success_dates = len(factor_returns)
    num_skipped_dates = total_dates_to_run - num_success_dates

    if num_success_dates < 20:
        raise ValueError(f"有效回归期数({num_success_dates})过少，无法进行可靠的统计检验。")

    # --- 计算“t值绝对值均值” ---
    fm_t_stats_series = pd.Series(factor_t_stats, index=pd.to_datetime(valid_dates),
                                   name='factor_t_stats')

    mean_abs_t_stat = fm_t_stats_series.abs().mean()
    fm_returns_series = pd.Series(factor_returns, index=pd.to_datetime(valid_dates), name='factor_returns')
    mean_factor_return = fm_returns_series.mean()

    # 修正：正确实现Newey-West t检验
    t_stat, p_value = np.nan, np.nan
    try:
        series_clean = fm_returns_series.dropna()
        n_obs = len(series_clean)

        # 构造回归：因子收益率 = 常数项 + 误差项
        # 检验常数项（即均值）是否显著不为0
        X_const = np.ones(n_obs).reshape(-1, 1)

        max_lags = min(int(n_obs ** 0.25), n_obs // 4)  # 防止lag过大

        nw_model = sm.OLS(series_clean, X_const).fit(
            cov_type='HAC',
            cov_kwds={'maxlags': max_lags, 'use_correction': True}
        )

        t_stat = nw_model.tvalues[0]
        p_value = nw_model.pvalues[0]
        # logger.info(f"已使用Newey-West(lags={max_lags})修正t检验。")

    except Exception as e:
        log_warning(f"Newey-West t检验计算失败: {e}。回退到标准t检验。")
        try:
            series_clean = fm_returns_series.dropna()
            t_stat, p_value = ttest_1samp(series_clean, 0)
        except Exception as e2:
            raise ValueError(f"标准t检验也失败: {e2}")

    # 显著性判断
    is_significant = abs(t_stat) > 1.96 if not np.isnan(t_stat) else False

    significance_level = ''
    significance_desc = '无法计算'

    if not np.isnan(t_stat):
        # 添加显著性评级
        if abs(t_stat) > 2.58:
            significance_level = "⭐⭐⭐"
            significance_desc = "1%显著"
        elif abs(t_stat) > 1.96:
            significance_level = "⭐⭐"
            significance_desc = "5%显著"
        elif abs(t_stat) > 1.64:
            significance_level = "⭐"
            significance_desc = "10%显著"
        else:
            significance_level = ""
            significance_desc = "不显著"

    logger.info(
        f"\t\t回归完成。总天数: {total_dates_to_run}, 成功回归天数: {num_success_dates} \t 因子平均收益率: {mean_factor_return:.6f}, t统计量: {t_stat:.4f}, 显著性: {significance_desc}")

    if is_significant:
        logger.info("\t\t结论: ✓ 因子有效性得到验证！")
    else:
        logger.info("\t\t结论: ✗ 无法在统计上拒绝因子无效的原假设。")
    fm_summary = {
        'mean_factor_return': mean_factor_return,
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_abs_t_stat': mean_abs_t_stat,  # <--- 【新增】存入我们新计算的指标
        'is_significant': is_significant,
        'significance_level': significance_level,
        'significance_desc': significance_desc,
        'num_total_periods': total_dates_to_run,
        'num_valid_periods': num_success_dates,
        'success_rate': num_success_dates / total_dates_to_run if total_dates_to_run > 0 else 0,  # 有多大比例的交易日成功地完成了回归
        # 'fm_returns_series': fm_returns_series,
        'skipped_dates': num_skipped_dates,
    }

    return fm_returns_series,fm_t_stats_series, fm_summary


def fama_macbeth(
        factor_data: pd.DataFrame,#以经shift处理过
        returns_calculator,
        close_df: pd.DataFrame,
        neutral_dfs: Dict[str, pd.DataFrame], #以经shift处理过
        forward_periods,
        circ_mv_df_shifted: pd.DataFrame,
        factor_name: str) -> Tuple[Dict[str, pd.DataFrame],Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Fama-MacBeth回归法测试（黄金标准）

    Args:
        factor_data: 预处理后的因子数据
        factor_name: 因子名称

    Returns:
        Fama-MacBeth回归结果字典
    """
    fm_summary_dict = {}
    fm_t_stats_series_dict = {}
    fm_returns_series_dict = {}
    for period in forward_periods:
        # 运行Fama-MacBeth回归
        fm_returns_series, fm_t_stats_series, fm_summary= fama_macbeth_regression(
            factor_df=factor_data,
            returns_calculator=returns_calculator,
            price_df=close_df,
            forward_returns_period=period,
            circ_mv_df_shifted=circ_mv_df_shifted,  # <-- 传入 流通市值作为权重，执行WLS
            neutral_factors=neutral_dfs  # <-- 传入市值和行业作为控制变量
        )
        fm_returns_series_dict[f'{period}d'] = fm_returns_series
        fm_t_stats_series_dict[f'{period}d'] = fm_t_stats_series
        fm_summary_dict[f'{period}d'] = fm_summary
    return fm_returns_series_dict,fm_t_stats_series_dict, fm_summary_dict


import pandas as pd
import numpy as np
from typing import Dict, List

# 假设 logger 已经配置好
import logging

logger = logging.getLogger(__name__)


# 只是用于绘图
def calculate_quantile_daily_returns(
        factor_df: pd.DataFrame,
        returns_calculator,  # 具体化Callable
        n_quantiles
) ->   pd.DataFrame:
    """
    计算因子分层组合的每日收益率。
    1. 修正了函数签名，防止因参数位置错误导致的TypeError。
    2. 增加了对因子数据类型的强制转换，确保qcut函数安全运行。

    Args:
        factor_df (pd.DataFrame): 因子值DataFrame (index=date, columns=stock)。
                                  这是T-1日的信息。
        price_df (pd.DataFrame): 每日收盘价矩阵 (index=date, columns=stock)。
        n_quantiles (int): (关键字参数) 要划分的分位数数量。

    Returns:
        Dict[str, pd.DataFrame]: 只有一个key的字典，值是分层组合的每日收益DataFrame。
    """
    logger.info("  > 正在计算分层组合的【每日】收益率 (用于绘图)...")
    forward_returns_1d = returns_calculator(period=1)
    # 2. 有效域掩码：显式定义分析样本
    # 单一事实来源 - 明确定义所有有效的(date, stock)坐标点
    valid_mask = factor_df.notna() & forward_returns_1d.notna()#好的合集

    # 应用掩码，确保因子和收益具有完全相同的NaN分布
    final_factor = factor_df.where(valid_mask)#坏的合集都为nan
    final_returns_1d = forward_returns_1d.where(valid_mask)#坏的合集都为nan stock进行操作，丢的nan都是一样的，就可有无脑concat了

    # 数据转换：从"宽表"到"长表"（现在是安全的）
    factor_long = final_factor.stack().rename('factor')
    returns_1d_long = final_returns_1d.stack().rename('return_1d')

    # 合并数据（不再需要dropna，因为已经完全对齐）
    merged_df = pd.concat([factor_long, returns_1d_long], axis=1)

    if merged_df.empty:
        raise ValueError("  > 因子和单日收益数据没有重叠，无法计算分层收益。")

    # 3. 确保因子值为数值类型
    merged_df['factor'] = pd.to_numeric(merged_df['factor'], errors='coerce')
    merged_df=merged_df.dropna(subset=['factor'], inplace=False)

    # 4. 稳健的分组
    merged_df['quantile'] = merged_df.groupby(level=0)['factor'].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop') + 1
    )

    # 5. 计算各分位数组合的每日平均收益
    daily_quantile_returns = merged_df.groupby([merged_df.index.get_level_values(0), 'quantile'])[
        'return_1d'].mean()

    # 6. 数据转换回“宽表”
    quantile_returns_wide = daily_quantile_returns.unstack()
    quantile_returns_wide.columns = [f'Q{int(col)}' for col in quantile_returns_wide.columns]

    # 7. 计算多空组合的每日收益（价差）
    top_q_col = f'Q{n_quantiles}'
    bottom_q_col = 'Q1'

    if top_q_col in quantile_returns_wide.columns and bottom_q_col in quantile_returns_wide.columns:
        quantile_returns_wide['TopMinusBottom'] = quantile_returns_wide[top_q_col] - quantile_returns_wide[
            bottom_q_col]
    else:
        quantile_returns_wide['TopMinusBottom'] = np.nan

    # 8. 返回结果
    # 我们用一个固定的key，比如 '21d'，让绘图函数能找到它
    return  quantile_returns_wide.sort_index(axis=1)


def _calculate_newey_west_tstat(ic_series: pd.Series) -> Tuple[float, float]:
        """
        计算 Newey-West 调整的 t-stat（针对 IC 均值的 HAC 标准误）
        Args:
            ic_series: pd.Series，IC 时间序列（可包含 NaN）
        Returns:
            (nw_t_stat, nw_p_value)：Newey-West t-stat 以及双侧 p-value
        """
        # 先去 NA 并计算样本数
        ic_nonan = ic_series.dropna()
        n = ic_nonan.size

        # 样本过小时直接返回不可显著
        if n < 10:
            return 0.0, 1.0

        try:
            ic_values = ic_nonan.values.astype(float)
            ic_mean = ic_values.mean()
            residuals = ic_values - ic_mean

            # Newey-West 推荐的 lag 选择（常用经验式）
            # lag = floor(4 * (n/100)^(2/9)), 并确保 <= n-1
            max_lag = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
            max_lag = max(1, min(max_lag, n - 1))

            # 计算 long-run variance (HAC)
            # gamma_0 = sum(residuals^2) / n
            gamma0 = np.sum(residuals ** 2) / n
            long_run_variance = gamma0

            for lag in range(1, max_lag + 1):
                # 自协方差 gamma_lag = sum_{t=lag}^{n-1} e_t e_{t-lag} / n
                gamma_lag = np.sum(residuals[:-lag] * residuals[lag:]) / n
                # Bartlett 权重
                weight = 1.0 - lag / (max_lag + 1.0)
                long_run_variance += 2.0 * weight * gamma_lag

            # 数值保护：不允许负数（可能由数值误差导致）
            long_run_variance = max(long_run_variance, 0.0)

            # 标准误（均值的方差估计）
            nw_se = math.sqrt(long_run_variance / n) if long_run_variance > 0 else 0.0

            if nw_se <= 0.0:
                return 0.0, 1.0

            nw_t_stat = float(ic_mean / nw_se)

            # p-value（双侧），这里用 t 分布 df = n-1（近似）
            nw_p_value = float(2.0 * (1.0 - stats.t.cdf(abs(nw_t_stat), df=n - 1)))

            return nw_t_stat, nw_p_value

        except Exception:
            # 作为兜底：回退到常规 t 统计量（样本均值 / (std/sqrt(n))）
            raise ValueError("无法计算Newey-West t-stat")
            # try:
            #     ic_vals = ic_nonan.values.astype(float)
            #     n2 = ic_vals.size
            #     if n2 < 2:
            #         return 0.0, 1.0
            #     ic_mean = ic_vals.mean()
            #     ic_std = ic_vals.std(ddof=1)  # 样本标准差
            #     if ic_std <= 0.0:
            #         return 0.0, 1.0
            #     t_stat = float(ic_mean / (ic_std / math.sqrt(n2)))
            #     p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n2 - 1)))
            #     return t_stat, p_value
            # except Exception:
            #     return 0.0, 1.0