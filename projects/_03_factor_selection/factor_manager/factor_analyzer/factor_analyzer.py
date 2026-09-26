"""因子有效性研究：processed 因子的 IC、分层和换手。"""

from functools import partial
from typing import Callable, Dict, Tuple

import pandas as pd
from pandas import DataFrame, Series

from projects._03_factor_selection.factor_manager.factor_composite.factor_synthesizer import (
    FactorSynthesizer,
)
from projects._03_factor_selection.utils.IndustryMap import PointInTimeIndustryMap
from projects._03_factor_selection.utils.factor_processor import FactorProcessor
from quant_lib import logger
from quant_lib.config.logger_config import log_flow_start
from quant_lib.evaluation.evaluation import (
    calculate_forward_returns_tradable_o2o,
    calculate_ic,
    calculate_quantile_daily_returns,
    calculate_quantile_returns,
    calculate_top_quantile_turnover_dict,
    quantile_stats_result,
)


# 执行 prepare_industry_dummies 对应逻辑。
def prepare_industry_dummies(
    pit_map: PointInTimeIndustryMap,
    trade_dates: pd.DatetimeIndex,
    stock_codes: list,
    level: str = "l1_code",
    drop_first: bool = True,
) -> Dict[str, pd.DataFrame]:
    """按时点行业映射生成预处理所需的行业哑变量。"""
    # 逐日读取当时可得的行业归属，拼成带日期的长表，避免使用当前行业覆盖历史。
    daily_maps = []
    for date in trade_dates:
        daily_map = pit_map.get_map_for_date(date)
        if not daily_map.empty:
            daily_map = daily_map.reset_index()
            daily_map["date"] = date
            daily_maps.append(daily_map)

    if not daily_maps:
        return {}

    # 按指定行业层级展开哑变量；date/ts_code 必须唯一，才能恢复为日期×股票宽表。
    long_frame = pd.concat(daily_maps)
    dummies = pd.get_dummies(
        long_frame[level], prefix="industry", dtype=float, drop_first=drop_first
    )
    dummy_frame = pd.concat([long_frame[["date", "ts_code"]], dummies], axis=1)
    if dummy_frame.duplicated(subset=["date", "ts_code"]).any():
        raise ValueError(f"行业映射存在重复的 date/ts_code，无法生成 {level} 哑变量")

    # 每个行业单独生成一张与因子网格一致的矩阵，非所属行业及补齐位置记为 0。
    result = {}
    for column in dummies.columns:
        pivoted = dummy_frame.pivot(index="date", columns="ts_code", values=column).fillna(0)
        result[column] = pivoted.reindex(index=trade_dates, columns=stock_codes).fillna(0)
    return result


class FactorAnalyzer:
    """运行正式的 processed 因子有效性研究。"""

    # 执行 __init__ 对应逻辑。
    def __init__(self, factor_manager):
        self.factor_manager = factor_manager
        self.config = factor_manager.data_manager.config
        evaluation = self.config["evaluation"]
        self.test_common_periods = evaluation["forward_periods"]
        self.n_quantiles = evaluation["quantiles"]
        self.factor_processor = FactorProcessor(self.config)

    # 执行 test_ic_analysis 对应逻辑。
    def test_ic_analysis(
        self,
        factor_data: pd.DataFrame,
        returns_calculator: Callable,
        close_df: pd.DataFrame,
    ) -> Tuple[Dict[str, Series], Dict[str, pd.DataFrame]]:
        # 逐周期计算非重叠 Spearman IC 及统计，截面至少需要 30 只因子/收益有效配对股票。
        return calculate_ic(
            factor_data,
            close_df,
            forward_periods=self.test_common_periods,
            method="spearman",
            returns_calculator=returns_calculator,
            min_stocks=30,
        )

    # 执行 test_quantile_backtest 对应逻辑。
    def test_quantile_backtest(
        self,
        factor_data: pd.DataFrame,
        returns_calculator: Callable,
        close_df: pd.DataFrame,
    ) -> Tuple[Dict[str, DataFrame], Dict[str, DataFrame]]:
        # 先按每日因子排序分层，计算各持有周期的组内等权收益及最高组减最低组收益。
        period_returns = calculate_quantile_returns(
            factor_data,
            returns_calculator,
            close_df,
            n_quantiles=self.n_quantiles,
            forward_periods=self.test_common_periods,
        )
        # 再按周期抽取非重叠节点，汇总收益、夏普、回撤和分层单调性。
        return quantile_stats_result(period_returns, self.n_quantiles)

    # 执行 test_turnover_result 对应逻辑。
    def test_turnover_result(self, factor_data: pd.DataFrame) -> dict:
        # 按各周期调仓日比较组合成员变化；下游封装当前固定统计第 5 组。
        turnover_by_period = calculate_top_quantile_turnover_dict(
            factor_df=factor_data,
            n_quantiles=self.n_quantiles,
            forward_periods=self.test_common_periods,
        )
        # 将逐期成员换手汇总为均值，再按每年 252 个交易日折算年化换手。
        return {
            period: {
                "turnover_mean": series.mean(),
                "turnover_annual": series.mean() * (252 / int(period[:-1])),
            }
            for period, series in turnover_by_period.items()
        }

    # 执行 analyze_processed_factor 对应逻辑。
    def analyze_processed_factor(
        self,
        factor_name: str,
        factor_data: pd.DataFrame,
        stock_pool_name: str,
        returns_calculator: Callable,
        already_processed: bool,
    ) -> dict:
        """生成正式研究所需且仅需的一组 processed 结果。"""
        # 复合因子在合成时已完成子因子预处理及合成后标准化；普通因子在此走完整预处理。
        if already_processed:
            processed = factor_data
        else:
            processed = self._process_single_factor(
                factor_name, factor_data, stock_pool_name
            )

        # 当前目标只在预处理完成后应用一次冻结方向；Inner 保留待研究方向。
        if self.config["stage"] != "inner":
            processed = processed * self.factor_manager.get_resolved_direction(factor_name)

        # 准备对齐后的价格矩阵供评估入口检查；实际 O2O 标签由已绑定开盘价的计算器生成。
        close_df = self.factor_manager.get_prepare_aligned_factor_for_analysis(
            "close_hfq", stock_pool_name, True
        )
        # 预处理只依赖 T 日信息；完成后按 T+1 开盘条件确定评价样本。
        entry_pool = self.factor_manager.data_manager.get_entry_pool(stock_pool_name)
        evaluation_factor = processed.where(entry_pool)
        log_flow_start(f"因子 {factor_name} 的 processed 信号进入 IC、分层和换手测试")
        # 三类评价共用同一可建仓样本：IC 衡量排序预测能力，分层比较组合收益，换手衡量成员变化。
        ic_series, ic_stats = self.test_ic_analysis(
            evaluation_factor, returns_calculator, close_df
        )
        quantile_returns, quantile_stats = self.test_quantile_backtest(
            evaluation_factor, returns_calculator, close_df
        )
        # 额外生成 period=1 的每日分层收益序列，供保存及绘图使用。
        quantile_daily_returns = calculate_quantile_daily_returns(
            evaluation_factor, returns_calculator, self.n_quantiles
        )
        return {
            # 保存实际评价方向的信号，不受 T+1 成交状态影响；合成子因子另行计算。
            "processed_factor_df": processed,
            "ic_series_periods_dict_processed": ic_series,
            "ic_stats_periods_dict_processed": ic_stats,
            "quantile_returns_series_periods_dict_processed": quantile_returns,
            "q_daily_returns_df_processed": quantile_daily_returns,
            "quantile_stats_periods_dict_processed": quantile_stats,
            "top_q_turnover_stats_periods_dict": self.test_turnover_result(evaluation_factor),
        }

    # 执行 _process_single_factor 对应逻辑。
    def _process_single_factor(
        self,
        factor_name: str,
        factor_data: pd.DataFrame,
        stock_pool_name: str,
    ) -> pd.DataFrame:
        # 先准备当前因子所需的市值、Beta 或行业风险变量，并取得风格分类。
        neutral_dfs, style_category = self.prepare_data_for_process_factor(
            factor_name,
            factor_data.index,
            factor_data.columns,
            stock_pool_name,
        )
        # 依配置执行去极值、中性化和标准化；行业步骤使用按日期查询的历史行业映射。
        return self.factor_processor.process_factor(
            factor_df=factor_data,
            expected_mask=self.factor_manager.data_manager.stock_pools_dict[stock_pool_name],
            target_factor_name=factor_name,
            neutral_dfs=neutral_dfs,
            style_category=style_category,
            pit_map=self.factor_manager.data_manager.get_preprocessing_industry_map(),
            need_standardize=True,
        )

    # 执行 prepare_data_for_process_factor 对应逻辑。
    def prepare_data_for_process_factor(
        self,
        factor_name: str,
        trade_dates: pd.DatetimeIndex,
        stock_codes: list,
        stock_pool_name: str,
    ) -> tuple[dict, str]:
        # 未启用中性化时无需准备回归矩阵；启用时按配置剔除目标因子自身对应的风险变量。
        style_category = self.factor_manager.get_style_category(factor_name)
        neutralization = self.factor_processor.preprocessing_config["neutralization"]
        if not neutralization["enable"]:
            return {}, style_category

        factors_to_neutralize = (
            self.factor_processor.get_regression_need_neutral_factor_list(factor_name)
        )
        neutral_dfs = {}

        # 市值变量采用与目标股票池对齐的对数流通市值。
        if "market_cap" in factors_to_neutralize:
            neutral_dfs["log_circ_mv"] = (
                self.factor_manager.get_prepare_aligned_factor_for_analysis(
                    "log_circ_mv", stock_pool_name, True
                )
            )

        # Beta 需要真实指数作为基准，因此先从股票池配置取得指数代码再请求计算。
        if "pct_chg_beta" in factors_to_neutralize:
            beta_request = (
                "beta",
                self.factor_manager.data_manager.get_stock_pool_index_code_by_name(
                    stock_pool_name
                ),
            )
            neutral_dfs["pct_chg_beta"] = (
                self.factor_manager.get_prepare_aligned_factor_for_analysis(
                    beta_request, stock_pool_name, True
                )
            )

        # 把指定层级的历史行业归属转为回归哑变量，默认删除一列以配合截距项。
        if "industry" in factors_to_neutralize:
            industry_level = neutralization["by_industry"]["industry_level"]
            industry_dummies = prepare_industry_dummies(
                self.factor_manager.data_manager.pit_map,
                trade_dates,
                stock_codes,
                level=industry_level,
            )
            neutral_dfs.update(industry_dummies)
        return neutral_dfs, style_category

    # 执行 prepare_data_for_entity_service 对应逻辑。
    def prepare_data_for_entity_service(
        self, factor_name: str, stock_pool_name: str
    ) -> tuple[pd.DataFrame, bool, dict]:
        data_manager = self.factor_manager.data_manager
        # 复合因子走子因子等权合成；普通因子走原始计算及研究股票池对齐。
        is_composite = data_manager.is_composite_factor(factor_name)
        if is_composite:#todo
            factor_data = FactorSynthesizer(
                self.factor_manager, self, self.factor_processor
            ).synthesize_equal_factor(factor_name, stock_pool_name)
        else:
            factor_data = self.factor_manager.get_prepare_aligned_factor_for_analysis(
                factor_name, stock_pool_name, True
            )

        configured_calculators = data_manager.config["evaluation"]["returns_calculator"]
        # 将后复权开盘价和次日买入资格对齐到信号网格，后续各周期共用。
        open_df = self.factor_manager.get_raw_factor("open_hfq").reindex(
            index=factor_data.index, columns=factor_data.columns
        )
        entry_mask = data_manager.get_entry_pool(stock_pool_name).reindex(
            index=factor_data.index, columns=factor_data.columns
        )
        # 用 partial 绑定价格和买入掩码，调用方只需传持有期；T 行标签从 T+1 开盘开始。
        calculators = {
            "o2o": partial(
                calculate_forward_returns_tradable_o2o,
                open_df=open_df,
                entry_mask=entry_mask,
            )
        }
        return (
            factor_data,
            is_composite,
            {name: calculators[name] for name in configured_calculators},
        )

    def evaluate_factor(
        self, factor_name: str, stock_pool_index_name: str
    ) -> Dict[str, dict]:
        """计算并返回 processed 评估结果，不保存文件。"""
        # 准备信号及收益计算器，同时标记复合因子，避免再次执行整套预处理。
        factor_data, is_composite, calculators = (
            self.prepare_data_for_entity_service(factor_name, stock_pool_index_name)
        )
        all_results = {}
        # 逐收益口径汇总 processed 评估结果，文件保存和方向确定交给 Runner。
        for calculator_name, calculator in calculators.items():
            results = self.analyze_processed_factor(
                factor_name,
                factor_data,
                stock_pool_index_name,
                calculator,
                already_processed=is_composite,
            )
            all_results[calculator_name] = results
        return all_results
