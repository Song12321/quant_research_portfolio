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
    daily_maps = []
    for date in trade_dates:
        daily_map = pit_map.get_map_for_date(date)
        if not daily_map.empty:
            daily_map = daily_map.reset_index()
            daily_map["date"] = date
            daily_maps.append(daily_map)

    if not daily_maps:
        return {}

    long_frame = pd.concat(daily_maps)
    dummies = pd.get_dummies(
        long_frame[level], prefix="industry", dtype=float, drop_first=drop_first
    )
    dummy_frame = pd.concat([long_frame[["date", "ts_code"]], dummies], axis=1)
    if dummy_frame.duplicated(subset=["date", "ts_code"]).any():
        raise ValueError(f"行业映射存在重复的 date/ts_code，无法生成 {level} 哑变量")

    result = {}
    for column in dummies.columns:
        pivoted = dummy_frame.pivot(index="date", columns="ts_code", values=column).fillna(0)
        result[column] = pivoted.reindex(index=trade_dates, columns=stock_codes).fillna(0)
    return result


class FactorAnalyzer:
    """运行正式的 processed 因子有效性研究。"""

    # 执行 __init__ 对应逻辑。
    def __init__(self, factor_manager):
        if factor_manager is None or factor_manager.data_manager is None:
            raise ValueError("FactorAnalyzer 必须传入已绑定 DataManager 的 FactorManager")

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
        period_returns = calculate_quantile_returns(
            factor_data,
            returns_calculator,
            close_df,
            n_quantiles=self.n_quantiles,
            forward_periods=self.test_common_periods,
        )
        return quantile_stats_result(period_returns, self.n_quantiles)

    # 执行 test_turnover_result 对应逻辑。
    def test_turnover_result(self, factor_data: pd.DataFrame) -> dict:
        turnover_by_period = calculate_top_quantile_turnover_dict(
            factor_df=factor_data,
            n_quantiles=self.n_quantiles,
            forward_periods=self.test_common_periods,
        )
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
        if already_processed:
            processed = factor_data
        else:
            processed = self._process_single_factor(
                factor_name, factor_data, stock_pool_name
            )

        close_df = self.factor_manager.get_prepare_aligned_factor_for_analysis(
            "close_hfq", stock_pool_name, True
        )
        # 预处理只依赖 T 日信息；完成后按 T+1 开盘条件确定评价样本。
        entry_pool = self.factor_manager.data_manager.get_entry_pool(stock_pool_name)
        evaluation_factor = processed.where(entry_pool)
        log_flow_start(f"因子 {factor_name} 的 processed 信号进入 IC、分层和换手测试")
        ic_series, ic_stats = self.test_ic_analysis(
            evaluation_factor, returns_calculator, close_df
        )
        quantile_returns, quantile_stats = self.test_quantile_backtest(
            evaluation_factor, returns_calculator, close_df
        )
        quantile_daily_returns = calculate_quantile_daily_returns(
            evaluation_factor, returns_calculator, self.n_quantiles
        )
        return {
            # 保留未受 T+1 成交状态影响的信号，供复合因子后续合成、预处理。
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
        neutral_dfs, style_category = self.prepare_data_for_process_factor(
            factor_name,
            factor_data.index,
            factor_data.columns,
            stock_pool_name,
        )
        return self.factor_processor.process_factor(
            factor_df=factor_data,
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
        style_category = self.factor_manager.get_style_category(factor_name)
        neutralization = self.factor_processor.preprocessing_config["neutralization"]
        if not neutralization["enable"]:
            return {}, style_category

        factors_to_neutralize = (
            self.factor_processor.get_regression_need_neutral_factor_list(factor_name)
        )
        neutral_dfs = {}

        if "market_cap" in factors_to_neutralize:
            neutral_dfs["log_circ_mv"] = (
                self.factor_manager.get_prepare_aligned_factor_for_analysis(
                    "log_circ_mv", stock_pool_name, True
                )
            )

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
        is_composite = data_manager.is_composite_factor(factor_name)
        if is_composite:
            factor_data = FactorSynthesizer(
                self.factor_manager, self, self.factor_processor
            ).synthesize_equal_factor(factor_name, stock_pool_name)
        else:
            factor_data = self.factor_manager.get_prepare_aligned_factor_for_analysis(
                factor_name, stock_pool_name, True
            )

        configured_calculators = data_manager.config["evaluation"]["returns_calculator"]
        unsupported = set(configured_calculators) - {"o2o"}
        if unsupported:
            raise ValueError(f"evaluation.returns_calculator 仅支持 o2o，实际: {sorted(unsupported)}")
        open_df = self.factor_manager.get_raw_factor("open_hfq").reindex(
            index=factor_data.index, columns=factor_data.columns
        )
        entry_mask = data_manager.get_entry_pool(stock_pool_name).reindex(
            index=factor_data.index, columns=factor_data.columns
        )
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
        factor_data, is_composite, calculators = (
            self.prepare_data_for_entity_service(factor_name, stock_pool_index_name)
        )
        all_results = {}
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
