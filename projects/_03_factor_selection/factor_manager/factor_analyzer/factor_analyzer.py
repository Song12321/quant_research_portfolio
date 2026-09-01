"""因子有效性研究：processed 因子的 IC、分层和换手。"""

from functools import partial
from typing import Callable, Dict, Tuple

import pandas as pd
from pandas import DataFrame, Series

from projects._03_factor_selection.factor_manager.factor_composite.factor_synthesizer import (
    FactorSynthesizer,
)
from projects._03_factor_selection.factor_manager.factor_manager import FactorResultsManager
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

    def __init__(self, factor_manager):
        if factor_manager is None or factor_manager.data_manager is None:
            raise ValueError("FactorAnalyzer 必须传入已绑定 DataManager 的 FactorManager")

        self.factor_manager = factor_manager
        self.config = factor_manager.data_manager.config
        evaluation = self.config["evaluation"]
        self.test_common_periods = evaluation["forward_periods"]
        self.n_quantiles = evaluation["quantiles"]
        self.factor_processor = FactorProcessor(self.config)
        self.factor_results_manager = FactorResultsManager()

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

    def analyze_processed_factor(
        self,
        factor_name: str,
        factor_data_shifted: pd.DataFrame,
        stock_pool_name: str,
        returns_calculator: Callable,
        already_processed: bool,
    ) -> dict:
        """生成正式研究所需且仅需的一组 processed 结果。"""
        if already_processed:
            processed = self._clip_composite_outliers(factor_data_shifted, factor_name)
        else:
            processed = self._process_single_factor(
                factor_name, factor_data_shifted, stock_pool_name
            )

        close_df = self.factor_manager.get_prepare_aligned_factor_for_analysis(
            "close_hfq", stock_pool_name, True
        )
        log_flow_start(f"因子 {factor_name} 的 processed 信号进入 IC、分层和换手测试")
        ic_series, ic_stats = self.test_ic_analysis(
            processed, returns_calculator, close_df
        )
        quantile_returns, quantile_stats = self.test_quantile_backtest(
            processed, returns_calculator, close_df
        )
        quantile_daily_returns = calculate_quantile_daily_returns(
            processed, returns_calculator, self.n_quantiles
        )
        return {
            "processed_factor_df": processed,
            "ic_series_periods_dict_processed": ic_series,
            "ic_stats_periods_dict_processed": ic_stats,
            "quantile_returns_series_periods_dict_processed": quantile_returns,
            "q_daily_returns_df_processed": quantile_daily_returns,
            "quantile_stats_periods_dict_processed": quantile_stats,
            "top_q_turnover_stats_periods_dict": self.test_turnover_result(processed),
        }

    def _process_single_factor(
        self,
        factor_name: str,
        factor_data_shifted: pd.DataFrame,
        stock_pool_name: str,
    ) -> pd.DataFrame:
        neutral_dfs, style_category = self.prepare_data_for_process_factor(
            factor_name,
            factor_data_shifted.index,
            factor_data_shifted.columns,
            stock_pool_name,
        )
        return self.factor_processor.process_factor(
            factor_df_shifted=factor_data_shifted,
            target_factor_name=factor_name,
            neutral_dfs=neutral_dfs,
            style_category=style_category,
            pit_map=self.factor_manager.data_manager.pit_map,
            need_standardize=True,
        )

    @staticmethod
    def _clip_composite_outliers(
        factor_df: pd.DataFrame, factor_name: str
    ) -> pd.DataFrame:
        """保留旧正式组合评价入口的逐日 1%/99% 极值处理。"""
        factor_flat = factor_df.stack().dropna()
        q01 = factor_flat.quantile(0.01)
        q99 = factor_flat.quantile(0.99)
        outlier_ratio = ((factor_flat < q01) | (factor_flat > q99)).mean()
        if outlier_ratio <= 0.02:
            return factor_df

        processed = factor_df.copy()
        for date in factor_df.index:
            daily_values = factor_df.loc[date].dropna()
            if len(daily_values) > 10:
                lower = daily_values.quantile(0.01)
                upper = daily_values.quantile(0.99)
                processed.loc[date] = daily_values.clip(lower=lower, upper=upper)
        logger.info(f"组合因子 {factor_name} 保留旧口径极值处理，极值比例 {outlier_ratio:.1%}")
        return processed

    def prepare_data_for_process_factor(
        self,
        factor_name: str,
        trade_dates: pd.DatetimeIndex,
        stock_codes: list,
        stock_pool_name: str,
    ) -> tuple[dict, str]:
        style_category = self.factor_manager.get_style_category(factor_name)
        neutralization = self.factor_processor.preprocessing_config["neutralization"]
        industry_level = neutralization["by_industry"]["industry_level"]
        industry_dummies = prepare_industry_dummies(
            self.factor_manager.data_manager.pit_map,
            trade_dates,
            stock_codes,
            level=industry_level,
        )
        beta_request = (
            "beta",
            self.factor_manager.data_manager.get_stock_pool_index_code_by_name(
                stock_pool_name
            ),
        )
        neutral_dfs = {
            "log_circ_mv": self.factor_manager.get_prepare_aligned_factor_for_analysis(
                "log_circ_mv", stock_pool_name, True
            ),
            "pct_chg_beta": self.factor_manager.get_prepare_aligned_factor_for_analysis(
                beta_request, stock_pool_name, True
            ),
            **{name: frame.shift(1, fill_value=0) for name, frame in industry_dummies.items()},
        }
        return neutral_dfs, style_category

    def prepare_data_for_entity_service(
        self, factor_name: str, stock_pool_name: str
    ) -> tuple[pd.DataFrame, bool, str, str, str, dict]:
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
        open_df = self.factor_manager.get_prepare_aligned_factor_for_analysis(
            "open_hfq", stock_pool_name, True
        )
        calculators = {
            "o2o": partial(calculate_forward_returns_tradable_o2o, open_df=open_df)
        }
        return (
            factor_data,
            is_composite,
            data_manager.config["backtest"]["start_date"],
            data_manager.config["backtest"]["end_date"],
            data_manager.get_stock_pool_index_code_by_name(stock_pool_name),
            {name: calculators[name] for name in configured_calculators},
        )

    def test_factor_entity_service_route(
        self, factor_name: str, stock_pool_index_name: str
    ) -> Dict[str, dict]:
        """正式入口：只评价并保存 processed 信号。"""
        factor_data, is_composite, start_date, end_date, pool_code, calculators = (
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
            self.factor_results_manager._save_factor_results(
                factor_name=factor_name,
                stock_index=pool_code,
                start_date=start_date,
                end_date=end_date,
                returns_calculator_func_name=calculator_name,
                results=results,
            )
            all_results[calculator_name] = results
        return all_results
