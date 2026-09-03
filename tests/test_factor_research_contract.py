import json

import numpy as np
import pandas as pd
import pytest

from projects._03_factor_selection.factory.enhanced_test_runner import EnhancedTestRunner
from projects._03_factor_selection.factor_manager.factor_composite.factor_synthesizer import (
    FactorSynthesizer,
)
from projects._03_factor_selection.factor_manager.factor_manager import (
    FactorManager,
    FactorResultsManager,
)


def build_results() -> dict:
    index = pd.date_range("2024-01-02", periods=2)
    factor = pd.DataFrame({"000001.SZ": [1.0, 2.0]}, index=index)
    ic_series = pd.Series([0.1, 0.2], index=index)
    quantile_returns = pd.DataFrame({"Q1": [0.01], "Q5": [0.02]}, index=index[:1])
    return {
        "processed_factor_df": factor,
        "ic_series_periods_dict_processed": {"5d": ic_series},
        "ic_stats_periods_dict_processed": {"5d": {"ic_mean": np.float64(0.15)}},
        "quantile_returns_series_periods_dict_processed": {"5d": quantile_returns},
        "q_daily_returns_df_processed": quantile_returns,
        "quantile_stats_periods_dict_processed": {"5d": {"mean": 0.015}},
        "top_q_turnover_stats_periods_dict": {"5d": {"turnover_mean": 0.1}},
    }


def test_result_writer_only_saves_processed_contract(tmp_path):
    manager = FactorResultsManager()
    manager.results_dir = tmp_path
    manager._save_factor_results(
        factor_name="demo",
        stock_index="000906",
        start_date="2024-01-01",
        end_date="2024-01-31",
        returns_calculator_func_name="o2o",
        results=build_results(),
    )

    output = tmp_path / "000906" / "demo" / "o2o" / "20240101_20240131"
    assert {path.name for path in output.iterdir()} == {
        "summary_stats.json",
        "processed_factor.parquet",
        "ic_series_processed_5d.parquet",
        "quantile_returns_processed_5d.parquet",
        "q_daily_returns_df_processed.parquet",
    }
    summary = json.loads((output / "summary_stats.json").read_text(encoding="utf-8"))
    assert set(summary) == {
        "ic_analysis_processed",
        "quantile_backtest_processed",
        "top_q_turnover",
    }


def test_result_writer_rejects_non_processed_fields(tmp_path):
    manager = FactorResultsManager()
    manager.results_dir = tmp_path
    results = build_results()
    results["raw_factor_df"] = results["processed_factor_df"]
    with pytest.raises(ValueError, match="多余字段"):
        manager._save_factor_results("demo", "000906", "2024-01-01", "2024-01-31", "o2o", results)


def test_result_writer_rejects_legacy_files(tmp_path):
    manager = FactorResultsManager()
    manager.results_dir = tmp_path
    output = tmp_path / "000906" / "demo" / "o2o" / "20240101_20240131"
    output.mkdir(parents=True)
    (output / "ic_series_raw_5d.parquet").write_bytes(b"legacy")
    with pytest.raises(RuntimeError, match="旧 raw/F-M 产物"):
        manager._save_factor_results(
            "demo", "000906", "2024-01-01", "2024-01-31", "o2o", build_results()
        )


def test_equal_average_propagates_missing_values():
    first = pd.DataFrame([[1.0, np.nan]], columns=["a", "b"])
    second = pd.DataFrame([[3.0, 5.0]], columns=["a", "b"])
    result = FactorSynthesizer.equal_average([first, second])
    expected = pd.DataFrame([[2.0, np.nan]], columns=["a", "b"])
    pd.testing.assert_frame_equal(result, expected, check_exact=True)


class _CompositeDataManager:
    def get_cal_require_base_fields_for_composite(self, name):
        assert name == "combo"
        return ["positive", "negative"]


class _CompositeFactorManager:
    data_manager = _CompositeDataManager()

    def __init__(self, directions):
        self.directions = directions

    def get_inner_resolved_direction(self, factor_name):
        if factor_name not in self.directions:
            raise ValueError(f"合成因子缺少本次 Inner 子因子方向：factor={factor_name}")
        return self.directions[factor_name]


class _IdentityProcessor:
    @staticmethod
    def _standardize_robust(factor):
        return factor


def test_composite_uses_resolved_child_directions():
    manager = _CompositeFactorManager({"positive": 1, "negative": -1})
    synthesizer = FactorSynthesizer(manager, object(), _IdentityProcessor())
    factors = {
        "positive": pd.DataFrame([[1.0, 3.0]], columns=["a", "b"]),
        "negative": pd.DataFrame([[2.0, -4.0]], columns=["a", "b"]),
    }
    synthesizer.get_processed_sub_factor = lambda name, _: factors[name]

    actual = synthesizer.synthesize_equal_factor("combo", "ZZ800")

    expected = pd.DataFrame([[-0.5, 3.5]], columns=["a", "b"])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_composite_rejects_missing_resolved_child_direction():
    manager = _CompositeFactorManager({"positive": 1})
    synthesizer = FactorSynthesizer(manager, object(), _IdentityProcessor())
    synthesizer.get_processed_sub_factor = lambda *_: pd.DataFrame([[1.0]], columns=["a"])

    with pytest.raises(ValueError, match="缺少本次 Inner 子因子方向.*negative"):
        synthesizer.synthesize_equal_factor("combo", "ZZ800")


def test_composite_dependencies_require_earlier_same_pool_children():
    definitions = [
        {"name": "child", "action": "technical_calcu"},
        {"name": "combo", "action": "composite", "cal_require_base_fields": ["child"]},
    ]
    experiments = [
        {"factor_name": "combo", "stock_pool_name": "ZZ800"},
        {"factor_name": "child", "stock_pool_name": "ZZ800"},
    ]

    with pytest.raises(ValueError, match="提前完成.*child"):
        EnhancedTestRunner._validate_composite_dependencies(experiments, definitions)


def test_composite_dependencies_reject_different_child_pool():
    definitions = [
        {"name": "child", "action": "technical_calcu"},
        {"name": "combo", "action": "composite", "cal_require_base_fields": ["child"]},
    ]
    experiments = [
        {"factor_name": "child", "stock_pool_name": "ZZ500"},
        {"factor_name": "combo", "stock_pool_name": "ZZ800"},
    ]

    with pytest.raises(ValueError, match="股票池必须一致.*child"):
        EnhancedTestRunner._validate_composite_dependencies(experiments, definitions)


def test_inner_direction_store_rejects_missing_and_duplicate_values():
    manager = FactorManager.__new__(FactorManager)
    manager.inner_resolved_directions = {}
    manager.store_inner_resolved_direction("child", -1)

    assert manager.get_inner_resolved_direction("child") == -1
    with pytest.raises(ValueError, match="已存在"):
        manager.store_inner_resolved_direction("child", 1)
    with pytest.raises(ValueError, match="缺少本次 Inner 子因子方向.*missing"):
        manager.get_inner_resolved_direction("missing")
