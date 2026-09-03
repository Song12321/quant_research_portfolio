import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from projects._03_factor_selection.config_manager.factor_definition_loader import (
    load_factor_definitions,
)
from projects._03_factor_selection.config_manager.inner_direction_store import (
    resolve_and_store_inner_direction,
)
from projects._03_factor_selection.factory.enhanced_test_runner import EnhancedTestRunner
from projects._03_factor_selection.factor_manager.factor_composite.factor_synthesizer import (
    FactorSynthesizer,
)
from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import (
    FactorAnalyzer,
)
from projects._03_factor_selection.factor_manager.factor_manager import (
    FactorManager,
    FactorResultsManager,
)
from projects._03_factor_selection.utils.factor_processor import FactorProcessor


FACTOR_DEFINITION_DIR = (
    Path(__file__).parents[1]
    / "projects"
    / "_03_factor_selection"
    / "configs"
    / "factors"
    / "definitions"
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


def test_inner_direction_uses_non_overlapping_sample_weights(tmp_path):
    output_path = tmp_path / "directions.yaml"
    output_path.write_text("factors: {}\n", encoding="utf-8")
    stats = {
        "5d": {"ic_mean": 0.1, "ic_Valid Days": 8},
        "10d": {"ic_mean": -0.1, "ic_Valid Days": 4},
        "20d": {"ic_mean": -0.1, "ic_Valid Days": 2},
        "40d": {"ic_mean": -0.1, "ic_Valid Days": 1},
    }

    direction = resolve_and_store_inner_direction(
        factor_name="factor",
        configured_periods=[5, 10, 20, 40],
        ic_stats_periods_dict_processed=stats,
        inner_run_id="run",
        output_path=output_path,
    )

    saved = yaml.safe_load(output_path.read_text(encoding="utf-8"))["factors"]["factor"]
    assert direction == 1
    assert saved["direction_score"] == pytest.approx(0.1 / 15)
    assert saved["ic_valid_days_by_period"] == {"5d": 8, "10d": 4, "20d": 2, "40d": 1}
    assert saved["direction_weight_by_period"] == pytest.approx(
        {"5d": 8 / 15, "10d": 4 / 15, "20d": 2 / 15, "40d": 1 / 15}
    )


def test_inner_direction_rejects_missing_or_invalid_non_overlapping_sample_count(tmp_path):
    output_path = tmp_path / "directions.yaml"
    output_path.write_text("factors: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="5d 缺少 ic_Valid Days"):
        resolve_and_store_inner_direction(
            factor_name="factor",
            configured_periods=[5],
            ic_stats_periods_dict_processed={"5d": {"ic_mean": 0.1}},
            inner_run_id="run",
            output_path=output_path,
        )

    with pytest.raises(ValueError, match="5d 的 ic_Valid Days=0"):
        resolve_and_store_inner_direction(
            factor_name="factor",
            configured_periods=[5],
            ic_stats_periods_dict_processed={"5d": {"ic_mean": 0.1, "ic_Valid Days": 0}},
            inner_run_id="run",
            output_path=output_path,
        )


def test_only_composites_declare_component_fields():
    definitions = load_factor_definitions(FACTOR_DEFINITION_DIR)

    for definition in definitions:
        assert "cal_require_base_fields_from_daily" not in definition
        if definition["action"] == "composite":
            assert definition["cal_require_base_fields"]
        else:
            assert "cal_require_base_fields" not in definition


def test_standardization_without_industry_config_uses_full_cross_section():
    processor = FactorProcessor({"preprocessing": {"standardization": {"method": "zscore"}}})
    factor = pd.DataFrame([[0.0, 2.0, 100.0, 104.0]], columns=list("abcd"))

    actual = processor._standardize_robust(factor)
    expected = factor.sub(factor.mean(axis=1), axis=0).div(factor.std(axis=1), axis=0)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.loc[0, ["a", "b"]].mean() != pytest.approx(0.0)


def test_zero_variance_standardization_preserves_missing_values():
    processor = FactorProcessor({"preprocessing": {"standardization": {"method": "zscore"}}})
    factor = pd.DataFrame([[3.0, np.nan]], columns=["a", "b"])

    actual = processor._standardize_robust(factor)

    assert actual.loc[0, "a"] == 0.0
    assert pd.isna(actual.loc[0, "b"])


def test_neutralization_factors_are_a_strict_yaml_contract():
    processor = FactorProcessor(
        {"preprocessing": {"neutralization": {"factors": ["market_cap", "industry"]}}}
    )

    assert processor.get_regression_need_neutral_factor_list("value_signal") == [
        "market_cap",
        "industry",
    ]

    missing = FactorProcessor({"preprocessing": {"neutralization": {}}})
    with pytest.raises(ValueError, match="必须是非空列表"):
        missing.get_configured_neutralization_factors()

    invalid = FactorProcessor(
        {"preprocessing": {"neutralization": {"factors": ["unknown"]}}}
    )
    with pytest.raises(ValueError, match="不支持的变量"):
        invalid.get_configured_neutralization_factors()


def test_industry_mad_skips_the_first_date_without_a_prior_industry_map():
    processor = FactorProcessor(
        {
            "preprocessing": {
                "winsorization": {
                    "method": "mad",
                    "mad_threshold": 3.0,
                    "by_industry": {
                        "primary_level": "l2_code",
                        "fallback_level": "l1_code",
                        "min_samples": 2,
                    },
                }
            }
        }
    )
    factor = pd.DataFrame([[1.0, 2.0]], columns=["a", "b"])

    actual = processor.winsorize_robust(factor, pit_industry_map=object())

    assert actual.loc[0].isna().all()


class _MarketCapOnlyDataManager:
    config = {
        "preprocessing": {
            "neutralization": {"enable": True, "factors": ["market_cap"]}
        }
    }

    @property
    def pit_map(self):
        raise AssertionError("未配置 industry 时不应读取行业映射")

    def get_stock_pool_index_code_by_name(self, _):
        raise AssertionError("未配置 pct_chg_beta 时不应读取基准指数")


class _MarketCapOnlyFactorManager:
    data_manager = _MarketCapOnlyDataManager()

    @staticmethod
    def get_style_category(_):
        return "value"

    @staticmethod
    def get_prepare_aligned_factor_for_analysis(factor_name, _, __):
        assert factor_name == "log_circ_mv"
        return pd.DataFrame([[1.0]], columns=["000001.SZ"])


def test_neutralization_yaml_limits_prepared_regressors():
    analyzer = FactorAnalyzer.__new__(FactorAnalyzer)
    analyzer.factor_manager = _MarketCapOnlyFactorManager()
    analyzer.factor_processor = FactorProcessor(analyzer.factor_manager.data_manager.config)

    neutral_dfs, style_category = analyzer.prepare_data_for_process_factor(
        "value_signal",
        pd.DatetimeIndex([pd.Timestamp("2024-01-02")]),
        ["000001.SZ"],
        "ZZ800",
    )

    assert style_category == "value"
    assert list(neutral_dfs) == ["log_circ_mv"]


class _AlignmentDataManager:
    def __init__(self, definitions):
        self.config = {"factor_definition": definitions}


@pytest.mark.parametrize(
    "factor_name",
    ["three_low_one_high_value", "three_low_one_high_improve"],
)
def test_three_low_one_high_daily_components_shift_before_o2o_label(factor_name):
    definitions = load_factor_definitions(FACTOR_DEFINITION_DIR)
    raw_factor = pd.DataFrame(
        {"000001.SZ": [1.0, 2.0, 3.0]},
        index=pd.date_range("2024-01-02", periods=3),
    )
    manager = FactorManager.__new__(FactorManager)
    manager.data_manager = _AlignmentDataManager(definitions)
    manager.get_factor_by_rule = lambda _: raw_factor

    actual = manager.get_raw_factor_for_analysis(factor_name)

    pd.testing.assert_frame_equal(actual, raw_factor.shift(1), check_exact=True)
