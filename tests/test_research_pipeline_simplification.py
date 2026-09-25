import json
from unittest.mock import Mock

import pandas as pd
import pytest

from projects._03_factor_selection.data_manager import data_manager as data_module
from projects._03_factor_selection.factory import enhanced_test_runner as runner_module
from projects._03_factor_selection.utils.factor_processor import FactorProcessor


def make_data_manager(preprocessing):
    return data_module.DataManager(
        {
            "research_window": {"start_date": "2024-01-01", "end_date": "2025-01-01"},
            "preprocessing": preprocessing,
            "stock_pool_name": "ALL",
            "experiments": [{"factor_name": "demo"}],
        },
    )


def test_config_and_experiments_are_passed_without_file_io(monkeypatch):
    reader = Mock(side_effect=AssertionError("Unexpected file read"))
    monkeypatch.setattr(data_module, "_load_file", reader)
    manager = make_data_manager({"neutralization": {"enable": False}})
    assert manager.get_experiments_factor_names() == ["demo"]
    reader.assert_not_called()


def test_data_constructor_does_not_load_data(monkeypatch):
    from quant_lib.data_loader import DataLoader
    loader = Mock(side_effect=AssertionError("Unexpected constructor IO"))
    monkeypatch.setattr(DataLoader, "_load_trade_cal", loader)
    monkeypatch.setattr(data_module, "IndexComponentLoader", loader)
    manager = make_data_manager({"neutralization": {"enable": False}})
    assert manager.data_loader.trade_cal is None
    assert not hasattr(manager.data_loader, "field_map")
    assert manager.raw_dfs == {}
    loader.assert_not_called()


def test_data_prepare_loads_calendar_before_basic_data(monkeypatch):
    manager = make_data_manager({"neutralization": {"enable": False}})
    manager.config["stock_pool_profiles"] = {"ALL": {"index_filter": {"enable": False}}}
    calls = []
    loader = Mock()
    loader._load_trade_cal.side_effect = lambda: calls.append("calendar")
    loader.get_trading_dates.side_effect = lambda *args: calls.append(args) or []
    manager.data_loader = loader
    monkeypatch.setattr(manager, "_resolve_buffer_start_date", lambda: "20230101")
    monkeypatch.setattr(manager, "_prepare_stock_pool", lambda: calls.append("pool"))
    manager.prepare()
    assert calls == [
        "calendar", ("2024-01-01", "2025-01-01"), ("20230101", "2025-01-01"), "pool"
    ]


def test_calendar_failure_stops_before_pool(tmp_path, monkeypatch):
    from quant_lib.data_loader import DataLoader
    loader = DataLoader(tmp_path / "not_created")
    manager = make_data_manager({"neutralization": {"enable": False}})
    manager.data_loader = loader
    pool = Mock()
    monkeypatch.setattr(manager, "_prepare_stock_pool", pool)
    with pytest.raises(FileNotFoundError):
        manager.prepare()
    pool.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_index_components_are_created_only_when_building_enabled_pool(monkeypatch, enabled):
    component = Mock()
    component.return_value.get_members_on_date.return_value = {"a"}
    monkeypatch.setattr(data_module, "IndexComponentLoader", component)
    manager = make_data_manager({"neutralization": {"enable": False}})
    manager.config["stock_pool_profiles"] = {"ALL": {
        "index_filter": {"enable": enabled, "index_code": "000300"},
        "filters": {"history_days": 0, "remove_st": False, "adapt_tradeable_matrix_by_suspend_resume": False},
    }}
    manager.data_loader = Mock()
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
    manager.trading_dates = frame.index[1:]
    manager.data_loader.get_raw_dfs_by_require_fields.return_value = {"close_raw": frame}
    component.assert_not_called()
    manager._prepare_stock_pool()
    assert component.call_count == int(enabled)
    if enabled:
        component.assert_called_once_with(index_codes=["000300"])


def test_runner_stage_order(monkeypatch):
    runner = runner_module.EnhancedTestRunner()
    config, results, calls = {}, [], []
    runner.data = Mock()
    runner.data.prepare.side_effect = lambda: calls.append("data")
    monkeypatch.setattr(runner, "_prepare_run", lambda _: calls.append("run") or config)
    monkeypatch.setattr(runner, "_initialize_components", lambda _: calls.append("components"))
    monkeypatch.setattr(runner, "_research_factors", lambda _: calls.append("research") or results)
    monkeypatch.setattr(runner, "_save_run_summary", lambda *args: calls.append("summary"))
    assert runner.run() is results
    assert calls == ["run", "components", "data", "research", "summary"]


def test_unknown_factor_stops_before_creating_run(monkeypatch):
    runner = runner_module.EnhancedTestRunner()
    config = runner._load_yaml_mapping(runner.research_config_path)
    config["experiments"] = [{"factor_name": "missing_test_factor"}]
    monkeypatch.setattr(runner, "_load_yaml_mapping", lambda _: config)
    create_run = Mock()
    monkeypatch.setattr(runner_module, "create_run_dir", create_run)

    with pytest.raises(KeyError, match="missing_test_factor"):
        runner._prepare_run("test")
    create_run.assert_not_called()


def test_missing_direction_stops_before_snapshot_write(tmp_path):
    runner = runner_module.EnhancedTestRunner()
    runner.run_dir = tmp_path
    runner.direction_output_path = tmp_path / "directions.yaml"
    runner.direction_output_path.write_text("factors: {}\n", encoding="utf-8")

    with pytest.raises(KeyError, match="demo"):
        runner._snapshot_direction_config([{"factor_name": "demo"}])
    assert not (tmp_path / "resolved_factors.yaml").exists()


def test_evaluator_returns_results_without_writing(tmp_path):
    evaluator = runner_module.FactorAnalyzer.__new__(runner_module.FactorAnalyzer)
    factor = pd.DataFrame([[1.0]])
    calculator = object()
    expected = {"processed_factor_df": factor}
    evaluator.prepare_data_for_entity_service = Mock(
        return_value=(factor, False, {"o2o": calculator})
    )
    evaluator.analyze_processed_factor = Mock(return_value=expected)
    assert evaluator.evaluate_factor("demo", "ALL") == {"o2o": expected}
    evaluator.analyze_processed_factor.assert_called_once_with(
        "demo", factor, "ALL", calculator, already_processed=False
    )
    assert not hasattr(evaluator, "factor_results_manager")


def test_result_write_failure_stops_before_direction_and_next_factor(tmp_path):
    runner = runner_module.EnhancedTestRunner()
    runner.run_dir = tmp_path
    runner.data = Mock()
    runner.factor_engine = Mock()
    runner.evaluator = Mock()
    runner.evaluator.evaluate_factor.return_value = {"o2o": {}}
    runner.result_store = Mock()
    runner.result_store._save_factor_results.side_effect = OSError("write failed")
    runner._store_direction = Mock()
    config = {
        "stock_pool_name": "ALL", "experiments": [{"factor_name": "a"}, {"factor_name": "b"}],
        "research_window": {"start_date": "2024-01-01", "end_date": "2025-01-01"},
    }
    with pytest.raises(OSError, match="write failed"):
        runner._research_factors(config)
    runner.evaluator.evaluate_factor.assert_called_once()
    runner._store_direction.assert_not_called()
    runner.factor_engine.clear_cache.assert_called_once()


def test_only_research_pool_is_built():
    manager = make_data_manager({"neutralization": {"enable": False}})
    manager.config["stock_pool_profiles"] = {
        "ALL": {"index_filter": {"enable": False}, "filters": {}},
        "OTHER": {"index_filter": {"enable": True}},
    }
    manager.create_stock_pool = Mock(return_value=object())
    manager.data_loader = Mock()
    manager.data_loader.get_raw_dfs_by_require_fields.return_value = {"close_raw": pd.DataFrame([[1.0]])}
    manager._prepare_stock_pool()
    manager.create_stock_pool.assert_called_once_with(
        manager.config["stock_pool_profiles"]["ALL"], "ALL"
    )
    assert manager.stock_pools_dict == {"ALL": manager.create_stock_pool.return_value}
    manager.data_loader.get_raw_dfs_by_require_fields.assert_not_called()


@pytest.mark.parametrize("liquidity", [0, 0.1])
@pytest.mark.parametrize("market_cap", [0, 0.05])
def test_pool_loads_only_enabled_filter_fields(liquidity, market_cap):
    manager = make_data_manager({"neutralization": {"enable": False}})
    manager.config["stock_pool_profiles"] = {"ALL": {
        "index_filter": {"enable": False},
        "filters": {
            "history_days": 0, "remove_st": False, "adapt_tradeable_matrix_by_suspend_resume": False,
            "min_liquidity_percentile": liquidity, "min_market_cap_percentile": market_cap,
        },
    }}
    expected = ["close_raw"]
    if liquidity:
        expected.append("turnover_rate")
    if market_cap:
        expected.append("circ_mv")
    manager.data_loader = Mock()
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
    manager.trading_dates = frame.index[1:]
    manager.data_loader.get_raw_dfs_by_require_fields.side_effect = (
        lambda **kwargs: {kwargs["fields"][0]: frame.copy()}
    )
    manager._prepare_stock_pool()
    assert [
        call.kwargs["fields"][0]
        for call in manager.data_loader.get_raw_dfs_by_require_fields.call_args_list
    ] == expected
    assert set(manager.raw_dfs) == {"close_raw"}
    assert set(manager.temporary_raw_dfs) == set(expected) - {"close_raw"}


def test_on_demand_reads_establish_base_and_reuse_cache_without_quality_scan(monkeypatch):
    manager = make_data_manager({"neutralization": {"enable": False}})
    dates = pd.date_range("2024-01-01", periods=2)
    close = pd.DataFrame({"a": [1.0, 2.0]}, index=dates)
    turnover = pd.DataFrame({"a": [0.5], "outside": [2.0]}, index=dates[1:])
    manager.data_loader = Mock()
    manager.data_loader.get_raw_dfs_by_require_fields.side_effect = (
        lambda **kwargs: {kwargs["fields"][0]: close if kwargs["fields"] == ["close_raw"] else turnover}
    )
    monkeypatch.setattr(data_module, "check_field_level_completeness", Mock(
        side_effect=AssertionError("Research must not run a completeness scan")
    ))
    actual = manager.get_raw_field("turnover_rate")
    pd.testing.assert_frame_equal(actual, turnover.reindex(index=dates, columns=["a"]))
    assert manager.get_raw_field("turnover_rate") is actual
    assert manager.get_raw_field("close_raw") is close
    assert manager.data_loader.get_raw_dfs_by_require_fields.call_count == 2
    manager.clear_temporary_raw_fields()
    assert manager.get_raw_field("close_raw") is close
    assert not manager.temporary_raw_dfs
    manager.get_raw_field("turnover_rate")
    assert manager.data_loader.get_raw_dfs_by_require_fields.call_count == 3


def test_pool_filter_order_is_unchanged(monkeypatch):
    manager = make_data_manager({"neutralization": {"enable": False}})
    close = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
    manager.trading_dates = close.index[1:]
    manager.get_raw_field = Mock(return_value=close)
    calls = []
    for name in [
        "_build_dynamic_index_universe", "_filter_by_history_days", "_filter_st_stocks",
        "_filter_tradeable_matrix_by_suspend_resume", "_filter_by_liquidity", "_filter_by_market_cap",
    ]:
        monkeypatch.setattr(manager, name, lambda pool, *args, name=name: calls.append(name) or pool)
    profile = {
        "index_filter": {"enable": True, "index_code": "000300"},
        "filters": {
            "history_days": 252, "remove_st": True, "adapt_tradeable_matrix_by_suspend_resume": True,
            "min_liquidity_percentile": 0.1, "min_market_cap_percentile": 0.05,
        },
    }
    actual = manager.create_stock_pool(profile, "ALL")
    assert calls == [
        "_build_dynamic_index_universe", "_filter_by_history_days", "_filter_st_stocks",
        "_filter_tradeable_matrix_by_suspend_resume", "_filter_by_liquidity", "_filter_by_market_cap",
    ]
    pd.testing.assert_frame_equal(actual, close.notna().reindex(manager.trading_dates))


def test_component_loader_reads_only_requested_index_files(monkeypatch):
    from projects._03_factor_selection.utils.component_loader import IndexComponentLoader
    frame = pd.DataFrame({
        "成分券代码": [1], "交易市场": ["XSHE"],
        "纳入日期": ["2020-01-01"], "剔除日期": [None],
    })
    reader = Mock(return_value=frame)
    monkeypatch.setattr(pd, "read_excel", reader)
    loader = IndexComponentLoader(
        ["000300"], {"000300": "selected.xlsx", "000905": "unneeded.xlsx"}
    )
    reader.assert_called_once_with("selected.xlsx")
    assert loader.get_members_on_date(pd.Timestamp("2024-01-01"), ["000300"]) == {"000001.SZ"}


@pytest.mark.parametrize("pool_name", ["UNKNOWN", "", None])
def test_invalid_research_pool_rejected_before_data_loading(monkeypatch, pool_name):
    runner = runner_module.EnhancedTestRunner()
    config = {
        "stage": "inner", "experiments": [{"factor_name": "demo"}],
        "evaluation": {"forward_periods": [5], "returns_calculator": ["o2o"]},
        "stock_pool_name": pool_name, "stock_pool_profiles": {"ALL": {}},
    }
    monkeypatch.setattr(runner, "_load_yaml_mapping", lambda _: config)
    with pytest.raises(ValueError, match="stock_pool_name"):
        runner._load_effective_config("demo")


def test_initialize_passes_config_and_does_not_clear_fresh_cache(tmp_path, monkeypatch):
    runner = runner_module.EnhancedTestRunner()
    runner.run_dir = tmp_path
    config = {"experiments": [{"factor_name": "demo"}], "stock_pool_name": "ALL"}
    data_factory = Mock()
    factor_factory = Mock()
    analyzer_factory = Mock()
    monkeypatch.setattr(runner_module, "DataManager", data_factory)
    monkeypatch.setattr(runner_module, "FactorManager", factor_factory)
    monkeypatch.setattr(runner_module, "FactorAnalyzer", analyzer_factory)
    runner._initialize_components(config)
    assert data_factory.call_args.args[0] is config
    data_factory.assert_called_once_with(config)
    data_factory.return_value.prepare.assert_not_called()
    data_factory.return_value._prepare_stock_pool.assert_not_called()
    factor_factory.return_value.clear_cache.assert_not_called()


def test_basic_data_does_not_eagerly_load_industry(monkeypatch):
    manager = make_data_manager({"neutralization": {
        "enable": True, "factors": ["industry"],
    }})
    manager.buffer_start_date = "20230101"
    manager.raw_dfs = {}
    manager.temporary_raw_dfs = {}
    manager.data_loader = Mock()
    manager.data_loader.get_raw_dfs_by_require_fields.return_value = {
        "close_raw": pd.DataFrame([[1.0]])
    }
    manager.config["stock_pool_profiles"] = {"ALL": {"index_filter": {"enable": False}, "filters": {}}}
    manager.create_stock_pool = Mock()
    loader = Mock(side_effect=AssertionError("Unexpected eager load"))
    monkeypatch.setattr(data_module, "PointInTimeIndustryMap", loader)
    manager._prepare_stock_pool()
    loader.assert_not_called()
    manager.create_stock_pool.assert_called_once()


@pytest.mark.parametrize("preprocessing", [
    {"neutralization": {"enable": False}},
    {"neutralization": {"enable": True, "factors": ["market_cap"]}},
])
def test_non_industry_processing_does_not_load_map(monkeypatch, preprocessing):
    loader = Mock(side_effect=AssertionError("Unexpected industry load"))
    monkeypatch.setattr(data_module, "PointInTimeIndustryMap", loader)
    manager = make_data_manager(preprocessing)
    assert manager.get_preprocessing_industry_map() is None
    loader.assert_not_called()


@pytest.mark.parametrize("step", ["winsorization", "standardization", "neutralization"])
def test_industry_map_is_loaded_once_and_shared(monkeypatch, step):
    preprocessing = {"neutralization": {"enable": False}}
    preprocessing[step] = (
        {"enable": True, "factors": ["industry"]}
        if step == "neutralization" else {"by_industry": {}}
    )
    loader = Mock(return_value=object())
    monkeypatch.setattr(data_module, "PointInTimeIndustryMap", loader)
    manager = make_data_manager(preprocessing)
    loader.assert_not_called()
    assert manager.get_preprocessing_industry_map() is manager.pit_map
    assert manager.get_preprocessing_industry_map() is loader.return_value
    loader.assert_called_once_with()


def test_industry_factor_can_request_map_without_industry_preprocessing(monkeypatch):
    loader = Mock(return_value=object())
    monkeypatch.setattr(data_module, "PointInTimeIndustryMap", loader)
    manager = make_data_manager({"neutralization": {"enable": False}})
    assert manager.pit_map is manager.pit_map
    loader.assert_called_once_with()


def test_industry_load_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        data_module, "PointInTimeIndustryMap", Mock(side_effect=ValueError("bad industry"))
    )
    manager = make_data_manager({"neutralization": {"enable": False}})
    with pytest.raises(ValueError, match="bad industry"):
        _ = manager.pit_map
    assert manager._pit_map is None


def test_processor_rejects_missing_required_map():
    processor = FactorProcessor({"preprocessing": {"winsorization": {"by_industry": {}}}})
    with pytest.raises(ValueError, match="行业"):
        processor.process_factor(pd.DataFrame([[1.0]]), "demo", {}, "value")


def test_processor_without_industry_preserves_full_section_result(monkeypatch):
    processor = FactorProcessor({"preprocessing": {
        "winsorization": {"method": "mad", "mad_threshold": 3.0},
        "standardization": {"method": "zscore"},
        "neutralization": {"enable": False},
    }})
    from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
    monkeypatch.setattr(FactorManager, "_validate_data_quality", Mock())
    monkeypatch.setattr(
        "projects._03_factor_selection.utils.factor_processor.PointInTimeIndustryMap",
        Mock(side_effect=AssertionError("Unexpected processor industry load")),
    )
    factor = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]])
    expected = processor._standardize_robust(processor.winsorize_robust(factor))
    actual = processor.process_factor(factor, "demo", {}, "value")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("fail", [False, True])
def test_runner_sequence_outputs_and_failure_cleanup(tmp_path, monkeypatch, fail):
    runner = runner_module.EnhancedTestRunner()
    config = {
        "output_root": str(tmp_path), "stage": "inner", "experiment_name": "demo",
        "experiments": [{"factor_name": "a"}, {"factor_name": "b"}],
        "stock_pool_name": "ALL",
        "research_window": {"start_date": "2024-01-01", "end_date": "2025-01-01"},
    }
    monkeypatch.setattr(runner, "_load_effective_config", lambda _: config)
    manager = Mock()
    analyzer = Mock()
    analyzer.evaluate_factor.side_effect = (
        [{"o2o": {}}, ValueError("evaluation failed")] if fail
        else [{"o2o": {}}, {"o2o": {}}]
    )

    def initialize(actual_config):
        assert actual_config is config
        runner.data = Mock()
        runner.data.get_stock_pool_storage_name_by_name.return_value = "ALL"
        runner.factor_engine = manager
        runner.evaluator = analyzer
        runner.result_store = Mock()

    monkeypatch.setattr(runner, "_initialize_components", initialize)
    store = Mock(return_value=1)
    snapshot = Mock()
    monkeypatch.setattr(runner, "_store_direction", store)
    monkeypatch.setattr(runner, "_snapshot_direction_config", snapshot)
    if fail:
        with pytest.raises(ValueError, match="evaluation failed"):
            runner.run()
        snapshot.assert_not_called()
        assert not (runner.run_dir / "summary.json").exists()
        assert store.call_count == 1
    else:
        results = runner.run()
        assert [row["factor_name"] for row in results] == ["a", "b"]
        snapshot.assert_called_once_with(results)
        assert (runner.run_dir / "summary.json").is_file()
        summary = json.loads((runner.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["stock_pool_name"] == "ALL"
        assert all(set(row) == {"factor_name", "direction"} for row in summary["factors"])
        assert store.call_count == 2
    assert manager.clear_cache.call_count == 2
    runner.data.prepare.assert_called_once_with()
    assert runner.result_store._save_factor_results.call_count == store.call_count
    assert [call.kwargs for call in analyzer.evaluate_factor.call_args_list] == [
        {"factor_name": name, "stock_pool_index_name": "ALL"} for name in ["a", "b"]
    ]
    assert manager.store_inner_resolved_direction.call_count == store.call_count
    assert (runner.run_dir / "effective_config.yaml").is_file()
    assert not (runner.run_dir / "experiments.yaml").exists()
    assert not (runner.run_dir / "artifacts" / "prices").exists()
