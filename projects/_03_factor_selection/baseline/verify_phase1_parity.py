"""重算第一阶段核心产物，并与 legacy 冻结包做精确对比。"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
LEGACY_DIR = (
    REPO_ROOT.parent
    / "quant_research_baselines"
    / "legacy_d8387bcf_20230710_20250710"
)
CONFIG_PATH = REPO_ROOT / "projects/_03_factor_selection/factory/config.yaml"
EXPERIMENTS_PATH = REPO_ROOT / "projects/_03_factor_selection/factory/experiments.yaml"
FACTORS = ("volatility_40d", "operating_accruals", "three_low_one_high")
START_DATE = "2023-07-10"
END_DATE = "2025-07-10"
PERIODS = (5, 21)
STOCK_POOL = "ZZ800"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_legacy_artifacts() -> None:
    manifest_path = LEGACY_DIR / "manifest.json"
    checksum_path = LEGACY_DIR / "manifest.sha256"
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(f"旧冻结包缺少 manifest: {LEGACY_DIR}")
    expected_hash = checksum_path.read_text(encoding="utf-8").strip()
    if sha256_file(manifest_path) != expected_hash:
        raise AssertionError("旧冻结包 manifest SHA-256 不一致")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["artifact_files"]:
        artifact = LEGACY_DIR / record["path"]
        if not artifact.is_file() or sha256_file(artifact) != record["sha256"]:
            raise AssertionError(f"旧冻结产物缺失或 SHA-256 不一致: {artifact}")


def effective_config(temp_dir: Path) -> Path:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["backtest"]["start_date"] = START_DATE
    config["backtest"]["end_date"] = END_DATE
    path = temp_dir / "config.yaml"
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return path


def configure_runtime() -> None:
    from projects._03_factor_selection.config_manager.function_load import load_config_file

    runtime = copy.deepcopy(load_config_file.trans_pram)
    runtime["period"] = (START_DATE, END_DATE)
    runtime["evaluation"].update(
        {
            "quantiles": 5,
            "forward_periods": list(PERIODS),
            "returns_calculator": ["o2o"],
        }
    )
    load_config_file.trans_pram = runtime


def initialize(config_path: Path):
    configure_runtime()
    from projects._03_factor_selection.data_manager.data_manager import DataManager
    from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import (
        FactorAnalyzer,
    )
    from projects._03_factor_selection.factor_manager.factor_manager import FactorManager

    data_manager = DataManager(str(config_path), str(EXPERIMENTS_PATH))
    data_manager.prepare_basic_data()
    factor_manager = FactorManager(data_manager)
    factor_manager.clear_cache()
    return FactorAnalyzer(factor_manager)


def assert_frame_equal(actual, expected_path: Path, label: str) -> None:
    expected = pd.read_parquet(expected_path)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, obj=label)


def assert_nested_equal(actual, expected, path: str = "statistics") -> None:
    if isinstance(actual, pd.Series):
        actual = actual.to_dict()
    elif isinstance(actual, pd.DataFrame):
        actual = actual.to_dict(orient="index")
    elif isinstance(actual, (pd.Timestamp, Path)):
        actual = str(actual)
    elif isinstance(actual, np.ndarray):
        actual = actual.tolist()
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{path} 字段不同: {set(actual)} != {set(expected)}")
        for key in expected:
            assert_nested_equal(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        for index, value in enumerate(expected):
            assert_nested_equal(actual[index], value, f"{path}[{index}]")
        return
    if isinstance(actual, (np.integer, np.floating)):
        actual = actual.item()
    if isinstance(expected, float) and math.isnan(expected):
        if not isinstance(actual, float) or not math.isnan(actual):
            raise AssertionError(f"{path}: {actual!r} != NaN")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def compare_factor(analyzer, factor_name: str) -> None:
    prepared, is_composite, start, end, _, calculators = (
        analyzer.prepare_data_for_entity_service(factor_name, STOCK_POOL)
    )
    if (start, end, tuple(calculators)) != (START_DATE, END_DATE, ("o2o",)):
        raise AssertionError(f"{factor_name} 运行契约变化")

    legacy = LEGACY_DIR / "artifacts" / factor_name
    assert_frame_equal(prepared, legacy / "prepared_factor.parquet", f"{factor_name}.prepared")
    result = analyzer.analyze_processed_factor(
        factor_name,
        prepared,
        STOCK_POOL,
        calculators["o2o"],
        already_processed=is_composite,
    )
    compare_result_frames(result, legacy / "processed", factor_name)
    compare_statistics(result, legacy / "processed" / "statistics.json", factor_name)


def compare_result_frames(result: dict, legacy: Path, factor_name: str) -> None:
    assert_frame_equal(
        result["processed_factor_df"], legacy / "factor.parquet", f"{factor_name}.factor"
    )
    for period, series in result["ic_series_periods_dict_processed"].items():
        assert_frame_equal(
            series.rename("IC").to_frame(),
            legacy / f"ic_series_{period}.parquet",
            f"{factor_name}.ic.{period}",
        )
    for period, frame in result["quantile_returns_series_periods_dict_processed"].items():
        assert_frame_equal(
            frame,
            legacy / f"quantile_period_returns_{period}.parquet",
            f"{factor_name}.quantile.{period}",
        )
    assert_frame_equal(
        result["q_daily_returns_df_processed"],
        legacy / "quantile_daily_returns_1d.parquet",
        f"{factor_name}.quantile_daily",
    )


def compare_statistics(result: dict, path: Path, factor_name: str) -> None:
    expected = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "ic": result["ic_stats_periods_dict_processed"],
        "quantile": result["quantile_stats_periods_dict_processed"],
        "turnover": result["top_q_turnover_stats_periods_dict"],
    }
    assert_nested_equal(actual, expected, f"{factor_name}.statistics")


def main() -> None:
    if not LEGACY_DIR.is_dir():
        raise FileNotFoundError(f"旧冻结包不存在: {LEGACY_DIR}")
    verify_legacy_artifacts()
    with tempfile.TemporaryDirectory(prefix="phase1_parity_") as temp_name:
        analyzer = initialize(effective_config(Path(temp_name)))
        for factor_name in FACTORS:
            compare_factor(analyzer, factor_name)
            print(f"PARITY_OK={factor_name}")
    print("PHASE1_PARITY_OK")


if __name__ == "__main__":
    main()
