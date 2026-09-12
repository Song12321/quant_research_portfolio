from pathlib import Path

import pandas as pd
import pytest

from quant_lib.config.constant_config import get_market_data_path
from utils.local_data_forensics import data_recon_manager
from utils.local_data_forensics.data_recon_manager import (
    DAILY_DATASETS,
    DataReconManager,
)


AUDIT_DAY = "2024-04-30"


def _write_dataset(root: Path, dataset: str, frame: pd.DataFrame) -> None:
    path = get_market_data_path(dataset, root)
    if dataset in DAILY_DATASETS:
        path = path / "year=2024" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _daily_frames(rows: list[dict] | None = None) -> dict[str, pd.DataFrame]:
    base_rows = rows if rows is not None else [{"ts_code": "000001.SZ", "trade_date": AUDIT_DAY}]
    payloads = {
        "daily": {"close": None, "amount": 1.0},
        "daily_hfq": {"close": 10.0},
        "daily_basic": {"total_mv": 100.0},
        "stk_limit": {"up_limit": 11.0},
    }
    result = {}
    for dataset, payload in payloads.items():
        result[dataset] = pd.DataFrame(
            [{**row, **payload} for row in base_rows],
            columns=["ts_code", "trade_date", *payload],
        )
    return result


def _reference_frames(
    calendar: list[dict] | None = None, suspend: list[dict] | None = None
) -> dict[str, pd.DataFrame]:
    trade_rows = calendar or [{"cal_date": AUDIT_DAY, "is_open": 1}]
    suspend_rows = suspend or []
    return {
        "trade_cal.parquet": pd.DataFrame(trade_rows),
        "stock_basic.parquet": pd.DataFrame([{
            "ts_code": "000001.SZ",
            "list_date": "2020-01-01",
            "delist_date": None,
            "name": "测试股票",
        }]),
        "suspend_d.parquet": pd.DataFrame(
            suspend_rows,
            columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"],
        ),
    }


def _build_root(
    root: Path,
    *,
    daily_rows: list[dict] | None = None,
    suspend: list[dict] | None = None,
    include_trade_cal: bool = True,
) -> None:
    frames = {}
    frames.update(_reference_frames(suspend=suspend))
    frames.update(_daily_frames(daily_rows))
    if not include_trade_cal:
        frames.pop("trade_cal.parquet")
    for dataset, frame in frames.items():
        _write_dataset(root, dataset, frame)


def _run(root: Path, output: Path, start: str = AUDIT_DAY, end: str = AUDIT_DAY):
    return DataReconManager(root).run(start, end, output)


def test_partial_payload_is_accepted_and_reports_are_written(tmp_path: Path) -> None:
    root = tmp_path / "market_data"
    output = tmp_path / "report"
    _build_root(root)

    summary = _run(root, output)

    assert summary["status"].eq("PASS").all()
    assert not summary["dataset"].str.contains("income|balancesheet|cashflow|fina_indicator").any()
    assert pd.read_csv(output / "issues.csv").empty
    assert (output / "summary.csv").is_file()


@pytest.mark.parametrize("amount", [0.0, None])
def test_no_positive_amount_is_unknown(tmp_path: Path, amount) -> None:
    root = tmp_path / "market_data"
    _build_root(root)
    frame = _daily_frames()["daily"]
    frame["amount"] = amount
    _write_dataset(root, "daily", frame)
    summary = _run(root, tmp_path / "report").set_index("dataset")
    assert summary.loc["daily", "unknown_rows"] == 1
    assert summary.loc["daily", "status"] == "UNKNOWN"
    assert summary.loc[list(DAILY_DATASETS[1:]), "expected_rows"].eq(0).all()
    assert summary["missing_rows"].eq(0).all()


def test_all_four_tables_missing_is_unknown_not_missing(tmp_path: Path) -> None:
    root = tmp_path / "market_data"
    _build_root(root, daily_rows=[])
    summary = _run(root, tmp_path / "report").set_index("dataset")
    assert summary.loc["daily", "unknown_rows"] == 1
    assert summary.loc["daily", "status"] == "UNKNOWN"
    assert summary["missing_rows"].eq(0).all()
    issues = pd.read_csv(tmp_path / "report" / "issues.csv")
    assert issues["issue_type"].tolist() == ["trading_status_unknown"]


def test_positive_amount_requires_other_tables_without_suspend_baseline(tmp_path: Path) -> None:
    root = tmp_path / "market_data"
    frames = _reference_frames()
    frames.pop("suspend_d.parquet")
    frames.update(_daily_frames())
    frames["daily_hfq"] = frames["daily_hfq"].iloc[:0]
    for dataset, frame in frames.items():
        _write_dataset(root, dataset, frame)
    summary = _run(root, tmp_path / "report").set_index("dataset")
    assert summary.loc["daily", "status"] == "PASS"
    assert summary.loc["daily_hfq", "missing_rows"] == 1
    assert summary.loc["daily_hfq", "status"] == "FAIL"
    assert summary.loc["daily_basic", "status"] == "PASS"


def test_row_with_no_business_value_is_invalid_not_missing(tmp_path: Path) -> None:
    root = tmp_path / "market_data"
    _build_root(root)
    empty_payload = pd.DataFrame([{
        "ts_code": "000001.SZ",
        "trade_date": AUDIT_DAY,
        "close": None,
        "amount": None,
    }])
    _write_dataset(root, "daily_hfq", empty_payload)

    summary = _run(root, tmp_path / "report")

    daily = summary.set_index("dataset").loc["daily_hfq"]
    assert daily["missing_rows"] == 0
    assert daily["invalid_rows"] == 1
    issues = pd.read_csv(tmp_path / "report" / "issues.csv")
    actual = issues.loc[issues["dataset"].eq("daily_hfq"), "issue_type"].tolist()
    assert actual == ["empty_payload"]


def test_listing_delisting_and_closed_day_boundaries(tmp_path: Path) -> None:
    root = tmp_path / "market_data"
    calendar = [
        {"cal_date": "2024-04-29", "is_open": 1},
        {"cal_date": AUDIT_DAY, "is_open": 1},
        {"cal_date": "2024-05-01", "is_open": 0},
    ]
    frames = _reference_frames(calendar=calendar)
    frames["stock_basic.parquet"].loc[0, "list_date"] = AUDIT_DAY
    frames["stock_basic.parquet"].loc[0, "delist_date"] = "2024-05-01"
    frames.update(_daily_frames())
    for dataset, frame in frames.items():
        _write_dataset(root, dataset, frame)

    summary = _run(root, tmp_path / "report", "2024-04-29", "2024-05-01")

    daily = summary.set_index("dataset").loc[list(DAILY_DATASETS)]
    assert daily["expected_rows"].eq(1).all()
    assert daily["missing_rows"].eq(0).all()


def test_missing_baseline_is_fatal_and_writes_no_report(tmp_path: Path) -> None:
    root = tmp_path / "market_data"
    output = tmp_path / "report"
    _build_root(root, include_trade_cal=False)

    with pytest.raises(FileNotFoundError):
        _run(root, output)

    assert not output.exists()


def test_invalid_explicit_date_stops_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _run(tmp_path / "market_data", tmp_path / "report", "not-a-date", AUDIT_DAY)


def test_missing_amount_column_stops_execution(tmp_path: Path) -> None:
    root = tmp_path / "market_data"
    _build_root(root)
    _write_dataset(root, "daily", _daily_frames()["daily"].drop(columns="amount"))
    with pytest.raises(KeyError, match="amount"):
        _run(root, tmp_path / "report")
    assert not (tmp_path / "report").exists()
