import pandas as pd
import pytest
from unittest.mock import Mock

from quant_lib.data_loader import DataLoader


@pytest.mark.parametrize("field,dataset,column", [
    *[(name + "_raw", "daily", name) for name in ("open", "close", "high", "low", "vol")],
    ("amount", "daily", "amount"),
    *[(name, "daily_basic", name) for name in ("circ_mv", "total_mv", "turnover_rate", "dv_ttm")],
    ("adj_factor", "adj_factor", "adj_factor"),
    ("up_limit", "stk_limit", "up_limit"),
    ("list_date", "stock_basic.parquet", "list_date"),
    ("delist_date", "stock_basic.parquet", "delist_date"),
])
def test_fixed_field_sources(tmp_path, field, dataset, column):
    loader = DataLoader(tmp_path)
    loader._read_panel = Mock(return_value=object())
    result = loader.read_field(field, "20240101", "20240103")
    assert result is loader._read_panel.return_value
    loader._read_panel.assert_called_once_with(dataset, column, "20240101", "20240103", None)


@pytest.mark.parametrize("field", ["unknown", "pe_ttm", "pb", "ps_ttm", "open_hfq"])
def test_unknown_fields_do_not_scan_or_guess(tmp_path, field):
    loader = DataLoader(tmp_path)
    loader._read_panel = Mock()
    with pytest.raises(ValueError, match="未定义字段读取方式"):
        loader.read_field(field, "20240101", "20240103")
    loader._read_panel.assert_not_called()


def test_requested_read_failure_propagates(tmp_path):
    loader = DataLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.read_field("close_raw", "20240101", "20240103")


def test_missing_column_does_not_use_another_dataset(tmp_path):
    from pyarrow import ArrowInvalid
    from quant_lib.config.constant_config import get_market_data_path
    path = get_market_data_path("daily", tmp_path) / "year=2024" / "data.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["a"], "trade_date": ["20240102"], "open": [1.0]}).to_parquet(path)
    pd.DataFrame({"close": [99.0]}).to_parquet(tmp_path / "stock" / "other.parquet")
    with pytest.raises(ArrowInvalid, match="close"):
        DataLoader(tmp_path).read_field("close_raw", "20240101", "20240103")


def test_daily_read_ignores_unrelated_corrupt_files(tmp_path):
    from quant_lib.config.constant_config import get_market_data_path
    path = get_market_data_path("daily", tmp_path) / "year=2024" / "data.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({
        "ts_code": ["a", "a", "a", "b"],
        "trade_date": ["20240102", "20240102", "20240103", "20240102"],
        "close": [1.0, 2.0, 3.0, 99.0],
    }).to_parquet(path)
    (tmp_path / "stock" / "broken.parquet").write_bytes(b"broken unrelated file")
    loader = DataLoader(tmp_path)
    loader.trade_cal = pd.DataFrame({
        "cal_date": pd.to_datetime(["20240102", "20240103"]), "is_open": [1, 1],
    })
    result = loader.read_field("close_raw", "20240102", "20240102", ["a"])
    assert result.shape == (1, 1)
    assert result.iloc[0, 0] == 2.0


def test_static_dates_broadcast_without_trade_date_column(tmp_path):
    from quant_lib.config.constant_config import get_market_data_path
    path = get_market_data_path("stock_basic.parquet", tmp_path)
    path.parent.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["a"], "list_date": ["20000101"]}).to_parquet(path)
    loader = DataLoader(tmp_path)
    dates = pd.to_datetime(["20240102", "20240103"])
    loader.trade_cal = pd.DataFrame({"cal_date": dates, "is_open": [1, 1]})
    result = loader.read_field("list_date", "20240102", "20240103")
    assert result["a"].tolist() == ["20000101", "20000101"]
    assert result.index.equals(dates)
