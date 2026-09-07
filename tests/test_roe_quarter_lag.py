import numpy as np
import pandas as pd
import pytest

from projects._03_factor_selection.factor_manager.factor_calculator import factor_calculator


class _DataManager:
    def __init__(self, trading_dates):
        self._prebuffer_trading_dates = pd.DatetimeIndex(trading_dates)


class _FactorManager:
    def __init__(self, trading_dates):
        self.data_manager = _DataManager(trading_dates)


def _income_rows(ts_code="000001.SZ"):
    return pd.DataFrame({
        "ts_code": [ts_code] * 5,
        "end_date": pd.to_datetime([
            "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31",
        ]),
        "f_ann_date": pd.to_datetime([
            "2023-04-20", "2023-08-15", "2023-10-31", "2024-03-20", "2024-04-27",
        ]),
        "n_income_attr_p": [10.0, 30.0, 60.0, 100.0, 50.0],
    })


def _equity_rows(ts_code="000001.SZ"):
    return pd.DataFrame({
        "ts_code": [ts_code] * 5,
        "end_date": pd.to_datetime([
            "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31",
        ]),
        "f_ann_date": pd.to_datetime([
            "2023-04-25", "2023-08-20", "2023-11-05", "2024-03-25", "2024-05-04",
        ]),
        "total_hldr_eqy_exc_min_int": [100.0, 900.0, 800.0, 700.0, 200.0],
    })


def _calculate(monkeypatch, income_df, equity_df):
    trading_dates = pd.to_datetime(["2024-04-29", "2024-05-03", "2024-05-06", "2024-05-07"])
    monkeypatch.setattr(factor_calculator, "load_income_df", lambda: income_df.copy())
    monkeypatch.setattr(factor_calculator, "load_balancesheet_df", lambda: equity_df.copy())
    calculator = factor_calculator.FactorCalculator(_FactorManager(trading_dates))
    return calculator._calculate_roe_ttm()


def test_roe_uses_exact_fifth_quarter_equity_and_latest_real_announcement(monkeypatch):
    result = _calculate(monkeypatch, _income_rows(), _equity_rows())

    expected_roe = 140.0 / ((200.0 + 100.0) / 2.0)
    assert np.isnan(result.loc[pd.Timestamp("2024-05-03"), "000001.SZ"])
    assert np.isclose(result.loc[pd.Timestamp("2024-05-06"), "000001.SZ"], expected_roe)
    assert np.isclose(result.loc[pd.Timestamp("2024-05-07"), "000001.SZ"], expected_roe)


def test_roe_is_nan_when_exact_four_quarter_prior_equity_is_missing(monkeypatch):
    income_df = _income_rows().iloc[:4].copy()
    equity_df = _equity_rows().iloc[:4].copy()
    result = _calculate(monkeypatch, income_df, equity_df)

    assert result["000001.SZ"].isna().all()


def test_roe_is_nan_when_profit_quarters_are_not_continuous(monkeypatch):
    income_df = _income_rows().loc[lambda frame: frame["end_date"] != pd.Timestamp("2023-09-30")]
    result = _calculate(monkeypatch, income_df, _equity_rows())

    assert result["000001.SZ"].isna().all()


def test_roe_rejects_missing_current_equity_record(monkeypatch):
    equity_df = _equity_rows().loc[
        lambda frame: frame["end_date"] != pd.Timestamp("2024-03-31")
    ]

    with pytest.raises(ValueError) as exc_info:
        _calculate(monkeypatch, _income_rows(), equity_df)

    message = str(exc_info.value)
    assert "ts_code=000001.SZ" in message
    assert "end_date=2024-03-31" in message
    assert "field=total_hldr_eqy_exc_min_int" in message


@pytest.mark.parametrize("source_name", ["income", "equity"])
def test_roe_rejects_missing_announcement_date(monkeypatch, source_name):
    income_df = _income_rows()
    equity_df = _equity_rows()
    source_df = income_df if source_name == "income" else equity_df
    source_df.loc[source_df.index[-1], "f_ann_date"] = pd.NaT

    with pytest.raises(ValueError) as exc_info:
        _calculate(monkeypatch, income_df, equity_df)

    message = str(exc_info.value)
    assert "ts_code=000001.SZ" in message
    assert "end_date=2024-03-31" in message
    assert "field=f_ann_date" in message


def test_roe_rejects_nonfinite_current_equity(monkeypatch):
    equity_df = _equity_rows()
    equity_df.loc[equity_df.index[-1], "total_hldr_eqy_exc_min_int"] = np.inf

    with pytest.raises(ValueError, match="field=total_hldr_eqy_exc_min_int"):
        _calculate(monkeypatch, _income_rows(), equity_df)
