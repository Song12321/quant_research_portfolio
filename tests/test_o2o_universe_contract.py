import pandas as pd
import pytest

from projects._03_factor_selection.factor_manager.factor_analyzer.factor_analyzer import (
    FactorAnalyzer,
)
from quant_lib.evaluation.evaluation import calculate_forward_returns_tradable_o2o


class _DataManager:
    def __init__(self, open_hfq: pd.DataFrame, entry_mask: pd.DataFrame):
        self.config = {
            "evaluation": {"returns_calculator": ["o2o"]},
            "research_window": {
                "start_date": "2024-01-02",
                "end_date": "2024-01-04",
            },
        }
        self.open_hfq = open_hfq
        self.stock_pools_dict = {"POOL": entry_mask}

    def get_entry_pool(self, stock_pool_name):
        return self.stock_pools_dict[stock_pool_name]

    @staticmethod
    def is_composite_factor(factor_name: str) -> bool:
        assert factor_name == "demo"
        return False

    @staticmethod
    def get_stock_pool_storage_name_by_name(stock_pool_name: str) -> str:
        assert stock_pool_name == "POOL"
        return "POOL"


class _FactorManager:
    def __init__(self, data_manager: _DataManager, factor_data: pd.DataFrame):
        self.data_manager = data_manager
        self.factor_data = factor_data

    def get_raw_factor(self, field_name):
        assert field_name == "open_hfq"
        return self.data_manager.open_hfq

    def get_prepare_aligned_factor_for_analysis(
        self, factor_name: str, stock_pool_name: str, for_test: bool
    ) -> pd.DataFrame:
        assert factor_name == "demo"
        assert stock_pool_name == "POOL"
        assert for_test is True
        return self.factor_data


def test_o2o_uses_raw_future_open_without_relaxing_t_day_pool_membership():
    dates = pd.date_range("2024-01-02", periods=3)
    stocks = ["STAYS", "EXITS_AFTER_T", "OUTSIDE_AT_T"]
    factor_data = pd.DataFrame(
        [
            [1.0, 2.0, float("nan")],
            [1.0, float("nan"), 3.0],
            [1.0, float("nan"), 3.0],
        ],
        index=dates,
        columns=stocks,
    )
    raw_open = pd.DataFrame(
        [
            [10.0, 20.0, 30.0, 40.0],
            [11.0, 22.0, 33.0, 44.0],
            [12.0, 24.0, 36.0, 48.0],
        ],
        index=dates,
        columns=stocks + ["NOT_IN_RESEARCH_GRID"],
    )
    entry_mask = pd.DataFrame(
        [
            [True, True, False],
            [True, False, True],
            [True, False, True],
        ],
        index=dates,
        columns=stocks,
    )
    analyzer = FactorAnalyzer.__new__(FactorAnalyzer)
    analyzer.factor_manager = _FactorManager(
        _DataManager(raw_open, entry_mask), factor_data
    )

    prepared_factor, _, calculators = analyzer.prepare_data_for_entity_service(
        "demo", "POOL"
    )
    forward_returns = calculators["o2o"](period=1)
    valid_sample = prepared_factor.notna() & forward_returns.notna()

    assert forward_returns.loc[dates[0], "EXITS_AFTER_T"] == pytest.approx(24 / 22 - 1)
    assert valid_sample.loc[dates[0], "EXITS_AFTER_T"]
    assert not valid_sample.loc[dates[0], "OUTSIDE_AT_T"]
    assert pd.isna(forward_returns.loc[dates[0], "OUTSIDE_AT_T"])
    assert list(forward_returns.columns) == stocks


def test_t_day_pool_mask_excludes_outlier_before_cross_sectional_winsorize():
    dates = pd.date_range("2024-01-02", periods=3)
    pool_stocks = [f"STOCK_{number:02d}" for number in range(40)]
    all_stocks = pool_stocks + ["OUTSIDE_EXTREME"]
    start_prices = [100.0] * len(all_stocks)
    end_prices = [101.0 + number for number in range(40)] + [100_000.0]
    open_with_outlier = pd.DataFrame(
        [start_prices, start_prices, end_prices], index=dates, columns=all_stocks
    )
    mask_with_outlier = pd.DataFrame(
        [[True] * 40 + [False]] * 3, index=dates, columns=all_stocks
    )

    with_outlier = calculate_forward_returns_tradable_o2o(
        period=1,
        open_df=open_with_outlier,
        entry_mask=mask_with_outlier,
    )
    without_outlier = calculate_forward_returns_tradable_o2o(
        period=1,
        open_df=open_with_outlier[pool_stocks],
        entry_mask=mask_with_outlier[pool_stocks],
    )

    pd.testing.assert_frame_equal(
        with_outlier[pool_stocks],
        without_outlier,
        check_exact=True,
    )
    assert pd.isna(with_outlier.loc[dates[0], "OUTSIDE_EXTREME"])


@pytest.mark.parametrize('period', [1, 2])
def test_signal_date_returns_start_next_trading_open_across_weekend(period):
    dates = pd.to_datetime(['20240105', '20240108', '20240109', '20240110'])
    prices = pd.DataFrame({'A': [100., 10., 12., 18.]}, index=dates)
    # Later pool exits must not erase T's valid label.
    mask = pd.DataFrame({'A': [True, False, False, False]}, index=dates)
    result = calculate_forward_returns_tradable_o2o(period, prices, mask)
    assert result.index.equals(dates)
    assert result.iloc[0, 0] == pytest.approx([0.2, 0.8][period - 1])
    assert result.iloc[1:, 0].isna().all()
    all_entries = pd.DataFrame(True, index=dates, columns=['A'])
    tail = calculate_forward_returns_tradable_o2o(period, prices, all_entries)
    assert tail.iloc[-(period + 1):].isna().all().all()
