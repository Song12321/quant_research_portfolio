from unittest.mock import Mock

import pandas as pd
import pytest

from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.data_manager.entry_pool import apply_open_buy_filter


def test_signal_pool_and_next_open_filter_use_different_dates():
    dates = pd.to_datetime(['20240105', '20240108', '20240109'])
    pool = pd.DataFrame({'UP_T': [True, True, True], 'UP_NEXT': [True, True, True],
                         'OUT_T': [False, True, True], 'EXIT_NEXT': [True, False, False]}, index=dates)
    opening = pd.DataFrame([[11, 10, 10, 10], [10, 11, 10, 10], [10, 10, 10, 10]],
                           index=dates, columns=pool.columns, dtype=float)
    limit = pd.DataFrame(11., index=dates, columns=pool.columns)
    tradeable = pd.DataFrame(True, index=dates, columns=pool.columns)
    actual = apply_open_buy_filter(pool, opening, limit, tradeable)
    # Friday limit up doesn't block Monday; Monday limit up does.
    assert actual.iloc[0].tolist() == [True, False, False, True]
    assert not actual.iloc[-1].any()


def test_pool_uses_current_positive_amount_and_st_without_suspend_filter():
    dates = pd.to_datetime(['20240105', '20240108', '20240109'])
    manager = DataManager({'research_window': {'start_date': '20240105', 'end_date': '20240109'}})
    manager.trading_dates = dates
    amount = pd.DataFrame({'A': [None, 10., 10.], 'B': [10., 10., 0.]}, index=dates)
    manager.get_base_field_df = Mock(return_value=amount)
    manager.st_matrix = pd.DataFrame({'A': [False, False, True], 'B': [False, True, False]}, index=dates)
    manager.build_st_period_from_namechange = Mock()
    manager._tradeable_matrix_by_suspend_resume = pd.DataFrame(
        {'A': [True, True, True], 'B': [False, True, True]}, index=dates)
    manager.build_tradeable_matrix_by_suspend_resume = Mock()
    profile = {'filters': {'history_days': 0, 'remove_st': True,
                           'adapt_tradeable_matrix_by_suspend_resume': True}}
    actual = manager.create_stock_pool(profile, 'ALL')
    expected = pd.DataFrame({'A': [False, True, False], 'B': [True, False, False]}, index=dates)
    pd.testing.assert_frame_equal(actual, expected)
    manager.get_base_field_df.assert_called_once_with('amount')
    manager.build_tradeable_matrix_by_suspend_resume.assert_not_called()


def test_liquidity_and_size_filters_read_signal_day_values():
    dates = pd.to_datetime(['20240105', '20240108'])
    manager = DataManager({'research_window': {'start_date': '20240105', 'end_date': '20240108'}})
    values = pd.DataFrame({'A': [1., 3.], 'B': [3., 1.]}, index=dates)
    manager.get_base_field_df = Mock(return_value=values)
    pool = pd.DataFrame(True, index=dates, columns=values.columns)
    expected = pd.DataFrame({'A': [False, True], 'B': [True, False]}, index=dates)
    pd.testing.assert_frame_equal(manager._filter_by_liquidity(pool.copy(), 0.5), expected)
    pd.testing.assert_frame_equal(manager._filter_by_market_cap(pool.copy(), 0.5), expected)


@pytest.mark.parametrize('method,field', [
    ('_filter_by_liquidity', 'turnover_rate'),
    ('_filter_by_market_cap', 'circ_mv'),
])
@pytest.mark.parametrize('case', ['partial', 'all', 'missing_column', 'missing_date'])
def test_pool_filters_reject_missing_active_values(method, field, case):
    dates = pd.to_datetime(['20240105'])
    values = pd.DataFrame([[1., float('nan')]], index=dates, columns=['A', 'B'])
    if case == 'all':
        values.loc[:, :] = float('nan')
    elif case == 'missing_column':
        values = values[['A']]
    elif case == 'missing_date':
        values = values.iloc[:0]
    manager = DataManager({'research_window': {'start_date': '20240105', 'end_date': '20240105'}})
    manager.get_base_field_df = Mock(return_value=values)
    pool = pd.DataFrame(True, index=dates, columns=['A', 'B'])
    original = pool.copy()
    with pytest.raises(ValueError, match=field + r' 缺失：日期=2024-01-05.*股票='):
        getattr(manager, method)(pool, 0.5)
    pd.testing.assert_frame_equal(pool, original)


@pytest.mark.parametrize('method', ['_filter_by_liquidity', '_filter_by_market_cap'])
def test_pool_filters_allow_missing_outside_pool_and_empty_day(method):
    dates = pd.to_datetime(['20240105', '20240108'])
    values = pd.DataFrame([[1., 3., float('nan')], [float('nan')] * 3],
                          index=dates, columns=['A', 'B', 'C'])
    pool = pd.DataFrame([[True, True, False], [False] * 3], index=dates, columns=values.columns)
    manager = DataManager({'research_window': {'start_date': '20240105', 'end_date': '20240108'}})
    manager.get_base_field_df = Mock(return_value=values)
    manager.show_stock_nums_for_per_day = Mock()
    expected = pd.DataFrame([[False, True, False], [False] * 3], index=dates, columns=values.columns)
    pd.testing.assert_frame_equal(getattr(manager, method)(pool, 0.5), expected)
