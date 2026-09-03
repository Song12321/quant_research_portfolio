import pandas as pd
import pytest

from projects._03_factor_selection.data_manager.stock_history import apply_history_days_filter


def test_history_days_excludes_251_observations_but_admits_252():
    dates = pd.bdate_range('2023-01-02', periods=253)
    pool = pd.DataFrame({'000001.SZ': True}, index=dates)
    close_hfq = pd.DataFrame({'000001.SZ': range(253)}, index=dates)

    result = apply_history_days_filter(pool, close_hfq, 252)

    assert result['000001.SZ'].iloc[-2:].tolist() == [False, True]


def test_history_days_inherits_observations_before_research_window():
    dates = pd.bdate_range('2023-01-02', periods=253)
    close_hfq = pd.DataFrame({'000001.SZ': range(253)}, index=dates)
    pool = pd.DataFrame({'000001.SZ': [True]}, index=dates[-1:])

    result = apply_history_days_filter(pool, close_hfq, 252)

    assert result['000001.SZ'].tolist() == [True]


def test_history_days_zero_keeps_pool_without_price_data():
    pool = pd.DataFrame({'000001.SZ': [True]}, index=pd.to_datetime(['2021-07-03']))
    result = apply_history_days_filter(pool, None, 0)

    pd.testing.assert_frame_equal(result, pool)


@pytest.mark.parametrize('history_days', [True, -1, 252.0, '252'])
def test_history_days_rejects_invalid_values(history_days):
    pool = pd.DataFrame({'000001.SZ': [True]}, index=pd.to_datetime(['2021-07-03']))
    close_hfq = pd.DataFrame({'000001.SZ': [1.0]}, index=pool.index)

    with pytest.raises(ValueError, match='history_days 必须是非负整数'):
        apply_history_days_filter(pool, close_hfq, history_days)


def test_history_days_does_not_count_missing_close_as_history():
    dates = pd.bdate_range('2024-01-02', periods=6)
    close_hfq = pd.DataFrame({'000001.SZ': [1.0, 1.0, None, 1.0, 1.0, 1.0]}, index=dates)
    pool = close_hfq.shift(1).notna()

    result = apply_history_days_filter(pool, close_hfq, 4)

    assert result['000001.SZ'].iloc[-2:].tolist() == [False, True]
