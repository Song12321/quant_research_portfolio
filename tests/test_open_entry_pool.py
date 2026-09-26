from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.data_manager.entry_pool import (
    apply_open_buy_filter,
    build_open_tradeable_mask,
)
from projects._03_factor_selection.factor_manager.factor_analyzer import factor_analyzer as analyzer_module
from quant_lib.evaluation.evaluation import calculate_forward_returns_tradable_o2o


def events(rows):
    return pd.DataFrame(rows, columns=['ts_code', 'trade_date', 'suspend_type', 'suspend_timing'])


def test_open_suspend_mask_handles_new_persistent_intraday_and_resume_events():
    dates = pd.date_range('2024-01-02', periods=3)
    source = events([
        ('NEW', '20240102', 'S', None),
        ('NEW', '20240104', 'R', None),
        ('PRIOR', '20231229', 'S', None),
        ('PRIOR', '20240103', 'R', None),
        ('OPEN', '20240102', 'S', '9:30-9:40,10:00:00-10:05:00'),
        ('LATER', '20240102', 'S', '09:31:07-09:41:07'),
        ('AFTERNOON', '20240102', 'S', '13:00-15:00'),
    ])
    stocks = ['NEW', 'PRIOR', 'OPEN', 'LATER', 'AFTERNOON', 'NORMAL']
    result = build_open_tradeable_mask(source, dates, stocks)
    assert result['NEW'].tolist() == [False, False, True]
    assert result['PRIOR'].tolist() == [False, True, True]
    assert result['OPEN'].tolist() == [False, True, True]
    assert result[['LATER', 'AFTERNOON', 'NORMAL']].to_numpy().all()


def panels():
    dates = pd.date_range('2024-01-02', periods=3)
    stocks = ['UP', 'DOWN', 'NORMAL', 'SUSPENDED', 'OUTSIDE']
    pool = pd.DataFrame(True, index=dates, columns=stocks)
    pool['OUTSIDE'] = False
    opening = pd.DataFrame([[10] * 5, [11 - 1e-14, 9, 10, np.nan, np.nan], [10] * 5], index=dates, columns=stocks, dtype=float)
    upper = pd.DataFrame(11., index=dates, columns=stocks)
    tradeable = pd.DataFrame(True, index=dates, columns=stocks)
    tradeable.loc[dates[1], 'SUSPENDED'] = False
    return pool, opening, upper, tradeable


def test_open_limit_up_excluded_same_day_limit_down_kept_and_pool_not_mutated():
    args = panels()
    original = args[0].copy()
    result = apply_open_buy_filter(*args)
    assert result.iloc[0].tolist() == [False, True, True, False, False]
    assert result.iloc[1].tolist() == [True, True, True, True, False]
    assert not result.iloc[2].any()
    pd.testing.assert_frame_equal(original, args[0])


def test_entry_pool_integrates_data_source_cache_and_future_exit_contract(monkeypatch):
    from projects._03_factor_selection.data_manager import data_manager as data_module

    pool, opening, upper, _ = panels()
    # Future exit at limit down must not erase today's return label.
    opening.loc[opening.index[2], 'DOWN'] = 8.1
    upper.loc[opening.index[2], 'DOWN'] = 9.9
    manager = DataManager({'research_window': {'start_date': '20240102', 'end_date': '20240103'}})
    manager.config['stock_pool_profiles'] = {'ALL': {'filters': {'remove_st': False}}}
    manager.stock_pools_dict = {'ALL': pool}
    fields = {'open_raw': opening, 'up_limit': upper}
    manager.get_base_field_df = Mock(side_effect=fields.__getitem__)
    monkeypatch.setattr(data_module, 'load_suspend_d_df', lambda: events([
        ('SUSPENDED', '20240103', 'S', None), ('SUSPENDED', '20240104', 'R', None),
    ]))
    entry = manager.get_entry_pool('ALL')
    assert manager.get_entry_pool('ALL') is entry
    assert manager.get_base_field_df.call_count == 2
    returns = calculate_forward_returns_tradable_o2o(1, opening * 2, entry)
    assert pd.isna(returns.iloc[0]['UP'])
    assert pd.isna(returns.iloc[0]['SUSPENDED'])
    assert returns.iloc[0]['DOWN'] == pytest.approx(-0.1)


@pytest.mark.parametrize('already_processed', [False, True])
def test_evaluation_masks_after_preprocessing_and_preserves_composite_inputs(monkeypatch, already_processed):
    pool, *_ = panels()
    entry = pool.copy()
    entry.iloc[0, 0] = False
    processed = pd.DataFrame(1., index=pool.index, columns=pool.columns).where(pool)
    analyzer = analyzer_module.FactorAnalyzer.__new__(analyzer_module.FactorAnalyzer)
    analyzer.config = {"stage": "inner"}
    analyzer.n_quantiles = 2
    analyzer.factor_manager = Mock()
    analyzer.factor_manager.data_manager.get_entry_pool.return_value = entry
    analyzer._process_single_factor = Mock(return_value=processed)
    analyzer.test_ic_analysis = Mock(return_value=({}, {}))
    analyzer.test_quantile_backtest = Mock(return_value=({}, {}))
    analyzer.test_turnover_result = Mock(return_value={})
    daily = Mock(return_value={})
    monkeypatch.setattr(analyzer_module, 'calculate_quantile_daily_returns', daily)
    result = analyzer.analyze_processed_factor('demo', processed, 'ALL', Mock(), already_processed)
    expected = processed.where(entry)
    for function in (analyzer.test_ic_analysis, analyzer.test_quantile_backtest, analyzer.test_turnover_result, daily):
        pd.testing.assert_frame_equal(function.call_args.args[0], expected)
    assert result['processed_factor_df'] is processed
    if not already_processed:
        pd.testing.assert_frame_equal(analyzer._process_single_factor.call_args.args[1], processed)
