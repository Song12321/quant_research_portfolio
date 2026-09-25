"""Local fixtures and mocked APIs only; never write real market data."""
from types import SimpleNamespace

import pandas as pd
import pytest
import tushare
from pandas.testing import assert_frame_equal

from projects._03_factor_selection.data_manager.data_manager import DataManager
from projects._03_factor_selection.factor_manager.factor_manager import FactorManager
from projects._03_factor_selection.factor_manager.factor_calculator.factor_calculator import FactorCalculator
from quant_lib.data_loader import DataLoader
from quant_lib.tushare.data import market_data_updater as updater


@pytest.fixture
def market(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    dates = ['20241231', '20250102']
    daily = pd.DataFrame([
        dict(ts_code=code, trade_date=date, open=10.12, close=10.37,
             high=10.50, low=10.01, pre_close=10.11, change=0.26,
             pct_chg=2.57, vol=100.0, amount=102.0)
        for date in dates for code in ['000001.SZ', '600000.SH']
    ])
    updater._save_daily_by_year('daily', daily)
    updater._save(updater._path('trade_cal.parquet'), pd.DataFrame({
        'exchange': ['SSE'] * 3,
        'cal_date': ['20241231', '20250101', '20250102'],
        'is_open': [1, 0, 1],
    }))
    factors = daily[['ts_code', 'trade_date']].copy()
    factors['adj_factor'] = [1.234, 2.345, 1.567, 2.789]
    calls = []

    def fetch(api, **params):
        assert api == 'adj_factor'
        assert set(params) == {'max_retries', 'trade_date'}
        calls.append(params['trade_date'])
        return factors.loc[factors['trade_date'].eq(params['trade_date'])].copy()

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fetch)
    return daily, factors, calls


def make_factor_manager():
    data = DataManager.__new__(DataManager)
    data.data_loader = DataLoader(data_root=updater.MARKET_DATA_ROOT)
    data.data_loader.trade_cal = data.data_loader._load_trade_cal()
    data.buffer_start_date = '20241231'
    data.research_end_date = '20250102'
    data._entry_pools = {}
    manager = FactorManager.__new__(FactorManager)
    manager.data_manager = data
    manager.factors_cache = {}
    manager.calculator = FactorCalculator(manager)
    return manager


def test_raw_storage_increment_and_legacy_file_untouched(market):
    daily, factors, calls = market
    legacy_path = updater._path('daily_hfq') / 'year=2024/data.parquet'
    updater._save(legacy_path, daily.assign(close=9999.0))
    original = legacy_path.read_bytes()
    assert updater.update_adj_factor('20241231', '20241231') == 2
    assert updater.update_adj_factor('20200101', '20250102') == 2
    assert calls == ['20241231', '20250102']
    actual = pd.concat([
        pd.read_parquet(updater._path('adj_factor') / f'year={year}/data.parquet')
        for year in [2024, 2025]
    ], ignore_index=True)
    assert_frame_equal(actual, factors)
    assert legacy_path.read_bytes() == original
    calls.clear()
    assert updater.update_adj_factor('20200101', '20250102') == 0
    assert calls == []


def test_calculated_prices_match_installed_pro_bar(market):
    daily, factors, calls = market
    expected = []
    for code in daily['ts_code'].unique():
        api = SimpleNamespace(
            daily=lambda **kw: daily.loc[daily['ts_code'].eq(code)].copy(),
            adj_factor=lambda **kw: factors.loc[
                factors['ts_code'].eq(code), ['trade_date', 'adj_factor']
            ].copy(),
        )
        expected.append(tushare.pro_bar(
            ts_code=code, api=api, start_date='20241231', end_date='20250102',
            adj='hfq', retry_count=1,
        ))
    expected = pd.concat(expected, ignore_index=True)
    expected['trade_date'] = pd.to_datetime(expected['trade_date'])
    assert updater.update_adj_factor('20241231', '20250102') == 4
    assert calls == ['20241231', '20250102']
    # A stale legacy file must not supply any raw fields.
    updater._save(updater._path('daily_hfq') / 'year=2024/data.parquet',
                  daily.assign(open=9999.0, close=9999.0, vol=9999.0))
    manager = make_factor_manager()
    assert not hasattr(manager.data_manager.data_loader, 'field_map')
    for column in ['open', 'close', 'high', 'low']:
        wanted = expected.pivot(index='trade_date', columns='ts_code', values=column)
        actual = manager.get_raw_factor(column + '_hfq')
        assert_frame_equal(actual, wanted)
    multiplier = manager.get_raw_factor('hfq_adj_factor')
    assert_frame_equal(multiplier, manager.get_raw_factor('adj_factor'))
    assert_frame_equal(manager.get_raw_factor('vol_hfq'),
                       manager.get_raw_factor('vol_raw') / multiplier)
    assert_frame_equal(manager.get_raw_factor('pct_chg'),
                       manager.get_raw_factor('close_hfq').pct_change())
    # Returned frames are copies of the existing factor cache.
    returned = manager.get_raw_factor('close_hfq')
    returned.iloc[0, 0] = -1
    assert manager.get_raw_factor('close_hfq').iloc[0, 0] > 0


@pytest.mark.parametrize('failure', ['empty', 'duplicate', 'wrong_date', 'api'])
def test_bad_response_stops_without_raw_write(market, monkeypatch, failure):
    _, factors, _ = market

    def fetch(api, **params):
        frame = factors.loc[factors['trade_date'].eq(params['trade_date'])].copy()
        if failure == 'api':
            raise RuntimeError('API failure')
        if failure == 'empty':
            return frame.iloc[:0]
        if failure == 'duplicate':
            return pd.concat([frame, frame.iloc[:1]])
        if failure == 'wrong_date':
            return frame.assign(trade_date='20250103')

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fetch)
    with pytest.raises((ValueError, RuntimeError)):
        updater.update_adj_factor('20241231', '20250102')
    assert not updater._path('adj_factor').exists()
    assert not updater._path('daily_hfq').exists()


@pytest.mark.parametrize('value', [float('nan'), 0, -1, float('inf')])
def test_invalid_factor_stops_calculation(market, value):
    updater.update_adj_factor('20241231', '20250102')
    manager = make_factor_manager()
    factors = manager.get_raw_factor('adj_factor')
    factors.iloc[0, 0] = value
    manager.factors_cache['adj_factor'] = factors
    with pytest.raises(ValueError, match='adj_factor'):
        manager.get_raw_factor('close_hfq')


def test_missing_raw_factors_do_not_fall_back_to_legacy(market):
    daily, _, _ = market
    updater._save(updater._path('daily_hfq') / 'year=2024/data.parquet', daily)
    manager = make_factor_manager()
    with pytest.raises(FileNotFoundError, match='adj_factor'):
        manager.get_raw_factor('close_hfq')


def test_suspension_nan_is_preserved(market, monkeypatch):
    updater.update_adj_factor('20241231', '20250102')
    manager = make_factor_manager()
    read_field = manager.data_manager.data_loader.read_base_field

    def read_with_missing_close(field, *args):
        frame = read_field(field, *args)
        if field == 'close_raw':
            frame.iloc[0, 0] = float('nan')
        return frame

    monkeypatch.setattr(manager.data_manager.data_loader, 'read_field', read_with_missing_close)
    factors = manager.get_raw_factor('adj_factor')
    factors.iloc[0, 0] = float('nan')
    manager.factors_cache['adj_factor'] = factors
    assert pd.isna(manager.get_raw_factor('close_hfq').iloc[0, 0])


def test_nontrading_day_needs_no_quotes_or_api(market):
    _, _, calls = market
    assert updater.update_adj_factor('20250101', '20250101') == 0
    assert calls == []


def test_download_does_not_require_daily(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    updater._save(updater._path('trade_cal.parquet'), pd.DataFrame({
        'exchange': ['SSE'], 'cal_date': ['20250102'], 'is_open': [1],
    }))
    raw = pd.DataFrame({
        'ts_code': ['000001.SZ'], 'trade_date': ['20250102'], 'adj_factor': [1.234567],
    })
    monkeypatch.setattr(updater, 'call_pro_tushare_api', lambda *a, **kw: raw.copy())
    assert updater.update_adj_factor('20250102', '20250102') == 1
    assert_frame_equal(
        pd.read_parquet(updater._path('adj_factor') / 'year=2025/data.parquet'), raw,
    )
    assert not updater._path('daily').exists()
    assert not updater._path('daily_hfq').exists()


def test_prepare_and_stock_pool_use_positive_amount(market, monkeypatch):
    daily, _, _ = market
    updater._save_daily_by_year(
        'daily_basic', daily[['ts_code', 'trade_date']].assign(circ_mv=100.0, turnover_rate=1.0),
    )
    updater._save(updater._path('stock_basic.parquet'), pd.DataFrame({
        'ts_code': daily['ts_code'].unique(), 'list_date': ['20200101', '20200101'],
    }))
    data = make_factor_manager().data_manager
    data.research_start_date = '20241231'
    data.trading_dates = data.data_loader.get_trading_dates('20241231', '20250102')
    data.config = {'stock_pool_name': 'ALL', 'stock_pool_profiles': {'ALL': {
        'index_filter': {'enable': False},
        'filters': {'history_days': 1, 'remove_st': False,
                    'adapt_tradeable_matrix_by_suspend_resume': False},
    }}}
    monkeypatch.setattr(data, 'show_stock_nums_for_per_day', lambda *args: None)
    module = 'projects._03_factor_selection.data_manager.data_manager'
    monkeypatch.setattr(module + '.PointInTimeIndustryMap', lambda: None)
    data._prepare_stock_pool()
    assert not hasattr(data, 'raw_dfs')
    expected = data.get_raw_field('amount').reindex(data.trading_dates).gt(0)
    assert_frame_equal(data.stock_pools_dict['ALL'], expected)
