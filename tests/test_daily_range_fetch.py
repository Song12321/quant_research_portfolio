from types import SimpleNamespace

import pandas as pd
import pytest

from quant_lib.tushare import api_wrapper as wrapper
from quant_lib.tushare.data import market_data_updater as updater


@pytest.mark.parametrize('rows,expected', [(5999, False), (6000, True), (6001, True)])
def test_pro_bar_row_limit(rows, expected):
    assert wrapper.reach_limit('pro_bar', pd.DataFrame(index=range(rows))) is expected


@pytest.mark.parametrize('kind', ['pro', 'ts'])
def test_row_limit_propagates_without_retry(monkeypatch, kind):
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({'x': range(6000)})

    def unexpected():
        pytest.fail('row limit must not sleep or refresh token')

    monkeypatch.setattr(wrapper.shared_rate_limiter, 'wait', lambda: None)
    monkeypatch.setattr(wrapper.time, 'sleep', lambda _: unexpected())
    monkeypatch.setattr(wrapper.TushareClient, 'refresh_pro', unexpected)
    if kind == 'pro':
        monkeypatch.setattr(wrapper.TushareClient, 'get_pro',
                            lambda: SimpleNamespace(daily=fetch))
        call, api = wrapper.call_pro_tushare_api, 'daily'
    else:
        monkeypatch.setattr(wrapper.TushareClient, 'get_ts',
                            lambda: SimpleNamespace(pro_bar=fetch))
        call, api = wrapper.call_ts_tushare_api, 'pro_bar'
    with pytest.raises(wrapper.RowLimitExceeded):
        call(api)
    assert len(calls) == 1


def test_bisection_has_no_missing_or_overlapping_dates():
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        if start != end:
            raise wrapper.RowLimitExceeded('full')
        return pd.DataFrame({'trade_date': [start]})

    frame = updater._fetch_by_range(fetch, '20241231', '20250102')
    assert calls == [
        ('20241231', '20250102'), ('20241231', '20250101'),
        ('20241231', '20241231'), ('20250101', '20250101'),
        ('20250102', '20250102'),
    ]
    assert frame['trade_date'].tolist() == ['20241231', '20250101', '20250102']


@pytest.mark.parametrize('error,end', [
    (wrapper.RowLimitExceeded('full'), '20250101'),
    (RuntimeError('failed'), '20250103'),
])
def test_unsplittable_or_other_error_stops(error, end):
    calls = []

    def fetch(start, end):
        calls.append((start, end))
        raise error

    with pytest.raises(type(error)) as caught:
        updater._fetch_by_range(fetch, '20250101', end)
    assert caught.value is error
    assert len(calls) == 1


@pytest.mark.parametrize('dataset', ['daily', 'daily_basic', 'stk_limit', 'daily_hfq'])
def test_full_increment_cross_year_save_and_rerun(tmp_path, monkeypatch, dataset):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    monkeypatch.setattr(updater, '_symbols', lambda: ['A'])
    calls = []

    def fetch(api, **params):
        calls.append(params)
        return pd.DataFrame({
            'ts_code': ['A', 'A'], 'trade_date': ['20241231', '20250101'],
        })

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fetch)
    monkeypatch.setattr(updater, 'call_ts_tushare_api', fetch)
    old_date = pd.Timestamp('20241230') if dataset == 'daily_hfq' else '20241230'
    updater._save(updater._path(dataset) / 'year=2024/data.parquet',
                  pd.DataFrame({'ts_code': ['A'], 'trade_date': [old_date]}))
    update = getattr(updater, f'update_{dataset}')
    assert update('20200101', '20250101') == 2
    assert len(calls) == 1
    assert calls[0]['start_date'] == '20241231'
    assert calls[0]['end_date'] == '20250101'
    assert len(pd.read_parquet(updater._path(dataset) / 'year=2024/data.parquet')) == 2
    assert len(pd.read_parquet(updater._path(dataset) / 'year=2025/data.parquet')) == 1
    assert update('20200101', '20250101') == 0
    assert len(calls) == 1


@pytest.mark.parametrize('dataset', ['daily', 'daily_basic', 'stk_limit', 'daily_hfq'])
def test_empty_daily_result_does_not_save(tmp_path, monkeypatch, dataset):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    monkeypatch.setattr(updater, '_symbols', lambda: ['A'])
    monkeypatch.setattr(updater, 'call_pro_tushare_api', lambda *a, **kw: pd.DataFrame())
    monkeypatch.setattr(updater, 'call_ts_tushare_api', lambda *a, **kw: pd.DataFrame())

    assert getattr(updater, f'update_{dataset}')('20250101', '20250101') == 0
    assert not list(tmp_path.rglob('*.parquet'))


def test_hfq_splits_only_affected_stock_and_failure_does_not_save(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    monkeypatch.setattr(updater, '_symbols', lambda: ['A', 'B'])
    calls = []

    def fetch(api, **params):
        code, start, end = params['ts_code'], params['start_date'], params['end_date']
        calls.append((code, start, end))
        if code == 'B':
            if start != end:
                raise wrapper.RowLimitExceeded('full')
            raise RuntimeError('failed')
        return pd.DataFrame({'ts_code': [code], 'trade_date': [end]})

    monkeypatch.setattr(updater, 'call_ts_tushare_api', fetch)
    with pytest.raises(RuntimeError, match='failed'):
        updater.update_daily_hfq('20241231', '20250101')
    assert calls == [
        ('A', '20241231', '20250101'),
        ('B', '20241231', '20250101'),
        ('B', '20241231', '20241231'),
    ]
    assert not list(tmp_path.rglob('*.parquet'))
