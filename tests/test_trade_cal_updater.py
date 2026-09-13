"""交易日历使用模拟接口和临时 Parquet，不访问真实数据。"""
import pandas as pd
import pytest

from quant_lib.tushare.data import market_data_updater as updater


def calendar():
    return pd.DataFrame({
        'exchange': ['SSE'] * 4,
        'cal_date': ['20260403', '20260404', '20260405', '20260406'],
        'is_open': [1, 0, 0, 0],
    })


def test_calendar_save_and_local_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)

    def fetch(api, **params):
        assert api == 'trade_cal'
        assert params['exchange'] == 'SSE'
        assert 'is_open' not in params
        return calendar()

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fetch)
    assert updater.update_trade_cal('20260403', '20260406') == 4
    assert len(pd.read_parquet(tmp_path / 'shared/trade_cal.parquet')) == 4
    assert updater.read_trade_dates('20260403', '20260406') == ['20260403']
    assert updater.read_trade_dates('20260404', '20260406') == []


@pytest.mark.parametrize('update', [
    updater.update_daily, updater.update_daily_basic, updater.update_stk_limit,
    updater.update_suspend, updater.update_hm_detail,
])
def test_fetches_only_request_local_open_dates(tmp_path, monkeypatch, update):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    updater._save(updater._path('trade_cal.parquet'), calendar())
    calls = []

    def fetch(api, **params):
        calls.append(params['trade_date'])
        return pd.DataFrame({'ts_code': ['000001.SZ'], 'trade_date': [params['trade_date']]})

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fetch)
    assert update('20260403', '20260406') == 1
    assert calls == ['20260403']


@pytest.mark.parametrize('invalid', ['missing', 'duplicate', 'flag'])
def test_invalid_calendar_stops_before_api(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    frame = calendar()
    if invalid == 'missing':
        frame = frame.iloc[:3]
    elif invalid == 'duplicate':
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[0, 'is_open'] = 2
    updater._save(updater._path('trade_cal.parquet'), frame)

    def unexpected(*args, **kwargs):
        pytest.fail('日历校验失败后不应请求接口')

    monkeypatch.setattr(updater, 'call_pro_tushare_api', unexpected)
    with pytest.raises(ValueError, match='trade_cal'):
        updater.update_daily('20260403', '20260406')


def test_missing_local_calendar_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        updater.read_trade_dates('20260403', '20260406')
