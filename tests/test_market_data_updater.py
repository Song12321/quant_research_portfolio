"""仅使用模拟接口和临时目录验证股票日更。"""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_lib.tushare.data import market_data_updater as updater


def install_fakes(monkeypatch, root):
    calls = []

    def pro(api, max_retries, **params):
        assert max_retries == 1
        calls.append((api, params))
        if api == 'stock_basic':
            return pd.DataFrame([{'ts_code': params['list_status'], 'list_status': params['list_status']}])
        if api == 'index_member_all':
            return pd.DataFrame([{'ts_code': params['ts_code'], 'in_date': '20200101',
                                  'out_date': None, 'is_new': params['is_new']}])
        if api in ('daily', 'daily_basic', 'stk_limit'):
            return pd.DataFrame([{'ts_code': 'L', 'trade_date': params['trade_date'], 'close': 20.0}])
        if api == 'suspend_d':
            return pd.DataFrame(columns=['ts_code', 'trade_date', 'suspend_type', 'suspend_timing'])
        if api in ('income_vip', 'balancesheet_vip', 'cashflow_vip', 'fina_indicator_vip'):
            assert set(params) == {'ann_date'}
            return pd.DataFrame([{'ts_code': 'L', 'end_date': '20240930',
                                  'ann_date': params['ann_date'], 'f_ann_date': params['ann_date'],
                                  'report_type': '1', 'update_flag': '1', 'value': 20.0}])
        if api == 'dividend':
            return pd.DataFrame([{'ts_code': 'L', 'end_date': '20241231', 'ann_date': '20250101',
                                  'div_proc': '实施', 'imp_ann_date': '20250102', 'cash_div': 1.0}])
        if api == 'namechange':
            return pd.DataFrame([{'ts_code': params['ts_code'], 'start_date': '20200101',
                                  'name': 'name', 'end_date': '20250102'}])
        raise AssertionError(api)

    def ts(api, max_retries, **params):
        assert api == 'pro_bar'
        assert max_retries == 1
        calls.append((api, params))
        dates = pd.date_range(params['start_date'], params['end_date']).strftime('%Y%m%d')
        return pd.DataFrame({'ts_code': params['ts_code'], 'trade_date': dates, 'close': 40.0})

    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', root)
    monkeypatch.setattr(updater, 'call_pro_tushare_api', pro)
    monkeypatch.setattr(updater, 'call_ts_tushare_api', ts)
    return calls


def test_independent_updates_cross_year_and_rerun(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, tmp_path)
    updater.update_stock_basic()
    updater.update_industry_record()
    updater.update_dividend()
    updater.update_namechange()
    updates = (
        updater.update_daily, updater.update_daily_hfq, updater.update_daily_basic,
        updater.update_stk_limit, updater.update_suspend, updater.update_balancesheet,
        updater.update_cashflow, updater.update_income, updater.update_fina_indicator,
    )
    for update in updates:
        update('20241231', '20250102')
    assert {p['ts_code'] for api, p in calls if api == 'pro_bar'} == {'L', 'D', 'P'}
    files = sorted(tmp_path.rglob('*.parquet'))
    assert len(files) == 17
    assert all(p.relative_to(tmp_path).parts[0] == 'stock' for p in files)
    before = {p: pd.read_parquet(p) for p in files}
    for update in updates:
        update('20241231', '20250102')
    for path, frame in before.items():
        assert_frame_equal(frame, pd.read_parquet(path))
    assert not list(tmp_path.rglob('*.tmp'))
    assert not list(tmp_path.rglob('*.bak'))


def test_daily_upsert_keeps_unreturned_rows(tmp_path):
    path = tmp_path / 'data.parquet'
    old = pd.DataFrame([{'ts_code': c, 'trade_date': '20250102', 'close': 1.0} for c in ('L', 'D')])
    old.to_parquet(path, index=False)
    new = old.iloc[:1].assign(close=2.0)
    updater._merge_save('daily', path, new)
    result = pd.read_parquet(path).set_index('ts_code')
    assert result.loc['L', 'close'] == 2.0
    assert result.loc['D', 'close'] == 1.0


def test_financial_revision_preserves_other_announcements(tmp_path):
    path = tmp_path / 'income.parquet'
    old = pd.DataFrame([
        {'ts_code': 'L', 'end_date': '20240930', 'ann_date': d, 'f_ann_date': d,
         'report_type': '1', 'value': 1.0} for d in ('20241030', '20250102')
    ])
    old.to_parquet(path, index=False)
    updater._merge_save('income.parquet', path, old.iloc[1:].assign(value=2.0))
    result = pd.read_parquet(path)
    assert result['value'].tolist() == [1.0, 2.0]


def test_empty_suspensions_remove_only_requested_dates(tmp_path, monkeypatch):
    install_fakes(monkeypatch, tmp_path)
    path = updater._path('suspend_d.parquet')
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.DataFrame([{'ts_code': 'L', 'trade_date': d, 'suspend_type': 'S'}
                        for d in ('20250101', '20250102')])
    old.to_parquet(path, index=False)
    # 固定请求窗口，验证搬移后的范围替换规则，包括合法空响应。
    monkeypatch.setattr(updater, '_incremental_start', lambda *args: '20250102')
    updater.update_suspend('20250101', '20250102')
    assert pd.read_parquet(path)['trade_date'].tolist() == ['20250101']


def test_failure_stops_without_rollback(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, tmp_path)
    original = updater.call_pro_tushare_api

    def fail(api, **params):
        if api == 'daily':
            raise RuntimeError('daily failed')
        return original(api, **params)

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fail)
    with pytest.raises(RuntimeError, match='daily failed'):
        updater.update_stock_basic()
        updater.update_industry_record()
        updater.update_daily('20250102', '20250102')
    assert updater.get_market_data_path('stock_basic.parquet', tmp_path).exists()
    assert updater.get_market_data_path('industry_record.parquet', tmp_path).exists()
    assert not any(api == 'income_vip' for api, _ in calls)


def test_write_failure_propagates(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', fail)
    with pytest.raises(OSError, match='disk full'):
        updater._save(tmp_path / 'data.parquet', pd.DataFrame({'x': [1]}))


def test_dividend_full_refresh_replaces_old_version(tmp_path, monkeypatch):
    install_fakes(monkeypatch, tmp_path)
    updater.update_stock_basic()
    path = updater._path('dividend.parquet')
    old = pd.DataFrame([{'ts_code': 'L', 'end_date': '20241231', 'ann_date': '20250101',
                         'div_proc': '预案', 'imp_ann_date': None, 'cash_div': 1.0}])
    updater._save(path, old)
    calls = []
    def dividend(api, **params):
        assert api == 'dividend'
        assert set(params) == {'ts_code', 'max_retries'}
        calls.append(params['ts_code'])
        if params['ts_code'] != 'L':
            return pd.DataFrame()
        return old.assign(div_proc='实施', imp_ann_date='20250601', cash_div=0.8)
    monkeypatch.setattr(updater, 'call_pro_tushare_api', dividend)
    updater.update_dividend()
    result = pd.read_parquet(path)
    assert calls == ['L', 'D', 'P']
    assert len(result) == 1
    assert result.iloc[0]['ann_date'] == '20250101'
    assert result.iloc[0]['imp_ann_date'] == '20250601'
    assert result.iloc[0]['div_proc'] == '实施'


@pytest.mark.parametrize('failure', ['limit', 'api', 'empty', 'type'])
def test_dividend_failure_preserves_file(tmp_path, monkeypatch, failure):
    install_fakes(monkeypatch, tmp_path)
    updater.update_stock_basic()
    path = updater._path('dividend.parquet')
    updater._save(path, pd.DataFrame({'old': [1]}))
    original = path.read_bytes()
    def dividend(api, **params):
        if failure == 'limit':
            return pd.DataFrame({'x': range(2000)})
        if failure == 'api':
            if params['ts_code'] == 'L':
                return pd.DataFrame({'x': [1]})
            raise RuntimeError('request failed')
        if failure == 'type':
            return None
        return pd.DataFrame()
    monkeypatch.setattr(updater, 'call_pro_tushare_api', dividend)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        updater.update_dividend()
    assert path.read_bytes() == original


def test_each_dataset_uses_own_max_date_and_skips(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, tmp_path)
    updater._save(updater._path('daily') / 'year=2025/data.parquet',
                  pd.DataFrame({'ts_code': ['L'], 'trade_date': ['20250101']}))
    updater._save(updater._path('daily_basic') / 'year=2025/data.parquet',
                  pd.DataFrame({'ts_code': ['L'], 'trade_date': ['20250102']}))
    updater.update_daily('20241201', '20250103')
    updater.update_daily_basic('20241201', '20250103')
    assert calls == [('daily', {'trade_date': '20250102'}),
                     ('daily', {'trade_date': '20250103'}),
                     ('daily_basic', {'trade_date': '20250103'})]
    calls.clear()
    updater.update_daily('20241201', '20250103')
    assert calls == []
    assert not updater._path('stock_basic.parquet').exists()


def test_financial_increment_uses_announcement_not_report_date(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, tmp_path)
    updater.update_income('20250101', '20250102')
    calls.clear()
    updater.update_income('20240101', '20250103')
    assert calls == [('income_vip', {'ann_date': '20250103'})]
    assert len(pd.read_parquet(updater._path('income.parquet'))) == 3


def test_max_date_reads_all_partitions_and_datetime(tmp_path, monkeypatch):
    install_fakes(monkeypatch, tmp_path)
    updater._save(updater._path('daily_hfq') / 'year=2024/data.parquet',
                  pd.DataFrame({'trade_date': pd.to_datetime(['2024-12-31'])}))
    updater._save(updater._path('daily_hfq') / 'year=2025/data.parquet',
                  pd.DataFrame({'trade_date': pd.Series([], dtype='datetime64[ns]')}))
    assert updater._incremental_start('daily_hfq', 'trade_date', '20200101', '20250103') == '20250101'


@pytest.mark.parametrize('values', [['bad'], [None]])
def test_invalid_local_date_stops(tmp_path, monkeypatch, values):
    calls = install_fakes(monkeypatch, tmp_path)
    updater._save(updater._path('daily') / 'year=2025/data.parquet',
                  pd.DataFrame({'trade_date': values}))
    with pytest.raises(ValueError):
        updater.update_daily('20250101', '20250103')
    assert calls == []


def test_missing_date_column_stops(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, tmp_path)
    updater._save(updater._path('income.parquet'), pd.DataFrame({'wrong': [1]}))
    with pytest.raises(Exception, match='ann_date'):
        updater.update_income('20250101', '20250103')
    assert calls == []


def test_missing_stock_list_stops(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        updater.update_dividend()
    assert calls == []


def hm_frame(date):
    return pd.DataFrame([
        {'trade_date': date, 'ts_code': '000001.SZ', 'ts_name': 'stock',
         'buy_amount': 10.0, 'sell_amount': 4.0, 'net_amount': 6.0,
         'hm_name': name, 'hm_orgs': 'org', 'tag': None}
        for name in ('a', 'b', 'b')
    ])


def test_hm_detail_cross_year_increment_and_raw_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    calls = []

    def fetch(api, max_retries, **params):
        assert api == 'hm_detail'
        assert max_retries == 1
        assert params['fields'].split(',') == list(hm_frame('20241231').columns)
        calls.append(params['trade_date'])
        return hm_frame(params['trade_date'])

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fetch)
    assert updater.update_hm_detail('20241231', '20250101') == 6
    assert calls == ['20241231', '20250101']
    root = tmp_path / 'stock/market_metrics/hm_detail'
    assert_frame_equal(pd.read_parquet(root / 'year=2024/data.parquet'), hm_frame('20241231'))
    assert_frame_equal(pd.read_parquet(root / 'year=2025/data.parquet'), hm_frame('20250101'))
    calls.clear()
    assert updater.update_hm_detail('20220801', '20250102') == 3
    assert calls == ['20250102']
    assert len(pd.read_parquet(root / 'year=2025/data.parquet')) == 6
    calls.clear()
    assert updater.update_hm_detail('20220801', '20250102') == 0
    assert calls == []


def test_hm_detail_empty_replaces_only_requested_range(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    path = updater._path('hm_detail') / 'year=2025/data.parquet'
    updater._save(path, pd.concat([hm_frame('20250101'), hm_frame('20250102')]))
    monkeypatch.setattr(updater, '_incremental_start', lambda *args: '20250102')
    monkeypatch.setattr(updater, 'call_pro_tushare_api',
                        lambda *args, **kwargs: hm_frame('20250102').iloc[:0])
    assert updater.update_hm_detail('20250101', '20250102') == 0
    assert_frame_equal(pd.read_parquet(path), hm_frame('20250101'))


@pytest.mark.parametrize('failure', ['fields', 'date', 'null_date', 'type', 'api', 'limit'])
def test_hm_detail_failure_keeps_completed_year(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from quant_lib.tushare import api_wrapper

    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    calls = []

    def fetch(**params):
        date = params['trade_date']
        calls.append(date)
        if date == '20241231':
            return hm_frame(date)
        if failure == 'fields':
            return hm_frame(date).drop(columns='tag')
        if failure == 'date':
            return hm_frame('20241230')
        if failure == 'null_date':
            return hm_frame(date).assign(trade_date=pd.NA)
        if failure == 'type':
            return None
        if failure == 'limit':
            return pd.concat([hm_frame(date)] * 667, ignore_index=True).iloc[:2000]
        raise RuntimeError('request failed')

    monkeypatch.setattr(api_wrapper.TushareClient, 'get_pro',
                        lambda: SimpleNamespace(hm_detail=fetch))
    monkeypatch.setattr(api_wrapper.shared_rate_limiter, 'wait', lambda: None)
    with pytest.raises((ValueError, TypeError)):
        updater.update_hm_detail('20241231', '20250102')
    assert calls == ['20241231', '20250101']
    root = updater._path('hm_detail')
    assert_frame_equal(pd.read_parquet(root / 'year=2024/data.parquet'), hm_frame('20241231'))
    assert not (root / 'year=2025/data.parquet').exists()
