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
        return pd.DataFrame([{'ts_code': params['ts_code'], 'trade_date': params['start_date'],
                              'close': 40.0}])

    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', root)
    monkeypatch.setattr(updater, 'call_pro_tushare_api', pro)
    monkeypatch.setattr(updater, 'call_ts_tushare_api', ts)
    return calls


def test_all_thirteen_datasets_cross_year_and_rerun(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch, tmp_path)
    result = updater.upsert_market_data('20241231', '20250102')
    assert tuple(result) == updater.ALL_DATASETS
    assert len(result) == 13
    assert {p['ts_code'] for api, p in calls if api == 'pro_bar'} == {'L', 'D', 'P'}
    files = sorted(tmp_path.rglob('*.parquet'))
    assert len(files) == 17  # 四类行情各两个年份，其他九类单文件。
    assert all(p.relative_to(tmp_path).parts[0] == 'stock' for p in files)
    before = {p: pd.read_parquet(p) for p in files}
    updater.upsert_market_data('20241231', '20250102')
    for path, frame in before.items():
        assert_frame_equal(frame, pd.read_parquet(path))
    assert not list(tmp_path.rglob('*.tmp'))
    assert not list(tmp_path.rglob('*.bak'))


def test_daily_upsert_keeps_unreturned_rows(tmp_path):
    path = tmp_path / 'data.parquet'
    old = pd.DataFrame([{'ts_code': c, 'trade_date': '20250102', 'close': 1.0} for c in ('L', 'D')])
    old.to_parquet(path, index=False)
    new = old.iloc[:1].assign(close=2.0)
    updater._merge_save('daily', path, new, '20250102', '20250102')
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
    updater._merge_save('income.parquet', path, old.iloc[1:].assign(value=2.0), '20250102', '20250102')
    result = pd.read_parquet(path)
    assert result['value'].tolist() == [1.0, 2.0]


def test_empty_suspensions_remove_only_requested_dates(tmp_path):
    path = tmp_path / 'suspend.parquet'
    old = pd.DataFrame([{'ts_code': 'L', 'trade_date': d, 'suspend_type': 'S'}
                        for d in ('20250101', '20250102')])
    old.to_parquet(path, index=False)
    updater._merge_save('suspend_d.parquet', path, pd.DataFrame(), '20250102', '20250102')
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
        updater.upsert_market_data('20250102', '20250102')
    assert updater.get_market_data_path('stock_basic.parquet', tmp_path).exists()
    assert updater.get_market_data_path('industry_record.parquet', tmp_path).exists()
    assert not any(api == 'income_vip' for api, _ in calls)


def test_write_failure_propagates(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', fail)
    with pytest.raises(OSError, match='disk full'):
        updater._save(tmp_path / 'data.parquet', pd.DataFrame({'x': [1]}))


def test_dividend_phases_are_distinct(tmp_path):
    path = tmp_path / 'dividend.parquet'
    old = pd.DataFrame([{'ts_code': 'L', 'end_date': '20241231', 'ann_date': '20250101',
                         'div_proc': '预案', 'imp_ann_date': None, 'cash_div': 1.0}])
    old.to_parquet(path, index=False)
    new = old.assign(div_proc='实施', imp_ann_date='20250102', cash_div=0.8)
    for _ in range(2):
        updater._merge_save('dividend.parquet', path, new, '20250102', '20250102')
    assert len(pd.read_parquet(path)) == 2
