"""Verify classification refreshes without network access."""

from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_lib.tushare import api_wrapper
from quant_lib.tushare.data import market_data_updater as updater


def classification(level):
    return pd.DataFrame([{
        'index_code': f'80100{level[-1]}.SI',
        'industry_name': f'industry-{level}',
        'parent_code': '0' if level == 'L1' else '801001',
        'level': level,
        'industry_code': f'110{level[-1]}00',
        'is_pub': '1',
        'src': 'SW2021',
    }])


def test_index_classify_full_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    calls = []

    def fetch(**params):
        calls.append(params)
        assert set(params) == {'src', 'fields'}
        assert params['src'] == 'SW2021'
        assert params['fields'].split(',') == list(classification('L1').columns)
        return pd.concat([classification(level) for level in ('L1', 'L2', 'L3')],
                         ignore_index=True)

    monkeypatch.setattr(api_wrapper.TushareClient, 'get_pro',
                        lambda: SimpleNamespace(index_classify=fetch))
    monkeypatch.setattr(api_wrapper.shared_rate_limiter, 'wait', lambda: None)
    path = tmp_path / 'index/shenwan/reference/index_classify.parquet'
    updater._save(path, pd.DataFrame({'old': [1]}))
    expected = pd.concat([classification(level) for level in ('L1', 'L2', 'L3')],
                         ignore_index=True)
    for _ in range(2):
        calls.clear()
        assert updater.update_index_classify() == 3
        assert_frame_equal(pd.read_parquet(path), expected)
        assert len(calls) == 1


@pytest.mark.parametrize('failure', [
    'empty', 'fields', 'level', 'src', 'null', 'code', 'duplicate', 'type', 'api',
])
def test_index_classify_failure_preserves_file(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', tmp_path)
    path = updater._path('index_classify.parquet')
    updater._save(path, pd.DataFrame({'old': [1]}))
    original = path.read_bytes()
    calls = []

    def fetch(api, **params):
        assert api == 'index_classify'
        assert 'level' not in params
        calls.append(params)
        frame = pd.concat([classification(level) for level in ('L1', 'L2', 'L3')],
                          ignore_index=True)
        if failure == 'empty':
            return frame.iloc[:0]
        if failure == 'fields':
            return frame.drop(columns='parent_code')
        if failure == 'level':
            return frame.assign(level='L4')
        if failure == 'src':
            return frame.assign(src='SW2014')
        if failure == 'null':
            return frame.assign(industry_name=None)
        if failure == 'code':
            return frame.assign(index_code='')
        if failure == 'duplicate':
            return frame.assign(index_code=classification('L1').iloc[0]['index_code'])
        if failure == 'type':
            return None
        raise RuntimeError('request failed')

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fetch)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        updater.update_index_classify()
    assert path.read_bytes() == original
    assert len(calls) == 1
