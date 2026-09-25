"""空批次和全空字段不触发 pandas 拼接弃用警告。"""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_lib.tushare.data import market_data_updater as updater


pytestmark = pytest.mark.filterwarnings('error::FutureWarning')


def test_concat_preserves_rows_columns_and_numeric_dtype():
    empty = pd.DataFrame(columns=['value', 'empty_only'])
    missing = pd.DataFrame({'value': [None], 'unused': [None]})
    data = pd.DataFrame({'value': [1.5], 'unused': [None]})
    result = updater._concat(iter([empty, missing, data]))
    expected = pd.DataFrame({'value': [float('nan'), 1.5],
                             'empty_only': [float('nan')] * 2,
                             'unused': [None, None]})
    assert_frame_equal(result, expected)
    assert missing['value'].dtype == object


def test_concat_empty_inputs_keep_schema():
    empty = pd.DataFrame({'value': pd.Series(dtype='float64')})
    assert_frame_equal(updater._concat([empty, empty]), empty)
    assert_frame_equal(updater._concat([]), pd.DataFrame())
    with pytest.raises(TypeError, match='DataFrame'):
        updater._concat([empty, None])


def test_merge_missing_values_preserves_history_and_new_revision(tmp_path):
    path = tmp_path / 'daily.parquet'
    old = pd.DataFrame({'ts_code': ['A', 'B'], 'trade_date': ['20250101'] * 2,
                        'close': [1.0, 2.0]})
    old.to_parquet(path, index=False)
    new = pd.DataFrame({'ts_code': ['A'], 'trade_date': ['20250101'], 'close': [None]})
    assert updater._merge_save('daily', path, new) == 1
    expected = pd.DataFrame({'ts_code': ['B', 'A'], 'trade_date': ['20250101'] * 2,
                             'close': [2.0, float('nan')]})
    assert_frame_equal(pd.read_parquet(path), expected)
