from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_lib.tushare import api_wrapper
from quant_lib.tushare.data import market_data_updater as updater


STOCK = '000001.SZ'
DAY = '20250102'


def _empty(dataset: str) -> pd.DataFrame:
    return pd.DataFrame(columns=updater._REQUIRED_COLUMNS[dataset])


def _fake_pro(api_name: str, **params) -> pd.DataFrame:
    if api_name == 'trade_cal':
        return pd.DataFrame([{'exchange': 'SSE', 'cal_date': DAY, 'is_open': 1}])
    if api_name == 'stock_basic':
        return pd.DataFrame([{
            'ts_code': STOCK, 'list_status': 'L', 'list_date': '19910403', 'delist_date': None,
        }])
    if api_name == 'index_basic':
        return pd.DataFrame([{'ts_code': '801010.SI', 'category': '一级行业指数'}])
    if api_name == 'daily':
        return pd.DataFrame([{'ts_code': STOCK, 'trade_date': DAY, 'open': 10.0, 'close': 11.0}])
    if api_name == 'adj_factor':
        return pd.DataFrame([{'ts_code': STOCK, 'trade_date': DAY, 'adj_factor': 2.0}])
    if api_name == 'daily_basic':
        return pd.DataFrame([{'ts_code': STOCK, 'trade_date': DAY, 'turnover_rate': 1.5}])
    if api_name == 'margin_detail':
        return pd.DataFrame([{'ts_code': STOCK, 'trade_date': DAY}])
    if api_name == 'stk_limit':
        return pd.DataFrame([{'ts_code': STOCK, 'trade_date': DAY, 'up_limit': 12.0, 'down_limit': 8.0}])
    if api_name == 'suspend_d':
        return _empty('suspend_d.parquet')
    if api_name == 'index_member_all':
        if params['is_new'] == 'Y':
            return _empty('industry_record.parquet')
        return pd.DataFrame([{'ts_code': STOCK, 'in_date': '20210101', 'out_date': None}])
    if api_name == 'dividend':
        assert set(params) in ({'ann_date'}, {'imp_ann_date'})
        return _empty('dividend.parquet')
    if api_name == 'namechange':
        assert params == {'start_date': DAY, 'end_date': DAY}
        return _empty('namechange.parquet')
    if api_name in {'balancesheet_vip', 'cashflow_vip', 'fina_indicator_vip'}:
        dataset = {'balancesheet_vip': 'balancesheet.parquet',
                   'cashflow_vip': 'cashflow.parquet',
                   'fina_indicator_vip': 'fina_indicator.parquet'}[api_name]
        if api_name == 'balancesheet_vip':
            assert set(params) == {'period'}
        elif api_name == 'cashflow_vip':
            assert set(params) == {'f_ann_date'}
        else:
            assert set(params) == {'ann_date'}
        return _empty(dataset)
    if api_name == 'income_vip':
        assert set(params) == {'f_ann_date'}
        return pd.DataFrame([{
            'ts_code': STOCK, 'ann_date': DAY, 'f_ann_date': DAY,
            'end_date': '20241231', 'update_flag': '1', 'n_income': 30.0,
        }])
    if api_name == 'index_daily':
        return pd.DataFrame([{'ts_code': params['ts_code'], 'trade_date': DAY, 'close': 100.0}])
    if api_name == 'index_weight':
        assert params['start_date'] == '20250101'
        assert params['end_date'] == '20250131'
        return pd.DataFrame([{
            'index_code': params['index_code'], 'con_code': STOCK,
            'trade_date': DAY, 'weight': 1.0,
        }])
    if api_name == 'sw_daily':
        return pd.DataFrame([{'ts_code': params['ts_code'], 'trade_date': DAY, 'close': 100.0}])
    raise AssertionError(f'未模拟的Tushare接口: {api_name}, params={params}')


def _fake_ts(api_name: str, **params) -> pd.DataFrame:
    assert api_name == 'pro_bar'
    return pd.DataFrame([{'ts_code': params['ts_code'], 'trade_date': DAY, 'open': 20.0, 'close': 22.0}])


def _write_existing_data(root: Path) -> Path:
    daily_path = updater.get_market_data_path('daily', root) / 'year=2025' / 'data.parquet'
    daily_path.parent.mkdir(parents=True)
    pd.DataFrame([
        {'ts_code': STOCK, 'trade_date': '20250101', 'open': 7.0, 'close': 8.0},
        {'ts_code': STOCK, 'trade_date': DAY, 'open': 8.0, 'close': 9.0},
    ]).to_parquet(daily_path, index=False)

    income_path = updater.get_market_data_path('income.parquet', root)
    income_path.parent.mkdir(parents=True)
    pd.DataFrame([
        {'ts_code': STOCK, 'ann_date': '20250101', 'f_ann_date': '20250101',
         'end_date': '20241231', 'update_flag': '0', 'n_income': 10.0},
        {'ts_code': STOCK, 'ann_date': DAY, 'f_ann_date': DAY,
         'end_date': '20241231', 'update_flag': '1', 'n_income': 20.0},
    ]).to_parquet(income_path, index=False)
    return daily_path


def _install_fakes(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(updater, 'MARKET_DATA_ROOT', root)
    monkeypatch.setattr(updater, 'call_pro_tushare_api', _fake_pro)
    monkeypatch.setattr(updater, 'call_ts_tushare_api', _fake_ts)


def _snapshot_parquet(root: Path) -> dict[str, pd.DataFrame]:
    return {str(path.relative_to(root)): pd.read_parquet(path)
            for path in sorted(root.rglob('*.parquet'))}


def test_upsert_all_datasets_is_idempotent_and_preserves_versions(tmp_path, monkeypatch):
    root = tmp_path / 'market_data'
    root.mkdir()
    daily_path = _write_existing_data(root)
    _install_fakes(monkeypatch, root)

    result = updater.upsert_market_data(DAY, DAY, [STOCK])

    assert tuple(result) == updater.ALL_DATASETS
    assert all(stats['files'] >= 1 for stats in result.values())
    assert result['daily']['removed_rows'] == 1
    daily = pd.read_parquet(daily_path)
    assert daily[['ts_code', 'trade_date']].duplicated().sum() == 0
    assert daily.loc[daily['trade_date'].eq(DAY), 'close'].item() == 11.0
    assert daily.loc[daily['trade_date'].eq('20250101'), 'close'].item() == 8.0

    income = pd.read_parquet(updater.get_market_data_path('income.parquet', root))
    assert set(income['ann_date']) == {'20250101', DAY}
    assert income.loc[income['ann_date'].eq(DAY), 'n_income'].item() == 30.0

    first = _snapshot_parquet(root)
    updater.upsert_market_data(DAY, DAY, [STOCK])
    second = _snapshot_parquet(root)
    assert first.keys() == second.keys()
    for path in first:
        assert_frame_equal(first[path], second[path])


def test_failure_before_commit_keeps_existing_files(tmp_path, monkeypatch):
    root = tmp_path / 'market_data'
    root.mkdir()
    daily_path = _write_existing_data(root)
    before = pd.read_parquet(daily_path)
    _install_fakes(monkeypatch, root)

    def fail_on_sw_daily(api_name: str, **params) -> pd.DataFrame:
        if api_name == 'sw_daily':
            raise ValueError('模拟申万行情接口失败')
        return _fake_pro(api_name, **params)

    monkeypatch.setattr(updater, 'call_pro_tushare_api', fail_on_sw_daily)
    with pytest.raises(ValueError, match='模拟申万行情接口失败'):
        updater.upsert_market_data(DAY, DAY, [STOCK])

    assert_frame_equal(before, pd.read_parquet(daily_path))
    assert not list(root.rglob('.*.tmp'))


def test_input_and_partition_boundaries_are_strict():
    with pytest.raises(ValueError, match='YYYYMMDD'):
        updater._validate_inputs('2025-01-01', DAY, [STOCK])
    with pytest.raises(ValueError, match='不得重复'):
        updater._validate_inputs(DAY, DAY, [STOCK, STOCK])
    assert updater._year_ranges('20241231', '20250102') == [
        (2024, '20241231', '20241231'),
        (2025, '20250101', '20250102'),
    ]
    assert updater._month_ranges('20250115', '20250203') == [
        ('20250101', '20250131'),
        ('20250201', '20250228'),
    ]
    assert updater._reporting_periods('20110101') == [
        '20100331', '20100630', '20100930', '20101231',
    ]


@pytest.mark.parametrize(
    ('api_name', 'rows', 'expected'),
    [
        ('index_member_all', 2000, True),
        ('sw_daily', 4000, True),
        ('stk_limit', 5800, True),
        ('daily', 5800, False),
        ('daily', 6000, True),
        ('index_basic', 6000, False),
        ('index_basic', 8000, True),
        ('suspend_d', 8000, False),
    ],
)
def test_api_row_limit_detection_is_interface_specific(api_name, rows, expected):
    assert api_wrapper.reach_limit(api_name, pd.DataFrame(index=range(rows))) is expected


def test_api_row_limit_detection_rejects_invalid_contract():
    with pytest.raises(TypeError, match='DataFrame'):
        api_wrapper.reach_limit('daily', None)
    with pytest.raises(ValueError, match='未配置'):
        api_wrapper.reach_limit('unknown_api', pd.DataFrame())


def test_commit_failure_rolls_back_prior_replacements(tmp_path, monkeypatch):
    targets = [tmp_path / 'one.parquet', tmp_path / 'two.parquet']
    temporaries = [tmp_path / '.one.tmp', tmp_path / '.two.tmp']
    for target in targets:
        pd.DataFrame({'value': ['old']}).to_parquet(target, index=False)
    for temporary in temporaries:
        pd.DataFrame({'value': ['new']}).to_parquet(temporary, index=False)
    stages = [
        updater._StagedWrite('daily', targets[index], temporaries[index], 1, 1, 1)
        for index in range(2)
    ]
    original_replace = updater.os.replace

    def fail_on_second_target(source, target):
        if Path(source) == temporaries[1] and Path(target) == targets[1]:
            raise OSError('模拟第二个正式文件替换失败')
        return original_replace(source, target)

    monkeypatch.setattr(updater.os, 'replace', fail_on_second_target)
    with pytest.raises(OSError, match='第二个正式文件替换失败'):
        updater._commit_staged(stages)

    for target in targets:
        assert pd.read_parquet(target)['value'].item() == 'old'
    assert not list(tmp_path.glob('*.bak'))


def test_daily_basic_may_be_a_strict_subset_of_prices():
    daily = pd.DataFrame([
        {'ts_code': STOCK, 'trade_date': DAY, 'open': 1.0, 'close': 1.0},
        {'ts_code': '000002.SZ', 'trade_date': DAY, 'open': 1.0, 'close': 1.0},
    ])
    frames = {
        'daily': daily,
        'daily_hfq': daily.copy(),
        'daily_basic': pd.DataFrame([
            {'ts_code': STOCK, 'trade_date': DAY, 'turnover_rate': 1.0},
        ]),
    }
    updater._validate_core_daily(frames)


def test_empty_response_without_schema_never_deletes_local_rows():
    existing = pd.DataFrame([{
        'ts_code': STOCK, 'trade_date': DAY, 'open': 1.0, 'close': 1.0,
    }])
    combined = updater._combine_range(
        'daily', existing, pd.DataFrame(), 'trade_date', DAY, DAY, [STOCK],
    )
    assert_frame_equal(existing, combined)


def test_required_market_dataset_rejects_schemaful_empty_response(monkeypatch):
    monkeypatch.setattr(
        updater,
        'call_pro_tushare_api',
        lambda api_name, **params: pd.DataFrame(columns=['ts_code', 'trade_date']),
    )
    with pytest.raises(ValueError, match='全市场接口返回空数据'):
        updater._fetch_by_dates('daily_basic', [DAY], [STOCK], 'trade_date', True)


def test_dividend_lifecycle_appends_new_version_without_growth_on_rerun():
    old = pd.DataFrame([{
        'ts_code': STOCK, 'ann_date': '20250101', 'end_date': '20241231',
        'div_proc': '预案', 'imp_ann_date': None, 'cash_div': 1.0,
    }])
    implemented = pd.DataFrame([{
        'ts_code': STOCK, 'ann_date': '20250101', 'end_date': '20241231',
        'div_proc': '实施', 'imp_ann_date': DAY, 'cash_div': 1.0,
    }])
    combined = updater._combine_dividend_versions(old, implemented)
    assert len(combined) == 2
    rerun = updater._combine_dividend_versions(combined, implemented)
    assert len(rerun) == 2
    corrected = implemented.assign(cash_div=2.0)
    corrected_result = updater._combine_dividend_versions(rerun, corrected)
    assert len(corrected_result) == 2
    assert corrected_result.loc[corrected_result['div_proc'].eq('实施'), 'cash_div'].item() == 2.0
