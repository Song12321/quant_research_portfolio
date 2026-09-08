"""全市场股票数据日更：拉取、合并、直接保存。"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_lib.config.constant_config import MARKET_DATA_ROOT, get_market_data_path
from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api


ALL_DATASETS = (
    'stock_basic.parquet', 'industry_record.parquet',
    'daily', 'daily_hfq', 'daily_basic', 'stk_limit', 'suspend_d.parquet',
    'balancesheet.parquet', 'cashflow.parquet', 'income.parquet',
    'fina_indicator.parquet', 'dividend.parquet', 'namechange.parquet',
)
_DAILY = ('daily', 'daily_hfq', 'daily_basic', 'stk_limit')
_STOCK_BASIC_FIELDS = (
    'ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,'
    'curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type'
)
_REPORT_KEY = ['ts_code', 'end_date', 'ann_date', 'f_ann_date', 'report_type']
_KEYS = {
    **{name: ['ts_code', 'trade_date'] for name in _DAILY},
    'balancesheet.parquet': _REPORT_KEY,
    'cashflow.parquet': _REPORT_KEY,
    'income.parquet': _REPORT_KEY,
    'fina_indicator.parquet': ['ts_code', 'end_date', 'ann_date'],
    'dividend.parquet': ['ts_code', 'end_date', 'ann_date', 'div_proc', 'imp_ann_date'],
    'namechange.parquet': ['ts_code', 'start_date', 'name'],
}
_DATETIMES = {
    'daily_hfq': ('trade_date',),
    'industry_record.parquet': ('in_date', 'out_date'),
    'fina_indicator.parquet': ('end_date',),
}


def _pro(api: str, **params) -> pd.DataFrame:
    return call_pro_tushare_api(api, max_retries=1, **params)


def _concat(frames) -> pd.DataFrame:
    parts = list(frames)
    if any(not isinstance(part, pd.DataFrame) for part in parts):
        raise TypeError('接口必须返回 DataFrame')
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _fetch(dataset: str, start: str, end: str, symbols: list[str]) -> pd.DataFrame:
    dates = pd.date_range(start, end).strftime('%Y%m%d')
    if dataset == 'industry_record.parquet':
        return _concat(
            _pro('index_member_all', ts_code=code, is_new=state)
            for code in symbols for state in ('N', 'Y')
        )
    if dataset == 'namechange.parquet':
        # 名称变更会补充旧名称的结束日，逐股获取完整记录再合并。
        return _concat(_pro('namechange', ts_code=code) for code in symbols)
    if dataset == 'daily_hfq':
        return _concat(
            call_ts_tushare_api('pro_bar', max_retries=1, ts_code=code,
                               start_date=start, end_date=end, adj='hfq', asset='E')
            for code in symbols
        )
    if dataset in ('daily', 'daily_basic', 'stk_limit', 'suspend_d.parquet'):
        return _concat(_pro(dataset.removesuffix('.parquet'), trade_date=date) for date in dates)
    if dataset == 'dividend.parquet':
        return _concat(
            _pro('dividend', **{field: date})
            for date in dates for field in ('ann_date', 'imp_ann_date')
        )
    # 四张财务表按公告日查询，不扫描全历史报告期，不改写公告日期。
    return _concat(
        _pro(dataset.removesuffix('.parquet') + '_vip', ann_date=date)
        for date in dates
    )


def _save(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _merge_save(dataset: str, path: Path, new: pd.DataFrame,
                start: str, end: str) -> int:
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if dataset == 'suspend_d.parquet':
        # 全市场范围替换，合法空结果也清除该日期范围的旧事件。
        if not old.empty:
            dates = pd.to_datetime(old['trade_date'], format='%Y%m%d')
            old = old.loc[~dates.between(pd.Timestamp(start), pd.Timestamp(end))]
        merged = pd.concat([old, new], ignore_index=True).drop_duplicates()
    else:
        merged = pd.concat([old, new], ignore_index=True).drop_duplicates(
            subset=_KEYS[dataset], keep='last',
        )
    _save(path, merged)
    return len(new)


def _update(dataset: str, start: str, end: str, symbols: list[str]) -> int:
    # 返回本次拉取的行数，不是去重后的行数，也不是相对旧文件净新增的行数。
    new = _fetch(dataset, start, end, symbols)
    path = get_market_data_path(dataset, MARKET_DATA_ROOT)
    # 普通表返回空时保留旧文件；行业表拉取的是全市场完整历史，空结果视为异常。
    # 停牌表不能在这里跳过：即使本次没有事件，也要在 _merge_save 中清除
    # [start, end] 内的旧事件，用本次查询结果替换这段日期范围。
    if new.empty and dataset != 'suspend_d.parquet':
        if dataset == 'industry_record.parquet':
            raise ValueError('industry_record: 全市场行业历史返回空，停止更新')
        return 0
    # 只转换已约定存为 datetime 的列，保持各表现有格式；未配置的表不做转换。
    for column in _DATETIMES.get(dataset, ()):
        new[column] = pd.to_datetime(new[column], format='%Y%m%d')
    if dataset == 'industry_record.parquet':
        # _fetch 不按 start/end 截取行业记录，而是逐股拉取历史和当前成员记录。
        # 因此整表去重后覆盖保存，让旧记录的退出日期等信息随本次结果刷新。
        _save(path, new.drop_duplicates())
        return len(new)
    if dataset in _DAILY:
        # 保持现有年份分区和后复权日期类型。
        # 临时解析 trade_date 只用于分组，不回写该列；后复权日期已在上面转换。
        # 每个年份只读取、合并并保存对应文件，未涉及的年份文件保持原样。
        years = pd.to_datetime(new['trade_date'], format='%Y%m%d').dt.year
        for year, part in new.groupby(years):
            _merge_save(dataset, path / f'year={year}' / 'data.parquet', part, start, end)
        return len(new)
    # 其余表存为单文件：普通表按 _KEYS 去重，同键以新记录覆盖旧记录，
    # 本次未返回的旧记录保留；停牌表则按上面说明替换指定日期范围。
    return _merge_save(dataset, path, new, start, end)


def upsert_market_data(start_date: str, end_date: str) -> dict[str, int]:
    """更新 stock 下的 13 类数据，返回各类拉取行数。

    调用或写入失败直接停止，已保存的数据不回滚。
    财务按公告日查询，不保证捕获公告日未变化的历史修订。
    """
    for date in (start_date, end_date):
        if not isinstance(date, str) or len(date) != 8 or not date.isdigit():
            raise ValueError(f'日期必须为 YYYYMMDD，实际为 {date!r}')
        datetime.strptime(date, '%Y%m%d')
    if start_date > end_date:
        raise ValueError('start_date 必须不晚于 end_date')

    basic = _concat(_pro('stock_basic', list_status=status, fields=_STOCK_BASIC_FIELDS)
                    for status in ('L', 'D', 'P'))
    if basic.empty:
        raise ValueError('stock_basic: 全市场股票名单返回空，停止更新')
    symbols = basic['ts_code'].drop_duplicates().tolist()
    _save(get_market_data_path('stock_basic.parquet', MARKET_DATA_ROOT), basic)
    result = {'stock_basic.parquet': len(basic)}
    for dataset in ALL_DATASETS[1:]:
        result[dataset] = 0
        if dataset in _DAILY:
            for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
                start = max(start_date, f'{year}0101')
                end = min(end_date, f'{year}1231')
                result[dataset] += _update(dataset, start, end, symbols)
        else:
            result[dataset] = _update(dataset, start_date, end_date, symbols)
    return result
