"""Tushare 市场数据增量更新入口。

公开函数只接收日期范围和股票列表。不同数据集采用显式的更新规则，
避免用一套模糊的去重逻辑处理行情、快照和公告版本数据。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq

from quant_lib.config.constant_config import MARKET_DATA_ROOT, get_market_data_path
from quant_lib.tushare.api_wrapper import call_pro_tushare_api, call_ts_tushare_api


ALL_DATASETS = (
    'trade_cal.parquet', 'stock_basic.parquet', 'industry_record.parquet',
    'daily', 'daily_hfq', 'adj_factor', 'daily_basic', 'margin_detail',
    'stk_limit', 'suspend_d.parquet', 'balancesheet.parquet',
    'cashflow.parquet', 'income.parquet', 'fina_indicator.parquet',
    'dividend.parquet', 'namechange.parquet', 'index_daily.parquet',
    'index_weights', 'sw_basic_info.parquet', 'sw_daily.parquet',
)

_INDEX_DAILY_CODES = ('000300', '000905', '000906', '000852', '000985')
_INDEX_WEIGHT_CODES = ('000300.SH', '000905.SH', '000906.SH', '000852.SH')
_STOCK_CODE_PATTERN = re.compile(r'^\d{6}\.(?:SH|SZ|BJ)$')
_STOCK_BASIC_FIELDS = (
    'ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,'
    'curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type'
)

_DAILY_PRO_APIS = {
    'daily': 'daily',
    'adj_factor': 'adj_factor',
    'daily_basic': 'daily_basic',
    'margin_detail': 'margin_detail',
    'stk_limit': 'stk_limit',
}
_FINANCIAL_APIS = {
    'cashflow.parquet': ('cashflow_vip', 'f_ann_date'),
    'income.parquet': ('income_vip', 'f_ann_date'),
    'fina_indicator.parquet': ('fina_indicator_vip', 'ann_date'),
}
_FINANCIAL_HISTORY_START = '20100331'
_DIVIDEND_VERSION_KEY = ('ts_code', 'end_date', 'ann_date', 'div_proc', 'imp_ann_date')
_REQUIRED_COLUMNS = {
    'trade_cal.parquet': ('exchange', 'cal_date', 'is_open'),
    'stock_basic.parquet': ('ts_code', 'list_status', 'list_date', 'delist_date'),
    'industry_record.parquet': ('ts_code', 'in_date', 'out_date'),
    'daily': ('ts_code', 'trade_date', 'open', 'close'),
    'daily_hfq': ('ts_code', 'trade_date', 'open', 'close'),
    'adj_factor': ('ts_code', 'trade_date', 'adj_factor'),
    'daily_basic': ('ts_code', 'trade_date', 'turnover_rate'),
    'margin_detail': ('ts_code', 'trade_date'),
    'stk_limit': ('ts_code', 'trade_date', 'up_limit', 'down_limit'),
    'suspend_d.parquet': ('ts_code', 'trade_date', 'suspend_type'),
    'balancesheet.parquet': ('ts_code', 'ann_date', 'f_ann_date', 'end_date', 'update_flag'),
    'cashflow.parquet': ('ts_code', 'ann_date', 'f_ann_date', 'end_date', 'update_flag'),
    'income.parquet': ('ts_code', 'ann_date', 'f_ann_date', 'end_date', 'update_flag'),
    'fina_indicator.parquet': ('ts_code', 'ann_date', 'end_date', 'update_flag'),
    'dividend.parquet': ('ts_code', 'ann_date', 'end_date', 'div_proc', 'imp_ann_date'),
    'namechange.parquet': ('ts_code', 'ann_date', 'start_date', 'name'),
    'index_daily.parquet': ('ts_code', 'trade_date', 'close'),
    'index_weights': ('index_code', 'con_code', 'trade_date', 'weight'),
    'sw_basic_info.parquet': ('ts_code', 'category'),
    'sw_daily.parquet': ('ts_code', 'trade_date', 'close'),
}
_UNIQUE_KEYS = {
    'trade_cal.parquet': ('exchange', 'cal_date'),
    'stock_basic.parquet': ('ts_code',),
    'daily': ('ts_code', 'trade_date'),
    'daily_hfq': ('ts_code', 'trade_date'),
    'adj_factor': ('ts_code', 'trade_date'),
    'daily_basic': ('ts_code', 'trade_date'),
    'margin_detail': ('ts_code', 'trade_date'),
    'stk_limit': ('ts_code', 'trade_date'),
    'suspend_d.parquet': ('ts_code', 'trade_date', 'suspend_type', 'suspend_timing'),
    'index_daily.parquet': ('ts_code', 'trade_date'),
    'index_weights': ('index_code', 'con_code', 'trade_date'),
    'sw_basic_info.parquet': ('ts_code',),
    'sw_daily.parquet': ('ts_code', 'trade_date'),
}
_DATETIME_COLUMNS = {
    'daily_hfq': ('trade_date',),
    'industry_record.parquet': ('in_date', 'out_date'),
    'fina_indicator.parquet': ('end_date',),
    'sw_daily.parquet': ('trade_date',),
}
_SORT_COLUMNS = {
    'trade_cal.parquet': ('exchange', 'cal_date'),
    'stock_basic.parquet': ('ts_code',),
    'industry_record.parquet': ('ts_code', 'in_date'),
    'dividend.parquet': ('ts_code', 'ann_date', 'end_date'),
    'namechange.parquet': ('ts_code', 'start_date', 'ann_date'),
    'index_weights': ('index_code', 'trade_date', 'con_code'),
}


@dataclass(frozen=True)
class _StagedWrite:
    dataset: str
    target: Path
    temporary: Path
    before_rows: int
    fetched_rows: int
    after_rows: int


def _validate_date(value: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'\d{8}', value):
        raise ValueError(f"更新参数错误: field={field}, actual={value!r}, expected=YYYYMMDD")
    try:
        datetime.strptime(value, '%Y%m%d')
    except ValueError as exc:
        raise ValueError(f"更新参数错误: field={field}, actual={value!r}, expected=有效日期") from exc
    return value


def _validate_inputs(start_date: str, end_date: str, symbols: list[str]) -> tuple[str, str, list[str]]:
    start_date = _validate_date(start_date, 'start_date')
    end_date = _validate_date(end_date, 'end_date')
    if start_date > end_date:
        raise ValueError(f"更新参数错误: start_date={start_date}, end_date={end_date}, expected=start_date<=end_date")
    if not isinstance(symbols, list) or not symbols:
        raise ValueError(f"更新参数错误: field=symbols, actual={symbols!r}, expected=非空list[str]")
    invalid = [symbol for symbol in symbols if not isinstance(symbol, str) or not _STOCK_CODE_PATTERN.fullmatch(symbol)]
    if invalid:
        raise ValueError(f"更新参数错误: field=symbols, invalid={invalid}, expected=000001.SZ格式")
    if len(symbols) != len(set(symbols)):
        raise ValueError(f"更新参数错误: field=symbols, actual={symbols!r}, expected=不得重复")
    return start_date, end_date, symbols.copy()


def _calendar_dates(start_date: str, end_date: str) -> list[str]:
    return pd.date_range(start_date, end_date, freq='D').strftime('%Y%m%d').tolist()


def _year_ranges(start_date: str, end_date: str) -> list[tuple[int, str, str]]:
    ranges = []
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        range_start = max(start_date, f'{year}0101')
        range_end = min(end_date, f'{year}1231')
        ranges.append((year, range_start, range_end))
    return ranges


def _month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    starts = pd.date_range(start_date[:6] + '01', end_date, freq='MS')
    ranges = []
    for month_start in starts:
        month_end = month_start + pd.offsets.MonthEnd(0)
        ranges.append((month_start.strftime('%Y%m%d'), month_end.strftime('%Y%m%d')))
    return ranges


def _month_scope(start_date: str, end_date: str) -> tuple[str, str]:
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date) + pd.offsets.MonthEnd(0)
    return start.strftime('%Y%m%d'), end.strftime('%Y%m%d')


def _reporting_periods(end_date: str) -> list[str]:
    if end_date < _FINANCIAL_HISTORY_START:
        return []
    periods = []
    for year in range(int(_FINANCIAL_HISTORY_START[:4]), int(end_date[:4]) + 1):
        periods.extend(f'{year}{suffix}' for suffix in ('0331', '0630', '0930', '1231'))
    return [period for period in periods if _FINANCIAL_HISTORY_START <= period <= end_date]


def _call_pro(api_name: str, **params) -> pd.DataFrame:
    frame = call_pro_tushare_api(api_name, **params)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Tushare返回类型错误: api={api_name}, actual={type(frame).__name__}, expected=DataFrame")
    return frame.copy()


def _call_ts(api_name: str, **params) -> pd.DataFrame:
    frame = call_ts_tushare_api(api_name, **params)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Tushare返回类型错误: api={api_name}, actual={type(frame).__name__}, expected=DataFrame")
    return frame.copy()


def _filter_symbols(frame: pd.DataFrame, symbols: list[str], api_name: str) -> pd.DataFrame:
    if frame.empty and 'ts_code' not in frame.columns:
        return frame
    if 'ts_code' not in frame.columns:
        raise ValueError(f"Tushare字段缺失: api={api_name}, missing=['ts_code']")
    return frame[frame['ts_code'].isin(symbols)].copy()


def _concat_frames(frames: list[pd.DataFrame], api_name: str) -> pd.DataFrame:
    populated = [frame for frame in frames if not frame.empty]
    if not populated:
        with_schema = [frame for frame in frames if len(frame.columns) > 0]
        return with_schema[0].iloc[0:0].copy() if with_schema else pd.DataFrame()
    columns = list(populated[0].columns)
    for frame in populated[1:]:
        if set(frame.columns) != set(columns):
            raise ValueError(f"Tushare字段漂移: api={api_name}, expected={columns}, actual={list(frame.columns)}")
    return pd.concat([frame.reindex(columns=columns) for frame in populated], ignore_index=True)


def _fetch_by_dates(api_name: str, dates: list[str], symbols: list[str],
                    date_param: str, require_market_data: bool = False) -> pd.DataFrame:
    frames = []
    for date in dates:
        frame = _call_pro(api_name, **{date_param: date})
        if require_market_data and frame.empty:
            raise ValueError(f"全市场接口返回空数据: api={api_name}, date={date}")
        frames.append(_filter_symbols(frame, symbols, api_name))
    return _concat_frames(frames, api_name)


def _fetch_namechange(dates: list[str], symbols: list[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        frame = _call_pro('namechange', start_date=date, end_date=date)
        frames.append(_filter_symbols(frame, symbols, 'namechange'))
    return _concat_frames(frames, 'namechange')


def _fetch_dividend_updates(dates: list[str], symbols: list[str]) -> pd.DataFrame:
    frames = []
    for date in dates:
        for date_param in ('ann_date', 'imp_ann_date'):
            frame = _call_pro('dividend', **{date_param: date})
            frames.append(_filter_symbols(frame, symbols, 'dividend'))
    combined = _concat_frames(frames, 'dividend')
    return combined.drop_duplicates().reset_index(drop=True)


def _fetch_daily_hfq(start_date: str, end_date: str, symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        frame = _call_ts('pro_bar', ts_code=symbol, start_date=start_date, end_date=end_date, adj='hfq', asset='E')
        if not frame.empty and set(frame['ts_code'].unique()) != {symbol}:
            raise ValueError(f"Tushare标的越界: api=pro_bar, requested={symbol}, actual={frame['ts_code'].unique().tolist()}")
        frames.append(frame)
    return _concat_frames(frames, 'pro_bar')


def _fetch_balancesheet_by_actual_date(start_date: str, end_date: str,
                                       symbols: list[str]) -> pd.DataFrame:
    frames = []
    for period in _reporting_periods(end_date):
        frame = _call_pro('balancesheet_vip', period=period)
        frames.append(_filter_symbols(frame, symbols, 'balancesheet_vip'))
    combined = _concat_frames(frames, 'balancesheet_vip')
    if combined.empty:
        return combined
    _validate_frame('balancesheet.parquet', combined)
    actual_dates = _date_values(combined['f_ann_date'], 'balancesheet.parquet', 'f_ann_date')
    return combined.loc[actual_dates.between(start_date, end_date)].copy()


def _fetch_stock_basic() -> pd.DataFrame:
    return _call_pro('stock_basic', list_status='L,D,P', fields=_STOCK_BASIC_FIELDS)


def _fetch_industry_history(symbols: list[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        history = _call_pro('index_member_all', ts_code=symbol, is_new='N')
        current = _call_pro('index_member_all', ts_code=symbol, is_new='Y')
        frames.extend([_filter_symbols(history, [symbol], 'index_member_all'),
                       _filter_symbols(current, [symbol], 'index_member_all')])
    return _concat_frames(frames, 'index_member_all')


def _fetch_index_daily(start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    for code in _INDEX_DAILY_CODES:
        frame = _call_pro('index_daily', ts_code=code, start_date=start_date, end_date=end_date)
        if not frame.empty:
            frame['ts_code'] = frame['ts_code'].astype(str).str.split('.').str[0]
        frames.append(frame)
    return _concat_frames(frames, 'index_daily')


def _fetch_index_weights(start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    result = {}
    for index_code in _INDEX_WEIGHT_CODES:
        frames = []
        for range_start, range_end in _month_ranges(start_date, end_date):
            frames.append(_call_pro('index_weight', index_code=index_code,
                                    start_date=range_start, end_date=range_end))
        combined = _concat_frames(frames, 'index_weight')
        if not combined.empty and set(combined['index_code']) != {index_code}:
            actual = sorted(combined['index_code'].astype(str).unique().tolist())
            raise ValueError(f"Tushare指数越界: api=index_weight, requested={index_code}, actual={actual}")
        result[index_code] = combined
    return result


def _fetch_sw_daily(sw_basic: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if 'category' not in sw_basic.columns or 'ts_code' not in sw_basic.columns:
        raise ValueError("申万基础信息缺少字段: required=['category', 'ts_code']")
    codes = sw_basic.loc[sw_basic['category'].eq('一级行业指数'), 'ts_code'].drop_duplicates().tolist()
    if not codes:
        raise ValueError("申万一级行业代码为空: category=一级行业指数")
    frames = [_call_pro('sw_daily', ts_code=code, start_date=start_date, end_date=end_date) for code in codes]
    return _concat_frames(frames, 'sw_daily')


def _date_values(series: pd.Series, dataset: str, column: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors='coerce')
    else:
        normalized = series.astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
        compact = normalized.str.replace('-', '', regex=False).str.slice(0, 8)
        parsed = pd.to_datetime(compact, format='%Y%m%d', errors='coerce')
    invalid = series.notna() & parsed.isna()
    if invalid.any():
        sample = series[invalid].astype(str).head(3).tolist()
        raise ValueError(f"数据日期非法: dataset={dataset}, column={column}, sample={sample}")
    return parsed.dt.strftime('%Y%m%d')


def _normalize_frame(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in _DATETIME_COLUMNS.get(dataset, ()):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors='raise')
    if dataset == 'index_daily.parquet' and 'ts_code' in frame.columns:
        frame['ts_code'] = frame['ts_code'].astype(str).str.split('.').str[0]
    return frame


def _validate_frame(dataset: str, frame: pd.DataFrame) -> None:
    if frame.empty and len(frame.columns) == 0:
        return
    required = _REQUIRED_COLUMNS[dataset]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"数据字段缺失: dataset={dataset}, missing={missing}, actual={list(frame.columns)}")
    if frame.duplicated(keep=False).any():
        raise ValueError(f"数据存在完全重复行: dataset={dataset}, rows={int(frame.duplicated(keep=False).sum())}")
    keys = _UNIQUE_KEYS.get(dataset)
    if keys and frame.duplicated(list(keys), keep=False).any():
        count = int(frame.duplicated(list(keys), keep=False).sum())
        raise ValueError(f"数据主键重复: dataset={dataset}, keys={list(keys)}, rows={count}")


def _align_schema(dataset: str, existing: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    fetched = _normalize_frame(dataset, fetched)
    if len(existing.columns) == 0:
        return fetched
    if fetched.empty and len(fetched.columns) == 0:
        return existing.iloc[0:0].copy()
    if set(existing.columns) != set(fetched.columns):
        raise ValueError(
            f"数据结构变化: dataset={dataset}, local={list(existing.columns)}, api={list(fetched.columns)}"
        )
    fetched = fetched.reindex(columns=existing.columns)
    for column in existing.columns:
        if pd.api.types.is_datetime64_any_dtype(existing[column]) and not pd.api.types.is_datetime64_any_dtype(fetched[column]):
            fetched[column] = pd.to_datetime(fetched[column], errors='raise')
    return fetched


def _sort_frame(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(_SORT_COLUMNS.get(dataset, ()))
    if not columns:
        columns = [column for column in ('ts_code', 'trade_date', 'ann_date', 'end_date') if column in frame.columns]
    return frame.sort_values(columns, kind='stable', na_position='last').reset_index(drop=True) if columns else frame.reset_index(drop=True)


def _combine_range(dataset: str, existing: pd.DataFrame, fetched: pd.DataFrame,
                   date_column: str, start_date: str, end_date: str,
                   symbols: list[str] | None) -> pd.DataFrame:
    if fetched.empty and len(fetched.columns) == 0:
        return existing.copy()
    fetched = _align_schema(dataset, existing, fetched)
    if date_column not in existing.columns and not existing.empty:
        raise ValueError(f"本地字段缺失: dataset={dataset}, missing={date_column}")
    if not fetched.empty:
        if fetched[date_column].isna().any():
            raise ValueError(f"接口数据缺少可用日期: dataset={dataset}, column={date_column}")
        dates = _date_values(fetched[date_column], dataset, date_column)
        if ((dates < start_date) | (dates > end_date)).any():
            raise ValueError(f"接口数据越界: dataset={dataset}, expected={start_date}..{end_date}")
    if existing.empty:
        return _sort_frame(dataset, fetched)
    in_range = _date_values(existing[date_column], dataset, date_column).between(start_date, end_date)
    if symbols is not None:
        in_range &= existing['ts_code'].isin(symbols)
    combined = pd.concat([existing.loc[~in_range], fetched], ignore_index=True)
    return _sort_frame(dataset, combined)


def _combine_symbol_snapshot(dataset: str, existing: pd.DataFrame,
                             fetched: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    fetched = _align_schema(dataset, existing, fetched)
    if not fetched.empty:
        unexpected = sorted(set(fetched['ts_code']) - set(symbols))
        if unexpected:
            raise ValueError(f"接口数据越界: dataset={dataset}, unexpected_symbols={unexpected[:5]}")
    if existing.empty:
        return _sort_frame(dataset, fetched)
    combined = pd.concat([existing.loc[~existing['ts_code'].isin(symbols)], fetched], ignore_index=True)
    return _sort_frame(dataset, combined)


def _dividend_key_index(frame: pd.DataFrame) -> pd.MultiIndex:
    keys = frame.loc[:, _DIVIDEND_VERSION_KEY].astype('string').fillna('<NULL>')
    return pd.MultiIndex.from_frame(keys)


def _combine_dividend_versions(existing: pd.DataFrame,
                               fetched: pd.DataFrame) -> pd.DataFrame:
    if fetched.empty and len(fetched.columns) == 0:
        return existing.copy()
    fetched = _align_schema('dividend.parquet', existing, fetched)
    fetched_keys = _dividend_key_index(fetched)
    if fetched_keys.duplicated(keep=False).any():
        count = int(fetched_keys.duplicated(keep=False).sum())
        raise ValueError(f"分红版本键重复: keys={list(_DIVIDEND_VERSION_KEY)}, rows={count}")
    if existing.empty:
        return _sort_frame('dividend.parquet', fetched)
    keep = ~_dividend_key_index(existing).isin(fetched_keys)
    combined = pd.concat([existing.loc[keep], fetched], ignore_index=True)
    return _sort_frame('dividend.parquet', combined)


def _stage_frame(dataset: str, target: Path, frame: pd.DataFrame,
                 before_rows: int, fetched_rows: int) -> _StagedWrite | None:
    _validate_frame(dataset, frame)
    if not target.exists() and frame.empty and len(frame.columns) == 0:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f'.{target.name}.{uuid4().hex}.tmp')
    try:
        frame.to_parquet(temporary, index=False)
        parquet_file = pq.ParquetFile(temporary)
        if parquet_file.metadata.num_rows != len(frame):
            raise ValueError(f"临时文件行数错误: dataset={dataset}, expected={len(frame)}, actual={parquet_file.metadata.num_rows}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _StagedWrite(dataset, target, temporary, before_rows, fetched_rows, len(frame))


def _read_existing(target: Path) -> pd.DataFrame:
    return pd.read_parquet(target) if target.exists() else pd.DataFrame()


def _stage_snapshot(stages: list[_StagedWrite], dataset: str,
                    target: Path, fetched: pd.DataFrame) -> None:
    if fetched.empty:
        raise ValueError(f"快照接口返回空数据: dataset={dataset}")
    existing = _read_existing(target)
    fetched = _align_schema(dataset, existing, fetched)
    staged = _stage_frame(dataset, target, _sort_frame(dataset, fetched), len(existing), len(fetched))
    if staged is not None:
        stages.append(staged)


def _stage_range(stages: list[_StagedWrite], dataset: str, target: Path,
                 fetched: pd.DataFrame, date_column: str, start_date: str,
                 end_date: str, symbols: list[str] | None) -> None:
    existing = _read_existing(target)
    combined = _combine_range(dataset, existing, fetched, date_column, start_date, end_date, symbols)
    staged = _stage_frame(dataset, target, combined, len(existing), len(fetched))
    if staged is not None:
        stages.append(staged)


def _stage_symbol_snapshot(stages: list[_StagedWrite], dataset: str,
                           target: Path, fetched: pd.DataFrame,
                           symbols: list[str]) -> None:
    existing = _read_existing(target)
    combined = _combine_symbol_snapshot(dataset, existing, fetched, symbols)
    staged = _stage_frame(dataset, target, combined, len(existing), len(fetched))
    if staged is not None:
        stages.append(staged)


def _stage_dividend(stages: list[_StagedWrite], target: Path,
                    fetched: pd.DataFrame) -> None:
    existing = _read_existing(target)
    combined = _combine_dividend_versions(existing, fetched)
    staged = _stage_frame('dividend.parquet', target, combined, len(existing), len(fetched))
    if staged is not None:
        stages.append(staged)


def _subset_range(frame: pd.DataFrame, dataset: str, date_column: str,
                  start_date: str, end_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    dates = _date_values(frame[date_column], dataset, date_column)
    return frame.loc[dates.between(start_date, end_date)].copy()


def _stage_partitioned(stages: list[_StagedWrite], dataset: str,
                       fetched: pd.DataFrame, start_date: str,
                       end_date: str, symbols: list[str]) -> None:
    root = get_market_data_path(dataset, MARKET_DATA_ROOT)
    for year, range_start, range_end in _year_ranges(start_date, end_date):
        target = root / f'year={year}' / 'data.parquet'
        year_frame = _subset_range(fetched, dataset, 'trade_date', range_start, range_end)
        _stage_range(stages, dataset, target, year_frame, 'trade_date', range_start, range_end, symbols)


def _validate_known_symbols(stock_basic: pd.DataFrame, symbols: list[str]) -> None:
    _validate_frame('stock_basic.parquet', stock_basic)
    unknown = sorted(set(symbols) - set(stock_basic['ts_code']))
    if unknown:
        raise ValueError(f"更新标的不在stock_basic中: symbols={unknown}")


def _open_dates(trade_cal: pd.DataFrame, start_date: str, end_date: str) -> list[str]:
    _validate_frame('trade_cal.parquet', trade_cal)
    dates = _date_values(trade_cal['cal_date'], 'trade_cal.parquet', 'cal_date')
    mask = dates.between(start_date, end_date) & trade_cal['is_open'].astype(int).eq(1)
    return sorted(dates[mask].drop_duplicates().tolist())


def _validate_core_daily(frames: dict[str, pd.DataFrame]) -> None:
    key_sets = {}
    for dataset in ('daily', 'daily_basic', 'daily_hfq'):
        _validate_frame(dataset, frames[dataset])
        frame = frames[dataset]
        if frame.empty:
            key_sets[dataset] = set()
            continue
        dates = _date_values(frame['trade_date'], dataset, 'trade_date')
        key_sets[dataset] = set(zip(frame['ts_code'].astype(str), dates))
    if key_sets['daily'] != key_sets['daily_hfq']:
        counts = {dataset: len(key_sets[dataset]) for dataset in ('daily', 'daily_hfq')}
        raise ValueError(f"价格与后复权数据覆盖不一致: key_counts={counts}")
    unexpected_basic = key_sets['daily_basic'] - key_sets['daily']
    if unexpected_basic:
        raise ValueError(f"daily_basic出现无行情主键: sample={sorted(unexpected_basic)[:3]}")


def _prepare_snapshots(stages: list[_StagedWrite], start_date: str,
                       end_date: str, symbols: list[str]) -> tuple[list[str], pd.DataFrame]:
    trade_cal = _call_pro('trade_cal', start_date=start_date, end_date=end_date)
    stock_basic = _fetch_stock_basic()
    sw_basic = _call_pro('index_basic', market='SW')
    _validate_known_symbols(stock_basic, symbols)
    _stage_range(stages, 'trade_cal.parquet', get_market_data_path('trade_cal.parquet', MARKET_DATA_ROOT),
                 trade_cal, 'cal_date', start_date, end_date, None)
    _stage_snapshot(stages, 'stock_basic.parquet', get_market_data_path('stock_basic.parquet', MARKET_DATA_ROOT), stock_basic)
    _stage_snapshot(stages, 'sw_basic_info.parquet', get_market_data_path('sw_basic_info.parquet', MARKET_DATA_ROOT), sw_basic)
    return _open_dates(trade_cal, start_date, end_date), sw_basic


def _prepare_daily_data(stages: list[_StagedWrite], start_date: str, end_date: str,
                        symbols: list[str], open_dates: list[str]) -> None:
    for _, range_start, range_end in _year_ranges(start_date, end_date):
        dates = [date for date in open_dates if range_start <= date <= range_end]
        required_market = {'daily', 'adj_factor', 'daily_basic', 'stk_limit'}
        frames = {dataset: _fetch_by_dates(api, dates, symbols, 'trade_date', dataset in required_market)
                  for dataset, api in _DAILY_PRO_APIS.items()}
        traded_symbols = sorted(frames['daily']['ts_code'].unique().tolist()) if not frames['daily'].empty else []
        frames['daily_hfq'] = (_fetch_daily_hfq(range_start, range_end, traded_symbols)
                               if traded_symbols else frames['daily'].iloc[0:0].copy())
        _validate_core_daily(frames)
        for dataset, frame in frames.items():
            _validate_frame(dataset, frame)
            _stage_partitioned(stages, dataset, frame, range_start, range_end, symbols)


def _prepare_stock_reference(stages: list[_StagedWrite], symbols: list[str]) -> None:
    industry = _fetch_industry_history(symbols)
    _validate_frame('industry_record.parquet', industry)
    target = get_market_data_path('industry_record.parquet', MARKET_DATA_ROOT)
    _stage_symbol_snapshot(stages, 'industry_record.parquet', target, industry, symbols)


def _prepare_stock_events(stages: list[_StagedWrite], start_date: str,
                          end_date: str, symbols: list[str]) -> None:
    dates = _calendar_dates(start_date, end_date)
    suspend = _fetch_by_dates('suspend_d', dates, symbols, 'trade_date')
    dividend = _fetch_dividend_updates(dates, symbols)
    namechange = _fetch_namechange(dates, symbols)
    for dataset, frame, date_column in (('suspend_d.parquet', suspend, 'trade_date'),
                                        ('namechange.parquet', namechange, 'ann_date')):
        _validate_frame(dataset, frame)
        target = get_market_data_path(dataset, MARKET_DATA_ROOT)
        _stage_range(stages, dataset, target, frame, date_column, start_date, end_date, symbols)
    _validate_frame('dividend.parquet', dividend)
    target = get_market_data_path('dividend.parquet', MARKET_DATA_ROOT)
    _stage_dividend(stages, target, dividend)


def _prepare_financials(stages: list[_StagedWrite], start_date: str,
                        end_date: str, symbols: list[str]) -> None:
    dates = _calendar_dates(start_date, end_date)
    balancesheet = _fetch_balancesheet_by_actual_date(start_date, end_date, symbols)
    _validate_frame('balancesheet.parquet', balancesheet)
    target = get_market_data_path('balancesheet.parquet', MARKET_DATA_ROOT)
    _stage_range(stages, 'balancesheet.parquet', target, balancesheet,
                 'f_ann_date', start_date, end_date, symbols)
    for dataset, (api_name, date_column) in _FINANCIAL_APIS.items():
        frame = _fetch_by_dates(api_name, dates, symbols, date_column)
        _validate_frame(dataset, frame)
        target = get_market_data_path(dataset, MARKET_DATA_ROOT)
        _stage_range(stages, dataset, target, frame, date_column, start_date, end_date, symbols)


def _prepare_index_weights(stages: list[_StagedWrite], start_date: str,
                           end_date: str, frames: dict[str, pd.DataFrame]) -> None:
    root = get_market_data_path('index_weights', MARKET_DATA_ROOT)
    for index_code, frame in frames.items():
        _validate_frame('index_weights', frame)
        index_root = root / index_code.replace('.', '_')
        for year, range_start, range_end in _year_ranges(start_date, end_date):
            year_frame = _subset_range(frame, 'index_weights', 'trade_date', range_start, range_end)
            target = index_root / f'year={year}' / 'data.parquet'
            _stage_range(stages, 'index_weights', target, year_frame, 'trade_date',
                         range_start, range_end, None)


def _prepare_indices(stages: list[_StagedWrite], start_date: str,
                     end_date: str, sw_basic: pd.DataFrame) -> None:
    index_daily = _fetch_index_daily(start_date, end_date)
    sw_daily = _fetch_sw_daily(sw_basic, start_date, end_date)
    index_weights = _fetch_index_weights(start_date, end_date)
    _validate_frame('index_daily.parquet', index_daily)
    _validate_frame('sw_daily.parquet', sw_daily)
    _stage_range(stages, 'index_daily.parquet', get_market_data_path('index_daily.parquet', MARKET_DATA_ROOT),
                 index_daily, 'trade_date', start_date, end_date, None)
    _stage_range(stages, 'sw_daily.parquet', get_market_data_path('sw_daily.parquet', MARKET_DATA_ROOT),
                 sw_daily, 'trade_date', start_date, end_date, None)
    weight_start, weight_end = _month_scope(start_date, end_date)
    _prepare_index_weights(stages, weight_start, weight_end, index_weights)


def _cleanup_staged(stages: list[_StagedWrite]) -> None:
    for staged in stages:
        staged.temporary.unlink(missing_ok=True)


def _commit_one(staged: _StagedWrite) -> Path | None:
    backup = None
    if staged.target.exists():
        backup = staged.target.with_name(f'.{staged.target.name}.{uuid4().hex}.bak')
        os.replace(staged.target, backup)
    try:
        os.replace(staged.temporary, staged.target)
    except Exception:
        if backup is not None:
            os.replace(backup, staged.target)
        raise
    return backup


def _rollback_committed(committed: list[tuple[_StagedWrite, Path | None]]) -> None:
    for staged, backup in reversed(committed):
        if backup is None:
            staged.target.unlink(missing_ok=True)
        else:
            os.replace(backup, staged.target)


def _empty_result() -> dict[str, dict[str, int]]:
    return {dataset: {'files': 0, 'before_rows': 0, 'fetched_rows': 0,
                      'removed_rows': 0, 'after_rows': 0}
            for dataset in ALL_DATASETS}


def _build_result(stages: list[_StagedWrite]) -> dict[str, dict[str, int]]:
    result = _empty_result()
    for staged in stages:
        stats = result[staged.dataset]
        stats['files'] += 1
        stats['before_rows'] += staged.before_rows
        stats['fetched_rows'] += staged.fetched_rows
        stats['removed_rows'] += staged.before_rows + staged.fetched_rows - staged.after_rows
        stats['after_rows'] += staged.after_rows
    return result


def _commit_staged(stages: list[_StagedWrite]) -> dict[str, dict[str, int]]:
    targets = [staged.target.resolve() for staged in stages]
    if len(targets) != len(set(targets)):
        raise ValueError("内部错误: 同一目标文件被重复暂存")
    committed: list[tuple[_StagedWrite, Path | None]] = []
    try:
        for staged in stages:
            committed.append((staged, _commit_one(staged)))
    except Exception:
        _rollback_committed(committed)
        _cleanup_staged(stages)
        raise
    for _, backup in committed:
        if backup is not None:
            backup.unlink()
    return _build_result(stages)


def upsert_market_data(start_date: str, end_date: str, symbols: list[str]) -> dict[str, dict[str, int]]:
    """抓取并合并全部已登记市场数据；任一准备步骤失败都不替换正式文件。"""
    start_date, end_date, symbols = _validate_inputs(start_date, end_date, symbols)
    if not MARKET_DATA_ROOT.is_dir():
        raise FileNotFoundError(f"市场数据根目录不存在: {MARKET_DATA_ROOT}")
    stages: list[_StagedWrite] = []
    try:
        open_dates, sw_basic = _prepare_snapshots(stages, start_date, end_date, symbols)
        _prepare_daily_data(stages, start_date, end_date, symbols, open_dates)
        _prepare_stock_reference(stages, symbols)
        _prepare_stock_events(stages, start_date, end_date, symbols)
        _prepare_financials(stages, start_date, end_date, symbols)
        _prepare_indices(stages, start_date, end_date, sw_basic)
        return _commit_staged(stages)
    except Exception:
        _cleanup_staged(stages)
        raise
