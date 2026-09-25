"""按需更新股票数据。返回拉取行数；失败直接停止，不回滚已保存的数据。"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from quant_lib.config.constant_config import MARKET_DATA_ROOT, get_market_data_path
from quant_lib.config.logger_config import setup_logger
from quant_lib.tushare.api_wrapper import (
    RowLimitExceeded, call_pro_tushare_api,
)


logger = setup_logger(__name__)


_STOCK_BASIC_FIELDS = (
    'ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,'
    'curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type'
)
_REPORT_KEY = ['ts_code', 'end_date', 'ann_date', 'f_ann_date', 'report_type','update_flag']
_HM_DETAIL_FIELDS = (
    'trade_date,ts_code,ts_name,buy_amount,sell_amount,net_amount,hm_name,hm_orgs,tag'
)
_KEYS = {
    'trade_cal.parquet': ['exchange', 'cal_date'],
    **{name: ['ts_code', 'trade_date'] for name in ('daily', 'adj_factor', 'daily_basic', 'stk_limit')},
    'balancesheet.parquet': _REPORT_KEY,
    'cashflow.parquet': _REPORT_KEY,
    'income.parquet': _REPORT_KEY,
    'fina_indicator.parquet': ['ts_code', 'end_date', 'ann_date'],
    'namechange.parquet': ['ts_code', 'start_date', 'name'],
}


def _path(dataset: str) -> Path:
    return get_market_data_path(dataset, MARKET_DATA_ROOT)


def _pro(api: str, **params) -> pd.DataFrame:
    frame = call_pro_tushare_api(api, max_retries=3, **params)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f'{api}: 接口必须返回 DataFrame')
    logger.info('接口:%s: params=%s  rows=%s', api, params, len(frame))
    return frame


def _concat(frames) -> pd.DataFrame:
    """按行拼接接口结果，避免空批次或局部全空列触发类型推断警告。"""
    # frames 可能是生成器；转成列表后才能多次遍历。
    parts = list(frames)
    if any(not isinstance(part, pd.DataFrame) for part in parts):
        raise TypeError('接口必须返回 DataFrame')
    # 连一个批次都没有时，返回无行、无列的空表。
    if not parts:
        return pd.DataFrame()

    # iloc[:0] 只取表头。合并这些表头，记录所有列及首次出现的顺序，
    # 包括那些只在空批次里出现的列，最后用来恢复完整字段。
    columns = pd.concat([part.iloc[:0] for part in parts], ignore_index=True).columns

    # notna().any() 判断每列是否至少有一个非空值；
    # set().union(...) 汇总出“在任意批次中有值”的列名集合。
    populated = set().union(*(set(part.columns[part.notna().any()]) for part in parts))

    # 零行批次不参与数据拼接；有行但整行全空的批次仍然保留。
    nonempty = [part for part in parts if len(part)]
    if not nonempty:
        # 所有批次都是零行时，仍保留它们的表头和空列类型。
        return pd.concat(parts, ignore_index=True)

    # 某列在当前批次全空、在其他批次有值时，临时移除当前批次的这列，
    # 避免它干扰 pandas 的类型推断。例如 [None] 与 [1.5] 拼接时，
    # 让有值的批次决定类型；concat 会给缺列批次的对应行补上缺失值。
    # 所有批次中都全空的列不在 populated 中，因此不会被这里删除。
    # drop 返回新表，不修改原始批次，也不删除任何行。
    cleaned = [part.drop(columns=[column for column in part.columns
                                 if column in populated and part[column].isna().all()])
               for part in nonempty]

    # 按行拼接并重新编号；恢复最初的列顺序，只在零行批次出现的列补为缺失值。
    return pd.concat(cleaned, ignore_index=True).reindex(columns=columns)


def _symbols() -> list[str]:
    # 逐股接口使用本地全市场名单；需要刷新时先调用 update_stock_basic。
    basic = pd.read_parquet(_path('stock_basic.parquet'), columns=['ts_code'])
    codes = basic['ts_code']
    valid = codes.map(lambda code: isinstance(code, str) and bool(code.strip()))
    if codes.empty or not valid.all():
        raise ValueError('stock_basic: 股票代码为空或非法，请先更新股票名单')
    return codes.drop_duplicates().tolist()


def _incremental_start(dataset: str, column: str, initial_date: str, end_date: str) -> str:
    # 初始日期仅用于本地无数据；已有数据从本表最大日期的下一天开始。
    for date in (initial_date, end_date):
        if not isinstance(date, str) or len(date) != 8 or not date.isdigit():
            raise ValueError(f'日期必须为 YYYYMMDD，实际为 {date!r}')
        datetime.strptime(date, '%Y%m%d')
    if initial_date > end_date:
        raise ValueError('initial_date 必须不晚于 end_date')

    path = _path(dataset)
    if path.is_dir():
        files = sorted(path.glob('year=*/data.parquet'))
    else:
        files = [path] if path.exists() else []
    max_date = None
    for file in files:
        values = pd.read_parquet(file, columns=[column])[column]
        if values.empty:
            continue
        # 保留现有字符串和 datetime 两种存储格式；非法或缺失日期不能作为增量依据。
        if pd.api.types.is_datetime64_any_dtype(values):
            dates = pd.to_datetime(values)
        else:
            dates = pd.to_datetime(values.astype(str), format='%Y%m%d')
        if dates.isna().any():
            raise ValueError(f'{file}: {column} 存在空日期')
        current = dates.max().normalize()
        max_date = current if max_date is None else max(max_date, current)
    return initial_date if max_date is None else (max_date + pd.Timedelta(days=1)).strftime('%Y%m%d')


def _save(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _merge_save(dataset: str, path: Path, new: pd.DataFrame) -> int:
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    # 同键保留新记录，未返回的历史记录继续保留。
    merged = _concat([old, new]).drop_duplicates(
        subset=_KEYS[dataset], keep='last',
    )
    _save(path, merged)
    return len(new)


def _validate_trade_cal(frame: pd.DataFrame, start: str, end: str) -> None:
    required = {'exchange', 'cal_date', 'is_open'}
    if not required.issubset(frame.columns):
        raise ValueError(f'trade_cal: 缺少字段 {sorted(required - set(frame.columns))}')
    for date in (start, end):
        if not isinstance(date, str) or len(date) != 8 or not date.isdigit():
            raise ValueError('trade_cal: 日期必须为 YYYYMMDD')
        datetime.strptime(date, '%Y%m%d')
    if start > end:
        raise ValueError('trade_cal: 起点必须不晚于截止日')
    if (frame[list(required)].isna().any().any()
            or not frame['exchange'].eq('SSE').all()
            or not frame['is_open'].isin([0, 1]).all()
            or frame['cal_date'].duplicated().any()):
        raise ValueError('trade_cal: 交易所、开市标记或日期非法、缺失或重复')
    dates = frame['cal_date']
    if not dates.map(lambda d: isinstance(d, str) and len(d) == 8 and d.isdigit()).all():
        raise ValueError('trade_cal: cal_date 必须为 YYYYMMDD 字符串')
    pd.to_datetime(dates, format='%Y%m%d', errors='raise')
    expected = set(pd.date_range(start, end).strftime('%Y%m%d'))
    if expected - set(dates):
        raise ValueError(f'trade_cal: 未完整覆盖 {start} 至 {end}，请先更新本地交易日历')


def update_trade_cal(initial_date: str, end_date: str) -> int:
    """获取 SSE 完整日历（含休市日），合并保存至 shared/trade_cal.parquet。"""
    # 按年请求，避免多年日历超过接口单次返回上限。
    datetime.strptime(initial_date, '%Y%m%d')
    datetime.strptime(end_date, '%Y%m%d')
    if initial_date > end_date:
        raise ValueError('initial_date 必须不晚于 end_date')
    frames = []
    for year in range(int(initial_date[:4]), int(end_date[:4]) + 1):
        start, end = max(initial_date, f'{year}0101'), min(end_date, f'{year}1231')
        frame = _pro('trade_cal', exchange='SSE', start_date=start, end_date=end,
                     fields='exchange,cal_date,is_open,pretrade_date')
        _validate_trade_cal(frame, start, end)
        frames.append(frame)
    return _merge_save('trade_cal.parquet', _path('trade_cal.parquet'), _concat(frames))


def read_trade_dates(start_date: str, end_date: str) -> list[str]:
    """只读取本地 SSE 日历，缺失或不完整时直接报错。"""
    frame = pd.read_parquet(_path('trade_cal.parquet'))
    frame = frame.loc[frame['exchange'] == 'SSE']
    _validate_trade_cal(frame, start_date, end_date)
    return frame.loc[
        frame['is_open'].eq(1) & frame['cal_date'].between(start_date, end_date),
        'cal_date',
    ].sort_values().tolist()

#自切割
def _fetch_by_range(fetch, start: str, end: str) -> pd.DataFrame:
    try:
        return fetch(start, end)
    except RowLimitExceeded:
        if start == end:
            raise
        first = datetime.strptime(start, '%Y%m%d')
        last = datetime.strptime(end, '%Y%m%d')
        mid = first + timedelta(days=(last - first).days // 2)
        logger.warning('【触发行数上限，自动分割】[%s, %s] -> [%s, %s] + [%s, %s]', start, end, start, mid.strftime('%Y%m%d'), (mid + timedelta(days=1)).strftime('%Y%m%d'), end)
        left = _fetch_by_range(fetch, start, mid.strftime('%Y%m%d'))
        right = _fetch_by_range(
            fetch, (mid + timedelta(days=1)).strftime('%Y%m%d'), end,
        )
        return _concat([left, right])


def _update_daily(dataset: str, initial_date: str, end_date: str) -> int:
    start = _incremental_start(dataset, 'trade_date', initial_date, end_date)
    if start > end_date:
        return 0

    new = _fetch_by_range(
        lambda first, last: _pro(dataset, start_date=first, end_date=last),
        start, end_date,
    )
    return _save_daily_by_year(dataset, new)


def _save_daily_by_year(dataset: str, new: pd.DataFrame) -> int:
    if new.empty:
        return 0

    rows = 0
    years = pd.to_datetime(new['trade_date'], format='%Y%m%d').dt.year
    for year, part in new.groupby(years):
        rows += _merge_save(dataset, _path(dataset) / f'year={year}' / 'data.parquet',
                            part)
    return rows


def update_daily(initial_date: str, end_date: str) -> int:
    return _update_daily('daily', initial_date, end_date)


def update_adj_factor(initial_date: str, end_date: str) -> int:
    """按交易日下载原始复权因子，按年增量保存，不生成复权行情。"""
    start = _incremental_start('adj_factor', 'trade_date', initial_date, end_date)
    if start > end_date:
        return 0
    frames = []
    for date in read_trade_dates(start, end_date):
        frame = _pro('adj_factor', trade_date=date)[['ts_code', 'trade_date', 'adj_factor']]
        if (frame.empty or not frame['trade_date'].eq(date).all()
                or frame.duplicated(['ts_code', 'trade_date']).any()):
            raise ValueError(f'adj_factor: {date} 返回为空、日期不符或主键重复')
        frames.append(frame)
    return _save_daily_by_year('adj_factor', _concat(frames))


def update_daily_basic(initial_date: str, end_date: str) -> int:
    return _update_daily('daily_basic', initial_date, end_date)


def update_stk_limit(initial_date: str, end_date: str) -> int:
    return _update_daily('stk_limit', initial_date, end_date)


def update_hm_detail(initial_date: str, end_date: str) -> int:
    """按日增量下载游资明细，按年保存；返回拉取行数，不自动补历史修订。"""
    start = _incremental_start('hm_detail', 'trade_date', initial_date, end_date)
    if start > end_date:
        logger.info(f'hm_detail: 无需更新，已有数据覆盖至 {end_date}')
        return 0

    trade_dates = read_trade_dates(start, end_date)
    total_days = len(trade_dates)
    completed_days = 0
    logger.info(f'hm_detail: 开始拉取 {start} 至 {end_date}，共 {total_days} 天')
    columns = _HM_DETAIL_FIELDS.split(',')
    rows = 0
    for year in range(int(start[:4]), int(end_date[:4]) + 1):
        year_start = max(start, f'{year}0101')
        year_end = min(end_date, f'{year}1231')
        frames = []
        for date in (date for date in trade_dates if year_start <= date <= year_end):
            logger.info(f'hm_detail: [{completed_days + 1}/{total_days}] 正在拉取 {date}')
            frame = _pro('hm_detail', trade_date=date, fields=_HM_DETAIL_FIELDS)
            # missing = set(columns) - set(frame.columns)
            # if missing:
            #     raise ValueError(f'hm_detail {date}: 缺少字段 {sorted(missing)}')

            frames.append(frame)
            completed_days += 1
            logger.info(f'hm_detail: [{completed_days}/{total_days}] {date} 返回 {len(frame)} 行')
        new = _concat(frames)
        path = _path('hm_detail') / f'year={year}' / 'data.parquet'
        old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=columns)
        # 文档未声明唯一键；按完整请求范围替换，保留所有原始明细。
        old = old.loc[~old['trade_date'].between(year_start, year_end)]
        merged = new if old.empty else old if new.empty else pd.concat([old, new], ignore_index=True)
        _save(path, merged)
        rows += len(new)
        logger.info(f'hm_detail: {year} 年已保存至 {path}，本年拉取 {len(new)} 行，累计 {rows} 行')
    logger.info(f'hm_detail: 更新完成，共处理 {completed_days} 天，拉取 {rows} 行')
    return rows


def update_suspend(initial_date: str, end_date: str) -> int:
    dataset = 'suspend_d.parquet'
    start = _incremental_start(dataset, 'trade_date', initial_date, end_date)
    if start > end_date:
        return 0
    dates = read_trade_dates(start, end_date)
    new = _concat(_pro('suspend_d', trade_date=date) for date in dates)
    path = _path(dataset)
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    # 合法空结果也替换请求范围；没有新事件时，最大事件日期不会前进。
    if not old.empty:
        dates = pd.to_datetime(old['trade_date'], format='%Y%m%d')
        old = old.loc[~dates.between(pd.Timestamp(start), pd.Timestamp(end_date))]
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates()
    _save(path, merged)
    return len(new)


def _update_financial(dataset: str, initial_date: str, end_date: str) -> int:
    # 按公告日增量，不保证捕获公告日未变化的历史修订。
    start = _incremental_start(dataset, 'ann_date', initial_date, end_date)
    if start > end_date:
        return 0
    dates = pd.date_range(start, end_date).strftime('%Y%m%d')
    api = dataset.removesuffix('.parquet') + '_vip'
    new = _concat(_pro(api, ann_date=date) for date in dates)
    if new.empty:
        return 0
    if dataset == 'fina_indicator.parquet':
        new['end_date'] = pd.to_datetime(new['end_date'], format='%Y%m%d')
    return _merge_save(dataset, _path(dataset), new)


def update_balancesheet(initial_date: str, end_date: str) -> int:
    return _update_financial('balancesheet.parquet', initial_date, end_date)


def update_cashflow(initial_date: str, end_date: str) -> int:
    return _update_financial('cashflow.parquet', initial_date, end_date)


def update_income(initial_date: str, end_date: str) -> int:
    return _update_financial('income.parquet', initial_date, end_date)


def update_fina_indicator(initial_date: str, end_date: str) -> int:
    return _update_financial('fina_indicator.parquet', initial_date, end_date)


def update_stock_basic() -> int:
    new = _concat(_pro('stock_basic', list_status=status, fields=_STOCK_BASIC_FIELDS)
                  for status in ('L', 'D', 'P'))
    if new.empty:
        raise ValueError('stock_basic: 全市场股票名单返回空，停止更新')
    _save(_path('stock_basic.parquet'), new)
    return len(new)


def update_index_classify() -> int:
    """全量刷新申万2021版三级行业分类，全部校验成功后保存，返回拉取行数。"""
    fields = 'index_code,industry_name,parent_code,level,industry_code,is_pub,src'
    new = _pro('index_classify', src='SW2021', fields=fields)
    missing = set(fields.split(',')) - set(new.columns)
    if missing:
        raise ValueError(f'index_classify: 缺少字段 {sorted(missing)}')
    if (new.empty or new[fields.split(',')].isna().any().any()
            or not new['level'].isin(['L1', 'L2', 'L3']).all()
            or not new['src'].eq('SW2021').all()
            or not new['index_code'].map(
                lambda code: isinstance(code, str) and bool(code.strip())).all()):
        raise ValueError('index_classify: 返回为空、字段缺失或分类数据非法')
    if new['index_code'].duplicated().any():
        raise ValueError('index_classify: 指数代码重复')
    _save(_path('index_classify.parquet'), new)
    return len(new)


def update_industry_record() -> int:
    """按本地 SW2021 三级行业刷新历史和当前成员，返回拉取行数。"""
    classify = pd.read_parquet(
        _path('index_classify.parquet'), columns=['index_code', 'level', 'src'],
    )
    if classify.empty or not classify['src'].eq('SW2021').fillna(False).all():
        raise ValueError('index_classify: 分类为空或不是 SW2021，请先更新行业分类')
    codes = classify.loc[classify['level'] == 'L3', 'index_code']
    if (codes.empty or codes.duplicated().any()
            or not codes.map(lambda code: isinstance(code, str) and bool(code.strip())).all()):
        raise ValueError('index_classify: L3 行业代码为空、非法或重复')
    required = {'ts_code', 'l1_code', 'l2_code', 'l3_code', 'in_date', 'out_date'}
    frames = []
    for code in codes:
        for state in ('N', 'Y'):
            frame = _pro('index_member_all', l3_code=code, is_new=state)
            if not frame.empty:
                if not frame['l3_code'].eq(code).fillna(False).all():
                    raise ValueError(f'industry_record: 返回的 L3 行业代码与请求 {code} 不一致')
            frames.append(frame)
    new = _concat(frames)
    if new.empty:
        raise ValueError('industry_record: 全市场行业历史返回空，停止更新')
    for column in ('in_date', 'out_date'):
        new[column] = pd.to_datetime(new[column], format='%Y%m%d', errors='raise')
    _save(_path('industry_record.parquet'), new.drop_duplicates())
    return len(new)


def update_namechange() -> int:
    # 逐股获取完整历史，更新旧名称的结束日。
    new = _concat(_pro('namechange', ts_code=code) for code in _symbols())
    if new.empty:
        return 0
    return _merge_save('namechange.parquet', _path('namechange.parquet'), new)


def update_dividend() -> int:
    # 不做日期增量：逐股拉取完整分红历史，全部成功后再覆盖保存。
    frames = []
    for code in _symbols():
        frame = _pro('dividend', ts_code=code)
        if len(frame) >= 2000:
            raise ValueError(f'{code}: 分红返回达到 2000 行上限，无法确认完整性')
        frames.append(frame)
    new = _concat(frames)
    if new.empty:
        raise ValueError('dividend: 全市场分红历史返回空，停止更新')
    _save(_path('dividend.parquet'), new.drop_duplicates())
    return len(new)
