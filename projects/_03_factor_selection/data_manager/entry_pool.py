"""开盘建仓资格；由调用方对齐到前一交易日的收盘信号。"""

from datetime import time

from projects._03_factor_selection.data_manager.suspend_state import (
    _aggregate_suspend_events_to_eod_states,
    _build_suspend_eod_matrix,
    _prepare_suspend_events,
)


def build_open_tradeable_mask(suspend_df, dates, stocks):
    """日级 R 按当日开盘复牌处理；日内停牌只检查是否覆盖 09:30。"""
    events = _prepare_suspend_events(suspend_df)
    daily = _aggregate_suspend_events_to_eod_states(suspend_df)
    # 全日停牌持续至 R；日内 S 不向后传播。这里读取 T 日状态，不 shift。
    tradeable = _build_suspend_eod_matrix(daily, dates, stocks)
    intraday = events.loc[
        events['trade_date'].isin(dates)
        & events['ts_code'].isin(stocks)
        & events['suspend_type'].eq('S') & events['_has_timing']
    ]
    for row in intraday.itertuples(index=False):
        blocked = False
        for interval in row.suspend_timing.split(','):
            start_text, end_text = interval.strip().split('-')
            start = time(*map(int, start_text.split(':')))
            end = time(*map(int, end_text.split(':')))
            blocked |= start <= time(9, 30) < end
        if blocked:
            tradeable.loc[row.trade_date, row.ts_code] = False
    return tradeable


def apply_open_buy_filter(pool, open_raw, up_limit, open_tradeable):
    """T 日基础池与 T+1 开盘资格取交集；跌停不限制买入。"""
    buyable = open_tradeable & open_raw.round(2).lt(up_limit.round(2))
    return pool & buyable.shift(-1, fill_value=False)
