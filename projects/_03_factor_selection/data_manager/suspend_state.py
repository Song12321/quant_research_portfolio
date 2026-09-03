"""停复牌日终状态的严格、确定性重建。

流程边界：
1. 校验 Tushare 日级停复牌事件，不猜测或静默丢弃关键字段。
2. 将同一股票同一天的多条 S/R 事件聚合成唯一的日终状态。
3. 把日终状态传播到研究期交易日，供上层 shift(1) 后约束次日股票池。

这里不处理公告何时可得，也不描述盘中任意时刻能否成交；实际成交约束属于执行层。
"""

import pandas as pd


def _prepare_suspend_events(suspend_df: pd.DataFrame) -> pd.DataFrame:
    """校验停复牌输入，并生成“是否具有日内停牌区间”的标记。"""
    # trade_date 和 suspend_type 决定状态何时改变；suspend_timing 允许为空，
    # 为空表示没有可识别的日内停牌区间，不得自行补造时间。
    required = ['ts_code', 'trade_date', 'suspend_type', 'suspend_timing']
    missing = sorted(set(required) - set(suspend_df.columns))
    if missing:
        raise ValueError(f"停复牌数据校验失败：缺少必填字段 {missing}。")

    events = suspend_df[required].copy()
    invalid_code = events['ts_code'].isna() | events['ts_code'].astype(str).str.strip().eq('')
    if invalid_code.any():
        sample = events.loc[invalid_code, required].head(5).to_dict('records')
        raise ValueError(f"停复牌数据校验失败：ts_code 不能为空，实际样例={sample}。")

    events['trade_date'] = pd.to_datetime(events['trade_date'], errors='coerce')
    invalid_date = events['trade_date'].isna()
    if invalid_date.any():
        # 日期为空时无法确定停牌作用区间，也无法证明其位于研究窗口之外。
        # 因此必须暴露数据源问题，禁止猜测日期、静默删除或跳过。
        sample = events.loc[invalid_date, required].head(5).to_dict('records')
        raise ValueError(f"停复牌数据校验失败：trade_date 无法解析或为空，实际样例={sample}。")

    invalid_type = ~events['suspend_type'].isin(['S', 'R'])
    if invalid_type.any():
        actual = events.loc[invalid_type, 'suspend_type'].drop_duplicates().tolist()
        raise ValueError(f"停复牌数据校验失败：suspend_type 必须为 S 或 R，实际值={actual}。")

    # Tushare 仅在日内停牌时提供 suspend_timing；空字符串与空值统一视为无区间。
    events['_has_timing'] = (
        events['suspend_timing'].notna()
        & events['suspend_timing'].astype(str).str.strip().ne('')
    )
    invalid_resume_timing = events['suspend_type'].eq('R') & events['_has_timing']
    if invalid_resume_timing.any():
        # suspend_timing 描述停牌区间，不应附着在 R 事件上；出现时语义冲突。
        sample = events.loc[invalid_resume_timing, required].head(5).to_dict('records')
        raise ValueError(f"停复牌数据校验失败：R 事件不应包含 suspend_timing，实际样例={sample}。")

    # 同日 S/R 可以合法并存；只有四个业务字段完全相同才属于无意义重复。
    duplicate = events.duplicated(required, keep=False)
    if duplicate.any():
        sample = events.loc[duplicate, required].head(5).to_dict('records')
        raise ValueError(f"停复牌数据校验失败：存在完全重复事件，实际样例={sample}。")
    return events


def _aggregate_suspend_events_to_eod_states(suspend_df: pd.DataFrame) -> pd.DataFrame:
    """将同日事件确定性聚合为收盘后的可交易状态。"""
    events = _prepare_suspend_events(suspend_df)
    # 先把每条事件转换为三个互斥语义，再按股票和日期做布尔聚合；
    # 这样结果只由事件集合决定，不再依赖 parquet 的物理行顺序。
    events = events.assign(
        has_resume=events['suspend_type'].eq('R'),
        has_full_day_suspend=events['suspend_type'].eq('S') & ~events['_has_timing'],
        has_intraday_suspend=events['suspend_type'].eq('S') & events['_has_timing'],
    )
    daily = events.groupby(['ts_code', 'trade_date'], as_index=False).agg(
        has_resume=('has_resume', 'any'),
        has_full_day_suspend=('has_full_day_suspend', 'any'),
        has_intraday_suspend=('has_intraday_suspend', 'any'),
    )
    # 同日既声称全日停牌又声称仅日内停牌、且没有 R 时，数据不足以确定日终状态。
    ambiguous = ~daily['has_resume'] & daily['has_full_day_suspend'] & daily['has_intraday_suspend']
    if ambiguous.any():
        sample = daily.loc[ambiguous, ['ts_code', 'trade_date']].head(5).to_dict('records')
        raise ValueError(
            "停复牌日终状态聚合失败：同日同时存在全日和日内 S 且没有 R，"
            f"无法判断日终状态，实际样例={sample}。"
        )

    # 同日 R 表示已经复牌；带时段的 S 只在日内暂停，二者收盘后均应可交易。
    # 只有无时段且没有 R 的 S 才代表需要向后传播的全日停牌状态。
    daily['is_tradeable_eod'] = daily['has_resume'] | daily['has_intraday_suspend']
    return daily[['ts_code', 'trade_date', 'is_tradeable_eod']].sort_values(
        ['ts_code', 'trade_date'], ignore_index=True
    )


def _build_suspend_eod_matrix(
        daily_states: pd.DataFrame,
        trading_dates: pd.DatetimeIndex,
        ts_codes: list[str],
) -> pd.DataFrame:
    """把离散的日终状态变更传播到研究期的连续交易日。"""
    # 状态只能沿时间向后传播，因此交易日必须严格递增且不得重复。
    trading_dates = pd.DatetimeIndex(trading_dates)
    if trading_dates.empty:
        raise ValueError("停复牌日终矩阵构建失败：研究期交易日不能为空。")
    if trading_dates.has_duplicates or not trading_dates.is_monotonic_increasing:
        raise ValueError("停复牌日终矩阵构建失败：研究期交易日必须严格递增且无重复。")

    # 研究期首日需要继承此前最后一个已知日终状态；更早事件只用于确定该初值。
    relevant = daily_states[daily_states['ts_code'].isin(ts_codes)]
    prior = relevant[relevant['trade_date'] < trading_dates[0]]
    prior = prior.sort_values(['ts_code', 'trade_date']).groupby('ts_code', sort=False).tail(1)
    prior = prior.set_index('ts_code')['is_tradeable_eod']
    # 某股票此前从未出现停复牌事件时，没有不可交易证据，延续原逻辑默认可交易。
    initial_status = pd.Series(True, index=ts_codes, dtype=bool)
    initial_status.update(prior)

    # 研究期内每天只写入当天发生变化的日终状态；其余日期随后使用 ffill 向后继承。
    # 第一行若恰有事件，以当日事件为准；否则使用研究期开始前的初始状态。
    in_period = relevant[relevant['trade_date'].isin(trading_dates)]
    updates = in_period.pivot(index='trade_date', columns='ts_code', values='is_tradeable_eod')
    matrix = updates.reindex(index=trading_dates, columns=ts_codes).astype('boolean')
    matrix.columns.name = None
    matrix.iloc[0] = matrix.iloc[0].combine_first(initial_status)
    return matrix.ffill().fillna(True).astype(bool)
