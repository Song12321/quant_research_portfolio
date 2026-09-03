"""停复牌日终状态的严格、确定性重建。"""

import pandas as pd


def _prepare_suspend_events(suspend_df: pd.DataFrame) -> pd.DataFrame:
    """校验停复牌输入，并生成日内停牌标记。"""
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
        sample = events.loc[invalid_date, required].head(5).to_dict('records')
        raise ValueError(f"停复牌数据校验失败：trade_date 无法解析或为空，实际样例={sample}。")

    invalid_type = ~events['suspend_type'].isin(['S', 'R'])
    if invalid_type.any():
        actual = events.loc[invalid_type, 'suspend_type'].drop_duplicates().tolist()
        raise ValueError(f"停复牌数据校验失败：suspend_type 必须为 S 或 R，实际值={actual}。")

    events['_has_timing'] = (
        events['suspend_timing'].notna()
        & events['suspend_timing'].astype(str).str.strip().ne('')
    )
    invalid_resume_timing = events['suspend_type'].eq('R') & events['_has_timing']
    if invalid_resume_timing.any():
        sample = events.loc[invalid_resume_timing, required].head(5).to_dict('records')
        raise ValueError(f"停复牌数据校验失败：R 事件不应包含 suspend_timing，实际样例={sample}。")

    duplicate = events.duplicated(required, keep=False)
    if duplicate.any():
        sample = events.loc[duplicate, required].head(5).to_dict('records')
        raise ValueError(f"停复牌数据校验失败：存在完全重复事件，实际样例={sample}。")
    return events


def _aggregate_suspend_events_to_eod_states(suspend_df: pd.DataFrame) -> pd.DataFrame:
    """将同日事件确定性聚合为收盘后的可交易状态。"""
    events = _prepare_suspend_events(suspend_df)
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
    """把日终状态变更传播到研究期交易日。"""
    trading_dates = pd.DatetimeIndex(trading_dates)
    if trading_dates.empty:
        raise ValueError("停复牌日终矩阵构建失败：研究期交易日不能为空。")
    if trading_dates.has_duplicates or not trading_dates.is_monotonic_increasing:
        raise ValueError("停复牌日终矩阵构建失败：研究期交易日必须严格递增且无重复。")

    relevant = daily_states[daily_states['ts_code'].isin(ts_codes)]
    prior = relevant[relevant['trade_date'] < trading_dates[0]]
    prior = prior.sort_values(['ts_code', 'trade_date']).groupby('ts_code', sort=False).tail(1)
    prior = prior.set_index('ts_code')['is_tradeable_eod']
    initial_status = pd.Series(True, index=ts_codes, dtype=bool)
    initial_status.update(prior)

    in_period = relevant[relevant['trade_date'].isin(trading_dates)]
    updates = in_period.pivot(index='trade_date', columns='ts_code', values='is_tradeable_eod')
    matrix = updates.reindex(index=trading_dates, columns=ts_codes).astype('boolean')
    matrix.columns.name = None
    matrix.iloc[0] = matrix.iloc[0].combine_first(initial_status)
    return matrix.ffill().fillna(True).astype(bool)
