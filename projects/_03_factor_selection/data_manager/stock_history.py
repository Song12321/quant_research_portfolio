"""股票有效交易日历史过滤。"""

import numpy as np
import pandas as pd


def apply_history_days_filter(
        stock_pool_df: pd.DataFrame,
        close_raw_df: pd.DataFrame | None,
        history_days: int,
) -> pd.DataFrame:
    """剔除截至 T 日收盘有效历史不足的候选股票；0 表示不应用该门槛。"""
    if (isinstance(history_days, (bool, np.bool_))
            or not isinstance(history_days, (int, np.integer)) or history_days < 0):
        raise ValueError(f"股票池 history_days 必须是非负整数，实际值={history_days!r}。")
    if history_days == 0:
        return stock_pool_df

    observable_closes = close_raw_df.notna().cumsum()
    history_counts = observable_closes.reindex(index=stock_pool_df.index, columns=stock_pool_df.columns)
    missing_for_candidate = stock_pool_df & history_counts.isna()
    if missing_for_candidate.any().any():
        sample = missing_for_candidate.stack()[lambda value: value].index.tolist()[:5]
        raise ValueError(f"股票池 history_days 过滤失败：候选股票缺少历史计数，样例={sample}。")
    return stock_pool_df & history_counts.ge(history_days)
