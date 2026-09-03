import pandas as pd
import pytest

from projects._03_factor_selection.data_manager.suspend_state import (
    _aggregate_suspend_events_to_eod_states,
    _build_suspend_eod_matrix,
)


def _events(rows):
    return pd.DataFrame(
        rows,
        columns=['ts_code', 'trade_date', 'suspend_type', 'suspend_timing'],
    )


def test_same_day_events_are_order_independent():
    rows = [
        ('A.SZ', '20240102', 'S', '09:30-09:40'),
        ('A.SZ', '20240102', 'R', None),
        ('B.SH', '20240102', 'S', None),
        ('B.SH', '20240102', 'R', None),
    ]

    forward = _aggregate_suspend_events_to_eod_states(_events(rows))
    reversed_rows = _aggregate_suspend_events_to_eod_states(_events(list(reversed(rows))))

    pd.testing.assert_frame_equal(forward, reversed_rows)
    assert forward['is_tradeable_eod'].tolist() == [True, True]


def test_intraday_suspend_recovers_but_full_day_suspend_persists_until_resume():
    events = _events([
        ('A.SZ', '20240102', 'S', '10:00-10:10'),
        ('B.SH', '20240102', 'S', None),
        ('B.SH', '20240104', 'R', None),
    ])

    daily = _aggregate_suspend_events_to_eod_states(events)
    matrix = _build_suspend_eod_matrix(
        daily,
        pd.date_range('2024-01-02', periods=4, freq='D'),
        ['A.SZ', 'B.SH', 'C.BJ'],
    )

    expected = pd.DataFrame(
        {
            'A.SZ': [True, True, True, True],
            'B.SH': [False, False, True, True],
            'C.BJ': [True, True, True, True],
        },
        index=pd.date_range('2024-01-02', periods=4, freq='D'),
    )
    pd.testing.assert_frame_equal(matrix, expected)


@pytest.mark.parametrize(
    ('events', 'message'),
    [
        (_events([('A.SZ', None, 'S', None)]), 'trade_date'),
        (_events([('A.SZ', '20240102', 'X', None)]), 'suspend_type'),
        (_events([('A.SZ', '20240102', 'R', '09:30-09:40')]), 'R 事件'),
        (
            _events([
                ('A.SZ', '20240102', 'S', None),
                ('A.SZ', '20240102', 'S', None),
            ]),
            '完全重复事件',
        ),
        (
            _events([
                ('A.SZ', '20240102', 'S', None),
                ('A.SZ', '20240102', 'S', '09:30-09:40'),
            ]),
            '无法判断日终状态',
        ),
    ],
)
def test_invalid_suspend_events_fail_strictly(events, message):
    with pytest.raises(ValueError, match=message):
        _aggregate_suspend_events_to_eod_states(events)
