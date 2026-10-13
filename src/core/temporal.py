"""Precision-aware dates. ISO storage anchors are never treated as exact dates."""
from __future__ import annotations
import calendar
from datetime import date


def bounds(value: str, precision: str = 'day') -> tuple[date, date]:
    """Inclusive possible-date bounds for an endpoint (not an interval end convention)."""
    parts = value.split('-'); y = int(parts[0])
    if precision == 'year':
        return date(y, 1, 1), date(y, 12, 31)
    m = int(parts[1])
    if precision == 'month':
        return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
    if precision != 'day':
        raise ValueError(f'Unknown precision: {precision}')
    v = date.fromisoformat(value)
    return v, v


def display(value: str | None, precision: str | None) -> str | None:
    if value is None:
        return None
    return value[:{'year': 4, 'month': 7, 'day': 10}[precision]]


def endpoint_bounds(fact: dict, endpoint: str):
    return bounds(fact['valid_from' if endpoint == 'start' else 'valid_to'],
                  fact['precision_start' if endpoint == 'start' else 'precision_end'])


def endpoint_text(fact: dict, endpoint: str):
    return display(fact['valid_from' if endpoint == 'start' else 'valid_to'],
                   fact['precision_start' if endpoint == 'start' else 'precision_end'])


def compare_bounds(left, right) -> str | None:
    if left[1] < right[0]: return 'before'
    if left[0] > right[1]: return 'after'
    if left[0] == left[1] == right[0] == right[1]: return 'same'
    return None  # Overlapping uncertainty ranges do not establish an ordering.


def exact_duration(fact: dict) -> int | None:
    if (fact['valid_to'] is None or fact['precision_start'] != 'day'
            or fact['precision_end'] != 'day'):
        return None
    return (date.fromisoformat(fact['valid_to']) - date.fromisoformat(fact['valid_from'])).days
