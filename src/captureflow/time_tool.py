"""Deterministic parsing for bounded local calendar times and durations."""

import re
from datetime import datetime, time, timedelta


_CHINESE_CLOCK_HOURS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "十一": 11,
    "十二": 12,
}
_MAX_DURATION_MINUTES = 12 * 60


def parse_time(expression: str) -> time | None:
    """Resolve one exact, supported local clock expression without guessing."""

    value = expression.strip().casefold().replace("：", ":")
    for pattern in ("%H:%M", "%I:%M %p", "%I %p"):
        try:
            return datetime.strptime(value, pattern).time()
        except ValueError:
            pass
    if value == "noon":
        return time(12)
    if value == "midnight":
        return time(0)
    return _parse_chinese_time(value)


def parse_duration(expression: str) -> timedelta | None:
    """Resolve one exact, positive, bounded duration without model arithmetic."""

    value = "".join(expression.strip().casefold().split())
    minutes = {"半小时": 30, "半个小时": 30, "一小时": 60, "halfanhour": 30, "halfhour": 30}.get(value)
    if minutes is None:
        chinese_match = re.fullmatch(r"([0-9]+)分钟", value)
        english_match = re.fullmatch(r"([0-9]+)(minutes?|hours?)", value)
        chinese_hours_match = re.fullmatch(r"([0-9]+)小时", value)
        if chinese_match is not None:
            minutes = int(chinese_match.group(1))
        elif chinese_hours_match is not None:
            minutes = int(chinese_hours_match.group(1)) * 60
        elif english_match is not None:
            amount, unit = english_match.groups()
            minutes = int(amount) * (60 if unit.startswith("hour") else 1)
    if minutes is None or not 0 < minutes <= _MAX_DURATION_MINUTES:
        return None
    return timedelta(minutes=minutes)


def _parse_chinese_time(value: str) -> time | None:
    match = re.fullmatch(
        r"(凌晨|上午|中午|下午|晚上)([0-9]{1,2}|[一二三四五六七八九十两]+)点(?:(半)|([0-9]{1,2})分?)?",
        value.replace(" ", ""),
    )
    if match is None:
        return None
    period, hour_text, half, minute_text = match.groups()
    hour = _chinese_clock_hour(hour_text)
    minute = 30 if half else int(minute_text or "0")
    if hour is None or minute > 59:
        return None
    resolved_hour = _period_hour(period, hour)
    if resolved_hour is None:
        return None
    return time(resolved_hour, minute)


def _period_hour(period: str, hour: int) -> int | None:
    """Map one bounded period and a 1-12 clock hour onto the 24-hour clock."""

    if period == "中午":
        return 12 if hour == 12 else None
    if hour == 12:
        # 晚上十二点 and 上午十二点 do not state which calendar day they name.
        return None
    if period in {"凌晨", "上午"}:
        return hour
    return hour + 12


def _chinese_clock_hour(value: str) -> int | None:
    hour = int(value) if value.isdigit() else _CHINESE_CLOCK_HOURS.get(value)
    return hour if hour is not None and 1 <= hour <= 12 else None
