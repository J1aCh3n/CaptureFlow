"""Plain date lookups: the calendar table the model reads, and the day a recall question names."""

import re
from datetime import date, timedelta

_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_CHINESE_WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
_CHINESE_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# Checked in order, so a longer phrase wins over the shorter one it contains.
_RELATIVE_DAYS = (
    ("the day after tomorrow", 2),
    ("大后天", 3),
    ("后天", 2),
    ("tomorrow", 1),
    ("明天", 1),
    ("today", 0),
    ("今天", 0),
)
_FULL_DATE = re.compile(
    r"\b(?P<iso_year>\d{4})-(?P<iso_month>\d{2})-(?P<iso_day>\d{2})\b"
    r"|(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
    r"(?:\s*[（(](?:周|星期|礼拜)(?P<weekday>[一二三四五六日天])[）)])?"
)


def calendar_reference(
    reference_date: date, days_before: int = 7, days_after: int = 27
) -> list[dict[str, str]]:
    """A bounded date table so a model looks weekday phrases up instead of computing them."""

    return [
        {
            "date": day.isoformat(),
            "weekday": _WEEKDAY_NAMES[day.weekday()],
            "weekday_zh": _CHINESE_WEEKDAY_NAMES[day.weekday()],
        }
        for day in (
            reference_date + timedelta(days=offset)
            for offset in range(-days_before, days_after + 1)
        )
    ]


def asked_date(query: str, reference_date: date) -> date | None:
    """The single day a question names: a relative day word or a full date, otherwise None.

    A full Chinese date whose bracketed weekday disagrees with it is treated as naming
    no day, rather than trusting either half.
    """

    normalized = query.casefold()
    for phrase, offset in _RELATIVE_DAYS:
        if phrase in normalized:
            return reference_date + timedelta(days=offset)
    match = _FULL_DATE.search(query)
    if match is None:
        return None
    if match["iso_year"] is not None:
        year, month, day = match["iso_year"], match["iso_month"], match["iso_day"]
    else:
        year, month, day = match["year"], match["month"], match["day"]
    try:
        named = date(int(year), int(month), int(day))
    except ValueError:
        return None
    if match["weekday"] is not None and named.weekday() != _CHINESE_WEEKDAYS[match["weekday"]]:
        return None
    return named
