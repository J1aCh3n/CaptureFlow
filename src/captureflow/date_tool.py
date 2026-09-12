import re
from datetime import date, timedelta

from pydantic import BaseModel


class DateResolution(BaseModel):
    expression: str
    reference_date: date
    timezone: str
    status: str
    resolved_date: date | None = None
    explanation: str


_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

_MONTH_NAMES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sept": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_CHINESE_WEEKDAYS = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}

_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_CHINESE_WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


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


def resolve_relative_date(expression: str, reference_date: date, timezone: str) -> DateResolution:
    """Resolve a small, explicit v0 vocabulary with future bare weekdays."""

    normalized = " ".join(expression.lower().split())
    if normalized.startswith("by "):
        normalized = normalized[3:]
    base = dict(expression=expression, reference_date=reference_date, timezone=timezone)
    if normalized == "today":
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date,
            explanation="Resolved relative to the supplied reference date.",
        )
    if normalized == "tomorrow":
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=1),
            explanation="Resolved relative to the supplied reference date.",
        )
    if normalized == "the day after tomorrow":
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=2),
            explanation="Resolved relative to the supplied reference date.",
        )
    if expression.strip() == "今天":
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date,
            explanation="根据提供的参考日期解析。",
        )
    if expression.strip() == "明天":
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=1),
            explanation="根据提供的参考日期解析。",
        )
    if expression.strip() == "后天":
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=2),
            explanation="根据提供的参考日期解析。",
        )
    if expression.strip() == "大后天":
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=3),
            explanation="根据提供的参考日期解析。",
        )
    if chinese_date := _parse_chinese_date(expression.strip(), reference_date):
        candidate, stated_weekday = chinese_date
        if stated_weekday is not None and candidate.weekday() != stated_weekday:
            return DateResolution(
                **base,
                status="ambiguous",
                explanation="日期与括号中的星期不一致，请确认具体日期。",
            )
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=candidate,
            explanation="解析了明确的中文年月日。",
        )
    if iso_date := _parse_iso_date(normalized):
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=iso_date,
            explanation="The expression is an ISO calendar date.",
        )
    if month_day := _parse_month_day(normalized, reference_date):
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=month_day,
            explanation=(
                "Resolved a month and day using the reference year, or the following year when "
                "that date has already passed."
            ),
        )
    if match := re.fullmatch(r"next (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", normalized):
        weekday = _WEEKDAYS[match.group(1)]
        days_ahead = (weekday - reference_date.weekday()) % 7 or 7
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=days_ahead),
            explanation="Resolved the explicit next-weekday expression.",
        )
    if match := re.fullmatch(r"(?:this week|this) (monday|tuesday|wednesday|thursday|friday|saturday|sunday)", normalized):
        weekday = _WEEKDAYS[match.group(1)]
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=weekday - reference_date.weekday()),
            explanation="Resolved the explicit weekday in the reference week.",
        )
    if match := re.fullmatch(r"(?:本周|这周|本星期|这星期)([一二三四五六日天])", expression.strip()):
        weekday = _CHINESE_WEEKDAYS[match.group(1)]
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=weekday - reference_date.weekday()),
            explanation="根据提供的参考日期解析了本周日期。",
        )
    if match := re.fullmatch(r"(?:下周|下星期)([一二三四五六日天])", expression.strip()):
        weekday = _CHINESE_WEEKDAYS[match.group(1)]
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=reference_date + timedelta(days=7 + weekday - reference_date.weekday()),
            explanation="根据提供的参考日期解析了下周日期。",
        )
    if normalized in _WEEKDAYS:
        candidate = _future_weekday(reference_date, _WEEKDAYS[normalized])
        if candidate is not None:
            return DateResolution(
                **base,
                status="resolved",
                resolved_date=candidate,
                explanation="Resolved the nearest future weekday from the supplied reference date.",
            )
        return DateResolution(
            **base,
            status="ambiguous",
            explanation="A bare weekday on the reference date needs an explicit date.",
        )
    if re.fullmatch(r"(?:周|星期|礼拜)[一二三四五六日天]", expression.strip()):
        weekday = _CHINESE_WEEKDAYS[expression.strip()[-1]]
        candidate = _future_weekday(reference_date, weekday)
        if candidate is not None:
            return DateResolution(
                **base,
                status="resolved",
                resolved_date=candidate,
                explanation="根据提供的参考日期解析了最近未来的星期。",
            )
        return DateResolution(
            **base,
            status="ambiguous",
            explanation="日期与所说星期相同，请确认具体日期。",
        )
    return DateResolution(
        **base,
        status="unsupported",
        explanation=(
            "This date tool supports bounded relative dates, ISO or Chinese full dates, "
            "explicit next weekdays, and future bare weekdays."
        ),
    )


def _future_weekday(reference_date: date, weekday: int) -> date | None:
    days_ahead = (weekday - reference_date.weekday()) % 7
    if days_ahead == 0:
        return None
    return reference_date + timedelta(days=days_ahead)


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_month_day(value: str, reference_date: date) -> date | None:
    month_day = _numeric_month_day(value) or _named_month_day(value)
    if month_day is None:
        return None
    month, day = month_day
    candidate = _calendar_date(reference_date.year, month, day)
    if candidate is None:
        return None
    if candidate < reference_date:
        return _calendar_date(reference_date.year + 1, month, day)
    return candidate


def _numeric_month_day(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})", value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _named_month_day(value: str) -> tuple[int, int] | None:
    names = "|".join(_MONTH_NAMES)
    match = re.fullmatch(rf"({names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?", value) or re.fullmatch(
        rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({names})\.?", value
    )
    if match is None:
        return None
    first, second = match.groups()
    month_text, day_text = (first, second) if first in _MONTH_NAMES else (second, first)
    return _MONTH_NAMES[month_text], int(day_text)


def _parse_chinese_date(value: str, reference_date: date) -> tuple[date, int | None] | None:
    match = re.fullmatch(
        r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})[日号]"
        r"(?:\s*[（(](?:周|星期|礼拜)([一二三四五六日天])[）)])?",
        value,
    )
    if match is None:
        return None
    year, month, day, weekday_text = match.groups()
    candidate = _calendar_date(
        reference_date.year if year is None else int(year), int(month), int(day)
    )
    if candidate is None:
        return None
    if year is None and candidate < reference_date:
        # A bare 9月20日 names the next time that day comes around.
        candidate = _calendar_date(reference_date.year + 1, int(month), int(day))
        if candidate is None:
            return None
    stated_weekday = None if weekday_text is None else _CHINESE_WEEKDAYS[weekday_text]
    return candidate, stated_weekday


def _calendar_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None
