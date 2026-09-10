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

def resolve_relative_date(expression: str, reference_date: date, timezone: str) -> DateResolution:
    """Resolve a small, explicit v0 vocabulary without guessing bare weekdays."""

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
    if chinese_date := _parse_chinese_date(expression.strip()):
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
                "Resolved M/D using the reference year, or the following year when that date "
                "has already passed."
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
        return DateResolution(
            **base,
            status="ambiguous",
            explanation="A bare weekday can refer to more than one upcoming date.",
        )
    if re.fullmatch(r"(?:周|星期|礼拜)[一二三四五六日天]", expression.strip()):
        return DateResolution(
            **base,
            status="ambiguous",
            explanation="未带周次的中文星期表达可能指不同的日期，请确认具体日期。",
        )
    return DateResolution(
        **base,
        status="unsupported",
        explanation=(
            "This date tool supports bounded relative dates, ISO or Chinese full dates, "
            "explicit next weekdays, and reports bare weekdays as ambiguous."
        ),
    )


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_month_day(value: str, reference_date: date) -> date | None:
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})", value)
    if not match:
        return None
    month, day = (int(part) for part in match.groups())
    try:
        candidate = date(reference_date.year, month, day)
    except ValueError:
        return None
    if candidate < reference_date:
        try:
            return date(reference_date.year + 1, month, day)
        except ValueError:
            return None
    return candidate


def _parse_chinese_date(value: str) -> tuple[date, int | None] | None:
    match = re.fullmatch(
        r"(\d{4})年(\d{1,2})月(\d{1,2})日"
        r"(?:\s*[（(](?:周|星期|礼拜)([一二三四五六日天])[）)])?",
        value,
    )
    if match is None:
        return None
    year, month, day, weekday_text = match.groups()
    try:
        candidate = date(int(year), int(month), int(day))
    except ValueError:
        return None
    stated_weekday = None if weekday_text is None else _CHINESE_WEEKDAYS[weekday_text]
    return candidate, stated_weekday
