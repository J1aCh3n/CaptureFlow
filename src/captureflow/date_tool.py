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


def resolve_relative_date(expression: str, reference_date: date, timezone: str) -> DateResolution:
    """Resolve a small, explicit v0 vocabulary without guessing bare weekdays."""

    normalized = " ".join(expression.lower().split())
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
    if iso_date := _parse_iso_date(normalized):
        return DateResolution(
            **base,
            status="resolved",
            resolved_date=iso_date,
            explanation="The expression is an ISO calendar date.",
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
    if normalized in _WEEKDAYS:
        return DateResolution(
            **base,
            status="ambiguous",
            explanation="A bare weekday can refer to more than one upcoming date.",
        )
    return DateResolution(
        **base,
        status="unsupported",
        explanation="This Phase 2 date tool supports today, tomorrow, ISO dates, and next weekdays.",
    )


def _parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
