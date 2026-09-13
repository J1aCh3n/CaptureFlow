"""Local-time configuration with an explicit, reproducible override."""

import os
from datetime import date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tzlocal import get_localzone_name


def configured_timezone() -> str:
    """Return an explicit IANA override or the computer's detected IANA timezone."""

    timezone = os.getenv("CAPTUREFLOW_TIMEZONE") or get_localzone_name()
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown timezone: {timezone}") from error
    return timezone


def configured_reference_now(timezone: str) -> datetime:
    """The current local date and time, read once so a turn never mixes two clocks."""

    try:
        return datetime.now(ZoneInfo(timezone))
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown timezone: {timezone}") from error


def configured_reference_date(timezone: str) -> date:
    return configured_reference_now(timezone).date()
