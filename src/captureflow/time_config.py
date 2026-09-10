"""Local-time configuration with an explicit, reproducible override."""

import os
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
