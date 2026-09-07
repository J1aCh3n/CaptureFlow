import unittest
from datetime import date

from captureflow.date_tool import resolve_relative_date


class DateToolTests(unittest.TestCase):
    def test_next_weekday_uses_explicit_reference_date(self) -> None:
        result = resolve_relative_date("next Wednesday", date(2026, 9, 6), "America/Vancouver")

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2026, 9, 9))

    def test_bare_weekday_is_reported_as_ambiguous(self) -> None:
        result = resolve_relative_date("Wednesday", date(2026, 9, 6), "America/Vancouver")

        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.resolved_date)
