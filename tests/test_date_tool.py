import unittest
from datetime import date

from captureflow.date_tool import resolve_relative_date


class DateToolTests(unittest.TestCase):
    def test_iso_date_with_leading_by_is_resolved_and_preserves_expression(self) -> None:
        result = resolve_relative_date(
            "by 2026-10-16", date(2026, 9, 8), "America/Vancouver"
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2026, 10, 16))
        self.assertEqual(result.expression, "by 2026-10-16")

    def test_next_weekday_uses_explicit_reference_date(self) -> None:
        result = resolve_relative_date("next Wednesday", date(2026, 9, 6), "America/Vancouver")

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2026, 9, 9))

    def test_bare_weekday_is_reported_as_ambiguous(self) -> None:
        result = resolve_relative_date("Wednesday", date(2026, 9, 6), "America/Vancouver")

        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.resolved_date)

    def test_bare_chinese_weekday_is_reported_as_ambiguous(self) -> None:
        result = resolve_relative_date("周二", date(2026, 9, 6), "America/Vancouver")

        self.assertEqual(result.status, "ambiguous")
        self.assertIn("中文星期", result.explanation)

    def test_explicit_chinese_week_is_resolved_without_treating_bare_weekdays_as_safe(self) -> None:
        result = resolve_relative_date("这周四", date(2026, 9, 9), "America/Vancouver")

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2026, 9, 10))

    def test_month_day_uses_reference_year_when_not_yet_passed(self) -> None:
        result = resolve_relative_date("9/8", date(2026, 9, 6), "America/Vancouver")

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2026, 9, 8))

    def test_month_day_rolls_to_next_year_when_passed(self) -> None:
        result = resolve_relative_date("9/8", date(2026, 9, 9), "America/Vancouver")

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2027, 9, 8))

    def test_day_after_tomorrow_is_resolved_in_both_languages(self) -> None:
        reference = date(2026, 9, 9)

        self.assertEqual(
            resolve_relative_date("后天", reference, "America/Vancouver").resolved_date,
            date(2026, 9, 11),
        )
        self.assertEqual(
            resolve_relative_date(
                "the day after tomorrow", reference, "America/Vancouver"
            ).resolved_date,
            date(2026, 9, 11),
        )
        self.assertEqual(
            resolve_relative_date("大后天", reference, "America/Vancouver").resolved_date,
            date(2026, 9, 12),
        )

    def test_chinese_full_date_with_matching_weekday_is_resolved(self) -> None:
        result = resolve_relative_date(
            "2026年9月11日 (周五)", date(2026, 9, 9), "America/Vancouver"
        )

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2026, 9, 11))

    def test_chinese_full_date_with_wrong_weekday_requires_confirmation(self) -> None:
        result = resolve_relative_date(
            "2026年9月11日（周四）", date(2026, 9, 9), "America/Vancouver"
        )

        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.resolved_date)
