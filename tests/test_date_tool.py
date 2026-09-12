import unittest
from datetime import date

from captureflow.date_tool import resolve_relative_date, states_recurrence


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

    def test_bare_weekday_resolves_to_the_nearest_future_occurrence(self) -> None:
        result = resolve_relative_date("Wednesday", date(2026, 9, 6), "America/Vancouver")

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2026, 9, 9))

    def test_bare_chinese_weekday_resolves_to_the_nearest_future_occurrence(self) -> None:
        result = resolve_relative_date("周二", date(2026, 9, 6), "America/Vancouver")

        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.resolved_date, date(2026, 9, 8))

    def test_bare_weekday_on_the_reference_date_still_requires_clarification(self) -> None:
        result = resolve_relative_date("周四", date(2026, 9, 10), "America/Vancouver")

        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.resolved_date)

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

    def test_chinese_month_day_without_a_year_uses_the_next_occurrence(self) -> None:
        reference = date(2026, 9, 11)
        cases = {
            "9月20日": date(2026, 9, 20),
            "9月20号": date(2026, 9, 20),
            "12月1日": date(2026, 12, 1),
            "3月1日": date(2027, 3, 1),
        }

        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = resolve_relative_date(expression, reference, "America/Vancouver")
                self.assertEqual(result.status, "resolved")
                self.assertEqual(result.resolved_date, expected)

    def test_english_month_names_resolve_like_numeric_month_day(self) -> None:
        reference = date(2026, 9, 11)
        cases = {
            "September 20": date(2026, 9, 20),
            "sept 20": date(2026, 9, 20),
            "20 September": date(2026, 9, 20),
            "Dec 1": date(2026, 12, 1),
            "March 1": date(2027, 3, 1),
        }

        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = resolve_relative_date(expression, reference, "America/Vancouver")
                self.assertEqual(result.status, "resolved")
                self.assertEqual(result.resolved_date, expected)

    def test_repeating_expressions_are_recognised_as_recurrence(self) -> None:
        for expression in ("每周一", "每天", "每个月", "每月15号", "every Monday", "weekly", "biweekly"):
            with self.subTest(expression=expression):
                self.assertIsNotNone(states_recurrence(expression))

        for expression in ("下周一", "周一", "monday", "2026年9月20日", "明天"):
            with self.subTest(expression=expression):
                self.assertIsNone(states_recurrence(expression))

    def test_impossible_and_unsupported_dates_stay_unresolved(self) -> None:
        for expression in ("2月30日", "13月1日", "下下周三", "月底", "九月二十日"):
            with self.subTest(expression=expression):
                result = resolve_relative_date(expression, date(2026, 9, 11), "America/Vancouver")
                self.assertNotEqual(result.status, "resolved")
                self.assertIsNone(result.resolved_date)

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
