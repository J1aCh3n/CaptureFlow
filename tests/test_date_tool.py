import unittest
from datetime import date

from captureflow.date_tool import asked_date, calendar_reference


class CalendarReferenceTests(unittest.TestCase):
    def test_calendar_reference_spans_last_week_to_four_weeks_ahead(self) -> None:
        table = calendar_reference(date(2026, 9, 12))

        self.assertEqual(len(table), 35)
        self.assertEqual(
            table[0], {"date": "2026-09-05", "weekday": "Saturday", "weekday_zh": "周六"}
        )
        self.assertEqual(
            table[7], {"date": "2026-09-12", "weekday": "Saturday", "weekday_zh": "周六"}
        )
        self.assertEqual(
            table[-1], {"date": "2026-10-09", "weekday": "Friday", "weekday_zh": "周五"}
        )


class AskedDateTests(unittest.TestCase):
    def test_relative_day_words_are_resolved_in_both_languages(self) -> None:
        reference = date(2026, 9, 9)
        cases = {
            "我今天有什么安排？": date(2026, 9, 9),
            "What is on my calendar today?": date(2026, 9, 9),
            "明天会议的内容是什么？": date(2026, 9, 10),
            "When is my meeting tomorrow?": date(2026, 9, 10),
            "我后天有什么安排？": date(2026, 9, 11),
            "Anything the day after tomorrow?": date(2026, 9, 11),
            "大后天有会吗？": date(2026, 9, 12),
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(asked_date(query, reference), expected)

    def test_full_dates_are_resolved(self) -> None:
        reference = date(2026, 9, 9)

        self.assertEqual(asked_date("What is on 2026-10-16?", reference), date(2026, 10, 16))
        self.assertEqual(asked_date("2026年9月11日有什么会？", reference), date(2026, 9, 11))
        self.assertEqual(
            asked_date("2026年9月11日 (周五) 的会议是什么？", reference), date(2026, 9, 11)
        )

    def test_a_date_whose_weekday_disagrees_names_no_day(self) -> None:
        self.assertIsNone(asked_date("2026年9月11日（周四）的会议是什么？", date(2026, 9, 9)))

    def test_impossible_and_unnamed_dates_name_no_day(self) -> None:
        for query in ("2026年2月30日有会吗？", "2026-13-01", "下周三有什么会？", "我的会议是什么？"):
            with self.subTest(query=query):
                self.assertIsNone(asked_date(query, date(2026, 9, 11)))
