import unittest
from datetime import time, timedelta

from captureflow.time_tool import parse_duration, parse_time


class TimeToolTests(unittest.TestCase):
    def test_supported_clock_expressions_are_resolved_deterministically(self) -> None:
        cases = {
            "下午四点": time(16),
            "下午六点": time(18),
            "晚上七点": time(19),
            "晚上七点半": time(19, 30),
            "上午九点半": time(9, 30),
            "19:00": time(19),
            "19：30": time(19, 30),
            "7 PM": time(19),
            "7:30 PM": time(19, 30),
        }

        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(parse_time(expression), expected)

    def test_malformed_or_ambiguous_clock_expressions_are_unresolved(self) -> None:
        for expression in ("晚上七点才开始", "25:00", "下午十三点", "七点"):
            with self.subTest(expression=expression):
                self.assertIsNone(parse_time(expression))

    def test_supported_durations_are_resolved_deterministically(self) -> None:
        cases = {
            "半个小时": 30,
            "半小时": 30,
            "30分钟": 30,
            "一小时": 60,
            "2小时": 120,
            "45 minutes": 45,
        }

        for expression, expected_minutes in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(parse_duration(expression), timedelta(minutes=expected_minutes))

    def test_zero_negative_and_unbounded_durations_are_rejected(self) -> None:
        for expression in ("0分钟", "-30分钟", "13小时", "随便一会儿"):
            with self.subTest(expression=expression):
                self.assertIsNone(parse_duration(expression))
