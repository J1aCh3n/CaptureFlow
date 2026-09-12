import unittest
from datetime import time, timedelta

from captureflow.time_tool import (
    parse_duration,
    parse_end_time,
    parse_time,
    stated_timezone,
    with_stated_period,
)


class TimeToolTests(unittest.TestCase):
    def test_supported_clock_expressions_are_resolved_deterministically(self) -> None:
        cases = {
            "下午四点": time(16),
            "下午六点": time(18),
            "晚上七点": time(19),
            "晚上七点半": time(19, 30),
            "上午九点半": time(9, 30),
            "中午十二点": time(12),
            "中午12点半": time(12, 30),
            "凌晨三点": time(3),
            "19:00": time(19),
            "19：30": time(19, 30),
            "7 PM": time(19),
            "7:30 PM": time(19, 30),
        }

        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(parse_time(expression), expected)

    def test_malformed_or_ambiguous_clock_expressions_are_unresolved(self) -> None:
        for expression in (
            "晚上七点才开始",
            "25:00",
            "下午十三点",
            "七点",
            "晚上十二点",
            "上午十二点",
            "凌晨十二点",
            "中午三点",
        ):
            with self.subTest(expression=expression):
                self.assertIsNone(parse_time(expression))

    def test_a_period_written_before_a_copied_bare_hour_is_restored(self) -> None:
        message = "把这场会议改到明天下午两点到三点。"

        self.assertEqual(with_stated_period("两点", message), "下午两点")
        self.assertEqual(with_stated_period("三点", message), "三点")
        self.assertEqual(with_stated_period("下午两点", message), "下午两点")
        self.assertEqual(with_stated_period("两点", "上午开会，两点结束"), "两点")
        self.assertIsNone(with_stated_period(None, message))

    def test_a_bare_range_end_shares_the_period_stated_on_its_start(self) -> None:
        cases = {
            ("五点", "下午四点"): time(17),
            ("三点", "下午两点"): time(15),
            ("十一点半", "上午十点"): time(11, 30),
            ("下午五点", "下午四点"): time(17),
            ("17:00", "下午四点"): time(17),
        }

        for (end, start), expected in cases.items():
            with self.subTest(end=end, start=start):
                self.assertEqual(parse_end_time(end, start), expected)

        for end, start in (
            ("一点", "晚上十一点"),
            ("三点", "下午四点"),
            ("五点", "16:00"),
            ("五点", None),
        ):
            with self.subTest(end=end, start=start):
                self.assertIsNone(parse_end_time(end, start))

    def test_named_timezones_are_recognised_without_false_positives(self) -> None:
        cases = {
            "纽约时间明天上午九点有电话会": ("纽约时间", "America/New_York"),
            "北京时间晚上八点开会": ("北京时间", "Asia/Shanghai"),
            "温哥华时间下午三点": ("温哥华时间", "America/Vancouver"),
            "Call at 9 AM PST": ("PST", "America/Los_Angeles"),
            "Sync at 14:00 UTC": ("UTC", "UTC"),
        }

        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(stated_timezone(expression), expected)

        for expression in (
            "会议时间定在下午三点",
            "开始时间是明天",
            "本地时间下午三点",
            "当地时间下午三点",
            "PSTN 测试明天下午三点",
            "上海时间表已经更新",
        ):
            with self.subTest(expression=expression):
                self.assertIsNone(stated_timezone(expression))

    def test_supported_durations_are_resolved_deterministically(self) -> None:
        cases = {
            "半个小时": 30,
            "半小时": 30,
            "半个钟头": 30,
            "30分钟": 30,
            "一小时": 60,
            "一个小时": 60,
            "一个半小时": 90,
            "1.5小时": 90,
            "两小时": 120,
            "两个小时": 120,
            "2小时": 120,
            "45 minutes": 45,
            "an hour": 60,
            "an hour and a half": 90,
            "half an hour": 30,
        }

        for expression, expected_minutes in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(parse_duration(expression), timedelta(minutes=expected_minutes))

    def test_zero_negative_and_unbounded_durations_are_rejected(self) -> None:
        for expression in ("0分钟", "-30分钟", "13小时", "随便一会儿", "半个半小时", "一二小时"):
            with self.subTest(expression=expression):
                self.assertIsNone(parse_duration(expression))
