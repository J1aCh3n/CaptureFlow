import os
import unittest
from datetime import timedelta
from unittest.mock import patch

from captureflow.time_config import configured_timezone


class TimeConfigTests(unittest.TestCase):
    def test_explicit_environment_override_wins_over_computer_timezone(self) -> None:
        with patch.dict(os.environ, {"CAPTUREFLOW_TIMEZONE": "America/Vancouver"}, clear=False):
            with patch("captureflow.time_config.get_localzone_name", return_value="America/Los_Angeles"):
                self.assertEqual(configured_timezone(), "America/Vancouver")

    def test_computer_timezone_is_used_without_an_override(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("captureflow.time_config.get_localzone_name", return_value="America/Toronto"):
                self.assertEqual(configured_timezone(), "America/Toronto")

    def test_invalid_explicit_override_is_not_silently_replaced(self) -> None:
        with patch.dict(os.environ, {"CAPTUREFLOW_TIMEZONE": "not/a-timezone"}, clear=False):
            with self.assertRaisesRegex(ValueError, "Unknown timezone"):
                configured_timezone()


class ReferenceClockTests(unittest.TestCase):
    def test_reference_now_is_timezone_aware_and_matches_the_reference_date(self) -> None:
        from captureflow.workflow import configured_reference_date, configured_reference_now

        now = configured_reference_now("America/Vancouver")

        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(str(now.tzinfo), "America/Vancouver")
        # Two clock reads may straddle midnight, so the date may be one day later.
        self.assertIn(
            configured_reference_date("America/Vancouver") - now.date(),
            {timedelta(0), timedelta(days=1)},
        )

    def test_an_unknown_timezone_is_rejected(self) -> None:
        from captureflow.workflow import configured_reference_now

        with self.assertRaises(ValueError):
            configured_reference_now("Mars/Olympus_Mons")
