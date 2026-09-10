import os
import unittest
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
