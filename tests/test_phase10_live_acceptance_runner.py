import unittest

from scripts.run_phase10_live_acceptance import _validate_model_selection


class PhaseTenLiveAcceptanceRunnerTests(unittest.TestCase):
    def test_frozen_protocol_model_is_not_an_override(self) -> None:
        self.assertFalse(
            _validate_model_selection("gpt-5.6-terra", "gpt-5.6-terra", False)
        )

    def test_different_model_requires_explicit_override(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--allow-model-override"):
            _validate_model_selection("qwen3.6-plus", "gpt-5.6-terra", False)

    def test_different_model_is_recorded_when_explicitly_allowed(self) -> None:
        self.assertTrue(
            _validate_model_selection("qwen3.6-plus", "gpt-5.6-terra", True)
        )


if __name__ == "__main__":
    unittest.main()
