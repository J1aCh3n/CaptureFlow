import unittest
from types import SimpleNamespace

from scripts.run_phase10_live_acceptance import (
    _validate_model_selection,
    answered_from_project_record,
    answered_from_stored_event,
)


class _Sources:
    def __init__(self, local_ids=(), record_ids=()):
        self.local_ids = local_ids
        self.record_ids = record_ids

    def list_local_context_answer_sources(self, _assistant_message_id):
        return [SimpleNamespace(reference=SimpleNamespace(source_id=source_id)) for source_id in self.local_ids]

    def list_project_answer_sources(self, _assistant_message_id):
        return [SimpleNamespace(id=record_id) for record_id in self.record_ids]


def _turn(outcome, operations=()):
    return SimpleNamespace(outcome=outcome, operations=operations, assistant_message=SimpleNamespace(id="reply"))


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


class EffectCheckTests(unittest.TestCase):
    def test_a_meeting_answer_passes_on_its_effect_whatever_the_intent_label(self) -> None:
        self.assertTrue(
            answered_from_stored_event(_Sources(local_ids=["event-1"]), _turn("grounded_local_context"), "event-1")
        )

    def test_a_meeting_answer_fails_without_the_created_event_or_with_a_write(self) -> None:
        for sources, turn, event_id in (
            (_Sources(local_ids=["other-event"]), _turn("grounded_local_context"), "event-1"),
            (_Sources(local_ids=["event-1"]), _turn("project_answer"), "event-1"),
            (_Sources(local_ids=["event-1"]), _turn("grounded_local_context", ("write",)), "event-1"),
            (_Sources(local_ids=["event-1"]), _turn("grounded_local_context"), None),
            (_Sources(local_ids=["event-1"]), None, "event-1"),
        ):
            with self.subTest(turn=turn, event_id=event_id):
                self.assertFalse(answered_from_stored_event(sources, turn, event_id))

    def test_a_project_answer_must_cite_the_tracked_record(self) -> None:
        self.assertTrue(
            answered_from_project_record(_Sources(record_ids=["task-1"]), _turn("project_answer"), "task-1")
        )
        self.assertFalse(
            answered_from_project_record(_Sources(record_ids=["task-2"]), _turn("project_answer"), "task-1")
        )
        self.assertFalse(
            answered_from_project_record(_Sources(record_ids=["task-1"]), _turn("project_answer"), None)
        )


if __name__ == "__main__":
    unittest.main()
