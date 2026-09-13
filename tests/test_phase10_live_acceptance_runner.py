import unittest
from datetime import date, datetime
from types import SimpleNamespace

from captureflow.models import CalendarEvent, CalendarEventUpdateOperation
from scripts.run_phase10_live_acceptance import (
    TokenCounter,
    _validate_model_selection,
    answered_from_project_record,
    answered_from_stored_event,
    asked_without_writing,
    moved_event,
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


class _Events:
    def __init__(self, event):
        self.event = event

    def get_calendar_event(self, _event_id):
        return self.event


def _event(start_at):
    return CalendarEvent(
        id="event-1",
        conversation_id="conversation-1",
        title="算法优化与代码审查会议",
        source_message_id="message-1",
        source_quote="会议",
        start_at=start_at,
        all_day=False,
        timezone="America/Vancouver",
    )


def _update(status="committed"):
    event = _event(datetime.fromisoformat("2026-09-10T14:00:00-07:00"))
    return CalendarEventUpdateOperation(
        id="update-1",
        source_message_id="message-2",
        action_index=0,
        event_id=event.id,
        before=event,
        after=event,
        status=status,
    )


class TargetCaseTests(unittest.TestCase):
    def test_an_update_passes_only_when_committed_and_the_event_moved_as_stated(self) -> None:
        moved = _Events(_event(datetime.fromisoformat("2026-09-17T15:00:00-07:00")))

        self.assertTrue(moved_event(moved, _turn("calendar_update", (_update(),)), "event-1", day=date(2026, 9, 17)))
        self.assertTrue(moved_event(moved, _turn("calendar_update", (_update(),)), "event-1", hour=15))
        for turn, expected in (
            (_turn("calendar_update", (_update(),)), {"hour": 14}),
            (_turn("calendar_update", (_update("review"),)), {"hour": 15}),
            (_turn("clarification_created"), {"hour": 15}),
            (None, {"hour": 15}),
        ):
            with self.subTest(turn=turn, expected=expected):
                self.assertFalse(moved_event(moved, turn, "event-1", **expected))

    def test_a_question_passes_only_without_a_write_or_a_changed_event(self) -> None:
        before = _event(datetime.fromisoformat("2026-09-10T14:00:00-07:00"))

        self.assertTrue(asked_without_writing(_Events(before), _turn("clarification_created"), before))
        self.assertFalse(asked_without_writing(_Events(before), _turn("calendar_update", (_update(),)), before))
        self.assertFalse(
            asked_without_writing(
                _Events(_event(datetime.fromisoformat("2026-09-10T15:00:00-07:00"))),
                _turn("clarification_created"),
                before,
            )
        )

    def test_token_counts_are_recorded_per_request_without_content(self) -> None:
        usage = SimpleNamespace(prompt_tokens=4200, completion_tokens=900)
        completions = SimpleNamespace(create=lambda **_: SimpleNamespace(usage=usage))
        model = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=completions)))
        counter = TokenCounter(model)

        model.client.chat.completions.create(messages=[])
        model.client.chat.completions.create(messages=[])

        self.assertEqual(counter.since(0), {"input_tokens": 8400, "output_tokens": 1800})
        self.assertEqual(counter.since(2), {"input_tokens": None, "output_tokens": None})


if __name__ == "__main__":
    unittest.main()
