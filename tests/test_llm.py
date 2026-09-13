import json
import unittest
from types import SimpleNamespace

from captureflow.llm import InvalidStructuredOutput, ModelSettings, OpenAICompatibleModel

_REPLY_ONLY = '{"intent":"reply_only","reply":"I can only save local records."}'
_MESSAGE = {"message_id": "message-1", "role": "user", "content": "What is the weather?"}


class RecordingCompletions:
    def __init__(self, content: str = _REPLY_ONLY) -> None:
        self.kwargs = None
        self.calls = 0
        self.content = content

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content
                    )
                )
            ]
        )


class OpenAICompatibleModelTests(unittest.TestCase):
    def test_request_omits_temperature_for_reasoning_model_compatibility(self) -> None:
        completions = RecordingCompletions()
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        decision = model.propose_conversation_decision(
            current_message=_MESSAGE,
            recent_messages=[_MESSAGE],
            candidate_events=[],
            candidate_saved_items=[],
            candidate_project_records=[],
            reference_date="2026-09-12",
            timezone="America/Vancouver",
        )

        self.assertEqual(decision.intent, "reply_only")
        self.assertNotIn("temperature", completions.kwargs)
        self.assertEqual(completions.kwargs["timeout"], 30)

    def test_output_outside_the_schema_is_reported_as_invalid(self) -> None:
        completions = RecordingCompletions('{"intent":"send_email_SECRET_VALUE"}')
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with self.assertRaises(InvalidStructuredOutput) as raised:
            model.propose_conversation_decision(
                current_message=_MESSAGE,
                recent_messages=[_MESSAGE],
                candidate_events=[],
                candidate_saved_items=[],
                candidate_project_records=[],
                reference_date="2026-09-12",
                timezone="America/Vancouver",
            )

        # The failure names the field and rule for diagnosis, but never echoes model output.
        self.assertIn("intent:", str(raised.exception))
        self.assertNotIn("SECRET_VALUE", str(raised.exception))

    def test_contextual_router_receives_the_layered_context_once(self) -> None:
        completions = RecordingCompletions(
            '{"intent":"calendar_create","action":{"action":"create_event",'
            '"content":"Algorithm & Code Review","description":"讨论前缀树与最小堆。",'
            '"date":"2026-09-10","start_time":"16:00","end_time":"17:00",'
            '"source_message_ids":["message-1"]},"project_changes":[],"reply":null}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        recent_messages = [
            {"message_id": "message-0", "role": "user", "reference": "long message, 900 characters, starting: 会议纪要…"},
            {"message_id": "message-1", "role": "user", "content": "我这周四下午四点到五点有算法优化与代码审查会议。"},
        ]

        decision = model.propose_conversation_decision(
            current_message=recent_messages[1],
            recent_messages=recent_messages,
            candidate_events=["event-1 | Existing meeting | 2026-09-10 10:00"],
            candidate_saved_items=["idea-1 | idea | An idea."],
            candidate_project_records=["record-1 | task | 排查延迟 | active"],
            truncated_candidate_lists=["events"],
            reference_date="2026-09-09",
            timezone="America/Vancouver",
        )

        prompt = completions.kwargs["messages"][0]["content"]
        payload = json.loads(completions.kwargs["messages"][1]["content"])["input"]
        self.assertEqual(decision.intent, "calendar_create")
        self.assertEqual(decision.action.source_message_ids, ["message-1"])
        self.assertEqual(completions.calls, 1)
        for rule in (
            "Stored state is current",
            "Only current_message can ask for a write",
            "It is found when the current conversation mentioned it",
            "An item that appeared only in an older, different conversation is not found",
            "clarify is a plain question in reply",
            "truncated_candidate_lists",
            "source_message_ids",
            "source_message_id",
            "in your own short words",
            "confirm a deletion",
            "calendar table",
            "never write a UTC offset",
            "due_date is YYYY-MM-DD",
            "mutually exclusive",
            "never kind, content, or rationale",
            "is transition_record, not a new record",
            "follow_up_question",
            "reference_time, when supplied, is the current local time",
        ):
            with self.subTest(rule=rule):
                self.assertIn(rule, prompt)
        for retired in ("pending", "deferred_action", "EvidenceReference", "_expression"):
            with self.subTest(retired=retired):
                self.assertNotIn(retired, prompt)
        self.assertEqual(payload["recent_messages"], recent_messages)
        self.assertEqual(payload["candidate_events"], ["event-1 | Existing meeting | 2026-09-10 10:00"])
        self.assertEqual(payload["candidate_project_records"], ["record-1 | task | 排查延迟 | active"])
        self.assertEqual(payload["truncated_candidate_lists"], ["events"])
        self.assertNotIn("pending_clarification", payload)
        self.assertEqual(payload["calendar"], [])
        self.assertIsNone(payload["reference_time"])

    def test_contextual_router_keeps_a_finished_meeting_as_a_note(self) -> None:
        content = "今天中午十二点的会议是关于系统架构与驱动集成，主要分析了驱动调试。"
        completions = RecordingCompletions(
            '{"intent":"save_note","action":{"action":"save_note",'
            '"content":"系统架构与驱动集成会议：主要分析了驱动调试。",'
            '"description":null,"date":null,"start_time":null,'
            '"end_time":null,"duration_minutes":null,'
            '"target_event_id":null,"source_message_ids":["minutes-message"]},'
            '"project_changes":[],"reply":null}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        message = {"message_id": "minutes-message", "role": "user", "content": content}

        decision = model.propose_conversation_decision(
            current_message=message,
            recent_messages=[message],
            candidate_events=[],
            candidate_saved_items=[],
            candidate_project_records=[],
            reference_date="2026-09-11",
            timezone="America/Vancouver",
        )

        prompt = completions.kwargs["messages"][0]["content"]
        self.assertEqual(decision.intent, "save_note")
        self.assertEqual(decision.action.action, "save_note")
        self.assertIn("already held or finished", prompt)
        self.assertIn("Keep its minutes with save_note", prompt)
        self.assertIn("still going to happen", prompt)

    def test_contextual_router_accepts_task_not_calendar_decision(self) -> None:
        completions = RecordingCompletions(
            '{"intent":"tracker_change","action":null,"project_changes":['
            '{"action":"create_record","kind":"task","content":"完成代码审查。",'
            '"rationale":null,"source_quote":"我这周四需要完成代码审查。",'
            '"target_record_id":null,"target_state":null}],"reply":null}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        message = {
            "message_id": "task-message",
            "role": "user",
            "content": "我这周四需要完成代码审查。",
        }

        decision = model.propose_conversation_decision(
            current_message=message,
            recent_messages=[message],
            candidate_events=[],
            candidate_saved_items=[],
            candidate_project_records=[],
            reference_date="2026-09-09",
            timezone="America/Vancouver",
        )

        self.assertEqual(decision.intent, "tracker_change")
        self.assertEqual(decision.project_changes[0].kind, "task")
        self.assertEqual(completions.calls, 1)

