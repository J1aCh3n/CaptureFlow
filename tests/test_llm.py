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
        completions = RecordingCompletions('{"intent":"send_email"}')
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with self.assertRaises(InvalidStructuredOutput):
            model.propose_conversation_decision(
                current_message=_MESSAGE,
                recent_messages=[_MESSAGE],
                candidate_events=[],
                candidate_saved_items=[],
                candidate_project_records=[],
                reference_date="2026-09-12",
                timezone="America/Vancouver",
            )

    def test_contextual_router_receives_bounded_context_and_candidates_once(self) -> None:
        completions = RecordingCompletions(
            '{"intent":"calendar_create","action":{"action":"create_event",'
            '"content":"Algorithm & Code Review","description":"讨论前缀树与最小堆。",'
            '"date":"2026-09-10","start_time":"16:00",'
            '"end_time":"17:00","target_event_id":null,'
            '"evidence":[{"message_id":"message-1",'
            '"source_quote":"我这周四下午四点到五点有算法优化与代码审查会议。"}]},'
            '"project_changes":[],"clarification_question":null,'
            '"pending_action":null,"reply":null}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        recent_messages = [
            {
                "message_id": "message-1",
                "role": "user",
                "content": "我这周四下午四点到五点有算法优化与代码审查会议。",
            }
        ]

        decision = model.propose_conversation_decision(
            current_message=recent_messages[0],
            recent_messages=recent_messages,
            candidate_events=[{"id": "event-1", "title": "Existing meeting"}],
            candidate_saved_items=[{"id": "idea-1", "kind": "idea", "content": "An idea."}],
            candidate_project_records=[{"id": "record-1", "kind": "task", "state": "active"}],
            reference_date="2026-09-09",
            timezone="America/Vancouver",
        )

        prompt = completions.kwargs["messages"][0]["content"]
        payload = json.loads(completions.kwargs["messages"][1]["content"])["input"]
        self.assertEqual(decision.intent, "calendar_create")
        self.assertEqual(decision.action.evidence[0].message_id, "message-1")
        self.assertEqual(completions.calls, 1)
        self.assertIn("calendar_create", prompt)
        self.assertIn("tracker_change", prompt)
        self.assertIn("exactly one supplied candidate_events ID", prompt)
        self.assertIn("duration_minutes", prompt)
        self.assertIn("calendar table", prompt)
        self.assertIn("never write a UTC offset", prompt)
        self.assertIn("due_date is YYYY-MM-DD", prompt)
        self.assertNotIn("_expression", prompt)
        self.assertIn("mutually exclusive", prompt)
        self.assertIn("deferred_action", prompt)
        self.assertIn("never kind, content, or rationale", prompt)
        self.assertIn("is transition_record, not a new record", prompt)
        self.assertIn("One turn commits exactly one local action", prompt)
        self.assertIn("must always quote current_message exactly", prompt)
        self.assertEqual(payload["current_message"]["message_id"], "message-1")
        self.assertEqual(payload["recent_messages"], recent_messages)
        self.assertEqual(payload["candidate_events"][0]["id"], "event-1")
        self.assertEqual(payload["candidate_saved_items"][0]["id"], "idea-1")
        self.assertEqual(payload["candidate_project_records"][0]["id"], "record-1")
        self.assertEqual(payload["calendar"], [])
        self.assertIsNone(payload["reference_time"])
        self.assertIn("reference_time, when supplied, is the current local time", prompt)

    def test_contextual_router_keeps_a_finished_meeting_as_a_note(self) -> None:
        content = "今天中午十二点的会议是关于系统架构与驱动集成，主要分析了驱动调试。"
        completions = RecordingCompletions(
            '{"intent":"save_note","action":{"action":"save_note",'
            '"content":"系统架构与驱动集成会议：主要分析了驱动调试。",'
            '"description":null,"date":null,"start_time":null,'
            '"end_time":null,"duration_minutes":null,'
            '"target_event_id":null,"evidence":[{"message_id":"minutes-message",'
            f'"source_quote":"{content}"}}]}},"project_changes":[],'
            '"clarification_question":null,"pending_action":null,"reply":null}'
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
            '"target_record_id":null,"target_state":null}],'
            '"clarification_question":null,"pending_action":null,"reply":null}'
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

    def test_contextual_router_rejects_an_event_outside_its_candidates(self) -> None:
        completions = RecordingCompletions(
            '{"intent":"calendar_update","action":{"action":"update_event",'
            '"content":null,"description":null,"date":null,'
            '"start_time":"14:00","end_time":null,'
            '"target_event_id":"foreign-event","evidence":[{"message_id":"message-1",'
            '"source_quote":"这个会议改到下午两点。"}]},"project_changes":[],'
            '"clarification_question":null,"pending_action":null,"reply":null}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        message = {
            "message_id": "message-1",
            "role": "user",
            "content": "这个会议改到下午两点。",
        }

        with self.assertRaisesRegex(InvalidStructuredOutput, "outside the supplied candidates"):
            model.propose_conversation_decision(
                current_message=message,
                recent_messages=[message],
                candidate_events=[{"id": "event-1", "title": "Known meeting"}],
                candidate_saved_items=[],
                candidate_project_records=[],
                reference_date="2026-09-09",
                timezone="America/Vancouver",
            )

        self.assertEqual(completions.calls, 1)
