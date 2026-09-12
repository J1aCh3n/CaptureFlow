import json
import unittest
from types import SimpleNamespace

from captureflow.llm import InvalidStructuredOutput, ModelSettings, OpenAICompatibleModel
from captureflow.models import CapturePlan, OrganizerDraft, Project


class RecordingCompletions:
    def __init__(self, content: str | None = None) -> None:
        self.kwargs = None
        self.calls = 0
        self.content = content or (
            '{"steps":[{"id":"task","action":"extract_task",'
            '"source_span":"Buy milk.","reason":"Explicit task."}]}'
        )

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

        plan = model.plan("Buy milk.")

        self.assertEqual(plan.steps[0].action, "extract_task")
        self.assertNotIn("temperature", completions.kwargs)
        self.assertEqual(completions.kwargs["timeout"], 30)

    def test_turn_router_uses_the_bounded_action_prompt(self) -> None:
        completions = RecordingCompletions('{"route":"reply_only","actions":[],"clarification_question":null,"pending_action":null,"reply":"I can save local records."}')
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        decision = model.route_turn("What is the weather?", "2026-09-08", "America/Vancouver")

        self.assertEqual(decision.route, "reply_only")
        prompt = completions.kwargs["messages"][0]["content"]
        self.assertIn("This is not a general chatbot", prompt)
        self.assertIn("exact, continuous source_quote", prompt)
        self.assertIn("definite first-person or subjectless statement", prompt)
        self.assertIn("rather than creating separate notes", prompt)
        self.assertIn("end_time_expression", prompt)
        self.assertIn("Never invent a default duration", prompt)

    def test_contextual_router_receives_bounded_context_and_candidates_once(self) -> None:
        completions = RecordingCompletions(
            '{"intent":"calendar_create","action":{"action":"create_event",'
            '"content":"Algorithm & Code Review","description":"讨论前缀树与最小堆。",'
            '"date_expression":"这周四","time_expression":"下午四点",'
            '"end_time_expression":"下午五点","target_event_id":null,'
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
        self.assertIn("duration_expression", prompt)
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

    def test_contextual_router_keeps_a_finished_meeting_as_a_note(self) -> None:
        content = "今天中午十二点的会议是关于系统架构与驱动集成，主要分析了驱动调试。"
        completions = RecordingCompletions(
            '{"intent":"save_note","action":{"action":"save_note",'
            '"content":"系统架构与驱动集成会议：主要分析了驱动调试。",'
            '"description":null,"date_expression":null,"time_expression":null,'
            '"end_time_expression":null,"duration_expression":null,'
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
            '"content":null,"description":null,"date_expression":null,'
            '"time_expression":"14:00","end_time_expression":null,'
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

    def test_project_responder_receives_only_bounded_project_context(self) -> None:
        completions = RecordingCompletions(
            '{"reply":"That is a follow-up.","turn_kind":"follow_up",'
            '"clarification_question":null}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        response = model.respond_to_project_turn(
            Project(id="project-1", name="CaptureFlow"),
            [
                {"role": "user", "content": "We kept SQLite."},
                {"role": "user", "content": "Why?"},
            ],
            False,
        )

        system_prompt = completions.kwargs["messages"][0]["content"]
        payload = json.loads(completions.kwargs["messages"][1]["content"])["input"]
        self.assertEqual(response.turn_kind, "follow_up")
        self.assertIn("cannot create, update, complete, delete", system_prompt)
        self.assertEqual(payload["project"]["name"], "CaptureFlow")
        self.assertEqual(payload["recent_messages"][-1]["content"], "Why?")
        self.assertFalse(payload["context_truncated"])

    def test_project_tracker_proposer_receives_only_supplied_candidate_records(self) -> None:
        completions = RecordingCompletions(
            '{"reply":"I will propose the completion.","turn_kind":"follow_up",'
            '"clarification_question":null,"changes":[{"action":"transition_record",'
            '"kind":null,"content":null,"rationale":null,"source_quote":"It is complete.",'
            '"target_record_id":"record-1","target_state":"completed"}]}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        response = model.propose_project_changes(
            Project(id="project-1", name="CaptureFlow"),
            [{"role": "user", "content": "It is complete."}],
            [{"id": "record-1", "kind": "task", "content": "Phase 6.", "state": "active"}],
            False,
        )

        system_prompt = completions.kwargs["messages"][0]["content"]
        payload = json.loads(completions.kwargs["messages"][1]["content"])["input"]
        self.assertEqual(response.changes[0].target_record_id, "record-1")
        self.assertIn("only propose", system_prompt)
        self.assertIn("Never invent a target ID", system_prompt)
        self.assertEqual(payload["candidate_records"][0]["id"], "record-1")

    def test_project_retrieval_responder_receives_bounded_records_and_requires_sources(self) -> None:
        completions = RecordingCompletions(
            '{"reply":"SQLite is the current local decision.","answer_kind":"history",'
            '"supporting_record_ids":["record-1"]}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        response = model.answer_project_query(
            Project(id="project-1", name="CaptureFlow"),
            [{"role": "user", "content": "Why SQLite?"}],
            [{"id": "record-1", "kind": "decision", "content": "Use SQLite.", "state": "active"}],
            "history",
            False,
        )

        system_prompt = completions.kwargs["messages"][0]["content"]
        payload = json.loads(completions.kwargs["messages"][1]["content"])["input"]
        self.assertEqual(response.supporting_record_ids, ["record-1"])
        self.assertIn("supporting_record_ids", system_prompt)
        self.assertIn("do not treat a superseded record as current", system_prompt)
        self.assertEqual(payload["retrieved_records"][0]["id"], "record-1")

    def test_local_context_responder_receives_only_selected_local_evidence(self) -> None:
        completions = RecordingCompletions(
            '{"outcome":"answer","selected_event_ids":["event-1"],'
            '"selected_item_ids":["note-1"],"summary_points":[{"text":"Use voting.",'
            '"source_ids":["note-1"]}],"preparation_suggestions":[],'
            '"missing_information":["preparation"],"clarification_question":null}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        response = model.answer_local_context(
            "What time is my meeting tomorrow?",
            {
                "reference_date": "2026-09-08",
                "timezone": "America/Vancouver",
                "target_date": "2026-09-09",
            },
            [{"source_id": "event-1", "title": "Workflow meeting", "start_at": "2026-09-09"}],
            [{"source_id": "note-1", "kind": "note", "content": "Use voting."}],
            [
                {
                    "source_id": "record-1",
                    "kind": "decision",
                    "content": "Use voting and idea merging.",
                    "state": "active",
                }
            ],
        )

        system_prompt = completions.kwargs["messages"][0]["content"]
        payload = json.loads(completions.kwargs["messages"][1]["content"])["input"]
        self.assertEqual(response.selected_event_ids, ["event-1"])
        self.assertIn("cannot access an external calendar", system_prompt)
        self.assertIn("must not create or change", system_prompt)
        self.assertEqual(payload["runtime_context"]["target_date"], "2026-09-09")
        self.assertEqual(payload["events"][0]["source_id"], "event-1")
        self.assertEqual(payload["saved_items"][0]["source_id"], "note-1")
        self.assertEqual(payload["project_records"][0]["source_id"], "record-1")
        self.assertIn("Only a Calendar event establishes", system_prompt)

    def test_coordinator_prompt_distinguishes_memory_from_tasks(self) -> None:
        completions = RecordingCompletions(
            '{"steps":[{"id":"memory","action":"inspect_memory",'
            '"source_span":"I decided to use PostgreSQL.",'
            '"reason":"A durable user decision."}]}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        plan = model.plan("I decided to use PostgreSQL.")

        system_prompt = completions.kwargs["messages"][0]["content"]
        self.assertEqual(plan.steps[0].action, "inspect_memory")
        self.assertIn(
            "durable user-specific preferences, decisions, facts, or ideas", system_prompt
        )
        self.assertIn("is not by itself a task", system_prompt)
        self.assertIn("Select both actions only", system_prompt)
        self.assertIn("third party in a news article, report, quote", system_prompt)
        self.assertIn("not a user task", system_prompt)
        self.assertIn("Duplicate checking happens downstream", system_prompt)
        self.assertIn("Never return an empty step list", system_prompt)

    def test_invalid_plan_preserves_raw_output_for_bounded_repair(self) -> None:
        raw_output = (
            '{"steps":[],"user_clarification_required":false,'
            '"clarification_question":null}'
        )
        completions = RecordingCompletions(raw_output)
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with self.assertRaises(InvalidStructuredOutput) as raised:
            model.plan("I prefer concise meeting notes.")

        self.assertEqual(raised.exception.raw_output, raw_output)

    def test_critic_does_not_treat_summary_usefulness_as_grounding(self) -> None:
        completions = RecordingCompletions('{"verdict":"pass","feedback":[]}')
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        draft = OrganizerDraft.model_validate(
            {
                "summary_claims": [
                    {"text": "A concise restatement.", "source_quote": "A source sentence."}
                ]
            }
        )

        report = model.critique("A source sentence.", draft)

        system_prompt = completions.kwargs["messages"][0]["content"]
        self.assertEqual(report.verdict, "pass")
        self.assertIn("Do not judge usefulness", system_prompt)
        self.assertNotIn("merely restates", system_prompt)

    def test_organizer_receives_idea_synthesis_boundary(self) -> None:
        completions = RecordingCompletions(
            '{"summary_claims":[{"text":"A product idea.",'
            '"source_quote":"A product idea."}],"tasks":[],"review_notes":[]}'
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        plan = CapturePlan.model_validate(
            {
                "steps": [
                    {
                        "id": "summary",
                        "action": "summarize",
                        "source_span": "A product idea.",
                        "reason": "Summarize the idea.",
                    }
                ]
            }
        )

        model.organize("A product idea.", plan)

        system_prompt = completions.kwargs["messages"][0]["content"]
        self.assertIn("input-to-outcome flow", system_prompt)
        self.assertIn("not committed implementation", system_prompt)
        self.assertIn("do not turn an idea into a task", system_prompt)
