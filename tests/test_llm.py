import json
import unittest
from types import SimpleNamespace

from captureflow.llm import InvalidStructuredOutput, ModelSettings, OpenAICompatibleModel
from captureflow.models import CapturePlan, OrganizerDraft, Project


class RecordingCompletions:
    def __init__(self, content: str | None = None) -> None:
        self.kwargs = None
        self.content = content or (
            '{"steps":[{"id":"task","action":"extract_task",'
            '"source_span":"Buy milk.","reason":"Explicit task."}]}'
        )

    def create(self, **kwargs):
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
