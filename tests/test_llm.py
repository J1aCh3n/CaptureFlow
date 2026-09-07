import unittest
from types import SimpleNamespace

from captureflow.llm import InvalidStructuredOutput, ModelSettings, OpenAICompatibleModel
from captureflow.models import CapturePlan, OrganizerDraft


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
