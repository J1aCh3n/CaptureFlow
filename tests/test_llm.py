import unittest
from types import SimpleNamespace

from captureflow.llm import ModelSettings, OpenAICompatibleModel


class RecordingCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"steps":[{"id":"task","action":"extract_task",'
                            '"source_span":"Buy milk.","reason":"Explicit task."}]}'
                        )
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
