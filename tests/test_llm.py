import hashlib
import json
import unittest
from types import SimpleNamespace

from captureflow.llm import InvalidStructuredOutput, ModelSettings, OpenAICompatibleModel

_ANSWER_AS_TEXT = '{"reply":"No matching records were found.","source_ids":[]}'


class RecordingCompletions:
    def __init__(self, content: str = _ANSWER_AS_TEXT) -> None:
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


class _AgentCompletions(RecordingCompletions):
    """Return one message whose tool calls are the given (name, raw arguments) pairs."""

    def __init__(self, calls) -> None:
        super().__init__()
        self.calls_to_return = calls

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        tool_calls = [
            SimpleNamespace(id=f"call-{index}", function=SimpleNamespace(name=name, arguments=arguments))
            for index, (name, arguments) in enumerate(self.calls_to_return)
        ]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))])


def _agent_model(completions) -> OpenAICompatibleModel:
    model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
    model.settings = ModelSettings(api_key="test-key", model="qwen-flash")
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return model


class OpenAICompatibleModelTests(unittest.TestCase):
    def test_read_agent_normalizes_tool_calls(self) -> None:
        completions = RecordingCompletions()
        completions.create = lambda **kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="search_calendar",
                                    arguments='{"start_date":"2026-09-18","end_date":"2026-09-18"}',
                                ),
                            )
                        ],
                    )
                )
            ]
        )
        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="qwen-flash")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        turn = model.next_read_agent_turn(
            messages=[{"role": "user", "content": "Friday?"}],
            tools=[{"type": "function", "function": {"name": "search_calendar"}}],
        )

        self.assertEqual(turn.tool_calls[0].name, "search_calendar")
        self.assertEqual(turn.tool_calls[0].arguments["start_date"], "2026-09-18")

    def test_read_agent_takes_its_final_response_from_the_respond_call(self) -> None:
        completions = _AgentCompletions(
            [("respond", '{"reply":"Friday.","source_ids":["event-1"]}')]
        )

        turn = _agent_model(completions).next_read_agent_turn(messages=[], tools=[])

        self.assertEqual(turn.tool_calls, [])
        self.assertEqual(turn.final_response.source_ids, ["event-1"])
        # The answer arrives as tool arguments, so the request no longer asks for JSON content.
        self.assertNotIn("response_format", completions.kwargs)

    def test_read_agent_refuses_respond_called_together_with_a_search(self) -> None:
        completions = _AgentCompletions(
            [
                ("search_calendar", '{"start_date":"2026-09-18","end_date":"2026-09-18"}'),
                ("respond", '{"reply":"No matching records were found.","source_ids":[]}'),
            ]
        )

        with self.assertRaisesRegex(InvalidStructuredOutput, "respond"):
            _agent_model(completions).next_read_agent_turn(messages=[], tools=[])

    def test_read_agent_refuses_invalid_respond_arguments(self) -> None:
        for arguments in (
            '{"reply":null,"source_ids":[]}',
            # The failure seen on qwen3.7-flash-2026-07-15: the schema echoed instead of an answer.
            '{"type":"object","title":"AgentFinalResponse","properties":{},"required":["response_kind"]}',
            '{"reply":"Incomplete"',
        ):
            with self.subTest(arguments=arguments), self.assertRaises(InvalidStructuredOutput):
                _agent_model(_AgentCompletions([("respond", arguments)])).next_read_agent_turn(messages=[], tools=[])

    def test_read_agent_refuses_an_answer_written_as_content_instead_of_respond(self) -> None:
        completions = RecordingCompletions('{"reply":"No matching records were found.","source_ids":[]}')

        with self.assertRaisesRegex(InvalidStructuredOutput, "respond"):
            _agent_model(completions).next_read_agent_turn(messages=[], tools=[])

    def test_dashscope_read_agent_keeps_search_tool_selection_optional(self) -> None:
        completions = _AgentCompletions([("respond", '{"reply":"No matching records were found.","source_ids":[]}')])
        model = _agent_model(completions)
        model.settings = ModelSettings(
            api_key="test-key",
            model="qwen3.8-flash",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        model.next_read_agent_turn(messages=[], tools=[
            {"type": "function", "function": {"name": "search_calendar"}},
            {"type": "function", "function": {"name": "respond"}},
        ])

        self.assertEqual(completions.kwargs["extra_body"], {"enable_thinking": False})
        self.assertNotIn("tool_choice", completions.kwargs)

    def test_read_agent_requests_a_respond_call_when_only_finishing_is_available(self) -> None:
        completions = _AgentCompletions([("respond", '{"reply":"The review is scheduled.","source_ids":["event-1"]}')])
        model = _agent_model(completions)
        model.settings = ModelSettings(
            api_key="test-key", model="qwen3.8-flash", base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

        turn = model.next_read_agent_turn(messages=[], tools=[{"type": "function", "function": {"name": "respond"}}])

        self.assertEqual(completions.kwargs["tool_choice"], {"type": "function", "function": {"name": "respond"}})
        self.assertEqual(completions.kwargs["extra_body"], {"enable_thinking": False})
        self.assertEqual(turn.final_response.source_ids, ["event-1"])

    def test_request_omits_temperature_for_reasoning_model_compatibility(self) -> None:
        completions = _AgentCompletions([("respond", '{"reply":"Nothing is stored.","source_ids":[]}')])
        model = _agent_model(completions)
        model.settings = ModelSettings(api_key="test-key", model="gpt-5.6-terra")

        turn = model.next_read_agent_turn(messages=[], tools=[])

        self.assertEqual(turn.final_response.reply, "Nothing is stored.")
        self.assertNotIn("temperature", completions.kwargs)
        self.assertEqual(completions.kwargs["timeout"], 30)

    def test_output_outside_the_schema_is_reported_as_invalid(self) -> None:
        completions = _AgentCompletions([("respond", '{"reply":"ok","source_ids":"SECRET_VALUE"}')])

        with self.assertRaises(InvalidStructuredOutput) as raised:
            _agent_model(completions).next_read_agent_turn(messages=[], tools=[])

        # The failure names the field and rule for diagnosis, but never echoes model output.
        self.assertIn("source_ids", str(raised.exception))
        self.assertNotIn("SECRET_VALUE", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
