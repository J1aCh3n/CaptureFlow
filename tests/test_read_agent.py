import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.models import CalendarEvent, Conversation, Message
from captureflow.read_agent import (
    RESPOND_TOOL,
    AgentFinalResponse,
    AgentModelTurn,
    AgentToolCall,
    ReadAgentLoop,
    ReadAgentUsage,
)
from captureflow.read_agent_tools import BoundReadTools
from captureflow.storage import CaptureStore


class ScriptedReadAgentModel:
    def __init__(self, turns: list[AgentModelTurn]) -> None:
        self.turns = list(turns)
        self.requests = []

    def next_read_agent_turn(self, messages, tools) -> AgentModelTurn:
        self.requests.append((messages, tools))
        return self.turns.pop(0)


class ReadAgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.conversation = self.store.save_conversation(Conversation(id="agent-chat", title="Agent"))
        source = self.store.save_message(
            Message(id="source", conversation_id=self.conversation.id, role="user", content="Review Friday.")
        )
        self.event = self.store.save_calendar_event(
            CalendarEvent(
                id="review-event",
                conversation_id=self.conversation.id,
                title="Design review",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=date(2026, 9, 18),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        self.tools = BoundReadTools.for_conversation(
            self.store, self.conversation.id, "local_answer"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_tool_result_returns_to_model_and_grounded_final_is_accepted(self) -> None:
        model = ScriptedReadAgentModel(
            [
                self._calendar_call(),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        response_kind="grounded_answer",
                        reply="The design review is Friday.",
                        source_ids=[self.event.id],
                    )
                ),
            ]
        )

        run = ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "What is on Friday?"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.tool_calls, 1)
        self.assertEqual(run.final_response.source_ids, [self.event.id])
        second_messages = model.requests[1][0]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertIn(self.event.id, second_messages[-1]["content"])

    def test_the_model_is_offered_respond_for_its_final_answer_instead_of_a_json_schema(self) -> None:
        model = ScriptedReadAgentModel([AgentModelTurn(final_response=AgentFinalResponse(response_kind="no_match"))])

        ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "Anything?"},
            tools=self.tools,
        )

        messages, tools = model.requests[0]
        respond = [tool for tool in tools if tool["function"]["name"] == RESPOND_TOOL]
        self.assertEqual(len(respond), 1)
        self.assertEqual(respond[0]["function"]["parameters"], AgentFinalResponse.model_json_schema())
        self.assertEqual(
            {tool["function"]["name"] for tool in tools} - {RESPOND_TOOL},
            {tool["function"]["name"] for tool in self.tools.definitions_for_model()},
        )
        self.assertIn(RESPOND_TOOL, messages[0]["content"])
        self.assertNotIn('"properties"', messages[0]["content"])

    def test_unobserved_source_gets_one_repair_then_succeeds(self) -> None:
        model = ScriptedReadAgentModel(
            [
                self._calendar_call(),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        response_kind="grounded_answer", source_ids=["invented-id"]
                    )
                ),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        response_kind="grounded_answer", source_ids=[self.event.id]
                    )
                ),
            ]
        )

        run = ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "What is on Friday?"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual(run.model_calls, 3)
        self.assertIn("no tool returned", model.requests[2][0][-1]["content"])

    def test_repeated_unobserved_source_fails_the_whole_turn(self) -> None:
        invented = AgentModelTurn(
            final_response=AgentFinalResponse(
                response_kind="grounded_answer", source_ids=["invented-id"]
            )
        )
        model = ScriptedReadAgentModel([invented, invented])

        run = ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "What is stored?"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "validation_failed")
        self.assertIsNone(run.final_response)
        self.assertEqual(run.model_calls, 2)

    def test_citation_repair_cannot_observe_an_invented_id_after_the_fact(self) -> None:
        model = ScriptedReadAgentModel(
            [
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        response_kind="grounded_answer", source_ids=["invented-id"]
                    )
                ),
                self._calendar_call(),
            ]
        )

        run = ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "What is stored?"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "validation_failed")
        self.assertEqual(run.tool_calls, 0)
        self.assertEqual(run.observed_resources, {})

    def test_policy_denial_stops_without_returning_control_to_model(self) -> None:
        model = ScriptedReadAgentModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(
                            id="call-secret",
                            name="search_saved_items",
                            arguments={"keywords": ["api key is sk-abcdefghijklmnop"]},
                        )
                    ]
                )
            ]
        )

        run = ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "Find it"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "policy_denied")
        self.assertEqual(run.model_calls, 1)
        self.assertEqual(run.tool_calls, 1)

    def test_tool_budget_is_enforced_before_a_batch_executes(self) -> None:
        model = ScriptedReadAgentModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(id=f"call-{index}", name="search_saved_items", arguments={})
                        for index in range(3)
                    ]
                )
            ]
        )

        run = ReadAgentLoop(model, max_tool_calls=2).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "Find notes"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "tool_budget_exhausted")
        self.assertEqual(run.tool_calls, 0)
        self.assertEqual(run.observed_resources, {})

    def test_model_budget_stops_a_retryable_tool_loop(self) -> None:
        malformed_call = AgentModelTurn(
            tool_calls=[
                AgentToolCall(
                    id="bad-range",
                    name="search_calendar",
                    arguments={"start_date": "2026-10-01", "end_date": "2026-09-01"},
                )
            ]
        )
        model = ScriptedReadAgentModel([malformed_call])

        run = ReadAgentLoop(model, max_model_calls=1).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "Find events"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "model_budget_exhausted")
        self.assertEqual(run.tool_calls, 1)

    def test_a_run_that_raises_still_reports_what_it_spent(self) -> None:
        class FailingSecondCall(ScriptedReadAgentModel):
            def next_read_agent_turn(self, messages, tools) -> AgentModelTurn:
                if self.requests:
                    self.requests.append((messages, tools))
                    raise RuntimeError("provider failed")
                return super().next_read_agent_turn(messages, tools)

        usage = ReadAgentUsage()

        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            ReadAgentLoop(FailingSecondCall([self._calendar_call()])).run(
                system_prompt="Answer only from tool results.",
                user_input={"question": "Friday?"},
                tools=self.tools,
                usage=usage,
            )

        self.assertEqual(usage.model_calls, 2)
        self.assertEqual(usage.tool_calls, 1)
        self.assertGreater(usage.model_seconds, 0)

    def _calendar_call(self) -> AgentModelTurn:
        return AgentModelTurn(
            tool_calls=[
                AgentToolCall(
                    id="calendar-call",
                    name="search_calendar",
                    arguments={"start_date": "2026-09-18", "end_date": "2026-09-18"},
                )
            ]
        )
