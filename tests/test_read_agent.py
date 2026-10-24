import tempfile
import unittest
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from captureflow.llm import InvalidStructuredOutput
from captureflow.models import CalendarEvent, Conversation, Message, SavedItem
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
    def __init__(self, turns) -> None:
        self.turns = list(turns)
        self.requests = []

    def next_read_agent_turn(self, messages, tools) -> AgentModelTurn:
        self.requests.append((messages, tools))
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


class AgentFinalResponseTests(unittest.TestCase):
    def test_a_reply_needs_no_answer_kind_or_sources(self) -> None:
        response = AgentFinalResponse(reply="  Which meeting do you mean?  ")

        self.assertEqual(response.reply, "Which meeting do you mean?")
        self.assertEqual(response.source_ids, [])

    def test_sources_are_deduplicated_without_truncating_the_answer(self) -> None:
        ids = [f"event-{index}" for index in range(25)]
        response = AgentFinalResponse(reply="These are the stored events.", source_ids=[*ids, ids[0]])

        self.assertEqual(response.source_ids, ids)

    def test_an_empty_reply_is_not_a_completed_answer(self) -> None:
        for reply in ("", "   "):
            with self.subTest(reply=reply), self.assertRaises(ValidationError):
                AgentFinalResponse(reply=reply)


class _StubWriter:
    """Stands in for the router's write path without touching storage."""

    def __init__(self) -> None:
        self.saved: list[str] = []

    @property
    def wrote(self) -> bool:
        return bool(self.saved)

    def save_item(self, kind: str, content: str) -> SavedItem:
        self.saved.append(content)
        return SavedItem(
            id=f"stub-{kind}-{len(self.saved)}",
            conversation_id="agent-chat",
            kind=kind,
            content=content,
            source_message_id="source",
            source_quote="Review Friday.",
        )


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
        self.tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_tool_result_returns_to_model_and_grounded_final_is_accepted(self) -> None:
        model = ScriptedReadAgentModel(
            [
                self._calendar_call(),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
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
        model = ScriptedReadAgentModel([AgentModelTurn(final_response=AgentFinalResponse(reply="No matching data was found."))])

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

    def test_unobserved_source_can_be_corrected_in_the_remaining_budget(self) -> None:
        model = ScriptedReadAgentModel(
            [
                self._calendar_call(),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        reply="The stored records answer the question.", source_ids=["invented-id"]
                    )
                ),
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        reply="The stored records answer the question.", source_ids=[self.event.id]
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
        self.assertTrue(any("no tool returned" in (message.get("content") or "") for message in model.requests[2][0]))

    def test_repeated_unobserved_source_fails_the_whole_turn(self) -> None:
        invented = AgentModelTurn(
            final_response=AgentFinalResponse(
                reply="The stored records answer the question.", source_ids=["invented-id"]
            )
        )
        model = ScriptedReadAgentModel([invented, invented, invented])

        run = ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "What is stored?"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "validation_failed")
        self.assertIsNone(run.final_response)
        self.assertEqual(run.model_calls, 3)

    def test_an_unobserved_citation_can_be_followed_by_searching_for_real_evidence(self) -> None:
        model = ScriptedReadAgentModel(
            [
                AgentModelTurn(
                    final_response=AgentFinalResponse(
                        reply="The design review is stored.", source_ids=[self.event.id]
                    )
                ),
                self._calendar_call(),
                AgentModelTurn(final_response=AgentFinalResponse(
                    reply="The design review is stored.", source_ids=[self.event.id]
                )),
            ]
        )

        run = ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "What is stored?"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual(run.tool_calls, 1)
        self.assertEqual(run.final_response.source_ids, [self.event.id])
        self.assertIn(self.event.id, model.requests[1][0][-1]["content"])

    def test_a_denied_search_returns_feedback_without_executing_the_search(self) -> None:
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
                ),
                AgentModelTurn(final_response=AgentFinalResponse(reply="That search was refused; I cannot confirm the result.")),
            ]
        )

        run = ReadAgentLoop(model).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "Find it"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.tool_calls, 1)
        self.assertEqual(run.observed_resources, {})
        self.assertIn("policy_denied", model.requests[1][0][-1]["content"])

    def test_an_oversized_batch_keeps_available_results_and_reports_skipped_searches(self) -> None:
        model = ScriptedReadAgentModel(
            [
                AgentModelTurn(
                    tool_calls=[
                        AgentToolCall(id=f"call-{index}", name="search_calendar", arguments={})
                        for index in range(3)
                    ]
                ),
                AgentModelTurn(final_response=AgentFinalResponse(
                    reply="The review was found, but the last search was not run.", source_ids=[self.event.id]
                )),
            ]
        )

        run = ReadAgentLoop(model, max_tool_calls=2).run(
            system_prompt="Answer only from tool results.",
            user_input={"question": "Find notes"},
            tools=self.tools,
        )

        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual(run.tool_calls, 2)
        self.assertEqual(run.model_calls, 2)
        self.assertEqual(run.final_response.source_ids, [self.event.id])
        feedback = [message for message in model.requests[1][0] if message["role"] == "tool"]
        self.assertEqual(len(feedback), 3)
        self.assertIn(self.event.id, feedback[0]["content"])
        self.assertIn("was not run", feedback[2]["content"])
        self.assertEqual([tool["function"]["name"] for tool in model.requests[1][1]], [RESPOND_TOOL])

    def test_a_search_returned_during_the_reserved_answer_call_is_not_executed(self) -> None:
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
        self.assertEqual(run.tool_calls, 0)
        self.assertEqual([tool["function"]["name"] for tool in model.requests[0][1]], [RESPOND_TOOL])

    def test_the_last_model_call_answers_using_the_results_of_two_search_rounds(self) -> None:
        model = ScriptedReadAgentModel([
            self._calendar_call(),
            AgentModelTurn(tool_calls=[AgentToolCall(id="notes", name="search_saved_items", arguments={})]),
            AgentModelTurn(final_response=AgentFinalResponse(
                reply="The review is scheduled; the notes search returned no preparation details.", source_ids=[self.event.id]
            )),
        ])

        run = ReadAgentLoop(model).run(system_prompt="Answer from tool results.", user_input={"question": "Review?"}, tools=self.tools)

        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual((run.model_calls, run.tool_calls), (3, 2))
        self.assertEqual(run.final_response.source_ids, [self.event.id])
        self.assertEqual([tool["function"]["name"] for tool in model.requests[2][1]], [RESPOND_TOOL])
        self.assertIn("unconfirmed", model.requests[2][0][-1]["content"])

    def test_invalid_tool_arguments_can_be_corrected_before_the_final_answer(self) -> None:
        model = ScriptedReadAgentModel([
            AgentModelTurn(tool_calls=[AgentToolCall(
                id="bad-range", name="search_calendar", arguments={"start_date": "2026-10-01", "end_date": "2026-09-01"}
            )]),
            self._calendar_call(),
            AgentModelTurn(final_response=AgentFinalResponse(reply="The review is scheduled.", source_ids=[self.event.id])),
        ])

        run = ReadAgentLoop(model).run(system_prompt="Answer from tool results.", user_input={"question": "Review?"}, tools=self.tools)

        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual((run.model_calls, run.tool_calls), (3, 2))
        self.assertIn("retryable", model.requests[1][0][-1]["content"])

    def test_malformed_model_output_can_be_corrected_without_a_separate_repair_state(self) -> None:
        model = ScriptedReadAgentModel([
            InvalidStructuredOutput("The model answered without calling respond."),
            self._calendar_call(),
            AgentModelTurn(final_response=AgentFinalResponse(reply="The review is scheduled.", source_ids=[self.event.id])),
        ])

        run = ReadAgentLoop(model).run(system_prompt="Answer from tool results.", user_input={"question": "Review?"}, tools=self.tools)

        self.assertEqual(run.stop_reason, "completed")
        self.assertEqual((run.model_calls, run.tool_calls), (3, 1))
        self.assertIn("could not be read", model.requests[1][0][-1]["content"])

    def test_a_malformed_final_answer_still_respects_the_model_budget(self) -> None:
        model = ScriptedReadAgentModel([{"final_response": {"reply": ""}}])

        run = ReadAgentLoop(model, max_model_calls=1).run(
            system_prompt="Answer from tool results.", user_input={"question": "Review?"}, tools=self.tools
        )

        self.assertEqual(run.stop_reason, "validation_failed")
        self.assertIsNone(run.final_response)
        self.assertEqual((run.model_calls, run.tool_calls), (1, 0))

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
