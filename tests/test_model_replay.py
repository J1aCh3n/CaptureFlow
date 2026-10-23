"""Recording a run once, then reproducing its effects from the recording for free.

The round trip is the point: a recording that cannot drive the same turn to the same
stored rows is not evidence of anything. These tests record from a scripted model, so
they prove the mechanism without a provider call.
"""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.conversation_router_service import ConversationRouterService
from captureflow.llm import InvalidStructuredOutput, ModelError
from captureflow.models import (
    CalendarEvent,
    Conversation,
    Message,
    Project,
    ProjectRecord,
)
from captureflow.read_agent import HAND_OFF_TOOL, AgentFinalResponse, AgentModelTurn, AgentToolCall
from captureflow.storage import CaptureStore
from scripts.model_replay import RECORDING_VERSION, RecordingModel, ReplayModel

from tests.test_conversation_router_service import ScriptedContextualModel


REFERENCE = date(2026, 9, 9)


def _scripted():
    """One agent read turn, then one write turn, as the model would answer them."""

    return ScriptedContextualModel(
        # Only the handed-back write reaches the router's decision.
        [
            {
                "intent": "save_note",
                "action": {
                    "action": "save_note",
                    "content": "Latency work starts after the review.",
                    "source_message_ids": ["replay-write"],
                },
            }
        ],
        agent_turns=[
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id="search-1",
                        name="search_calendar",
                        arguments={"start_date": "2026-09-18", "end_date": "2026-09-18"},
                    )
                ]
            ),
            AgentModelTurn(
                final_response=AgentFinalResponse(
                    reply="The design review is on the 18th.", source_ids=["replay-event"]
                )
            ),
            AgentModelTurn(
                tool_calls=[
                    AgentToolCall(
                        id="hand-off",
                        name=HAND_OFF_TOOL,
                        arguments={"request": "Save a note about latency work."},
                    )
                ]
            ),
        ],
    )


class ModelReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _store(self, name: str) -> CaptureStore:
        """A fresh database holding the rows both the recorded and replayed runs read."""

        store = CaptureStore(Path(self.temporary_directory.name) / f"{name}.sqlite3")
        store.initialize()
        project = store.save_project(Project(id="replay-project", name="Replay"))
        conversation = store.save_conversation(Conversation(id="replay-chat", title="Replay"))
        store.assign_conversation_project(conversation.id, project.id)
        source = store.save_message(
            Message(
                id="replay-seed",
                conversation_id=conversation.id,
                role="user",
                content="Design review on September 18.",
            )
        )
        store.save_calendar_event(
            CalendarEvent(
                id="replay-event",
                conversation_id=conversation.id,
                title="Design review",
                source_message_id=source.id,
                source_quote=source.content,
                start_at=date(2026, 9, 18),
                all_day=True,
                timezone="America/Vancouver",
            )
        )
        store.save_project_record(
            ProjectRecord(
                id="replay-task",
                project_id=project.id,
                conversation_id=conversation.id,
                kind="task",
                content="Measure publisher latency",
                state="active",
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        return store

    def _run(self, store: CaptureStore, model) -> list:
        router = ConversationRouterService(store, model, REFERENCE)
        return [
            router.submit("replay-chat", "What is scheduled on September 18?", "replay-read"),
            router.submit("replay-chat", "Note that latency work starts after the review.", "replay-write"),
        ]

    def _effects(self, store: CaptureStore, turns: list) -> list[dict]:
        return [
            {
                "outcome": turn.outcome,
                "reply": turn.assistant_message.content,
                "operations": len(turn.operations),
                "model_calls": turn.model_calls,
                "tool_calls": turn.tool_calls,
                "stop_reason": turn.agent_stop_reason,
                "saved_items": [item.content for item in store.list_saved_items(workspace_id="default")],
            }
            for turn in turns
        ]

    def test_a_recording_reproduces_the_run_it_recorded(self) -> None:
        recorded_store = self._store("recorded")
        recorder = RecordingModel(_scripted())
        recorded = self._effects(recorded_store, self._run(recorded_store, recorder))
        recording = recorder.recording(model="scripted", commit="test")

        replayed_store = self._store("replayed")
        replayed = self._effects(
            replayed_store, self._run(replayed_store, ReplayModel(recording))
        )

        self.assertEqual(recorded, replayed)
        self.assertEqual(recording["version"], RECORDING_VERSION)
        self.assertEqual(
            [call["method"] for call in recording["calls"]],
            [
                "next_read_agent_turn",
                "next_read_agent_turn",
                "next_read_agent_turn",
                "propose_conversation_decision",
            ],
        )

    def test_a_recording_survives_being_written_and_read_as_json(self) -> None:
        recorded_store = self._store("recorded")
        recorder = RecordingModel(_scripted())
        self._run(recorded_store, recorder)
        destination = Path(self.temporary_directory.name) / "recording.json"
        destination.write_text(
            json.dumps(recorder.recording(model="scripted"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        replayed_store = self._store("replayed")
        turns = self._run(
            replayed_store,
            ReplayModel(json.loads(destination.read_text(encoding="utf-8"))),
        )

        self.assertEqual(turns[0].outcome, "agent_answer")
        self.assertEqual(turns[1].operations[0].tool_name, "save_note")

    def test_a_run_that_sends_a_different_request_is_refused(self) -> None:
        # The recorded answer may not be what the model would now say, so a changed
        # request must fail loudly instead of replaying a stale answer.
        recorder = RecordingModel(_scripted())
        self._run(self._store("recorded"), recorder)
        recording = recorder.recording(model="scripted")

        replay = ReplayModel(recording)
        with self.assertRaisesRegex(AssertionError, "different request"):
            replay.next_read_agent_turn(messages=[{"role": "user", "content": "other"}], tools=[])

    def test_a_run_that_calls_a_different_method_is_refused(self) -> None:
        recorder = RecordingModel(_scripted())
        self._run(self._store("recorded"), recorder)
        replay = ReplayModel(recorder.recording(model="scripted"))

        with self.assertRaisesRegex(AssertionError, "is next_read_agent_turn"):
            replay.propose_conversation_decision(current_message={}, recent_messages=[])

    def test_a_run_longer_than_its_recording_is_refused(self) -> None:
        recorder = RecordingModel(_scripted())
        store = self._store("recorded")
        self._run(store, recorder)
        replay = ReplayModel(recorder.recording(model="scripted"))
        self._run(self._store("replayed"), replay)

        with self.assertRaisesRegex(AssertionError, "no call 5"):
            replay.next_read_agent_turn(messages=[], tools=[])

    def test_a_recorded_model_failure_is_raised_again_as_its_own_class(self) -> None:
        # The agent loop treats InvalidStructuredOutput differently from any other error,
        # so a recording that flattened them would replay the wrong control flow.
        for error, expected in (
            (InvalidStructuredOutput("unreadable"), InvalidStructuredOutput),
            (ModelError("provider refused"), ModelError),
        ):
            with self.subTest(error=type(error).__name__):
                recorder = RecordingModel(ScriptedContextualModel([], agent_turns=[error]))
                with self.assertRaises(expected):
                    recorder.next_read_agent_turn(messages=[{"role": "user"}], tools=[])
                replay = ReplayModel(recorder.recording(model="scripted"))
                with self.assertRaises(expected):
                    replay.next_read_agent_turn(messages=[{"role": "user"}], tools=[])


if __name__ == "__main__":
    unittest.main()
