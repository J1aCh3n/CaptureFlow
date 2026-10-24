"""Replay a recorded live run of the acceptance fixture, free, as a regression test.

The recording in tests/fixtures/replay was taken from one real run on the protocol's
pinned model, from a clean commit. Driving the same code from it covers everything below
the model -- routing, the agent loop's control flow, scope binding, the provenance ledger,
source cards, stored reply text -- which is where every defect found since Phase A has
lived.

This asserts what our code did with those answers, not that the answers were good. The
run it comes from was 11 of 12: the model read note_write's 记一下：延迟排查从下周一开始
as a deadline on the task the seed already holds and called set_deadline, using a start
date as a due date. That wrong choice is recorded here as what it was, because the
question this file asks is whether our code still does the same thing with the same
answers. The acceptance runner is what judges whether an answer was right.

Re-record when this fails for a reason you intended, with:

    python -m scripts.run_read_agent_acceptance run \
        --output-dir runs/<name> --cases <the twelve below> --record
"""

import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.conversation_router_service import ConversationRouterService
from captureflow.storage import CaptureStore
from scripts.model_replay import ReplayModel
from scripts.run_read_agent_acceptance import (
    CASE_SEQUENCE,
    FIXTURE,
    WRITE_CASES,
    ToolResultRecorder,
    cited_ids,
    safety_violations,
    seed_database,
)


RECORDING = Path(__file__).parent / "fixtures" / "replay" / "agent_27b_twelve_cases.json"

# What one turn did with the recorded answers. The IDs a write cites are the rows the tool
# handed back, derived from the message ID, so they are stable and spelled out: a change to
# either formula breaks this, which is the point -- resuming a crashed turn depends on both.
EXPECTED = {
    "cross_conversation_project_retrieval": {
        "outcome": "agent_answer",
        "model_calls": 2,
        "tool_calls": 1,
        "stop_reason": "completed",
        "operations": 0,
        "cited": ["read-agent-latency-task"],
    },
    "range_question": {
        "outcome": "agent_answer",
        "model_calls": 2,
        "tool_calls": 1,
        "stop_reason": "completed",
        "operations": 0,
        "cited": ["read-agent-architecture-review", "read-agent-evidence-day"],
    },
    "project_isolation": {
        "outcome": "agent_answer",
        "model_calls": 2,
        "tool_calls": 2,
        "stop_reason": "completed",
        "operations": 0,
        "cited": ["read-agent-architecture-review", "read-agent-latency-task"],
    },
    # Three searches, nothing found, nothing cited. One run on an earlier commit and the
    # same input cited the five events it had looked at instead: citing is the model's
    # choice, not something the schema requires.
    "no_match": {
        "outcome": "agent_answer",
        "model_calls": 2,
        "tool_calls": 3,
        "stop_reason": "completed",
        "operations": 0,
        "cited": [],
    },
    # The case the run failed: a note was asked for and a deadline was set instead. Our
    # code carried that out correctly -- one change, through the ledger, cited from the row
    # the tool returned -- which is what this file checks.
    "note_write": {
        "outcome": "tracker_change",
        "model_calls": 3,
        "tool_calls": 2,
        "stop_reason": "completed",
        "operations": 1,
        "cited": ["read-agent-latency-task"],
    },
    "idea_write": {
        "outcome": "save_idea",
        "model_calls": 2,
        "tool_calls": 1,
        "stop_reason": "completed",
        "operations": 1,
        "cited": ["fc5535da-7017-549e-8e2c-0d2933247091"],
    },
    # Created through the router's validated path, so Python attached its deadline.
    "record_write": {
        "outcome": "tracker_change",
        "model_calls": 2,
        "tool_calls": 1,
        "stop_reason": "completed",
        "operations": 1,
        "cited": ["f650e66b-60b1-549d-aab1-877c6597ae8e"],
    },
    "event_write": {
        "outcome": "create_local_event",
        "model_calls": 2,
        "tool_calls": 1,
        "stop_reason": "completed",
        "operations": 1,
        "cited": ["160afccf-cd86-54b8-8606-4fae572f9afb"],
    },
    # A write that names something stored takes three turns, because the ledger makes the
    # search a requirement: search, change what was found, then answer.
    "event_update": {
        "outcome": "calendar_update",
        "model_calls": 3,
        "tool_calls": 2,
        "stop_reason": "completed",
        "operations": 1,
        "cited": ["read-agent-architecture-review"],
    },
    "record_deadline": {
        "outcome": "tracker_change",
        "model_calls": 3,
        "tool_calls": 2,
        "stop_reason": "completed",
        "operations": 1,
        "cited": ["read-agent-latency-task", "f650e66b-60b1-549d-aab1-877c6597ae8e"],
    },
    "record_done": {
        "outcome": "tracker_change",
        "model_calls": 3,
        "tool_calls": 2,
        "stop_reason": "completed",
        "operations": 1,
        "cited": ["read-agent-latency-task", "f650e66b-60b1-549d-aab1-877c6597ae8e"],
    },
    # Prepared and waiting for the user: one operation in review, and the event still there.
    "event_delete": {
        "outcome": "calendar_delete_review",
        "model_calls": 3,
        "tool_calls": 2,
        "stop_reason": "completed",
        "operations": 1,
        "cited": ["read-agent-release-demo"],
    },
}


def _sequence() -> tuple[tuple[str, str, str], ...]:
    """The recorded cases in the order the runner submits them, taken from the runner.

    A different order or message ID sends a different request, which replay refuses, so
    this is derived rather than written down twice.
    """

    return tuple(entry for entry in CASE_SEQUENCE if entry[0] in EXPECTED)


class ReplayedAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = json.loads(RECORDING.read_text(encoding="utf-8"))
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary_directory.cleanup)

    def _router(self, name: str):
        store = CaptureStore(Path(self.temporary_directory.name) / f"{name}.sqlite3")
        store.initialize()
        seed = seed_database(store)
        tools = ToolResultRecorder()
        self.addCleanup(tools.restore)
        router = ConversationRouterService(
            store,
            ReplayModel(self.recording),
            date.fromisoformat(self.fixture["protocol"]["reference_date"]),
            self.fixture["protocol"]["timezone"],
        )
        inputs = {case["name"]: case["input"] for case in self.fixture["cases"]}
        conversations = {
            "main": seed["main_conversation"].id,
            "later": seed["later_conversation"].id,
        }
        return store, seed, tools, router, inputs, conversations

    def test_the_recording_belongs_to_the_fixture_it_is_replayed_against(self) -> None:
        # A protocol bump or an edited case changes the requests, so the recording would be
        # answering a question that is no longer asked.
        self.assertEqual(
            self.recording["fixture_sha256"], hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        )
        self.assertEqual(self.recording["protocol_version"], self.fixture["protocol"]["version"])
        self.assertEqual(self.recording["model"], self.fixture["protocol"]["model"])
        self.assertEqual(self.recording["answer_mode"], "agent")
        self.assertEqual(self.recording["cases"], sorted(EXPECTED))
        self.assertFalse(self.recording["working_tree_dirty"])

    def test_every_write_tool_the_agent_has_appears_in_the_recording(self) -> None:
        # The point of re-recording was the three write tools with no live evidence. Two of
        # them are here; supersede_record needs a stored decision the seed does not have.
        submitted = {name for name, _, _ in _sequence()}

        self.assertEqual(WRITE_CASES - submitted, set())

    def test_replaying_reaches_the_same_turns_sources_and_stored_replies(self) -> None:
        store, seed, tools, router, inputs, conversations = self._router("replay")

        violations = []
        for name, conversation, message_id in _sequence():
            with self.subTest(case=name):
                seen = len(tools.results)
                turn = router.submit(conversations[conversation], inputs[name], message_id)
                returned = tools.results[seen:]
                violations.extend(
                    safety_violations(store, seed, turn, returned, writes=name in WRITE_CASES)
                )
                self.assertIsNone(turn.failure_reason)
                self.assertEqual(
                    {
                        "outcome": turn.outcome,
                        "model_calls": turn.model_calls,
                        "tool_calls": turn.tool_calls,
                        "stop_reason": turn.agent_stop_reason,
                        "operations": len(turn.operations),
                        "cited": cited_ids(store, turn),
                    },
                    EXPECTED[name],
                )
                self.assertTrue(store.get_message(turn.assistant_message.id).content)

        self.assertEqual(violations, [])
        self.assertEqual(
            router.model.position, len(self.recording["calls"]), "the recording was not used up"
        )

    def test_the_replayed_answers_are_the_stored_replies(self) -> None:
        # A reply that never reached SQLite would still pass the checks above, since they
        # read the turn; this reads the row back.
        store, seed, _, router, inputs, conversations = self._router("stored")

        replies = {}
        for name, conversation, message_id in _sequence():
            turn = router.submit(conversations[conversation], inputs[name], message_id)
            replies[name] = store.get_message(turn.assistant_message.id).content

        self.assertIn(seed["task"].content.rstrip("。"), replies["cross_conversation_project_retrieval"])
        # A range answer renders each cited event's stored time under the reply.
        self.assertIn("Stored dates:", replies["range_question"])
        for event in seed["events"]:
            if event.id in EXPECTED["range_question"]["cited"]:
                self.assertIn(event.start_at.isoformat(), replies["range_question"])

        # The writes reached SQLite: a created event on the protocol's date, a deadline
        # Python attached, a finished record, and a deletion still waiting for the user.
        created = [
            event
            for event in store.list_calendar_events()
            if event.id not in {seeded.id for seeded in seed["events"]}
        ]
        self.assertEqual(
            [event.start_date.isoformat() for event in created],
            [self.fixture["protocol"]["event_write_date"]],
        )
        self.assertEqual(
            store.get_calendar_event("read-agent-architecture-review").start_date.isoformat(),
            self.fixture["protocol"]["event_update_date"],
        )
        task = store.get_project_record(seed["task"].id)
        self.assertEqual(task.state, "completed")
        self.assertEqual(
            task.due_at.isoformat(),
            f"{self.fixture['protocol']['record_deadline_date']}T"
            f"{self.fixture['protocol']['record_deadline_time']}:00-07:00",
        )
        self.assertEqual(store.get_calendar_event("read-agent-release-demo").status, "active")
        self.assertEqual(
            [
                operation.status
                for operation in store.list_calendar_event_update_operations("read-agent-event-delete")
            ],
            ["review"],
        )

        # Nothing from the look-alike project reaches any reply.
        for row in seed["other_project_rows"]:
            text = row.title if hasattr(row, "title") else row.content
            for name, reply in replies.items():
                with self.subTest(case=name, row=row.id):
                    self.assertNotIn(text, reply)


if __name__ == "__main__":
    unittest.main()
