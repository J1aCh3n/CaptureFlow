"""Replay a recorded live run of the acceptance fixture, free, as a regression test.

The recording in tests/fixtures/replay was taken from one real run on the protocol's
pinned model. Driving the same code from it covers everything below the model -- routing,
the agent loop's control flow, scope binding, the provenance ledger, source cards, stored
reply text -- which is where every defect found since Phase A has lived.

This asserts what our code did with those answers, not that the answers were good. One of
them was not: the router classified the no-match question as reply_only, so it never
reached the agent. That is the router failure recorded in the Phase A canary report, and
it is what work item 3 of docs/refactor-plan-zh.md removes. Locking it here means a change
in how we handle it cannot pass unnoticed.

Re-record when this fails for a reason you intended, with:

    python -m scripts.run_read_agent_acceptance run --answer-mode agent \
        --output-dir runs/<name> --cases no_match project_isolation \
        cross_conversation_project_retrieval range_question --record
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
    FIXTURE,
    ToolResultRecorder,
    cited_ids,
    safety_violations,
    seed_database,
)


RECORDING = Path(__file__).parent / "fixtures" / "replay" / "agent_27b_four_cases.json"

# The order run_fixture submits these cases in, with the conversation and message ID each
# one uses. A different order or ID would send a different request, which replay refuses.
SEQUENCE = (
    ("cross_conversation_project_retrieval", "later", "read-agent-cross-project"),
    ("range_question", "main", "read-agent-range"),
    ("project_isolation", "main", "read-agent-isolation"),
    ("no_match", "main", "read-agent-no-match"),
)

EXPECTED = {
    "cross_conversation_project_retrieval": {
        "outcome": "agent_project_answer",
        "intent": "project_answer",
        "model_calls": 3,
        "tool_calls": 1,
        "stop_reason": "completed",
        "operations": 0,
        "cited": ["read-agent-latency-task"],
    },
    "range_question": {
        "outcome": "agent_local_answer",
        "intent": "local_answer",
        "model_calls": 3,
        "tool_calls": 1,
        "stop_reason": "completed",
        "operations": 0,
        "cited": ["read-agent-architecture-review", "read-agent-evidence-day"],
    },
    "project_isolation": {
        "outcome": "agent_project_answer",
        "intent": "project_answer",
        "model_calls": 3,
        "tool_calls": 2,
        "stop_reason": "completed",
        "operations": 0,
        "cited": ["read-agent-architecture-review", "read-agent-latency-task"],
    },
    # The router answered this one itself instead of letting the agent search.
    "no_match": {
        "outcome": "reply_only",
        "intent": "reply_only",
        "model_calls": 1,
        "tool_calls": 0,
        "stop_reason": None,
        "operations": 0,
        "cited": [],
    },
}


class ReplayedAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recording = json.loads(RECORDING.read_text(encoding="utf-8"))
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.temporary_directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temporary_directory.cleanup)

    def test_the_recording_belongs_to_the_fixture_it_is_replayed_against(self) -> None:
        # A protocol bump or an edited case changes the requests, so the recording would be
        # answering a question that is no longer asked.
        self.assertEqual(
            self.recording["fixture_sha256"], hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        )
        self.assertEqual(self.recording["protocol_version"], self.fixture["protocol"]["version"])
        self.assertEqual(self.recording["model"], self.fixture["protocol"]["model"])
        self.assertEqual(self.recording["answer_mode"], "agent")
        self.assertEqual(self.recording["cases"], sorted(name for name, _, _ in SEQUENCE))
        self.assertFalse(self.recording["working_tree_dirty"])

    def test_replaying_reaches_the_same_turns_sources_and_stored_replies(self) -> None:
        store = CaptureStore(Path(self.temporary_directory.name) / "replay.sqlite3")
        store.initialize()
        seed = seed_database(store)
        model = ReplayModel(self.recording)
        tools = ToolResultRecorder()
        self.addCleanup(tools.restore)
        router = ConversationRouterService(
            store,
            model,
            date.fromisoformat(self.fixture["protocol"]["reference_date"]),
            self.fixture["protocol"]["timezone"],
            answer_mode="agent",
        )
        inputs = {case["name"]: case["input"] for case in self.fixture["cases"]}
        conversations = {
            "main": seed["main_conversation"].id,
            "later": seed["later_conversation"].id,
        }

        violations = []
        for name, conversation, message_id in SEQUENCE:
            with self.subTest(case=name):
                seen = len(tools.results)
                turn = router.submit(conversations[conversation], inputs[name], message_id)
                returned = tools.results[seen:]
                violations.extend(safety_violations(store, seed, turn, returned))
                self.assertIsNone(turn.failure_reason)
                self.assertEqual(
                    {
                        "outcome": turn.outcome,
                        "intent": None if turn.decision is None else turn.decision.intent,
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
            model.position, len(self.recording["calls"]), "the recording was not used up"
        )

    def test_the_replayed_answers_are_the_stored_replies(self) -> None:
        # A reply that never reached SQLite would still pass the checks above, since they
        # read the turn; this reads the row back.
        store = CaptureStore(Path(self.temporary_directory.name) / "stored.sqlite3")
        store.initialize()
        seed = seed_database(store)
        tools = ToolResultRecorder()
        self.addCleanup(tools.restore)
        router = ConversationRouterService(
            store,
            ReplayModel(self.recording),
            date.fromisoformat(self.fixture["protocol"]["reference_date"]),
            self.fixture["protocol"]["timezone"],
            answer_mode="agent",
        )
        inputs = {case["name"]: case["input"] for case in self.fixture["cases"]}
        conversations = {"main": seed["main_conversation"].id, "later": seed["later_conversation"].id}

        replies = {}
        for name, conversation, message_id in SEQUENCE:
            turn = router.submit(conversations[conversation], inputs[name], message_id)
            replies[name] = store.get_message(turn.assistant_message.id).content

        self.assertIn(seed["task"].content.rstrip("。"), replies["cross_conversation_project_retrieval"])
        # A range answer renders each cited event's stored time under the reply.
        self.assertIn("Stored dates:", replies["range_question"])
        for event in seed["events"]:
            if event.id in EXPECTED["range_question"]["cited"]:
                self.assertIn(event.start_at.isoformat(), replies["range_question"])
        # Nothing from the look-alike project reaches any reply.
        for row in seed["other_project_rows"]:
            text = row.title if hasattr(row, "title") else row.content
            for name, reply in replies.items():
                with self.subTest(case=name, row=row.id):
                    self.assertNotIn(text, reply)


if __name__ == "__main__":
    unittest.main()
