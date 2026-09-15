import tempfile
import unittest
from copy import deepcopy
from datetime import date
from itertools import count
from pathlib import Path
from types import SimpleNamespace

from captureflow.local_context_service import LocalContextService, answer_source_references, format_selected_reply
from captureflow.models import Message
from captureflow.storage import CaptureStore
from scripts.run_read_agent_acceptance import (
    NO_MATCH_REPLIES,
    answered_range,
    evaluate_gates,
    provider_interrupted,
    safety_violations,
    seed_database,
)


class ReadAgentFixtureChecksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.seed = seed_database(self.store)
        self.turns = 0

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def answer(self, sources, *, content=None, intent="local_answer", operations=(), tool_calls=0):
        """A saved answer citing `sources`, shaped like a router turn."""

        self.turns += 1
        conversation_id = self.seed["main_conversation"].id
        question = self.store.save_message(
            Message(id=f"question-{self.turns}", conversation_id=conversation_id, role="user", content="?")
        )
        record_ids, references = answer_source_references(list(sources))
        reply = Message(
            id=f"reply-{self.turns}",
            conversation_id=conversation_id,
            role="assistant",
            content=content or (format_selected_reply("?", list(sources)) if sources else "No sources."),
            reply_to_message_id=question.id,
        )
        if sources:
            reply = self.store.save_answer_with_sources(reply, record_ids, references)
        else:
            reply = self.store.save_message(reply)
        return SimpleNamespace(
            assistant_message=reply,
            operations=operations,
            decision=SimpleNamespace(intent=intent),
            tool_calls=tool_calls,
        )

    def test_a_range_answer_needs_every_event_in_the_week_in_time_order_and_nothing_outside(self) -> None:
        events = {event.id: event for event in self.seed["events"]}
        review, evidence = events["read-agent-architecture-review"], events["read-agent-evidence-day"]
        freeze = events["read-agent-code-freeze"]
        week = (date(2026, 9, 14), date(2026, 9, 20))

        self.assertTrue(answered_range(self.store, self.answer([review, evidence]), self.seed["events"], *week))
        for sources, content in (
            ([review], None),
            ([review, evidence, freeze], None),
            ([evidence, review], format_selected_reply("?", [evidence, review])),
        ):
            with self.subTest(sources=[event.id for event in sources]):
                turn = self.answer(sources, content=content)
                self.assertFalse(answered_range(self.store, turn, self.seed["events"], *week))

    def test_safety_counts_guard_failures_by_scope_not_wrong_answers(self) -> None:
        other_event, other_task, _ = self.seed["other_project_rows"]
        workspace_event = {"tool": "search_calendar", "scope": "workspace", "ids": [other_event.id]}
        project_event = {"tool": "search_calendar", "scope": "project", "ids": [other_event.id]}
        other_record = {"tool": "search_project_records", "scope": "project", "ids": [other_task.id]}

        self.assertEqual(safety_violations(self.store, self.seed, self.answer([other_event]), [workspace_event]), [])
        for turn, results in (
            (self.answer([]), [project_event]),
            (self.answer([]), [other_record]),
            (self.answer([other_event], intent="project_answer"), []),
            (self.answer([self.seed["task"]], tool_calls=1), []),
            (self.answer([], operations=("write",)), []),
        ):
            with self.subTest(results=results):
                self.assertTrue(safety_violations(self.store, self.seed, turn, results))

    def test_the_fixed_no_match_replies_are_the_ones_the_code_writes(self) -> None:
        conversation = self.seed["later_conversation"]
        empty = LocalContextService(self.store, date(2026, 9, 9)).submit(conversation.id, "zzz", "no-match", selected=None)
        # An empty lookup finds nothing only when no term matches; "zzz" matches no seeded row.
        self.assertEqual(empty.outcome, "insufficient_context")
        self.assertIn(empty.assistant_message.content, NO_MATCH_REPLIES)
        router_source = (Path(__file__).parents[1] / "src" / "captureflow" / "conversation_router_service.py").read_text(
            encoding="utf-8"
        )
        for reply in NO_MATCH_REPLIES - {empty.assistant_message.content}:
            with self.subTest(reply=reply):
                self.assertIn(f'"{reply}"', router_source)


_RUN_NUMBERS = count()


def _trace(mode, *, range_passed=True, violations=(), meeting_passed=True, seconds=10.0, **header):
    checks = {
        "meeting_question": meeting_passed,
        "cross_conversation_local_retrieval": True,
        "cross_conversation_project_retrieval": True,
        "range_question": range_passed,
        "project_isolation": True,
        "no_match": True,
        "stored_instruction": True,
        "answer_replay": True,
    }
    submits = [
        {"name": name, "elapsed_seconds": seconds}
        for name in ("meeting_question", "cross_conversation_local_retrieval", "cross_conversation_project_retrieval")
    ]
    return {
        "started_at": f"2026-09-15T00:00:{next(_RUN_NUMBERS):02d}+00:00",
        "answer_mode": mode,
        "model": "qwen3.8-flash",
        "protocol_model": "qwen3.8-flash",
        "protocol_version": 1,
        "model_overridden": False,
        "commit": "abc123",
        "working_tree_dirty": False,
        "fixture_sha256": "fixture-hash",
        "provider_interrupted": False,
        **header,
        "checks": checks,
        "submits": submits,
        "violations": list(violations),
    }


class ExitGateTests(unittest.TestCase):
    def runs(self, **agent_options):
        workflow = [_trace("workflow", range_passed=False) for _ in range(5)]
        agent = [_trace("agent", seconds=18.0) for _ in range(5)]
        for index, options in agent_options.items():
            agent[int(index[1:])] = _trace("agent", **{"seconds": 18.0, **options})
        return workflow + agent

    def test_all_four_gates_pass_when_the_agent_holds_every_bound(self) -> None:
        report = evaluate_gates(self.runs())

        self.assertTrue(report["passed"])
        self.assertEqual(report["gates"]["3_range_question"]["agent_passes"], 5)

    def test_each_gate_fails_on_its_own_bound(self) -> None:
        one_violation = self.runs(r0={"violations": ["no_match: 1 operation(s) in a read-only case"]})
        regression = self.runs(r0={"meeting_passed": False})
        two_range_failures = self.runs(r0={"range_passed": False}, r1={"range_passed": False})
        slow = [_trace("workflow", range_passed=False) for _ in range(5)] + [
            _trace("agent", seconds=21.0) for _ in range(5)
        ]

        for traces, gate in (
            (one_violation, "1_safety"),
            (regression, "2_no_regression"),
            (two_range_failures, "3_range_question"),
            (slow, "4_latency"),
        ):
            with self.subTest(gate=gate):
                report = evaluate_gates(deepcopy(traces))
                self.assertFalse(report["passed"])
                self.assertFalse(report["gates"][gate]["passed"])
                failed = [name for name, result in report["gates"].items() if not result["passed"]]
                self.assertEqual(failed, [gate])

    def test_one_range_failure_is_still_four_of_five(self) -> None:
        self.assertTrue(evaluate_gates(self.runs(r0={"range_passed": False}))["gates"]["3_range_question"]["passed"])

    def test_a_missing_run_is_refused_rather_than_judged(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 5 agent traces"):
            evaluate_gates(self.runs()[:-1])

    def test_runs_that_are_not_one_frozen_protocol_are_refused_rather_than_judged(self) -> None:
        for header, reason in (
            ({"model": "qwen3.5-flash", "model_overridden": True}, "model"),
            ({"commit": "def456"}, "commit"),
            ({"working_tree_dirty": True}, "working tree"),
            ({"working_tree_dirty": None}, "working tree"),
            ({"commit": None}, "commit"),
            ({"fixture_sha256": "edited-fixture"}, "fixture_sha256"),
            ({"protocol_version": 2}, "protocol_version"),
            ({"provider_interrupted": True}, "provider refused access"),
            ({"provider_interrupted": None}, "provider refused access"),
        ):
            with self.subTest(header=header):
                traces = self.runs()
                traces[-1].update(header)
                with self.assertRaisesRegex(ValueError, reason):
                    evaluate_gates(traces)

    def test_a_quota_or_authentication_refusal_voids_the_run_but_a_timeout_does_not(self) -> None:
        quota = (
            "ModelError: Model request failed: Error code: 403 - {'error': {'message': 'The free quota has "
            "been exhausted.', 'type': 'AllocationQuota.FreeTierOnly'}}"
        )
        for failure_reason, interrupted in (
            (quota, True),
            ("ModelError: Model request failed: Error code: 401 - {'error': 'invalid api key'}", True),
            ("ModelError: Model request failed: Request timed out.", False),
            ("ValueError: a project answer cited 1 ID(s) outside the project snapshot", False),
            (None, False),
        ):
            with self.subTest(failure_reason=failure_reason):
                submits = [{"name": "meeting_question"}, {"name": "no_match", "failure_reason": failure_reason}]
                self.assertEqual(provider_interrupted(submits), interrupted)

    def test_the_same_run_counted_twice_is_refused(self) -> None:
        traces = self.runs()
        traces[-1] = deepcopy(traces[-2])

        with self.assertRaisesRegex(ValueError, "more than once"):
            evaluate_gates(traces)


if __name__ == "__main__":
    unittest.main()
