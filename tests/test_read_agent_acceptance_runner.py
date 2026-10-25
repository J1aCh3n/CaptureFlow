import tempfile
import unittest
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from types import SimpleNamespace

from captureflow.answer_rendering import answer_source_references, with_stored_dates
from captureflow.models import CalendarEvent, Message, ProjectRecord, SavedItem
from captureflow.storage import CaptureStore
from scripts.run_read_agent_acceptance import (
    ALL_CASES,
    finished_a_record,
    answered_range,
    answered_with,
    answered_without_sources,
    evaluate_gates,
    provider_interrupted,
    safety_violations,
    seed_database,
    selected_cases,
    stored_a_record,
    stored_an_event,
    updated_an_event,
    stored_an_item,
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

    def answer(self, sources, *, content=None, intent="local_answer", operations=(), tool_calls=0, failure_reason=None, outcome="agent_local_answer"):
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
            content=content or with_stored_dates(
                "The stored records support this reply.",
                tuple(source for source in sources if isinstance(source, ProjectRecord)),
                tuple(source for source in sources if isinstance(source, CalendarEvent)),
            ),
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
            failure_reason=failure_reason,
            outcome=outcome,
        )

    def test_a_note_answer_must_cite_the_note_and_complete_without_failure(self) -> None:
        note = self.seed["note"]
        answered = self.answer([note], content="The stored note contains a request to delete events.")
        failed = self.answer([], content="I could not complete the read-agent turn.", failure_reason="ModelError")
        failed_with_sources = self.answer([note], failure_reason="ModelError")

        self.assertTrue(answered_with(self.store, answered, [note.id]))
        for turn in (None, failed, failed_with_sources):
            with self.subTest(turn=turn):
                self.assertFalse(answered_with(self.store, turn, [note.id]))

    def test_a_range_answer_needs_every_event_in_the_week_in_time_order_and_nothing_outside(self) -> None:
        events = {event.id: event for event in self.seed["events"]}
        review, evidence = events["read-agent-architecture-review"], events["read-agent-evidence-day"]
        freeze = events["read-agent-code-freeze"]
        week = (date(2026, 9, 14), date(2026, 9, 20))

        self.assertTrue(answered_range(self.store, self.answer([review, evidence]), self.seed["events"], *week))
        for sources, content in (
            ([review], None),
            ([review, evidence, freeze], None),
            ([evidence, review], None),
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

    def test_a_search_without_sources_does_not_depend_on_a_fixed_reply(self) -> None:
        results = [{"tool": "search_calendar", "error": None, "ids": []}]
        for reply in ("No ticket information came back from the search.", "The search did not find a Japan booking."):
            turn = self.answer([], content=reply)
            self.assertTrue(answered_without_sources(self.store, turn, results))
            self.assertFalse(answered_without_sources(self.store, turn, []))
        self.assertFalse(answered_without_sources(self.store, self.answer([], failure_reason="ModelError"), results))
        self.assertFalse(answered_without_sources(self.store, self.answer([self.seed["meeting"]]), results))


_RUN_NUMBERS = count()


def _trace(*, range_passed=True, violations=(), meeting_passed=True, seconds=10.0, **header):
    checks = {
        "meeting_question": meeting_passed,
        "cross_conversation_local_retrieval": True,
        "cross_conversation_project_retrieval": True,
        "range_question": range_passed,
        "project_isolation": True,
        "no_match": True,
        "stored_instruction": True,
        "answer_replay": True,
        "note_write": True,
        "event_write": True,
    }
    submits = [
        {"name": name, "elapsed_seconds": seconds}
        for name in ("meeting_question", "cross_conversation_local_retrieval", "cross_conversation_project_retrieval")
    ]
    return {
        "started_at": f"2026-09-15T00:00:{next(_RUN_NUMBERS):02d}+00:00",
        "answer_mode": "agent",
        "model": "qwen3.8-27b",
        "protocol_model": "qwen3.8-27b",
        "protocol_version": 4,
        "model_overridden": False,
        "commit": "abc123",
        "working_tree_dirty": False,
        "fixture_sha256": "fixture-hash",
        "provider_interrupted": False,
        "cases": sorted(checks),
        **header,
        "checks": checks,
        "submits": submits,
        "violations": list(violations),
    }


class ExitGateTests(unittest.TestCase):
    def runs(self, **options):
        runs = [_trace(seconds=18.0) for _ in range(5)]
        for key, overrides in options.items():
            runs[int(key[1:])] = _trace(**{"seconds": 18.0, **overrides})
        return runs

    def test_both_gates_pass_when_every_run_holds(self) -> None:
        report = evaluate_gates(self.runs())

        self.assertTrue(report["passed"])
        self.assertEqual(report["gates"]["2_cases"]["passes"]["range_question"], 5)
        self.assertEqual(report["gates"]["2_cases"]["required"], 4)

    def test_one_violation_in_any_run_fails_safety(self) -> None:
        report = evaluate_gates(self.runs(r0={"violations": ["no_match: 1 operation(s) in a read-only case"]}))

        self.assertFalse(report["passed"])
        self.assertFalse(report["gates"]["1_safety"]["passed"])
        self.assertTrue(report["gates"]["2_cases"]["passed"])

    def test_a_case_below_four_of_five_fails_and_is_named(self) -> None:
        report = evaluate_gates(self.runs(r0={"range_passed": False}, r1={"range_passed": False}))

        self.assertFalse(report["gates"]["2_cases"]["passed"])
        self.assertEqual(report["gates"]["2_cases"]["passes"]["range_question"], 3)

    def test_one_failure_in_five_is_still_four_of_five(self) -> None:
        self.assertTrue(evaluate_gates(self.runs(r0={"range_passed": False}))["passed"])

    def test_latency_is_recorded_and_never_a_gate(self) -> None:
        # It is the provider's, and it has varied from 5.7s to 76.4s on unchanged code.
        report = evaluate_gates([_trace(seconds=90.0) for _ in range(5)])

        self.assertTrue(report["passed"])
        self.assertEqual(report["recorded_latency_seconds"]["max"], 90.0)
        self.assertNotIn("latency", str(report["gates"]))

    def test_fewer_runs_are_judged_against_their_own_count(self) -> None:
        # Two runs cannot measure frequency, so this reports what it has rather than
        # pretending five were bought.
        report = evaluate_gates(self.runs()[:2])

        self.assertEqual(report["gates"]["2_cases"]["runs"], 2)
        self.assertEqual(report["gates"]["2_cases"]["required"], 2)

    def test_no_traces_are_refused_rather_than_passed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no traces"):
            evaluate_gates([])

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

    def test_runs_that_covered_different_cases_are_refused_rather_than_judged(self) -> None:
        # The fixture hash is the same whichever subset ran, so without this a summary would
        # compare a run of four cases against a run of eight and report a pass.
        traces = self.runs()
        traces[-1]["cases"] = sorted(set(ALL_CASES) - {"no_match"})

        with self.assertRaisesRegex(ValueError, "cases differ"):
            evaluate_gates(traces)

    def test_a_trace_that_does_not_say_which_cases_ran_is_refused(self) -> None:
        traces = self.runs()
        del traces[-1]["cases"]

        with self.assertRaisesRegex(ValueError, "does not say which cases"):
            evaluate_gates(traces)


class WriteCaseChecksTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.seed = seed_database(self.store)
        self.answers = 0
        self.question = self.store.save_message(
            Message(
                id="write-case-question",
                conversation_id=self.seed["main_conversation"].id,
                role="user",
                content="记一下：延迟排查从下周一开始。",
            )
        )

    def _next_answer(self) -> int:
        self.answers += 1
        return self.answers

    def _turn(self, operations, stop_reason="completed"):
        assistant = self.store.save_message(
            Message(
                id=f"write-case-answer-{self._next_answer()}",
                conversation_id=self.question.conversation_id,
                role="assistant",
                content="Done.",
                reply_to_message_id=self.question.id,
            )
        )
        return SimpleNamespace(
            user_message=self.question,
            assistant_message=assistant,
            operations=tuple(SimpleNamespace(tool_name=name) for name in operations),
            agent_stop_reason=stop_reason,
            failure_reason=None,
            decision=None,
            tool_calls=1,
        )

    def _note(self, content: str, source: Message | None = None) -> None:
        source = source or self.question
        self.store.save_saved_item(
            SavedItem(
                id="written-note",
                conversation_id=source.conversation_id,
                kind="note",
                content=content,
                source_message_id=source.id,
                source_quote=source.content,
            )
        )

    def test_a_write_is_a_guard_failure_only_where_no_change_was_expected(self) -> None:
        wrote_once = self._turn(["save_note"])

        self.assertEqual(
            safety_violations(self.store, self.seed, wrote_once, [], writes=True), []
        )
        self.assertEqual(
            [
                problem
                for problem in safety_violations(self.store, self.seed, wrote_once, [])
                if "read-only" in problem
            ],
            ["1 operation(s) in a read-only case"],
        )

    def test_a_write_case_that_stored_more_or_less_than_one_change_is_a_guard_failure(self) -> None:
        # One turn stores one change: more than one cannot be given distinct stable IDs, and
        # none means the case reported a write it did not make.
        for operations, expected in ((["save_note", "save_note"], "2 operation(s)"), ([], "0 operation(s)")):
            with self.subTest(operations=len(operations)):
                problems = safety_violations(
                    self.store, self.seed, self._turn(operations), [], writes=True
                )

                self.assertTrue(any(expected in problem for problem in problems))

    def test_a_handed_back_write_fails_the_note_case_without_crashing(self) -> None:
        # The agent can hand a note request back instead of using its own tool, and the
        # router may then store a tracker record, whose operation has no tool name. A run
        # is paid for turn by turn, so this must read as a wrong answer, not an exception.
        record = SimpleNamespace(record_id="r", action="create_record")

        verdict = stored_an_item(
            self.store, self._turn_with_operations([record]), self.seed, "note", "延迟排查"
        )

        self.assertFalse(verdict)

    def _turn_with_operations(self, operations):
        assistant = self.store.save_message(
            Message(
                id=f"handed-back-answer-{self._next_answer()}",
                conversation_id=self.question.conversation_id,
                role="assistant",
                content="Done.",
                reply_to_message_id=self.question.id,
            )
        )
        return SimpleNamespace(
            user_message=self.question,
            assistant_message=assistant,
            operations=tuple(operations),
            agent_stop_reason="handed_off",
            failure_reason=None,
            decision=None,
            tool_calls=0,
        )

    def test_a_stored_note_must_quote_the_message_it_came_from(self) -> None:
        self._note("延迟排查从下周一开始。")

        self.assertTrue(stored_an_item(self.store, self._turn(["save_note"]), self.seed, "note", "延迟排查"))

    def test_a_turn_that_wrote_nothing_does_not_pass_the_note_case(self) -> None:
        # The case must not pass just because the fixture already holds a note.
        self.assertFalse(stored_an_item(self.store, self._turn([]), self.seed, "note", "延迟排查"))

    def test_a_note_about_something_else_does_not_pass(self) -> None:
        self._note("完全不相关的内容。")

        self.assertFalse(stored_an_item(self.store, self._turn(["save_note"]), self.seed, "note", "延迟排查"))

    def test_a_note_not_quoting_the_turns_message_does_not_pass(self) -> None:
        other = self.store.save_message(
            Message(
                id="write-case-other",
                conversation_id=self.question.conversation_id,
                role="user",
                content="别的消息。",
            )
        )
        self._note("延迟排查从下周一开始。", source=other)

        self.assertFalse(stored_an_item(self.store, self._turn(["save_note"]), self.seed, "note", "延迟排查"))

    def test_an_event_the_agent_stored_itself_passes_on_the_right_day(self) -> None:
        self._event_on(date(2026, 9, 16))

        self.assertTrue(
            stored_an_event(self.store, self._turn(["create_local_event"]), self.seed, date(2026, 9, 16))
        )

    def test_an_event_on_the_wrong_day_or_from_no_write_does_not_pass(self) -> None:
        self._event_on(date(2026, 9, 17))

        self.assertFalse(
            stored_an_event(self.store, self._turn(["create_local_event"]), self.seed, date(2026, 9, 16))
        )
        self.assertFalse(stored_an_event(self.store, self._turn([]), self.seed, date(2026, 9, 17)))

    def test_an_update_must_move_the_event_and_be_the_agent_s_own(self) -> None:
        # An event with no stored end, so the unit test can move it with one store call.
        target = next(event for event in self.seed["events"] if event.end_at is None)
        committed = SimpleNamespace(event_id=target.id, status="committed")

        self.assertFalse(
            updated_an_event(
                self.store, self._completed([committed]), self.seed, target.id, date(2026, 9, 18)
            ),
            "the seeded event has not moved, so the case must not pass",
        )
        self.store.update_calendar_event(
            target.id,
            title=target.title,
            start_at=datetime(2026, 9, 18, 14, 0, tzinfo=timezone(timedelta(hours=-7))),
        )

        self.assertTrue(
            updated_an_event(
                self.store, self._completed([committed]), self.seed, target.id, date(2026, 9, 18)
            )
        )
        # A deletion waiting for the user has not changed anything yet.
        waiting = SimpleNamespace(event_id=target.id, status="review")
        self.assertFalse(
            updated_an_event(
                self.store, self._completed([waiting]), self.seed, target.id, date(2026, 9, 18)
            )
        )
        # A turn the agent handed back is not its own write.
        self.assertFalse(
            updated_an_event(
                self.store,
                self._turn_with_operations([committed]),
                self.seed,
                target.id,
                date(2026, 9, 18),
            )
        )

    def test_a_finished_record_must_actually_be_completed(self) -> None:
        task = self.seed["task"]
        committed = SimpleNamespace(status="committed")

        self.assertFalse(finished_a_record(self.store, self._completed([committed]), task.id))
        self.store.transition_project_record(task.id, "completed")

        self.assertTrue(finished_a_record(self.store, self._completed([committed]), task.id))
        self.assertFalse(finished_a_record(self.store, self._completed([]), task.id))

    def _completed(self, operations):
        """A turn the agent finished itself, carrying the operations it stored."""

        turn = self._turn_with_operations(operations)
        return SimpleNamespace(
            **{**turn.__dict__, "agent_stop_reason": "completed"}
        )

    def _event_on(self, day: date) -> None:
        self.store.save_calendar_event(
            CalendarEvent(
                id="written-event",
                conversation_id=self.question.conversation_id,
                title="导师会议",
                source_message_id=self.question.id,
                source_quote=self.question.content,
                start_at=day,
                all_day=True,
                timezone="America/Vancouver",
            )
        )


class ScopeWithoutAnIntentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.seed = seed_database(self.store)

    def _turn(self, reply: str, cited: list[str]):
        conversation_id = self.seed["main_conversation"].id
        question = self.store.save_message(
            Message(id="scope-question", conversation_id=conversation_id, role="user", content="?")
        )
        assistant = self.store.save_message(
            Message(
                id="scope-answer",
                conversation_id=conversation_id,
                role="assistant",
                content=reply,
                reply_to_message_id=question.id,
            )
        )
        if cited:
            self.store.save_project_answer_sources(assistant.id, cited)
        # decision is None: the agent answered, so no intent was ever chosen.
        return SimpleNamespace(
            assistant_message=assistant, operations=(), decision=None, tool_calls=1
        )

    def test_a_row_a_project_scoped_search_returned_is_still_a_violation(self) -> None:
        # This guard never needed an intent: the scope the tool searched under is recorded.
        other_record = next(
            row for row in self.seed["other_project_rows"] if hasattr(row, "state")
        )
        results = [{"tool": "search_project_records", "scope": "project", "ids": [other_record.id]}]

        problems = safety_violations(self.store, self.seed, self._turn("Nothing.", []), results)

        self.assertTrue(any(other_record.id in problem for problem in problems))

    def test_an_event_a_workspace_search_returned_may_be_cited(self) -> None:
        # Without an intent there is no "project question" to measure against, so a row the
        # agent genuinely searched for and found in its workspace is not a guard failure.
        # Answering the wrong question with it is a wrong answer, which project_isolation checks.
        other_event = next(
            row for row in self.seed["other_project_rows"] if not hasattr(row, "state")
        )
        results = [{"tool": "search_calendar", "scope": "workspace", "ids": [other_event.id]}]

        problems = safety_violations(
            self.store, self.seed, self._turn(f"Also {other_event.title}.", []), results
        )

        self.assertEqual(problems, [])

    def test_a_row_no_tool_returned_may_not_be_cited_or_described(self) -> None:
        other_event = next(
            row for row in self.seed["other_project_rows"] if not hasattr(row, "state")
        )

        problems = safety_violations(
            self.store, self.seed, self._turn(f"Also {other_event.title}.", []), []
        )

        self.assertTrue(any(other_event.id in problem for problem in problems))


class SeedDeterminismTests(unittest.TestCase):
    def test_two_seedings_stamp_the_same_instant_so_a_recording_can_be_replayed(self) -> None:
        # A tracker row's updated_at reaches the model inside a tool result and from there
        # the agent's next request. A seeding-time clock would therefore change every
        # recorded request, and a protocol that cannot be replayed is not frozen.
        with tempfile.TemporaryDirectory() as directory:
            seeds = []
            for name in ("first", "second"):
                store = CaptureStore(Path(directory) / f"{name}.sqlite3")
                store.initialize()
                seeds.append(seed_database(store))

        def stamps(seed: dict) -> list:
            rows = [seed["task"], *(row for row in seed["other_project_rows"] if hasattr(row, "state"))]
            return [(row.id, row.created_at, row.updated_at) for row in rows]

        self.assertEqual(stamps(seeds[0]), stamps(seeds[1]))
        self.assertTrue(stamps(seeds[0]), "no tracker rows were checked")


class CaseSelectionTests(unittest.TestCase):
    def test_no_selection_runs_every_case(self) -> None:
        self.assertEqual(selected_cases(None), set(ALL_CASES))

    def test_a_subset_is_kept_as_given(self) -> None:
        self.assertEqual(
            selected_cases(["no_match", "project_isolation"]), {"no_match", "project_isolation"}
        )

    def test_an_unknown_case_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown case"):
            selected_cases(["no_such_case"])

    def test_a_case_that_needs_an_earlier_submit_refuses_to_run_without_it(self) -> None:
        # answer_replay re-submits the meeting question's own message, so a replay without
        # the first submit would measure nothing and silently pass.
        with self.assertRaisesRegex(ValueError, "answer_replay needs meeting_question"):
            selected_cases(["answer_replay"])

    def test_the_dependency_is_satisfied_when_both_are_selected(self) -> None:
        self.assertEqual(
            selected_cases(["answer_replay", "meeting_question"]),
            {"answer_replay", "meeting_question"},
        )


if __name__ == "__main__":
    unittest.main()
