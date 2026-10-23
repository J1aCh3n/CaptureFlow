import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.models import CalendarEvent, Conversation, Message, Project, ProjectRecord, SavedItem
from captureflow.read_agent_tools import BoundReadTools
from captureflow.storage import CaptureStore


class _RecordingWriter:
    """Stands in for the router's write path: stores the row and remembers what it wrote."""

    def __init__(self, store: CaptureStore, conversation_id: str, source_message_id: str) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.source_message_id = source_message_id
        self.saved: list[str] = []

    @property
    def wrote(self) -> bool:
        return bool(self.saved)

    def save_item(self, kind: str, content: str) -> SavedItem:
        source = self.store.get_message(self.source_message_id)
        item = self.store.save_saved_item(
            SavedItem(
                id=f"written-{len(self.saved)}",
                conversation_id=self.conversation_id,
                kind=kind,
                content=content,
                source_message_id=source.id,
                source_quote=source.content,
            )
        )
        self.saved.append(content)
        return item


class BoundReadToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.conversation = self.store.save_conversation(
            Conversation(id="current-conversation", title="Current")
        )
        self.project = self.store.save_project(Project(id="current-project", name="Current"))
        self.store.assign_conversation_project(self.conversation.id, self.project.id)
        self.source = self.store.save_message(
            Message(
                id="source-message",
                conversation_id=self.conversation.id,
                role="user",
                content="Project planning source.",
            )
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_model_definitions_never_expose_scope_identifiers(self) -> None:
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        definitions = tools.definitions_for_model()
        serialized = repr(definitions)

        self.assertEqual(
            {definition["function"]["name"] for definition in definitions},
            {"search_calendar", "search_project_records", "search_saved_items"},
        )
        self.assertNotIn("workspace_id", serialized)
        self.assertNotIn("project_id", serialized)
        self.assertNotIn("conversation_id", serialized)
        calendar_schema = next(
            definition["function"]["parameters"]
            for definition in definitions
            if definition["function"]["name"] == "search_calendar"
        )
        self.assertIn("scope", calendar_schema["properties"])

    def test_an_unassigned_conversation_does_not_offer_project_records(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="unassigned", title="Inbox"))
        tools = BoundReadTools.for_conversation(self.store, conversation.id)

        names = {definition["function"]["name"] for definition in tools.definitions_for_model()}

        self.assertEqual(names, {"search_calendar", "search_saved_items"})
        denied = tools.invoke("search_project_records", {})
        self.assertFalse(denied.ok)
        self.assertEqual(denied.error.category, "policy_denied")

    def test_calendar_scope_defaults_to_the_project_and_widens_only_within_the_workspace(self) -> None:
        same_project = self._event("same-project-event", "Design review", date(2026, 9, 20))
        other_conversation = self.store.save_conversation(
            Conversation(id="other-conversation", title="Other")
        )
        other_project = self.store.save_project(Project(id="other-project", name="Other"))
        self.store.assign_conversation_project(other_conversation.id, other_project.id)
        other_source = self.store.save_message(
            Message(
                id="other-source",
                conversation_id=other_conversation.id,
                role="user",
                content="Other source.",
            )
        )
        other_event = self._event(
            "other-project-event",
            "Design review elsewhere",
            date(2026, 9, 21),
            conversation_id=other_conversation.id,
            source_message_id=other_source.id,
            source_quote=other_source.content,
        )
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        # No scope given: the conversation's own project, as the router's project route used
        # to force. Widening is the model's explicit choice and is recorded per row.
        default_result = tools.invoke(
            "search_calendar", {"start_date": "2026-09-20", "end_date": "2026-09-21"}
        )

        self.assertEqual([item["id"] for item in default_result.items], [same_project.id])
        self.assertEqual(default_result.scope, "project")
        self.assertEqual(default_result.total, 1)
        self.assertEqual(tools.observed_resources[same_project.id].scope_kind, "project")
        self.assertNotIn(other_event.id, tools.observed_resources)

        widened = tools.invoke(
            "search_calendar",
            {"start_date": "2026-09-20", "end_date": "2026-09-21", "scope": "workspace", "limit": 1},
        )

        self.assertEqual([item["id"] for item in widened.items], [same_project.id])
        self.assertTrue(widened.truncated)
        self.assertEqual(widened.scope, "workspace")
        self.assertEqual(widened.total, 2)
        # Widening stays inside the workspace, and only rows actually returned are ledgered.
        self.assertNotIn(other_event.id, tools.observed_resources)

    def test_a_conversation_without_a_project_is_not_offered_a_scope_to_choose(self) -> None:
        # Offering a parameter whose only legal value is the default invites a guaranteed
        # error, so an unassigned conversation simply searches its workspace.
        conversation = self.store.save_conversation(Conversation(id="inbox", title="Inbox"))
        source = self.store.save_message(
            Message(
                id="inbox-source",
                conversation_id=conversation.id,
                role="user",
                content="Roadmap 2026-09-22",
            )
        )
        event = self._event(
            "inbox-event",
            "Roadmap",
            date(2026, 9, 22),
            conversation_id=conversation.id,
            source_message_id=source.id,
            source_quote=source.content,
        )
        tools = BoundReadTools.for_conversation(self.store, conversation.id)
        calendar_schema = next(
            definition["function"]["parameters"]
            for definition in tools.definitions_for_model()
            if definition["function"]["name"] == "search_calendar"
        )

        result = tools.invoke(
            "search_calendar", {"start_date": "2026-09-22", "end_date": "2026-09-22"}
        )

        self.assertNotIn("scope", calendar_schema["properties"])
        self.assertEqual(result.scope, "workspace")
        self.assertEqual([item["id"] for item in result.items], [event.id])

    def test_calendar_can_search_by_keyword_without_inventing_a_date_range(self) -> None:
        matching = self._event("japan-flight", "Japan flight booking", date(2027, 4, 10))
        self._event("design-review", "Design review", date(2026, 9, 22))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        result = tools.invoke("search_calendar", {"keywords": ["Japan", "flight"]})

        self.assertTrue(result.ok)
        self.assertEqual([item["id"] for item in result.items], [matching.id])
        self.assertEqual(result.scope, "project")

    def test_calendar_can_list_all_dates_or_use_open_ended_ranges(self) -> None:
        past = self._event("past", "Past review", date(2026, 9, 1))
        future = self._event("future", "Future review", date(2027, 4, 10))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        for query, expected in (
            ({}, [past.id, future.id]),
            ({"start_date": "2026-09-09"}, [future.id]),
            ({"end_date": "2026-09-09"}, [past.id]),
            ({"start_date": "2026-09-01", "end_date": "2027-04-10"}, [past.id, future.id]),
        ):
            with self.subTest(query=query):
                result = tools.invoke("search_calendar", query)
                self.assertTrue(result.ok)
                self.assertEqual([row["id"] for row in result.items], expected)

    def test_record_and_saved_item_filters_are_scoped_and_deterministic(self) -> None:
        record = self.store.save_project_record(
            ProjectRecord(
                id="matching-record",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Measure agent latency",
                state="active",
                source_message_id=self.source.id,
                source_quote=self.source.content,
            )
        )
        item = self.store.save_saved_item(
            SavedItem(
                id="matching-note",
                conversation_id=self.conversation.id,
                kind="note",
                content="Latency budget is three model calls",
                source_message_id=self.source.id,
                source_quote=self.source.content,
            )
        )
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        records = tools.invoke(
            "search_project_records", {"kinds": ["task"], "keywords": ["latency"]}
        )
        notes = tools.invoke(
            "search_saved_items", {"kinds": ["note"], "keywords": ["budget"]}
        )

        self.assertEqual([row["id"] for row in records.items], [record.id])
        self.assertEqual([row["id"] for row in notes.items], [item.id])
        self.assertEqual(tools.observed_resources[record.id].resource_type, "project_record")
        self.assertEqual(tools.observed_resources[item.id].resource_type, "saved_item")

    def test_listing_tasks_uses_kind_and_state_without_content_keywords(self) -> None:
        task = self.store.save_project_record(
            ProjectRecord(
                id="latency-task", project_id=self.project.id, conversation_id=self.conversation.id,
                kind="task", content="排查 ROS 2 节点延迟", state="active",
                source_message_id=self.source.id, source_quote=self.source.content,
            )
        )
        self.store.save_project_record(task.model_copy(update={"id": "closed-task", "state": "completed"}))
        self.store.save_project_record(task.model_copy(update={"id": "decision", "kind": "decision"}))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        listed = tools.invoke("search_project_records", {"kinds": ["task"]})
        narrowed = tools.invoke("search_project_records", {"kinds": ["task"], "keywords": ["数据库"]})

        self.assertEqual([row["id"] for row in listed.items], [task.id])
        # The kind filter still holds; the unmatched keyword only stops ordering the rows.
        self.assertEqual([row["id"] for row in narrowed.items], [task.id])
        self.assertEqual(narrowed.keyword_matches, 0)

    def test_chinese_keywords_match_phrases_instead_of_shared_characters(self) -> None:
        flight = self._event("flight", "日本机票预订", date(2026, 9, 20))
        self._event("robot", "机械臂延迟评审会议", date(2026, 9, 21))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        result = tools.invoke("search_calendar", {"keywords": ["日本", "机票"]})

        self.assertTrue(result.ok)
        self.assertEqual([row["id"] for row in result.items], [flight.id])

    def test_keywords_match_visible_record_fields_not_other_items_in_the_source(self) -> None:
        source = self.store.save_message(
            Message(id="mixed-source", conversation_id=self.conversation.id, role="user",
                    content="Book a Japan flight and schedule a design review.")
        )
        self._event("review", "Design review", date(2026, 9, 20),
                    source_message_id=source.id, source_quote=source.content)
        self._note("review-note", "Design review notes", self.conversation.id, source.id)
        self.store.save_project_record(
            ProjectRecord(
                id="review-task", project_id=self.project.id, conversation_id=self.conversation.id,
                kind="task", content="Prepare design review", state="active",
                source_message_id=source.id, source_quote=source.content,
            )
        )
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        for name in ("search_calendar", "search_project_records", "search_saved_items"):
            with self.subTest(tool=name):
                result = tools.invoke(name, {"keywords": ["Japan"]})
                self.assertTrue(result.ok)
                # "Japan" is only in the source message, not in any item's own text, so
                # nothing matches and the scope is reported instead of claimed empty.
                self.assertEqual(result.keyword_matches, 0)
                self.assertEqual(result.total, 1)

    def test_latin_keyword_phrases_are_case_insensitive_and_do_not_match_partial_words(self) -> None:
        exact = self._event("exact", "ROS 2 CAT review", date(2026, 9, 20))
        self._event("unrelated", "ROS 1 category review", date(2026, 9, 21))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        for keywords in (["ros 2"], ["cat"]):
            with self.subTest(keywords=keywords):
                result = tools.invoke("search_calendar", {"keywords": keywords})
                self.assertEqual([row["id"] for row in result.items], [exact.id])

    def test_calendar_pages_reach_all_matches_in_order(self) -> None:
        events = [self._event(f"event-{index:02}", "Review", date(2026, 9, 20)) for index in range(25)]
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)
        query = {"start_date": "2026-09-20", "end_date": "2026-09-20", "limit": 20}

        first = tools.invoke("search_calendar", query)
        second = tools.invoke("search_calendar", {**query, "offset": 20})
        end = tools.invoke("search_calendar", {**query, "offset": 25})

        self.assertTrue(first.truncated)
        self.assertTrue(second.ok)
        self.assertFalse(second.truncated)
        self.assertEqual([row["id"] for row in [*first.items, *second.items]], [event.id for event in events])
        self.assertTrue(end.ok)
        self.assertEqual(end.items, [])
        self.assertFalse(end.truncated)
        self.assertEqual({row.id for row in tools.resolve_observed([event.id for event in events])},
                         {event.id for event in events})

    def test_record_and_note_pages_apply_filters_before_offset(self) -> None:
        record = ProjectRecord(
            project_id=self.project.id, conversation_id=self.conversation.id, kind="task",
            content="Latency measurement", state="active",
            source_message_id=self.source.id, source_quote=self.source.content,
        )
        for index in range(5):
            self.store.save_project_record(record.model_copy(update={"id": f"task-{index}"}))
            self._note(f"note-{index}", "Latency measurement", self.conversation.id, self.source.id)
        self.store.save_project_record(record.model_copy(update={"id": "other", "content": "Design review"}))
        self._note("other-note", "Design review", self.conversation.id, self.source.id)
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        for name in ("search_project_records", "search_saved_items"):
            with self.subTest(tool=name):
                query = {"keywords": ["latency"], "limit": 3}
                first = tools.invoke(name, query)
                second = tools.invoke(name, {**query, "offset": 3})
                ids = [row["id"] for row in [*first.items, *second.items]]
                self.assertTrue(second.ok)
                self.assertTrue(first.truncated)
                self.assertFalse(second.truncated)
                self.assertEqual(len(ids), 5)
                self.assertEqual(len(set(ids)), 5)
                self.assertNotIn("other", ids)
                self.assertNotIn("other-note", ids)

    def test_keywords_that_match_nothing_return_the_scope_they_searched(self) -> None:
        # An empty result claims the scope holds nothing, which is a different fact from
        # "your words do not appear in it". Judging whether a row answers the question is
        # the model's work, so the tool reports and does not decide.
        task = self.store.save_project_record(
            ProjectRecord(
                id="ros-task", project_id=self.project.id, conversation_id=self.conversation.id,
                kind="task", content="排查 ROS 2 时间同步", state="active",
                source_message_id=self.source.id, source_quote=self.source.content,
            )
        )
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        result = tools.invoke("search_project_records", {"keywords": ["待办"]})

        self.assertTrue(result.ok)
        self.assertEqual([row["id"] for row in result.items], [task.id])
        self.assertEqual(result.keyword_matches, 0)
        self.assertEqual(result.total, 1)

    def test_rows_matching_more_keywords_rank_first_and_are_counted(self) -> None:
        both = self._event("both", "Japan flight booking", date(2026, 9, 20))
        one = self._event("one", "Flight log review", date(2026, 9, 21))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        result = tools.invoke("search_calendar", {"keywords": ["Japan", "flight"]})

        self.assertEqual([row["id"] for row in result.items], [both.id, one.id])
        self.assertEqual(result.keyword_matches, 2)
        self.assertEqual(result.total, 2)

    def test_a_search_without_keywords_reports_no_keyword_count(self) -> None:
        first = self._event("first", "Design review", date(2026, 9, 20))
        second = self._event("second", "Budget review", date(2026, 9, 21))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        result = tools.invoke("search_calendar", {})

        self.assertEqual([row["id"] for row in result.items], [first.id, second.id])
        self.assertIsNone(result.keyword_matches)
        self.assertEqual(result.total, 2)

    def test_total_counts_the_selected_set_and_not_the_returned_page(self) -> None:
        for index in range(5):
            self._event(f"match-{index}", "Latency review", date(2026, 9, 20))
        self._event("other", "Design review", date(2026, 9, 20))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)
        query = {"keywords": ["latency"], "limit": 3}

        first = tools.invoke("search_calendar", query)
        second = tools.invoke("search_calendar", {**query, "offset": 3})
        past_end = tools.invoke("search_calendar", {**query, "offset": 5})

        for page in (first, second, past_end):
            with self.subTest(offset=page.items):
                self.assertEqual(page.total, 5)
                self.assertEqual(page.keyword_matches, 5)
        self.assertTrue(first.truncated)
        self.assertFalse(second.truncated)
        # Paging past the end stays empty: the fallback answers an unmatched keyword, not an
        # exhausted page.
        self.assertEqual(past_end.items, [])

    def test_unmatched_keywords_never_widen_the_structural_filters(self) -> None:
        self._event("past", "Past review", date(2026, 9, 1))
        future = self._event("future", "Future review", date(2027, 4, 10))
        closed = ProjectRecord(
            id="closed-task", project_id=self.project.id, conversation_id=self.conversation.id,
            kind="task", content="Measure latency", state="completed",
            source_message_id=self.source.id, source_quote=self.source.content,
        )
        self.store.save_project_record(closed)
        self.store.save_project_record(closed.model_copy(update={"id": "open-decision", "kind": "decision", "state": "active"}))
        open_task = self.store.save_project_record(closed.model_copy(update={"id": "open-task", "state": "active"}))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        dated = tools.invoke("search_calendar", {"keywords": ["待办"], "start_date": "2026-09-09"})
        typed = tools.invoke("search_project_records", {"keywords": ["待办"], "kinds": ["task"]})

        self.assertEqual([row["id"] for row in dated.items], [future.id])
        self.assertEqual(dated.total, 1)
        self.assertEqual(dated.keyword_matches, 0)
        # The fallback returns what kind and state allowed, and nothing they excluded.
        self.assertEqual([row["id"] for row in typed.items], [open_task.id])
        self.assertEqual(typed.total, 1)
        self.assertEqual(typed.keyword_matches, 0)

    def test_a_failed_result_reports_no_counts(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="unassigned", title="Unassigned"))
        tools = BoundReadTools.for_conversation(self.store, conversation.id)

        refused = tools.invoke("search_project_records", {})

        self.assertFalse(refused.ok)
        self.assertEqual(refused.error.category, "policy_denied")
        self.assertIsNone(refused.total)
        self.assertIsNone(refused.keyword_matches)

    # The first write tool

    def test_a_note_cannot_be_saved_without_a_writer(self) -> None:
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        offered = {definition["function"]["name"] for definition in tools.definitions_for_model()}
        refused = tools.invoke("save_note", {"content": "Latency work starts Monday."})

        self.assertNotIn("save_note", offered)
        self.assertFalse(refused.ok)
        self.assertEqual(refused.error.category, "policy_denied")

    def test_a_saved_note_is_stored_and_can_be_cited_as_its_own_source(self) -> None:
        writer = _RecordingWriter(self.store, self.conversation.id, self.source.id)
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id, writer=writer)

        offered = {definition["function"]["name"] for definition in tools.definitions_for_model()}
        result = tools.invoke("save_note", {"content": "Latency work starts Monday."})

        self.assertIn("save_note", offered)
        self.assertTrue(result.ok)
        stored = self.store.list_saved_items(workspace_id=self.conversation.workspace_id, kinds=("note",))
        self.assertEqual([item.content for item in stored], ["Latency work starts Monday."])
        # The new row is ledgered, so the reply may cite the note it just saved.
        self.assertEqual([item["id"] for item in result.items], [stored[0].id])
        self.assertEqual(tools.resolve_observed([stored[0].id])[0].id, stored[0].id)

    def test_a_note_the_schema_refuses_writes_nothing(self) -> None:
        for content, reason in (("", "at least 1"), ("   ", "blank"), ("x" * 4_001, "at most 4000")):
            with self.subTest(content=content[:12]):
                writer = _RecordingWriter(self.store, self.conversation.id, self.source.id)
                tools = BoundReadTools.for_conversation(self.store, self.conversation.id, writer=writer)

                refused = tools.invoke("save_note", {"content": content})

                self.assertFalse(refused.ok)
                self.assertEqual(refused.error.category, "retryable")
                self.assertIn(reason, refused.error.message)
                self.assertEqual(writer.saved, [])

    def test_a_note_carrying_a_credential_is_refused_before_it_is_written(self) -> None:
        writer = _RecordingWriter(self.store, self.conversation.id, self.source.id)
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id, writer=writer)

        refused = tools.invoke("save_note", {"content": "The api key is sk-abcdefghijklmnop"})

        self.assertFalse(refused.ok)
        self.assertEqual(refused.error.category, "policy_denied")
        self.assertEqual(writer.saved, [])

    def test_only_one_write_happens_per_turn(self) -> None:
        # One local action per turn is what keeps a stable ID unique and a crash
        # recoverable, so the second call is refused rather than written.
        writer = _RecordingWriter(self.store, self.conversation.id, self.source.id)
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id, writer=writer)

        first = tools.invoke("save_note", {"content": "First note."})
        second = tools.invoke("save_note", {"content": "Second note."})

        self.assertTrue(first.ok)
        self.assertFalse(second.ok)
        self.assertEqual(second.error.category, "conflict")
        self.assertEqual(writer.saved, ["First note."])
        self.assertTrue(tools.wrote_this_turn)

    def test_decision_history_exposes_replacement_and_update_time(self) -> None:
        original = self.store.save_project_record(
            ProjectRecord(
                id="original-decision",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="decision",
                content="Use SQLite",
                state="active",
                source_message_id=self.source.id,
                source_quote=self.source.content,
            )
        )
        replacement = self.store.save_project_record(
            original.model_copy(
                update={"id": "replacement-decision", "content": "Use PostgreSQL", "supersedes_record_id": original.id}
            )
        )
        original = self.store.transition_project_record(original.id, "superseded")
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        result = tools.invoke("search_project_records", {"kinds": ["decision"], "states": ["active", "superseded"]})
        rows = {row["id"]: row for row in result.items}

        self.assertEqual(rows[replacement.id]["supersedes_record_id"], original.id)
        self.assertIsNone(rows[original.id]["supersedes_record_id"])
        self.assertEqual(rows[replacement.id]["updated_at"], replacement.updated_at.isoformat())
        self.assertEqual(rows[original.id]["updated_at"], original.updated_at.isoformat())
        self.assertEqual({row.id for row in tools.resolve_observed(list(rows))}, {original.id, replacement.id})

    def test_invalid_or_credential_bearing_arguments_return_typed_errors(self) -> None:
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        invalid_range = tools.invoke(
            "search_calendar", {"start_date": "2026-12-01", "end_date": "2026-09-01"}
        )
        credential = tools.invoke(
            "search_saved_items", {"keywords": ["api key is sk-abcdefghijklmnop"]}
        )

        self.assertFalse(invalid_range.ok)
        self.assertEqual(invalid_range.error.category, "retryable")
        self.assertFalse(credential.ok)
        self.assertEqual(credential.error.category, "policy_denied")
        self.assertEqual(tools.observed_resources, {})

    def test_calendar_validation_explains_the_rule_without_echoing_values(self) -> None:
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)

        reversed_range = tools.invoke("search_calendar", {"start_date": "2026-09-02", "end_date": "2026-09-01"})
        malformed = tools.invoke(
            "search_calendar",
            {
                "start_date": "private-date-sentinel",
                "end_date": "2026-09-02",
            },
        )

        self.assertIn("end_date must not be before start_date", reversed_range.error.message)
        self.assertIn("start_date", malformed.error.message)
        self.assertNotIn("private-date-sentinel", malformed.error.message)

    def test_notes_stay_in_the_project_for_a_local_answer_too(self) -> None:
        own = self._note("own-note", "Latency budget for this project", self.conversation.id, self.source.id)
        other_conversation = self.store.save_conversation(Conversation(id="other-notes", title="Other"))
        other_project = self.store.save_project(Project(id="other-notes-project", name="Other"))
        self.store.assign_conversation_project(other_conversation.id, other_project.id)
        other_source = self.store.save_message(
            Message(id="other-note-source", conversation_id=other_conversation.id, role="user", content="Other.")
        )
        other = self._note("other-note", "Latency budget elsewhere", other_conversation.id, other_source.id)
        inbox = self.store.save_conversation(Conversation(id="inbox", title="Inbox"))

        in_project = BoundReadTools.for_conversation(self.store, self.conversation.id)
        project_result = in_project.invoke("search_saved_items", {"keywords": ["latency"]})
        unassigned = BoundReadTools.for_conversation(self.store, inbox.id)
        workspace_result = unassigned.invoke("search_saved_items", {"keywords": ["latency"]})

        self.assertEqual(project_result.scope, "project")
        self.assertEqual([row["id"] for row in project_result.items], [own.id])
        self.assertNotIn(other.id, in_project.observed_resources)
        self.assertEqual(workspace_result.scope, "workspace")
        self.assertEqual({row["id"] for row in workspace_result.items}, {own.id, other.id})

    def test_a_citation_is_read_again_and_refused_once_it_left_its_state_or_scope(self) -> None:
        event = self._event("fading-event", "Design review", date(2026, 9, 20))
        note = self._note("moving-note", "Design review notes", self.conversation.id, self.source.id)
        kept = self._event("kept-event", "Design review follow-up", date(2026, 9, 21))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)
        tools.invoke("search_calendar", {"start_date": "2026-09-20", "end_date": "2026-09-21"})
        tools.invoke("search_saved_items", {"keywords": ["design"]})

        self.store.set_calendar_event_status(event.id, "deleted")
        with self.assertRaisesRegex(ValueError, "state or scope"):
            tools.resolve_observed([event.id])

        elsewhere = self.store.save_project(Project(id="elsewhere", name="Elsewhere"))
        self.store.assign_conversation_project(self.conversation.id, elsewhere.id)
        with self.assertRaisesRegex(ValueError, "state or scope"):
            tools.resolve_observed([note.id])
        with self.assertRaisesRegex(ValueError, "not returned by a tool"):
            tools.resolve_observed(["never-returned"])

        self.store.assign_conversation_project(self.conversation.id, self.project.id)
        self.assertEqual([row.id for row in tools.resolve_observed([kept.id])], [kept.id])

    def test_a_citation_whose_fields_changed_after_the_tool_returned_it_is_refused(self) -> None:
        event = self._event("moved-event", "Design review", date(2026, 9, 20))
        note = self._note("edited-note", "Budget is three model calls", self.conversation.id, self.source.id)
        task = self.store.save_project_record(
            ProjectRecord(
                id="finished-task",
                project_id=self.project.id,
                conversation_id=self.conversation.id,
                kind="task",
                content="Measure agent latency",
                state="active",
                source_message_id=self.source.id,
                source_quote=self.source.content,
            )
        )
        unchanged = self._event("unchanged-event", "Design review follow-up", date(2026, 9, 21))
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id)
        tools.invoke("search_calendar", {"start_date": "2026-09-20", "end_date": "2026-09-21"})
        tools.invoke("search_saved_items", {})
        tools.invoke("search_project_records", {})

        self.store.update_calendar_event(event.id, title=event.title, start_at=date(2026, 9, 28))
        self.store.update_saved_item(note.id, content="Budget is ten model calls")
        self.store.transition_project_record(task.id, "completed")

        for changed in (event.id, note.id, task.id):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "changed after"):
                tools.resolve_observed([changed])
        self.assertEqual([row.id for row in tools.resolve_observed([unchanged.id])], [unchanged.id])

    def test_an_unassigned_conversation_binds_to_its_workspace_without_a_route(self) -> None:
        # The agent runs before any intent exists, so the scope comes from the conversation
        # itself rather than from a route the router chose.
        conversation = self.store.save_conversation(Conversation(id="no-project", title="No project"))

        tools = BoundReadTools.for_conversation(self.store, conversation.id)

        self.assertIsNone(tools.project)
        self.assertEqual(
            {definition["function"]["name"] for definition in tools.definitions_for_model()},
            {"search_calendar", "search_saved_items"},
        )
        self.assertEqual(tools.invoke("search_calendar", {}).scope, "workspace")

    def _note(self, item_id: str, content: str, conversation_id: str, source_message_id: str) -> SavedItem:
        source = self.store.get_message(source_message_id)
        return self.store.save_saved_item(
            SavedItem(
                id=item_id,
                conversation_id=conversation_id,
                kind="note",
                content=content,
                source_message_id=source.id,
                source_quote=source.content,
            )
        )

    def _event(
        self,
        event_id: str,
        title: str,
        start_at: date,
        *,
        conversation_id: str | None = None,
        source_message_id: str | None = None,
        source_quote: str | None = None,
    ) -> CalendarEvent:
        return self.store.save_calendar_event(
            CalendarEvent(
                id=event_id,
                conversation_id=conversation_id or self.conversation.id,
                title=title,
                source_message_id=source_message_id or self.source.id,
                source_quote=source_quote or self.source.content,
                start_at=start_at,
                all_day=True,
                timezone="America/Vancouver",
            )
        )
