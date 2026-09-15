import tempfile
import unittest
from datetime import date
from pathlib import Path

from captureflow.models import CalendarEvent, Conversation, Message, Project, ProjectRecord, SavedItem
from captureflow.read_agent_tools import BoundReadTools
from captureflow.storage import CaptureStore


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
        tools = BoundReadTools.for_conversation(
            self.store, self.conversation.id, "local_answer"
        )

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
        self.assertIn("this_project_only", calendar_schema["properties"])

    def test_unassigned_local_answer_does_not_offer_project_records(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="unassigned", title="Inbox"))
        tools = BoundReadTools.for_conversation(self.store, conversation.id, "local_answer")

        names = {definition["function"]["name"] for definition in tools.definitions_for_model()}

        self.assertEqual(names, {"search_calendar", "search_saved_items"})
        denied = tools.invoke("search_project_records", {})
        self.assertFalse(denied.ok)
        self.assertEqual(denied.error.category, "policy_denied")

    def test_calendar_scope_can_only_be_narrowed_and_ledgers_returned_rows(self) -> None:
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
        tools = BoundReadTools.for_conversation(
            self.store, self.conversation.id, "local_answer"
        )

        workspace_result = tools.invoke(
            "search_calendar",
            {"start_date": "2026-09-20", "end_date": "2026-09-21", "limit": 1},
        )
        project_result = tools.invoke(
            "search_calendar",
            {
                "start_date": "2026-09-20",
                "end_date": "2026-09-21",
                "this_project_only": True,
            },
        )

        self.assertEqual([item["id"] for item in workspace_result.items], [same_project.id])
        self.assertTrue(workspace_result.truncated)
        self.assertEqual(workspace_result.scope, "workspace")
        self.assertEqual([item["id"] for item in project_result.items], [same_project.id])
        self.assertEqual(project_result.scope, "project")
        self.assertIn(same_project.id, tools.observed_resources)
        self.assertNotIn(other_event.id, tools.observed_resources)

    def test_project_answer_forces_project_scope_and_hides_narrowing_parameter(self) -> None:
        event = self._event("project-event", "Roadmap", date(2026, 9, 22))
        tools = BoundReadTools.for_conversation(
            self.store, self.conversation.id, "project_answer"
        )
        calendar_schema = next(
            definition["function"]["parameters"]
            for definition in tools.definitions_for_model()
            if definition["function"]["name"] == "search_calendar"
        )

        result = tools.invoke(
            "search_calendar", {"start_date": "2026-09-22", "end_date": "2026-09-22"}
        )

        self.assertNotIn("this_project_only", calendar_schema["properties"])
        self.assertEqual(result.scope, "project")
        self.assertEqual([item["id"] for item in result.items], [event.id])

    def test_calendar_can_search_by_keyword_without_inventing_a_date_range(self) -> None:
        matching = self._event("japan-flight", "Japan flight booking", date(2027, 4, 10))
        self._event("design-review", "Design review", date(2026, 9, 22))
        tools = BoundReadTools.for_conversation(
            self.store, self.conversation.id, "local_answer"
        )

        result = tools.invoke("search_calendar", {"keywords": ["Japan", "flight"]})

        self.assertTrue(result.ok)
        self.assertEqual([item["id"] for item in result.items], [matching.id])
        self.assertEqual(result.scope, "workspace")

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
        tools = BoundReadTools.for_conversation(
            self.store, self.conversation.id, "project_answer"
        )

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

    def test_invalid_or_credential_bearing_arguments_return_typed_errors(self) -> None:
        tools = BoundReadTools.for_conversation(
            self.store, self.conversation.id, "local_answer"
        )

        invalid_range = tools.invoke(
            "search_calendar", {"start_date": "2026-09-01", "end_date": "2026-12-01"}
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
        tools = BoundReadTools.for_conversation(
            self.store, self.conversation.id, "local_answer"
        )

        unpaired = tools.invoke("search_calendar", {"start_date": "2026-09-01"})
        unbounded = tools.invoke("search_calendar", {})
        malformed = tools.invoke(
            "search_calendar",
            {
                "start_date": "private-date-sentinel",
                "end_date": "2026-09-02",
            },
        )

        self.assertIn("start_date and end_date", unpaired.error.message)
        self.assertIn("keywords", unbounded.error.message)
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

        in_project = BoundReadTools.for_conversation(self.store, self.conversation.id, "local_answer")
        project_result = in_project.invoke("search_saved_items", {"keywords": ["latency"]})
        unassigned = BoundReadTools.for_conversation(self.store, inbox.id, "local_answer")
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
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id, "project_answer")
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
        tools = BoundReadTools.for_conversation(self.store, self.conversation.id, "local_answer")
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

    def test_project_answer_requires_an_assigned_project(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="no-project", title="No project"))

        with self.assertRaisesRegex(ValueError, "assigned project"):
            BoundReadTools.for_conversation(self.store, conversation.id, "project_answer")

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
