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

    def test_project_answer_requires_an_assigned_project(self) -> None:
        conversation = self.store.save_conversation(Conversation(id="no-project", title="No project"))

        with self.assertRaisesRegex(ValueError, "assigned project"):
            BoundReadTools.for_conversation(self.store, conversation.id, "project_answer")

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
