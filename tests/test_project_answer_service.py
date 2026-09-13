import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

from captureflow.llm import ModelSettings, OpenAICompatibleModel
from captureflow.models import CalendarEvent, Conversation, Message, Project, ProjectAnswer
from captureflow.project_answer_service import build_project_snapshot
from captureflow.storage import CaptureStore


class ProjectSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = CaptureStore(Path(self.temporary_directory.name) / "captureflow.sqlite3")
        self.store.initialize()
        self.project = self.store.save_project(Project(id="project", name="CaptureFlow"))
        self.other_project = self.store.save_project(Project(id="other-project", name="Other"))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_snapshot_keeps_only_upcoming_events_of_its_own_project(self) -> None:
        kept = self._event("kept", "project", "2026-09-10T14:00:00-07:00")
        self._event("past", "project", "2026-09-08T14:00:00-07:00")
        self._event("other", "other-project", "2026-09-10T15:00:00-07:00")

        snapshot = build_project_snapshot(self.store, self.project, date(2026, 9, 9))

        self.assertEqual([event.id for event in snapshot.upcoming_events], [kept.id])
        self.assertEqual(snapshot.for_model()["upcoming_events"][0]["start_at"], kept.start_at.isoformat())

    def _event(self, event_id: str, project_id: str, start: str) -> CalendarEvent:
        conversation = self.store.save_conversation(Conversation(id=f"{event_id}-chat", title=event_id))
        self.store.assign_conversation_project(conversation.id, project_id)
        source = self.store.save_message(
            Message(id=f"{event_id}-source", conversation_id=conversation.id, role="user", content="有个会议。")
        )
        return self.store.save_calendar_event(
            CalendarEvent(
                id=event_id,
                conversation_id=conversation.id,
                title=event_id,
                source_message_id=source.id,
                source_quote=source.content,
                start_at=datetime.fromisoformat(start),
                all_day=False,
                timezone="America/Vancouver",
            )
        )


class ProjectAnswerModelTests(unittest.TestCase):
    def test_the_answer_schema_tolerates_structural_noise(self) -> None:
        answer = ProjectAnswer.model_validate(
            {"reply": " 还剩一项。 ", "source_ids": None, "confidence": "high"}
        )
        repeated = ProjectAnswer.model_validate({"reply": "两项。", "source_ids": ["a", "b", "a"]})

        self.assertEqual((answer.reply, answer.source_ids), ("还剩一项。", []))
        self.assertEqual(repeated.source_ids, ["a", "b"])

    def test_the_second_call_sends_the_snapshot_and_forbids_restating_dates(self) -> None:
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            content = '{"reply":"还剩排查延迟一项。","source_ids":["task-1"]}'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        model = OpenAICompatibleModel.__new__(OpenAICompatibleModel)
        model.settings = ModelSettings(api_key="test-key", model="qwen3.8-flash")
        model.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        snapshot = {"project": {"name": "CaptureFlow"}, "open_records": [{"id": "task-1"}]}

        answer = model.answer_project_question(
            question={"message_id": "m1", "content": "项目还剩什么？"},
            recent_messages=[],
            project_snapshot=snapshot,
            reference_date="2026-09-09",
        )

        prompt = requests[0]["messages"][0]["content"]
        payload = json.loads(requests[0]["messages"][1]["content"])["input"]
        self.assertEqual(answer.source_ids, ["task-1"])
        self.assertIn("Do not write dates or clock times", prompt)
        self.assertIn("superseded", prompt)
        self.assertEqual(payload["project_snapshot"], snapshot)
