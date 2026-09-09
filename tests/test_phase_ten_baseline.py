import json
import unittest
from pathlib import Path

from captureflow.local_context_service import (
    is_local_context_query,
    is_local_event_statement,
    is_local_write_request,
)
from captureflow.retrieval import classify_project_query


def legacy_ui_service_for_assigned_project(prompt: str) -> str:
    """Mirror the current non-shortcut branch in ui.render_chat_input()."""

    if is_local_write_request(prompt) or is_local_event_statement(prompt):
        return "model_router"
    if is_local_context_query(prompt):
        return "local_context"
    return "project_retrieval" if classify_project_query(prompt) is not None else "project_tracker"


class PhaseTenBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "phase10_contextual_router_cases.json"
        cls.fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    def test_fixture_reproduces_current_assigned_project_dispatch(self) -> None:
        for case in self.fixture["cases"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(
                    legacy_ui_service_for_assigned_project(case["input"]),
                    case["observed_service"],
                )

    def test_fixture_declares_one_bounded_target_for_every_case(self) -> None:
        valid_intents = {
            "calendar_create",
            "calendar_update",
            "tracker_change",
            "local_answer",
            "clarify",
        }
        valid_writes = {"create_local_event", "update_local_event", "project_record", "none"}
        for case in self.fixture["cases"]:
            with self.subTest(case=case["name"]):
                self.assertIn(case["target_intent"], valid_intents)
                self.assertIn(case["expected_write"], valid_writes)

