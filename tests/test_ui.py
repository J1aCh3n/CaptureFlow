import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from captureflow.models import Capture, Conversation, MemoryItem
from captureflow.storage import CaptureStore


class UiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.previous_database_path = os.environ.get("CAPTUREFLOW_DATABASE_PATH")
        self.previous_model_settings = {
            key: os.environ.get(key)
            for key in (
                "CAPTUREFLOW_API_KEY",
                "CAPTUREFLOW_MODEL",
                "CAPTUREFLOW_BASE_URL",
                "CAPTUREFLOW_TIMEZONE",
            )
        }
        os.environ["CAPTUREFLOW_DATABASE_PATH"] = str(
            Path(self.temporary_directory.name) / "captureflow.sqlite3"
        )
        for key in self.previous_model_settings:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        if self.previous_database_path is None:
            os.environ.pop("CAPTUREFLOW_DATABASE_PATH", None)
        else:
            os.environ["CAPTUREFLOW_DATABASE_PATH"] = self.previous_database_path
        for key, value in self.previous_model_settings.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.temporary_directory.cleanup()

    def test_phase_three_shell_renders_without_automatic_model_behavior(self) -> None:
        app = AppTest.from_file("app.py").run()

        self.assertEqual(app.exception, [])
        self.assertEqual(app.title[0].value, "CaptureFlow")
        self.assertIn("Phase 3", app.caption[0].value)

    def test_memory_page_displays_and_undoes_an_automatic_memory(self) -> None:
        app = AppTest.from_file("app.py").run()
        store = CaptureStore(os.environ["CAPTUREFLOW_DATABASE_PATH"])
        store.initialize()
        conversation = store.save_conversation(Conversation(id="conversation-memory", title="Memory"))
        capture = store.save_capture(
            Capture(
                id="capture-memory",
                conversation_id=conversation.id,
                source_text="I prefer concise notes.",
            )
        )
        memory = store.save_memory(
            MemoryItem(
                id="memory-auto",
                memory_type="preference",
                content="Prefers concise notes.",
                source_capture_id=capture.id,
                source_quote="I prefer concise notes.",
                write_mode="auto",
                confidence=0.95,
            ),
            event_action="created_auto",
        )

        app.sidebar.radio[0].set_value("Memory").run()

        self.assertEqual(app.exception, [])
        self.assertTrue(any(memory.content in element.value for element in app.markdown))
        next(button for button in app.button if button.key == f"undo-{memory.id}").click().run()
        self.assertEqual(store.get_memory(memory.id).status, "deleted")

    def test_manual_capture_can_be_saved_and_deleted_from_shell(self) -> None:
        source_text = "Keep this hand-authored capture unchanged."
        app = AppTest.from_file("app.py").run()

        app.text_area[0].input(source_text).run()
        app.button[1].click().run()

        self.assertEqual(app.exception, [])
        self.assertIn(source_text, [element.value for element in app.text])

        delete_button = next(
            button for button in app.button if (button.key or "").startswith("delete-")
        )
        delete_button.click().run()

        self.assertEqual(app.exception, [])
        self.assertNotIn(source_text, [element.value for element in app.text])

    def test_analyze_without_model_settings_saves_a_visible_configuration_failure(self) -> None:
        app = AppTest.from_file("app.py").run()
        app.text_area[0].input("A capture that needs analysis.").run()
        next(button for button in app.button if button.label == "Save capture").click().run()

        next(
            button for button in app.button if (button.key or "").startswith("analyze-")
        ).click().run()

        self.assertEqual(app.exception, [])
        self.assertIn("configuration", app.error[0].value)
        self.assertIn("CAPTUREFLOW_API_KEY", app.error[0].value)
