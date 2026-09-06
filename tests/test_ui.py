import unittest

from streamlit.testing.v1 import AppTest


class UiTests(unittest.TestCase):
    def test_phase_one_shell_renders_without_model_behavior(self) -> None:
        app = AppTest.from_file("app.py").run()

        self.assertEqual(app.exception, [])
        self.assertEqual(app.title[0].value, "CaptureFlow")
        self.assertIn("No model is called.", app.caption[0].value)

    def test_manual_capture_can_be_saved_and_deleted_from_shell(self) -> None:
        source_text = "Keep this hand-authored capture unchanged."
        app = AppTest.from_file("app.py").run()

        app.text_area[0].input(source_text).run()
        app.button[1].click().run()

        self.assertEqual(app.exception, [])
        self.assertIn(source_text, [element.value for element in app.text])

        app.button[-1].click().run()

        self.assertEqual(app.exception, [])
        self.assertEqual(app.info[0].value, "No captures in this conversation yet.")
