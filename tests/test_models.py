import unittest

from pydantic import ValidationError

from captureflow.models import Capture, Conversation, MAX_CAPTURE_CHARACTERS


class ModelTests(unittest.TestCase):
    def test_conversation_title_is_trimmed(self) -> None:
        conversation = Conversation(title="  Reading list  ")

        self.assertEqual(conversation.title, "Reading list")

    def test_blank_capture_is_invalid(self) -> None:
        with self.assertRaises(ValidationError):
            Capture(conversation_id="conversation-1", source_text="   ")

    def test_capture_limit_is_enforced(self) -> None:
        with self.assertRaises(ValidationError):
            Capture(
                conversation_id="conversation-1",
                source_text="x" * (MAX_CAPTURE_CHARACTERS + 1),
            )
