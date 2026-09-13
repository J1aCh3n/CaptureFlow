import unittest

from captureflow.credentials import contains_credential_disclosure


class CredentialGuardTests(unittest.TestCase):
    def test_disclosures_are_detected_in_both_languages(self) -> None:
        for text in (
            "my api key is sk-test-123",
            "api key: sk-test-123",
            "我的 api key 是 sk-test-123",
            "我的密钥是 sk-test-123",
            "密码是 hunter2",
            "口令：hunter2",
            "我的password是abc123",
            "token=sk-test-123",
            "访问令牌为 abc123",
        ):
            with self.subTest(text=text):
                self.assertTrue(contains_credential_disclosure(text))

    def test_known_credential_shapes_are_detected_without_a_noun(self) -> None:
        for text in (
            "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
            "AKIAIOSFODNN7EXAMPLE",
            "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5",
        ):
            with self.subTest(text=text):
                self.assertTrue(contains_credential_disclosure(text))

    def test_ordinary_text_is_not_treated_as_a_disclosure(self) -> None:
        for text in (
            "明天下午三点开需求评审",
            "密码学是一门学科",
            "我们讨论了密钥管理的方案",
            "tokenizer is fast",
            "the meeting is at 3",
            "Team sync tomorrow at 2 PM",
        ):
            with self.subTest(text=text):
                self.assertFalse(contains_credential_disclosure(text))
