import unittest
from unittest.mock import Mock, patch

from google.api_core.exceptions import DeadlineExceeded

from utils import TokenCounter


class TokenCounterRetryTests(unittest.TestCase):
    def test_transient_timeout_is_retried_without_regenerating(self):
        counter = TokenCounter.__new__(TokenCounter)
        counter.model_name = "models/gemini-3-flash-preview"
        counter.model = Mock()
        counter.model.count_tokens.side_effect = [
            DeadlineExceeded("temporary timeout"),
            Mock(total_tokens=123),
        ]

        with patch("utils.time.sleep") as sleep:
            self.assertEqual(counter.token_count("prompt"), 123)

        self.assertEqual(counter.model.count_tokens.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_non_transient_error_is_not_retried(self):
        counter = TokenCounter.__new__(TokenCounter)
        counter.model_name = "models/gemini-3-flash-preview"
        counter.model = Mock()
        counter.model.count_tokens.side_effect = ValueError("bad prompt")

        with patch("utils.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "bad prompt"):
                counter.token_count("prompt")

        counter.model.count_tokens.assert_called_once_with("prompt")
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
