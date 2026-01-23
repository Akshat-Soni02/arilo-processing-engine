import unittest
from unittest.mock import MagicMock, patch
from impl.gemini import GeminiProvider


class TestGeminiProvider(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.provider = GeminiProvider(self.mock_client)
        self.input_data = {
            "model": "gemini-test",
            "token_limit": 100,
            "prompt": "test prompt",
            "system_instruction": "test instruction",
            "response_schema": None,
            "input_type": None,
            "user_data": None,
        }

    @patch("impl.gemini.time.time")
    def test_process_no_input(self, mock_time):
        """Verify process works with no input_type or user_data."""
        mock_time.return_value = 1000.0

        # Mock generate_content response
        mock_response = MagicMock()
        mock_response.text = '{"output": "success"}'
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        mock_response.usage_metadata.candidates_token_count = 10
        mock_response.usage_metadata.thoughts_token_count = 0
        self.mock_client.models.generate_content.return_value = mock_response

        # Mock count_tokens (called twice: prompt_part and inside calculate_metrics)
        self.mock_client.models.count_tokens.return_value.total_tokens = 5

        response_json, metrics = self.provider.process(self.input_data)

        self.assertEqual(response_json, {"output": "success"})
        self.assertEqual(metrics["model"], "gemini-test")
        self.assertEqual(metrics["input_tokens"], 0)  # input_part is None
        self.assertEqual(metrics["prompt_tokens"], 5)

        # Verify generate_content was called with only prompt in contents
        args, kwargs = self.mock_client.models.generate_content.call_args
        contents = kwargs["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(len(contents[0].parts), 1)
        self.assertEqual(contents[0].parts[0].text, "test prompt")

    @patch("impl.gemini.time.time")
    def test_process_empty_string_input(self, mock_time):
        """Verify process works with empty string input_type and user_data."""
        mock_time.return_value = 1000.0
        self.input_data["input_type"] = ""
        self.input_data["user_data"] = b""

        mock_response = MagicMock()
        mock_response.text = '{"output": "success"}'
        mock_response.candidates = [MagicMock(finish_reason="STOP")]
        self.mock_client.models.generate_content.return_value = mock_response
        self.mock_client.models.count_tokens.return_value.total_tokens = 5

        response_json, metrics = self.provider.process(self.input_data)

        self.assertEqual(response_json, {"output": "success"})

        # Verify generate_content was called with only prompt in contents
        args, kwargs = self.mock_client.models.generate_content.call_args
        contents = kwargs["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(len(contents[0].parts), 1)
        self.assertEqual(contents[0].parts[0].text, "test prompt")


if __name__ == "__main__":
    unittest.main()
