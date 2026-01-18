import unittest
from unittest.mock import patch
from impl.llm_input import get_llm_input
from config.config import Llm_Call, User_Input_Type


class TestLlmInput(unittest.TestCase):
    @patch("impl.llm_input.read_file")
    def test_get_llm_input_structure(self, mock_read_file):
        """Verify get_llm_input returns correct structure for Smart call."""
        mock_read_file.side_effect = lambda path, is_json=False: (
            "test_content" if not is_json else {"schema": "test"}
        )

        # Patch config values
        with patch("impl.llm_input.Context_Call_Config") as mock_config:
            mock_config.PROMPT_FILE_PATH = "prompt.txt"
            mock_config.SYSTEM_INSTRUCTION_FILE_PATH = "sys.txt"
            mock_config.RESPONSE_SCHEMA_FILE_PATH = "schema.json"
            mock_config.MODEL = "gemini-test"
            mock_config.TOKEN_LIMIT = 500

            result = get_llm_input(Llm_Call.SMART)

            self.assertEqual(result["model"], "gemini-test")
            self.assertEqual(result["token_limit"], 500)
            self.assertEqual(result["prompt"], "test_content")
            self.assertEqual(result["system_instruction"], "test_content")
            self.assertEqual(result["response_schema"], {"schema": "test"})

    @patch("impl.llm_input.read_file")
    def test_replacements(self, mock_read_file):
        """Verify replacements work in prompts and system instructions."""
        mock_read_file.return_value = "Hello {{name}}"

        replace = [
            {"type": "prompt", "replace_key": "{{name}}", "replace_value": "World"},
            {"type": "sys", "replace_key": "{{name}}", "replace_value": "System"},
        ]

        with patch("impl.llm_input.Stt_Call_Config") as mock_config:
            mock_config.PROMPT_FILE_PATH = "p.txt"
            mock_config.SYSTEM_INSTRUCTION_FILE_PATH = "s.txt"
            mock_config.RESPONSE_SCHEMA_FILE_PATH = None
            mock_config.MODEL = "test"
            mock_config.TOKEN_LIMIT = 100

            result = get_llm_input(Llm_Call.STT, replace=replace)

            self.assertEqual(result["prompt"], "Hello World")
            self.assertEqual(result["system_instruction"], "Hello System")

    @patch("impl.llm_input.read_file")
    def test_missing_config_files(self, mock_read_file):
        """Verify behavior when config file paths are None."""
        with patch("impl.llm_input.Noteback_Call_Config") as mock_config:
            mock_config.PROMPT_FILE_PATH = None
            mock_config.SYSTEM_INSTRUCTION_FILE_PATH = None
            mock_config.RESPONSE_SCHEMA_FILE_PATH = None
            mock_config.MODEL = "test"
            mock_config.TOKEN_LIMIT = 100

            result = get_llm_input(Llm_Call.NOTEBACK)

            self.assertIsNone(result["prompt"])
            self.assertIsNone(result["system_instruction"])
            self.assertIsNone(result["response_schema"])
            mock_read_file.assert_not_called()

    def test_invalid_llm_call(self):
        """Verify None returned for unknown LLM call type."""
        result = get_llm_input("UNKNOWN_TYPE")
        self.assertIsNone(result)

    @patch("impl.llm_input.read_file")
    def test_with_input_data(self, mock_read_file):
        """Verify input data and type are included in result."""
        mock_read_file.return_value = "content"

        with patch("impl.llm_input.Context_Call_Config") as mock_config:
            mock_config.PROMPT_FILE_PATH = "p.txt"
            mock_config.SYSTEM_INSTRUCTION_FILE_PATH = None
            mock_config.RESPONSE_SCHEMA_FILE_PATH = None
            mock_config.MODEL = "test"
            mock_config.TOKEN_LIMIT = 100

            input_bytes = b"audio_data"
            input_type = User_Input_Type.AUDIO_WAV

            result = get_llm_input(Llm_Call.SMART, input=input_bytes, input_type=input_type)

            self.assertEqual(result["user_data"], input_bytes)
            self.assertEqual(result["input_type"], input_type)
