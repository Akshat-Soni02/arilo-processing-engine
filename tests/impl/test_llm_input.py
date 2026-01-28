import unittest
from unittest.mock import patch
from impl.llm_input import get_llm_input
from config.config import Llm_Call, User_Input_Type, Plan_Type


class TestLlmInput(unittest.TestCase):
    @patch("impl.llm_input.read_file")
    def test_get_llm_input_structure(self, mock_read_file):
        """Verify get_llm_input returns correct structure based on LLM_CONFIG."""
        mock_read_file.side_effect = lambda path, is_json=False: (
            "test_content" if not is_json else {"schema": "test"}
        )

        # We don't necessarily need to patch LLM_CONFIG if we use default values,
        # but let's verify it uses the values from the dictionary.
        result = get_llm_input(Llm_Call.SMART, plan_type=Plan_Type.FREE)

        # FREE SMART uses Gemini Flash
        self.assertEqual(result["model"], "gemini-2.5-flash")
        self.assertEqual(result["prompt"], "test_content")
        self.assertEqual(result["response_schema"], {"schema": "test"})

    @patch("impl.llm_input.read_file")
    def test_plan_type_overrides(self, mock_read_file):
        """Verify PRO plan uses different model than FREE."""
        mock_read_file.return_value = "content"

        free_result = get_llm_input(Llm_Call.SMART, plan_type=Plan_Type.FREE)
        pro_result = get_llm_input(Llm_Call.SMART, plan_type=Plan_Type.PRO)

        self.assertEqual(free_result["model"], "gemini-2.5-flash")
        self.assertEqual(pro_result["model"], "gemini-2.5-pro")
        self.assertEqual(pro_result["token_limit"], 100000)

    @patch("impl.llm_input.read_file")
    def test_replacements(self, mock_read_file):
        """Verify replacements work in prompts and system instructions."""
        mock_read_file.return_value = "Hello {{name}}"

        replace = [
            {"type": "prompt", "replace_key": "{{name}}", "replace_value": "World"},
            {"type": "sys", "replace_key": "{{name}}", "replace_value": "System"},
        ]

        result = get_llm_input(Llm_Call.STT, replace=replace)

        self.assertEqual(result["prompt"], "Hello World")
        self.assertEqual(result["system_instruction"], "Hello System")

    def test_invalid_llm_call(self):
        """Verify None returned for unknown LLM call type."""
        # We need to suppress the key error or handle it in get_llm_input
        result = get_llm_input("UNKNOWN_TYPE")
        self.assertIsNone(result)

    @patch("impl.llm_input.read_file")
    def test_with_input_data(self, mock_read_file):
        """Verify input data and type are included in result."""
        mock_read_file.return_value = "content"

        input_bytes = b"audio_data"
        input_type = User_Input_Type.AUDIO_WAV

        result = get_llm_input(Llm_Call.SMART, input=input_bytes, input_type=input_type)

        self.assertEqual(result["user_data"], input_bytes)
        self.assertEqual(result["input_type"], input_type)
