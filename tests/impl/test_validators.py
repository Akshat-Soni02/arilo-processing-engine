import unittest
from pipeline.exceptions import TransientPipelineError
from impl.validators import (
    is_latin_script,
    is_snake_case,
    validate_stt_response,
    validate_smart_context_response,
    validate_noteback_response,
)


class TestValidators(unittest.TestCase):

    # --- Helper Tests ---
    def test_is_latin_script(self):
        self.assertTrue(is_latin_script("Hello World"))
        self.assertTrue(is_latin_script("Café Déjà vu"))  # Latin-1 Supplement
        self.assertTrue(is_latin_script("12345 !@#$%"))
        self.assertFalse(is_latin_script("नमस्ते"))  # Devanagari
        self.assertFalse(is_latin_script("Hello नमस्ते"))  # Mixed
        self.assertFalse(is_latin_script("你好"))  # CJK

    def test_is_snake_case(self):
        self.assertTrue(is_snake_case("hello_world"))
        self.assertTrue(is_snake_case("test"))
        self.assertTrue(is_snake_case("test_123"))
        self.assertFalse(is_snake_case("Hello_World"))  # Uppercase
        self.assertFalse(is_snake_case("hello world"))  # Space
        self.assertFalse(is_snake_case("hello-world"))  # Hyphen

    # --- STT Validation Tests ---
    def test_validate_stt_valid(self):
        valid_response = {
            "stt": "Hello world",
            "tasks": ["do this"],
            "anxiety_score": 3,
            "language": "en",
            "tags": ["tag_one", "tag_two"],
        }
        try:
            validate_stt_response(valid_response)
        except TransientPipelineError:
            self.fail("validate_stt_response raised error on valid input")

    def test_validate_stt_invalid_script(self):
        invalid_response = {
            "stt": "Hello नमस्ते",
            "tasks": [],
            "anxiety_score": 1,
            "language": "hi",
            "tags": [],
        }
        with self.assertRaises(TransientPipelineError):
            validate_stt_response(invalid_response)

    def test_validate_stt_invalid_tag_format(self):
        invalid_response = {
            "stt": "Hello",
            "tasks": [],
            "anxiety_score": 1,
            "language": "en",
            "tags": ["camelCase"],
        }
        with self.assertRaises(TransientPipelineError):
            validate_stt_response(invalid_response)

    def test_validate_stt_missing_keys(self):
        with self.assertRaises(TransientPipelineError):
            validate_stt_response({"stt": "hi"})

    def test_validate_stt_anxiety_range(self):
        invalid_response = {
            "stt": "Hello",
            "tasks": [],
            "anxiety_score": 6,
            "language": "en",
            "tags": [],
        }
        with self.assertRaises(TransientPipelineError):
            validate_stt_response(invalid_response)

    # --- SMART Context Validation Tests ---
    def test_validate_smart_valid(self):
        valid_response = {
            "input_to_sentences": [{"sentence": "Hello", "importance_score": 0.5}],
            "search_anchors": ["anchor_one"],
        }
        try:
            validate_smart_context_response(valid_response)
        except TransientPipelineError:
            self.fail("validate_smart_context_response raised error on valid input")

    def test_validate_smart_invalid_sentence_script(self):
        invalid_response = {
            "input_to_sentences": [{"sentence": "नमस्ते", "importance_score": 0.5}],
            "search_anchors": ["anchor"],
        }
        with self.assertRaises(TransientPipelineError):
            validate_smart_context_response(invalid_response)

    def test_validate_smart_invalid_score_range(self):
        invalid_response = {
            "input_to_sentences": [{"sentence": "Hello", "importance_score": 1.5}],
            "search_anchors": ["anchor"],
        }
        with self.assertRaises(TransientPipelineError):
            validate_smart_context_response(invalid_response)

    def test_validate_smart_anchor_limit(self):
        invalid_response = {"input_to_sentences": [], "search_anchors": []}  # Min 1 required
        with self.assertRaises(TransientPipelineError):
            validate_smart_context_response(invalid_response)

    # --- Noteback Validation Tests ---
    def test_validate_noteback_valid(self):
        valid_response = {"noteback": "This is a note.", "reasoning_trace": "Because."}
        try:
            validate_noteback_response(valid_response)
        except TransientPipelineError:
            self.fail("validate_noteback_response raised error on valid input")

    def test_validate_noteback_invalid_script(self):
        invalid_response = {"noteback": "Note containing नमस्ते.", "reasoning_trace": "Trace"}
        with self.assertRaises(TransientPipelineError):
            validate_noteback_response(invalid_response)


if __name__ == "__main__":
    unittest.main()
