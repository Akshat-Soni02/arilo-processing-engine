import unittest
from unittest.mock import MagicMock
from impl.context_utils import format_sentences, prepare_context_for_noteback
from pipeline.exceptions import FatalPipelineError


class TestContextUtils(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()

    # --- format_sentences tests ---

    def test_format_sentences_valid(self):
        """Verify sentences are formatted correctly with importance scores."""
        context_response = {
            "input_to_sentences": [
                {"sentence": "Test sentence 1", "importance_score": 0.9},
                {"sentence": "Test sentence 2", "importance_score": 0.5},
            ]
        }
        expected = [
            "Test sentence 1, importance_score: 0.90",
            "Test sentence 2, importance_score: 0.50",
        ]

        result = format_sentences(context_response)
        self.assertEqual(result, expected)

    def test_format_sentences_malformed(self):
        """Verify FatalPipelineError on missing fields."""
        context_response = {
            "input_to_sentences": [
                {"sentence": "Missing score"},
            ]
        }

        with self.assertRaises(FatalPipelineError):
            format_sentences(context_response)

    def test_format_sentences_invalid_types(self):
        """Verify FatalPipelineError on invalid types."""
        # Generic function catching errors in loop and raising FatalPipelineError
        context_response = {
            "input_to_sentences": [
                {"sentence": 123, "importance_score": 0.9},
            ]
        }
        with self.assertRaises(FatalPipelineError):
            format_sentences(context_response)

    def test_format_sentences_empty(self):
        """Verify FatalPipelineError on empty sentences list."""
        context_response = {"input_to_sentences": []}
        with self.assertRaises(FatalPipelineError):
            format_sentences(context_response)

    # --- prepare_context_for_noteback tests ---

    def test_prepare_context_success(self):
        """Verify correct context preparation from vector DB results."""
        context_response = {"search_anchors": ["anchor1"]}

        # Mock vector DB response
        self.mock_db.similarity_search.return_value = (
            [{"sentence_text": "similar text", "combined_score": 0.88}],
            100,  # chars used
        )

        expected = ["sentence_text: similar text, value_score: 0.88"]

        result = prepare_context_for_noteback(context_response, self.mock_db, "test_user")

        self.assertEqual(result, expected)
        self.mock_db.similarity_search.assert_called_once()

    def test_prepare_context_no_anchors(self):
        """Verify FatalPipelineError when no search anchors exist."""
        context_response = {"search_anchors": []}
        with self.assertRaises(FatalPipelineError):
            prepare_context_for_noteback(context_response, self.mock_db, "test_user")

    def test_prepare_context_db_uninitialized(self):
        """Verify FatalPipelineError when DB is None."""
        with self.assertRaises(FatalPipelineError):
            prepare_context_for_noteback({"search_anchors": ["a"]}, None, "test_user")

    def test_prepare_context_search_failure(self):
        """Verify FatalPipelineError when similarity search fails completely."""
        context_response = {"search_anchors": ["anchor1"]}
        self.mock_db.similarity_search.side_effect = Exception("DB Error")

        with self.assertRaises(FatalPipelineError):
            prepare_context_for_noteback(context_response, self.mock_db, "test_user")
