import unittest
from unittest.mock import MagicMock
from impl.llm_processor import call_llm
from pipeline.exceptions import FatalPipelineError, TransientPipelineError


class TestLlmProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_provider = MagicMock()
        self.valid_input = {"model": "test-model", "prompt": "test prompt"}
        self.call_name = "TEST_CALL"

    def test_call_llm_success(self):
        """Verify successful LLM call returns response and metrics."""
        expected_response = {"text": "response"}
        expected_metrics = {"latency": 100}
        self.mock_provider.process.return_value = (expected_response, expected_metrics)

        response, metrics = call_llm(self.mock_provider, self.valid_input, self.call_name)

        self.assertEqual(response, expected_response)
        self.assertEqual(metrics, expected_metrics)
        self.mock_provider.process.assert_called_once_with(self.valid_input)

    def test_call_llm_missing_provider(self):
        """Verify None returned if provider is missing."""
        response, metrics = call_llm(None, self.valid_input, self.call_name)
        self.assertIsNone(response)
        self.assertIsNone(metrics)

    def test_call_llm_invalid_input(self):
        """Verify None returned if input is invalid."""
        response, metrics = call_llm(self.mock_provider, None, self.call_name)
        self.assertIsNone(response)
        self.assertIsNone(metrics)

        response, metrics = call_llm(self.mock_provider, "not-a-dict", self.call_name)
        self.assertIsNone(response)
        self.assertIsNone(metrics)

    def test_call_llm_provider_returns_none(self):
        """Verify None response handled correctly."""
        self.mock_provider.process.return_value = (None, {"latency": 100})

        response, metrics = call_llm(self.mock_provider, self.valid_input, self.call_name)

        self.assertIsNone(response)
        self.assertEqual(metrics, {"latency": 100})

    def test_call_llm_generic_exception(self):
        """Verify generic exceptions are caught and return None."""
        self.mock_provider.process.side_effect = Exception("Unexpected error")

        response, metrics = call_llm(self.mock_provider, self.valid_input, self.call_name)

        self.assertIsNone(response)
        self.assertIsNone(metrics)

    def test_call_llm_propagates_fatal_error(self):
        """Verify FatalPipelineError is re-raised."""
        self.mock_provider.process.side_effect = FatalPipelineError("Fatal error")

        with self.assertRaises(FatalPipelineError):
            call_llm(self.mock_provider, self.valid_input, self.call_name)

    def test_call_llm_propagates_transient_error(self):
        """Verify TransientPipelineError is re-raised."""
        self.mock_provider.process.side_effect = TransientPipelineError("Transient error")

        with self.assertRaises(TransientPipelineError):
            call_llm(self.mock_provider, self.valid_input, self.call_name)
