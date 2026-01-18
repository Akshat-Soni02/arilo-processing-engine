import unittest
from unittest.mock import MagicMock, patch
from pipeline.smart import SmartPipeline
from pipeline.exceptions import FatalPipelineError, TransientPipelineError
from config.config import Llm_Call


class TestSmartPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_smart_provider = MagicMock()
        self.mock_noteback_provider = MagicMock()
        self.mock_db = MagicMock()

        self.pipeline = SmartPipeline(
            smart_provider=self.mock_smart_provider,
            noteback_provider=self.mock_noteback_provider,
            db=self.mock_db,
        )

        self.context = {"pipeline_stage_id": "test_stage", "input_type": "AUDIO_WAV"}
        self.input_data = b"test_input"

    # --- Success Case ---

    @patch("pipeline.smart.get_llm_input")
    @patch("pipeline.smart.call_llm")
    @patch("pipeline.smart.prepare_context_for_noteback")
    @patch("pipeline.smart.format_sentences")
    def test_process_success(self, mock_format, mock_prep_context, mock_call_llm, mock_get_input):
        """Verify successful pipeline execution."""
        # Setup mocks
        smart_input = {"prompt": "smart"}
        noteback_input = {"prompt": "noteback"}
        mock_get_input.side_effect = [smart_input, noteback_input]

        smart_response = {"search_anchors": ["a1"]}
        smart_metrics = {"lat": 10}
        noteback_response = {"note": "final note"}
        noteback_metrics = {"lat": 20}

        # call_llm called twice: Smart, then Noteback
        mock_call_llm.side_effect = [
            (smart_response, smart_metrics),
            (noteback_response, noteback_metrics),
        ]

        mock_prep_context.return_value = ["formatted context"]
        mock_format.return_value = ["formatted note"]

        # Execute
        response, metrics = self.pipeline._process(self.input_data, self.context)

        # Assertions
        self.assertEqual(response, noteback_response)
        self.assertEqual(metrics, noteback_metrics)

        # Verify call order and args
        # 1. get_llm_input for Smart
        mock_get_input.assert_any_call(Llm_Call.SMART, self.input_data, "AUDIO_WAV")
        # 2. call_llm for Smart
        # 3. prepare_context & format_sentences
        mock_prep_context.assert_called_with(smart_response, self.mock_db)
        # 4. get_llm_input for Noteback (with replacements)
        # 5. call_llm for Noteback

    # --- Failure Cases ---

    def test_process_empty_input(self):
        """Verify FatalPipelineError on empty input."""
        with self.assertRaises(FatalPipelineError):
            self.pipeline._process(None, self.context)

    @patch("pipeline.smart.get_llm_input")
    def test_input_preparation_failure(self, mock_get_input):
        """Verify FatalPipelineError when input preparation fails."""
        mock_get_input.side_effect = Exception("Input error")
        with self.assertRaises(FatalPipelineError):
            self.pipeline._process(self.input_data, self.context)

    @patch("pipeline.smart.get_llm_input")
    @patch("pipeline.smart.call_llm")
    def test_smart_llm_failure(self, mock_call_llm, mock_get_input):
        """Verify propagation of FatalPipelineError from Smart LLM."""
        mock_get_input.return_value = {}
        mock_call_llm.side_effect = FatalPipelineError("LLM failed")

        with self.assertRaises(FatalPipelineError):
            self.pipeline._process(self.input_data, self.context)

    @patch("pipeline.smart.get_llm_input")
    @patch("pipeline.smart.call_llm")
    def test_context_prep_returned_none(self, mock_call_llm, mock_get_input):
        """Verify TransientPipelineError when context response is None."""
        mock_get_input.return_value = {}
        mock_call_llm.return_value = (None, None)

        with self.assertRaises(TransientPipelineError):
            self.pipeline._process(self.input_data, self.context)

    @patch("pipeline.smart.get_llm_input")
    @patch("pipeline.smart.call_llm")
    @patch("pipeline.smart.prepare_context_for_noteback")
    def test_context_utils_failure(self, mock_prep_context, mock_call_llm, mock_get_input):
        """Verify FatalPipelineError from context utils is propagated."""
        mock_get_input.return_value = {}
        mock_call_llm.return_value = ({"ok": True}, {})
        mock_prep_context.side_effect = FatalPipelineError("Bad context")

        with self.assertRaises(FatalPipelineError):
            self.pipeline._process(self.input_data, self.context)

    @patch("pipeline.smart.get_llm_input")
    @patch("pipeline.smart.call_llm")
    @patch("pipeline.smart.prepare_context_for_noteback")
    @patch("pipeline.smart.format_sentences")
    def test_noteback_llm_failure(
        self, mock_format, mock_prep_context, mock_call_llm, mock_get_input
    ):
        """Verify TransientPipelineError from Noteback LLM."""
        mock_get_input.return_value = {}
        mock_prep_context.return_value = []
        mock_format.return_value = []

        # Smart succeeds, Noteback fails
        mock_call_llm.side_effect = [({"ok": True}, {}), TransientPipelineError("Noteback busy")]

        with self.assertRaises(TransientPipelineError):
            self.pipeline._process(self.input_data, self.context)
