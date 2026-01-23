"""
Custom exception hierarchy for the processing pipeline.
Defines base, fatal, and transient error types for standardized error handling.
"""


class PipelineError(Exception):
    """Base class for all pipeline-related errors."""

    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error

    def __str__(self):
        if self.original_error:
            return f"{super().__str__()}: {str(self.original_error)}"
        return super().__str__()


class FatalPipelineError(PipelineError):
    """
    Represents unrecoverable errors that should stop pipeline execution.
    Examples: Missing configuration, unauthorized access, severe data corruption.
    """

    pass


class TransientPipelineError(PipelineError):
    """
    Represents recoverable or temporary errors.
    Examples: Network timeouts, service unavailability, transient upstream failures.
    """

    pass
