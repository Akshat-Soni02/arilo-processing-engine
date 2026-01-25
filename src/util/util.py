"""
Utility functions for external service communications.
Primarily handles data transmission to upstream systems via HTTP.
"""

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from common.logging import get_logger
from config.settings import UPSTREAM_URL as API_URL

logger = get_logger(__name__)


def upstream_call(upstream_output: dict):
    """
    Transmit processed pipeline output to the upstream API with exponential retry.

    Args:
        upstream_output (dict): The complete payload to send upstream.
    """
    logger.debug("Transmitting output to upstream", extra={"data": upstream_output})

    try:
        _do_post(upstream_output)
        logger.debug("Upstream transmission successful")
    except requests.exceptions.HTTPError as e:
        try:
            error_details = e.response.json()
            error_message = error_details.get(
                "error", error_details.get("message", "Unknown API error")
            )
        except (ValueError, AttributeError):
            error_message = f"HTTP {e.response.status_code}: {e.response.text[:100]}"

        logger.error(
            "Upstream HTTP error (final failure)",
            extra={"error": error_message, "status_code": e.response.status_code},
        )
    except requests.exceptions.RequestException as e:
        logger.critical("Upstream request failed (final failure)", extra={"error": str(e)})


def is_retryable_exception(exception):
    """
    Retry on connection issues, timeouts, or 5xx server errors.
    """
    if isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exception, requests.exceptions.HTTPError):
        # Retry on 5xx Server Errors
        return 500 <= exception.response.status_code < 600
    return False


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=is_retryable_exception,
    reraise=True,
)
def _do_post(payload: dict):
    """
    Perform the actual POST request with retry logic.
    """
    # Increased timeout to 60s
    response = requests.post(
        f"{API_URL}/api/v1/notes/engine/callback",
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response
