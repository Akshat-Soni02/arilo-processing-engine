from common.logging import get_logger
from config.settings import UPSTREAM_URL as API_URL

import requests

logger = get_logger(__name__)


def upstream_call(upstream_output: dict):
    logger.info(f"passing output to upstream: {upstream_output}")
    try:
        response = requests.post(f"{API_URL}/processed-output", json=upstream_output)
        response.raise_for_status()
        logger.info(f"Successfully pushed response to upstream: {response.text}")
    except requests.exceptions.HTTPError as e:
        error_message = e.response.json().get("error", "Unknown API error")
        logger.error(f"Error pushing response to upstream: {error_message}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error pushing response to upstream: {e}")
