"""
validators.py

Ensures LLM outputs meet strict schema, language, and structural requirements.
Violations raise TransientPipelineError to trigger retries.
"""

import re
from pipeline.exceptions import TransientPipelineError


def is_latin_script(text: str) -> bool:
    """
    Check if text contains only Latin script based characters (English, French, German, etc.).
    Reject if it contains scripts like Devanagari, CJK, Arabic, etc.
    Allows: Basic Latin, Latin-1 Supplement, Latin Extended-A, General Punctuation.
    """
    if not text:
        return True

    # Range includes:
    # \u0000-\u007F: Basic Latin (ASCII)
    # \u0080-\u00FF: Latin-1 Supplement
    # \u0100-\u017F: Latin Extended-A
    # \u2000-\u206F: General Punctuation
    return bool(re.match(r"^[\u0000-\u007F\u0080-\u00FF\u0100-\u017F\u2000-\u206F\s]+$", text))


def is_snake_case(text: str) -> bool:
    """Check if text is in snake_case (lowercase, underscores, alphanumeric)."""
    return bool(re.match(r"^[a-z0-9_]+$", text))


def validate_schema(data: dict, required_keys: list, type_map: dict) -> None:
    """Generic schema validation helper."""
    for key in required_keys:
        if key not in data:
            raise TransientPipelineError(f"Missing required key: {key}")

    for key, expected_type in type_map.items():
        if key in data and not isinstance(data[key], expected_type):
            raise TransientPipelineError(
                f"Invalid type for {key}: expected {expected_type.__name__}, got {type(data[key]).__name__}"
            )


def validate_stt_response(response: dict) -> None:
    """Validate STT pipeline output."""
    required = ["stt", "tasks", "anxiety_score", "language", "tags"]
    type_map = {"stt": str, "tasks": list, "anxiety_score": int, "language": str, "tags": list}

    validate_schema(response, required, type_map)

    # 1. Content Checks
    if not is_latin_script(response["stt"]):
        raise TransientPipelineError("STT text contains non-Latin script characters")

    # 2. Tasks Check
    for task in response["tasks"]:
        if not isinstance(task, str):
            raise TransientPipelineError("Task items must be strings")
        if not is_latin_script(task):
            raise TransientPipelineError(f"Task contains non-Latin script: {task}")

    # 3. Anxiety Score Range
    if not (1 <= response["anxiety_score"] <= 5):
        raise TransientPipelineError(
            f"Anxiety score {response['anxiety_score']} out of range (1-5)"
        )

    # 4. Tags Check
    for tag in response["tags"]:
        if not isinstance(tag, str):
            raise TransientPipelineError("Tags must be strings")
        if not is_latin_script(tag):
            raise TransientPipelineError(f"Tag contains non-Latin script: {tag}")
        if not is_snake_case(tag):
            raise TransientPipelineError(f"Tag not in snake_case: {tag}")


def validate_smart_context_response(response: dict) -> None:
    """Validate SMART Context pipeline output."""
    required = ["input_to_sentences", "search_anchors"]
    type_map = {"input_to_sentences": list, "search_anchors": list}

    validate_schema(response, required, type_map)

    # 1. Input to Sentences
    for item in response["input_to_sentences"]:
        if not isinstance(item, dict):
            raise TransientPipelineError("input_to_sentences items must be objects")

        if "sentence" not in item or "importance_score" not in item:
            raise TransientPipelineError("Missing keys in input_to_sentences item")

        sentence = item["sentence"]
        score = item["importance_score"]

        if not isinstance(sentence, str):
            raise TransientPipelineError("Sentence must be a string")
        if not is_latin_script(sentence):
            raise TransientPipelineError("Sentence contains non-Latin script")

        if not (isinstance(score, (int, float)) and 0 <= score <= 1):
            raise TransientPipelineError(f"Importance score {score} out of range (0-1)")

    # 2. Search Anchors
    if not (1 <= len(response["search_anchors"]) <= 3):
        raise TransientPipelineError("search_anchors must contain 1-3 items")

    for anchor in response["search_anchors"]:
        if not isinstance(anchor, str):
            raise TransientPipelineError("Search anchors must be strings")
        if not is_latin_script(anchor):
            raise TransientPipelineError(f"Search anchor contains non-Latin script: {anchor}")


def validate_noteback_response(response: dict) -> None:
    """Validate Noteback pipeline output."""
    required = ["noteback", "reasoning_trace"]
    type_map = {"noteback": str, "reasoning_trace": str}

    validate_schema(response, required, type_map)

    if not is_latin_script(response["noteback"]):
        raise TransientPipelineError("Noteback contains non-Latin script")
