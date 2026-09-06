"""
utils.py
--------
Defensive utility helpers for MediGuide AI:

  - safe_parse_json(): robustly extract/clean/parse a JSON object out of
    an LLM's raw text response, even if it wrapped the JSON in markdown
    code fences or added stray commentary.
  - validate_assessment_schema(): sanity-check that a parsed dict has the
    fields MediGuide AI's UI expects, filling safe defaults for anything
    missing rather than crashing.
  - contains_emergency_keywords(): a client-side, pre-LLM safety net that
    scans free-text patient input for obvious emergency red-flag phrases.
  - clean_text(): basic whitespace/string cleanup helper.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.config import EMERGENCY_KEYWORDS, URGENCY_LEVELS

logger = logging.getLogger(__name__)

# Fields the app's dashboard expects on every assessment result, and the
# safe fallback value used if the model's response is missing that field.
_DEFAULT_ASSESSMENT_FIELDS: dict[str, Any] = {
    "summary": "No summary was returned. Please try submitting the assessment again.",
    "possible_conditions": [],
    "urgency_level": "MEDIUM",
    "recommended_next_steps": [
        "We couldn't fully process this assessment — please consult a "
        "healthcare professional to review your symptoms."
    ],
    "questions_for_doctor": [],
    "warning_signs": [],
}


def clean_text(text: str) -> str:
    """Trim and collapse excess whitespace in a string."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _strip_markdown_fences(raw: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` code fences that LLMs sometimes
    wrap around JSON output, despite instructions not to.
    """
    raw = raw.strip()
    fence_pattern = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
    match = fence_pattern.match(raw)
    if match:
        return match.group(1).strip()
    return raw


def _extract_json_substring(raw: str) -> str:
    """
    As a last resort, extract the substring between the first '{' and the
    last '}' — this rescues cases where the model added a stray sentence
    before or after an otherwise-valid JSON object.
    """
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start : end + 1]
    return raw


def safe_parse_json(raw_response: str) -> dict[str, Any] | None:
    """
    Attempt to parse a JSON object out of a raw LLM text response,
    defensively handling common formatting issues.

    Returns:
        The parsed dict on success, or None if parsing failed after all
        recovery attempts (caller should show a friendly error, never crash).
    """
    if not raw_response or not isinstance(raw_response, str):
        logger.warning("safe_parse_json received empty or non-string input.")
        return None

    candidate = _strip_markdown_fences(raw_response)

    # First attempt: parse as-is.
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.info("First JSON parse attempt failed (%s); trying substring extraction.", exc)

    # Second attempt: isolate the outermost { ... } block and retry.
    candidate2 = _extract_json_substring(candidate)
    try:
        return json.loads(candidate2)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("JSON parsing failed after cleanup attempts: %s", exc)
        return None


def validate_assessment_schema(parsed: dict[str, Any] | None) -> dict[str, Any]:
    """
    Ensure a parsed assessment dict has every field the UI needs, with
    safe fallback defaults for anything missing or malformed. Never
    raises — always returns a usable dict.
    """
    if parsed is None or not isinstance(parsed, dict):
        logger.warning("validate_assessment_schema received invalid input; using all defaults.")
        return dict(_DEFAULT_ASSESSMENT_FIELDS)

    result: dict[str, Any] = {}
    for field, default in _DEFAULT_ASSESSMENT_FIELDS.items():
        value = parsed.get(field, default)
        if value is None:
            value = default
        result[field] = value

    # Normalize / validate urgency_level specifically, since the UI's
    # color-coding logic depends on it being one of the known enum values.
    urgency = str(result.get("urgency_level", "MEDIUM")).strip().upper()
    if urgency not in URGENCY_LEVELS:
        logger.warning("Unrecognized urgency_level '%s'; defaulting to MEDIUM.", urgency)
        urgency = "MEDIUM"
    result["urgency_level"] = urgency

    # Defensive type coercion for list fields.
    for list_field in ("possible_conditions", "recommended_next_steps", "questions_for_doctor", "warning_signs"):
        if not isinstance(result.get(list_field), list):
            result[list_field] = []

    return result


def contains_emergency_keywords(*text_fields: str) -> list[str]:
    """
    Scan one or more free-text fields for known emergency red-flag
    phrases, as a client-side defense-in-depth check that runs BEFORE any
    LLM call. This never overrides the model's own assessment — it only
    supplements it with an immediate, guaranteed-visible warning.

    Returns:
        A list of the matched keyword phrases (empty if none found).
    """
    combined = " ".join(t for t in text_fields if t).lower()
    matches = [kw for kw in EMERGENCY_KEYWORDS if kw in combined]
    return matches
