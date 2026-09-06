"""
chains.py
---------
LLM client setup and chain construction for MediGuide AI.

Provides:
  - build_llm(): construct a ChatOpenAI client from the current session's
    API key / model choice.
  - build_structured_chain(): a chain that returns raw JSON text per the
    schema in prompts.py, suitable for utils.safe_parse_json().
  - build_narrative_chain(): a chain for the human-readable narrative.
  - stream_narrative(): a generator that yields narrative text chunks live,
    intended for use with st.write_stream() in app.py.
  - run_structured_assessment(): convenience wrapper that invokes the
    structured chain and returns the raw text response.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from langchain_openai import ChatOpenAI

from src.config import DEFAULT_MODEL_NAME, DEFAULT_TEMPERATURE, MAX_TOKENS
from src.prompts import build_narrative_prompt, build_structured_assessment_prompt

logger = logging.getLogger(__name__)


def build_llm(
    api_key: str,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    streaming: bool = False,
) -> ChatOpenAI:
    """
    Construct a ChatOpenAI client.

    Args:
        api_key: The user's OpenAI API key (from the login gate).
        model_name: Which chat model to use.
        temperature: Sampling temperature — kept low by default for
            consistent, cautious medical-education output.
        streaming: Whether this client will be used with `.stream()`.

    Returns:
        A configured ChatOpenAI instance.
    """
    if not api_key:
        raise ValueError("An OpenAI API key is required to build the LLM client.")

    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        streaming=streaming,
    )


def build_structured_chain(llm: ChatOpenAI):
    """
    Build the structured-JSON assessment chain: prompt | llm.

    Note: we intentionally do NOT attach an output parser here so that
    app.py can pass the raw text into utils.safe_parse_json(), which is
    more defensive than LangChain's built-in JSON parsers against models
    that ignore formatting instructions.
    """
    prompt = build_structured_assessment_prompt()
    return prompt | llm


def build_narrative_chain(llm: ChatOpenAI):
    """Build the streamed narrative-guidance chain: prompt | llm."""
    prompt = build_narrative_prompt()
    return prompt | llm


def _intake_kwargs_from_dict(patient_data: dict) -> dict:
    """
    Normalize a patient_data dict (as assembled in app.py) into the exact
    keyword arguments expected by prompts.PATIENT_INTAKE_TEMPLATE.
    """
    return {
        "age": patient_data.get("age", "Not provided"),
        "gender": patient_data.get("gender", "Not provided"),
        "symptoms_selected": patient_data.get("symptoms_selected", "None selected"),
        "symptoms_freetext": patient_data.get("symptoms_freetext", "None provided"),
        "duration": patient_data.get("duration", "Not provided"),
        "severity": patient_data.get("severity", "Not provided"),
        "existing_conditions": patient_data.get("existing_conditions", "None reported"),
        "current_medications": patient_data.get("current_medications", "None reported"),
        "additional_notes": patient_data.get("additional_notes", "None"),
        "language": patient_data.get("language", "English"),
    }


def run_structured_assessment(llm: ChatOpenAI, patient_data: dict) -> str:
    """
    Invoke the structured-JSON chain synchronously and return the raw
    text content of the model's response (to be parsed by
    utils.safe_parse_json()).
    """
    chain = build_structured_chain(llm)
    kwargs = _intake_kwargs_from_dict(patient_data)
    logger.info("Running structured assessment for patient intake.")
    response = chain.invoke(kwargs)
    return getattr(response, "content", str(response))


def stream_narrative(llm: ChatOpenAI, patient_data: dict) -> Iterator[str]:
    """
    Stream the human-readable narrative guidance chunk-by-chunk.

    This is a generator intended to be passed directly to Streamlit's
    `st.write_stream()`, e.g.:

        st.write_stream(stream_narrative(llm, patient_data))

    Args:
        llm: A ChatOpenAI client constructed with streaming=True.
        patient_data: The same patient intake dict used for the
            structured assessment.

    Yields:
        Successive string chunks of the narrative response.
    """
    chain = build_narrative_chain(llm)
    kwargs = _intake_kwargs_from_dict(patient_data)
    logger.info("Streaming narrative guidance for patient intake.")
    try:
        for chunk in chain.stream(kwargs):
            content = getattr(chunk, "content", None)
            if content:
                yield content
    except Exception as exc:  # noqa: BLE001 - surface any stream error as text
        logger.error("Error while streaming narrative: %s", exc)
        yield (
            "\n\n⚠️ We hit an error while generating your narrative guidance. "
            "Please try again, and consult a healthcare professional if "
            "symptoms are concerning.\n"
        )
