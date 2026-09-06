"""
prompts.py
----------
LangChain prompt construction for MediGuide AI.

This module defines:
  - The JSON schema the structured-assessment chain must return.
  - A strict SystemMessage that encodes MediGuide AI's safety behavior
    (no confirmed diagnoses, emergency escalation, cautious language).
  - A ChatPromptTemplate for the structured JSON assessment.
  - A separate ChatPromptTemplate for the streamed, human-readable
    narrative guidance shown live to the user.

Keeping the JSON schema as a Python constant (not just prose in the
prompt) lets utils.py validate the parsed response against the same
source of truth used to instruct the model.
"""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

# --------------------------------------------------------------------------
# JSON Schema (also embedded verbatim into the system prompt below so the
# model sees exactly the shape utils.py will validate against)
# --------------------------------------------------------------------------
ASSESSMENT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A brief, neutral summary of the patient's reported situation.",
        },
        "possible_conditions": {
            "type": "array",
            "description": (
                "Educational, non-diagnostic possibilities only. Never state "
                "these as confirmed."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "reason"],
            },
        },
        "urgency_level": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH", "EMERGENCY"],
        },
        "recommended_next_steps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "questions_for_doctor": {
            "type": "array",
            "items": {"type": "string"},
        },
        "warning_signs": {
            "type": "array",
            "description": "Signs that should prompt the patient to seek urgent/emergency care.",
            "items": {"type": "string"},
        },
    },
    "required": [
        "summary",
        "possible_conditions",
        "urgency_level",
        "recommended_next_steps",
        "questions_for_doctor",
        "warning_signs",
    ],
}

# A pretty-printed string version, for direct interpolation into prompts.
ASSESSMENT_JSON_SCHEMA_STR = json.dumps(ASSESSMENT_JSON_SCHEMA, indent=2)


# --------------------------------------------------------------------------
# System behavior contract shared by both the JSON and narrative chains
# --------------------------------------------------------------------------
SAFETY_SYSTEM_INSTRUCTIONS = """\
You are MediGuide AI, an EDUCATIONAL medical-information assistant. You are
NOT a licensed clinician and you are NOT providing medical care. You must
follow these rules at all times, without exception:

1. NEVER present a confirmed diagnosis. Only ever describe "possible",
   "commonly associated", or "educational" considerations. Use hedging
   language such as "may be associated with", "could potentially indicate",
   "one possibility among several is".
2. ALWAYS evaluate for emergency red-flag symptoms (e.g., chest pain with
   shortness of breath, signs of stroke, severe bleeding, difficulty
   breathing, suicidal ideation, anaphylaxis, loss of consciousness,
   severe abdominal pain, coughing up blood). If ANY plausible emergency
   is present, set urgency_level to "EMERGENCY" and make the very first
   recommended_next_step an instruction to seek emergency care immediately
   (e.g., call local emergency services or go to the nearest ER).
3. Err on the side of caution: when uncertain between two urgency levels,
   choose the higher one.
4. Never recommend specific prescription medications, dosages, or
   controlled substances. You may mention general categories of care
   (e.g., "rest and hydration", "seeing a primary care physician",
   "over-the-counter options as directed by a pharmacist") without naming
   specific drugs or doses.
5. Always encourage follow-up with a qualified healthcare professional
   regardless of assessed urgency.
6. Be respectful, calm, and clear. Avoid alarming language except when
   flagging a genuine emergency, where clarity and urgency both matter.
7. Consider the patient's stated age, existing conditions, and medications
   as context that can change urgency (e.g., symptoms in infants, elderly
   patients, or immunocompromised patients are generally treated as
   higher urgency).
8. If the patient's language preference is not English, write your
   response content in the requested language, while keeping any JSON
   keys in English exactly as specified by the schema.
"""

STRUCTURED_OUTPUT_INSTRUCTIONS = f"""\
You must respond with STRICT, VALID JSON and NOTHING ELSE — no markdown
code fences, no preamble, no trailing commentary. The JSON MUST conform
exactly to this schema (all fields required):

{ASSESSMENT_JSON_SCHEMA_STR}
"""

SYSTEM_MESSAGE_STRUCTURED = SystemMessage(
    content=SAFETY_SYSTEM_INSTRUCTIONS + "\n" + STRUCTURED_OUTPUT_INSTRUCTIONS
)

SYSTEM_MESSAGE_NARRATIVE = SystemMessage(
    content=SAFETY_SYSTEM_INSTRUCTIONS
    + """
Write a warm, clear, plain-language narrative (not JSON) that walks the
patient through: what their symptoms might broadly suggest (non-diagnostic),
what they should consider doing next, when they should seek urgent or
emergency care, and what to bring up with their doctor. Use short
paragraphs and, where helpful, bullet points. Keep it concise — aim for
roughly 200-400 words unless the situation is complex.
"""
)


# --------------------------------------------------------------------------
# Human-facing patient intake template (shared field list for both chains)
# --------------------------------------------------------------------------
PATIENT_INTAKE_TEMPLATE = """\
Please assess the following patient-reported information.

- Age: {age}
- Gender: {gender}
- Primary symptoms (selected): {symptoms_selected}
- Additional symptoms (free text): {symptoms_freetext}
- Duration of symptoms: {duration}
- Self-reported severity (1-10 scale): {severity}
- Existing medical conditions: {existing_conditions}
- Current medications: {current_medications}
- Additional notes from patient: {additional_notes}
- Requested response language: {language}

Remember: educational, non-diagnostic guidance only, and escalate clearly
to emergency care if warranted.
"""


# --------------------------------------------------------------------------
# ChatPromptTemplate builders
# --------------------------------------------------------------------------
def build_structured_assessment_prompt() -> ChatPromptTemplate:
    """
    Build the ChatPromptTemplate used for the structured JSON assessment
    chain. Combines the strict safety/JSON system message with the
    patient intake human message template.
    """
    return ChatPromptTemplate.from_messages(
        [
            SYSTEM_MESSAGE_STRUCTURED,
            ("human", PATIENT_INTAKE_TEMPLATE),
        ]
    )


def build_narrative_prompt() -> ChatPromptTemplate:
    """
    Build the ChatPromptTemplate used for the streamed, human-readable
    narrative guidance shown live to the user via st.write_stream().
    """
    return ChatPromptTemplate.from_messages(
        [
            SYSTEM_MESSAGE_NARRATIVE,
            ("human", PATIENT_INTAKE_TEMPLATE),
        ]
    )
