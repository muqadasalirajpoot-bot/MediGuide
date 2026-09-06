"""
config.py
---------
Central configuration for MediGuide AI: app metadata, default UI option
lists, model settings, and the mandatory safety disclaimer text.

Keeping all of this in one module means the disclaimers and option lists
are defined exactly once and imported everywhere else, so they can never
drift out of sync between the login page, sidebar, and main dashboard.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# App metadata
# --------------------------------------------------------------------------
APP_NAME = "MediGuide AI"
APP_TAGLINE = "AI-Powered Medical Symptom Assessment and Patient Guidance Assistant"
APP_ICON = "🩺"

# --------------------------------------------------------------------------
# Model configuration
# --------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "gpt-4o-mini"
AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
DEFAULT_TEMPERATURE = 0.2  # Low temperature: favor consistent, cautious output
MAX_TOKENS = 2000

# --------------------------------------------------------------------------
# Mandatory safety disclaimers
# --------------------------------------------------------------------------
# These strings are displayed verbatim across every screen of the app.
SHORT_DISCLAIMER = (
    "⚠️ Educational prototype only — NOT a substitute for professional "
    "medical advice, diagnosis, or treatment."
)

FULL_DISCLAIMER = (
    "**IMPORTANT NOTICE:** MediGuide AI is an **educational prototype ONLY**. "
    "It is **NOT a doctor**, does **NOT provide confirmed diagnoses**, and "
    "**CANNOT replace professional healthcare or emergency services**. "
    "If you are experiencing a medical emergency, call your local emergency "
    "number (e.g., 911, 112, 1122) immediately or go to the nearest "
    "emergency room."
)

EMERGENCY_BANNER_TEXT = (
    "🚨 **POSSIBLE EMERGENCY DETECTED** 🚨\n\n"
    "Based on what you entered, this may be a medical emergency. "
    "**Please stop using this app and call your local emergency number "
    "immediately, or go to the nearest emergency room.** "
    "MediGuide AI cannot call for help on your behalf and cannot confirm "
    "a diagnosis."
)

LOGIN_PAGE_FOOTER_NOTE = (
    "Your API key is stored only in this browser session "
    "(`st.session_state`) and is never written to disk or transmitted "
    "anywhere other than directly to OpenAI's API for generating your "
    "requested guidance."
)

# --------------------------------------------------------------------------
# UI option lists
# --------------------------------------------------------------------------
GENDER_OPTIONS = ["Female", "Male", "Non-binary", "Prefer not to say"]

COMMON_SYMPTOMS = [
    "Fever", "Cough", "Headache", "Sore throat", "Fatigue",
    "Shortness of breath", "Chest pain", "Abdominal pain", "Nausea",
    "Vomiting", "Diarrhea", "Constipation", "Dizziness", "Rash",
    "Joint pain", "Muscle aches", "Runny nose", "Congestion",
    "Loss of appetite", "Back pain", "Anxiety", "Insomnia",
    "Swelling", "Numbness or tingling", "Blurred vision",
]

DURATION_OPTIONS = [
    "Less than a day",
    "1-2 days",
    "3-6 days",
    "1-2 weeks",
    "2-4 weeks",
    "More than a month",
    "Chronic / ongoing for months or years",
]

OUTPUT_LANGUAGES = [
    "English", "Urdu", "Spanish", "French", "Arabic",
    "Hindi", "Chinese (Simplified)", "German", "Portuguese",
]

URGENCY_LEVELS = ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]

# Maps each urgency level to the Streamlit alert function name that should
# render it, per the required color-coded scheme.
URGENCY_TO_STREAMLIT_FN = {
    "LOW": "info",
    "MEDIUM": "warning",
    "HIGH": "warning",
    "EMERGENCY": "error",
}

URGENCY_EMOJI = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "EMERGENCY": "🔴",
}

# Free-text keywords that, if present in the user's symptom description,
# trigger an immediate client-side emergency banner *before* the LLM call.
# This is a defense-in-depth safety net; it does not replace the model's
# own emergency classification, and it does not diagnose anything itself.
EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "severe bleeding", "unconscious", "unresponsive", "suicidal",
    "suicide", "stroke", "slurred speech", "face drooping",
    "severe allergic reaction", "anaphylaxis", "heart attack",
    "not breathing", "seizure", "overdose", "coughing up blood",
    "severe head injury", "poisoning",
]

# --------------------------------------------------------------------------
# Cache configuration
# --------------------------------------------------------------------------
CACHE_BACKENDS = ["memory", "sqlite"]
DEFAULT_CACHE_BACKEND = "memory"
SQLITE_CACHE_PATH = ".mediguide_cache.db"
