"""
app.py
------
MediGuide AI — Main Streamlit UI entrypoint.

Flow:
  1. Login gate: block all app functionality until a valid-looking
     OpenAI API key is provided (either via .env / environment variable,
     or entered by the user on the Login Page).
  2. Main dashboard: patient intake form -> structured JSON assessment
     (color-coded urgency, conditions, next steps, questions, warning
     signs) + a streamed narrative explanation.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from src import config
from src.cache_manager import cache_status_label, configure_cache
from src.chains import build_llm, run_structured_assessment, stream_narrative
from src.utils import contains_emergency_keywords, safe_parse_json, validate_assessment_schema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mediguide_ai.app")

load_dotenv()  # Populate os.environ from a local .env file, if present.

st.set_page_config(
    page_title=config.APP_NAME,
    page_icon=config.APP_ICON,
    layout="wide",
)


# ==========================================================================
# Session state initialization
# ==========================================================================
def _init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "openai_api_key": "",
        "cache_backend": config.DEFAULT_CACHE_BACKEND,
        "model_name": config.DEFAULT_MODEL_NAME,
        "last_assessment": None,
        "last_patient_data": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

# Treat a pre-set environment variable (e.g., from .env) as already
# authenticated, so developers/deployers can skip the login page entirely.
if not st.session_state["authenticated"] and os.environ.get("OPENAI_API_KEY"):
    st.session_state["openai_api_key"] = os.environ["OPENAI_API_KEY"]
    st.session_state["authenticated"] = True


# ==========================================================================
# Shared UI fragments
# ==========================================================================
def render_disclaimer_banner() -> None:
    """Render the mandatory full safety disclaimer. Shown on every screen."""
    st.warning(config.FULL_DISCLAIMER)


def is_valid_api_key_format(key: str) -> bool:
    """
    Lightweight format validation for an OpenAI API key. This does NOT
    call the API — it only rejects obviously malformed input before
    attempting to use it, so the user gets fast feedback.
    """
    if not key:
        return False
    key = key.strip()
    # OpenAI keys are typically 40+ chars and start with "sk-".
    return key.startswith("sk-") and len(key) >= 20


# ==========================================================================
# Login / Authentication Gate
# ==========================================================================
def render_login_page() -> None:
    st.title(f"{config.APP_ICON} {config.APP_NAME}")
    st.caption(config.APP_TAGLINE)

    render_disclaimer_banner()

    st.markdown("### 🔑 Login / Start Session")
    st.write(
        "To use MediGuide AI, please enter your **OpenAI API key**. "
        "This key powers the AI assessment engine and is required to "
        "continue."
    )

    with st.form("login_form", clear_on_submit=False):
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your key is kept only in this browser session and is never stored to disk.",
        )
        submitted = st.form_submit_button("Login / Start Session", use_container_width=True)

    if submitted:
        if is_valid_api_key_format(api_key_input):
            st.session_state["openai_api_key"] = api_key_input.strip()
            st.session_state["authenticated"] = True
            os.environ["OPENAI_API_KEY"] = api_key_input.strip()
            st.success("Login successful — starting your session...")
            st.rerun()
        else:
            st.error(
                "That doesn't look like a valid OpenAI API key. Keys "
                "typically start with 'sk-' and are at least 20 "
                "characters long. Please check and try again."
            )

    st.divider()
    st.caption(config.LOGIN_PAGE_FOOTER_NOTE)


# ==========================================================================
# Sidebar (shown only once authenticated)
# ==========================================================================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"## {config.APP_ICON} {config.APP_NAME}")
        st.caption(config.SHORT_DISCLAIMER)

        st.divider()
        st.markdown("### ⚙️ Settings")

        st.session_state["model_name"] = st.selectbox(
            "Model",
            options=config.AVAILABLE_MODELS,
            index=config.AVAILABLE_MODELS.index(st.session_state["model_name"])
            if st.session_state["model_name"] in config.AVAILABLE_MODELS
            else 0,
        )

        cache_choice = st.radio(
            "Response Caching",
            options=config.CACHE_BACKENDS,
            index=config.CACHE_BACKENDS.index(st.session_state["cache_backend"]),
            format_func=lambda b: "In-Memory" if b == "memory" else "SQLite (persistent)",
            help=(
                "In-Memory caching is fast but clears on restart. "
                "SQLite caching persists identical-prompt responses "
                "across restarts to save on repeat API calls."
            ),
        )
        if cache_choice != st.session_state["cache_backend"]:
            st.session_state["cache_backend"] = cache_choice
        configure_cache(st.session_state["cache_backend"])
        st.caption(cache_status_label())

        st.divider()
        if st.button("🚪 Logout / Change Key", use_container_width=True):
            for key in ("authenticated", "openai_api_key", "last_assessment", "last_patient_data"):
                st.session_state[key] = False if key == "authenticated" else None
            os.environ.pop("OPENAI_API_KEY", None)
            st.rerun()

        st.divider()
        st.caption(config.SHORT_DISCLAIMER)


# ==========================================================================
# Patient intake form
# ==========================================================================
def render_intake_form() -> dict | None:
    st.markdown("### 📋 Patient Intake")

    with st.form("intake_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        with col2:
            gender = st.selectbox("Gender", options=config.GENDER_OPTIONS)
        with col3:
            duration = st.selectbox("Duration of symptoms", options=config.DURATION_OPTIONS)

        symptoms_selected = st.multiselect(
            "Select any symptoms that apply",
            options=config.COMMON_SYMPTOMS,
        )
        symptoms_freetext = st.text_area(
            "Describe any additional symptoms in your own words",
            placeholder="e.g., sharp pain behind the right eye that started this morning...",
            height=80,
        )

        severity = st.slider(
            "Self-rated severity (1 = mild, 10 = worst imaginable)",
            min_value=1, max_value=10, value=5,
        )

        col4, col5 = st.columns(2)
        with col4:
            existing_conditions = st.text_area(
                "Existing medical conditions (if any)",
                placeholder="e.g., asthma, type 2 diabetes...",
                height=80,
            )
        with col5:
            current_medications = st.text_area(
                "Current medications (if any)",
                placeholder="e.g., metformin, albuterol inhaler...",
                height=80,
            )

        additional_notes = st.text_area(
            "Additional notes",
            placeholder="Anything else you'd like to mention...",
            height=80,
        )

        language = st.selectbox("Output language", options=config.OUTPUT_LANGUAGES)

        submitted = st.form_submit_button("🔎 Get Guidance", use_container_width=True)

    if not submitted:
        return None

    if not symptoms_selected and not symptoms_freetext.strip():
        st.error("Please select at least one symptom or describe your symptoms in the text box.")
        return None

    return {
        "age": age,
        "gender": gender,
        "symptoms_selected": ", ".join(symptoms_selected) if symptoms_selected else "None selected",
        "symptoms_freetext": symptoms_freetext.strip() or "None provided",
        "duration": duration,
        "severity": severity,
        "existing_conditions": existing_conditions.strip() or "None reported",
        "current_medications": current_medications.strip() or "None reported",
        "additional_notes": additional_notes.strip() or "None",
        "language": language,
    }


# ==========================================================================
# Dashboard / results rendering
# ==========================================================================
def render_urgency_badge(urgency_level: str) -> None:
    fn_name = config.URGENCY_TO_STREAMLIT_FN.get(urgency_level, "info")
    emoji = config.URGENCY_EMOJI.get(urgency_level, "ℹ️")
    message = f"{emoji} **Urgency Level: {urgency_level}**"

    streamlit_fn = getattr(st, fn_name, st.info)
    streamlit_fn(message)

    if urgency_level == "EMERGENCY":
        st.error(config.EMERGENCY_BANNER_TEXT)


def render_assessment_dashboard(assessment: dict, patient_data: dict) -> None:
    st.markdown("## 📊 Assessment Dashboard")

    render_urgency_badge(assessment["urgency_level"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Reported Severity", f"{patient_data.get('severity', '—')}/10")
    m2.metric("Duration", patient_data.get("duration", "—"))
    m3.metric("Possible Considerations", len(assessment.get("possible_conditions", [])))

    st.markdown("#### Summary")
    st.write(assessment.get("summary", ""))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🩻 Possible Considerations", "✅ Next Steps", "❓ Ask Your Doctor", "🚩 Warning Signs"]
    )

    with tab1:
        conditions = assessment.get("possible_conditions", [])
        if not conditions:
            st.caption("No specific considerations were returned.")
        for cond in conditions:
            name = cond.get("name", "Unknown") if isinstance(cond, dict) else str(cond)
            reason = cond.get("reason", "") if isinstance(cond, dict) else ""
            with st.expander(f"🔹 {name}"):
                st.write(reason or "No further detail provided.")
        st.caption(
            "These are educational, non-diagnostic possibilities only — "
            "not a confirmed diagnosis."
        )

    with tab2:
        steps = assessment.get("recommended_next_steps", [])
        if steps:
            for step in steps:
                st.markdown(f"- {step}")
        else:
            st.caption("No next steps were returned.")

    with tab3:
        questions = assessment.get("questions_for_doctor", [])
        if questions:
            for q in questions:
                st.markdown(f"- {q}")
        else:
            st.caption("No suggested questions were returned.")

    with tab4:
        warnings_list = assessment.get("warning_signs", [])
        if warnings_list:
            for w in warnings_list:
                st.markdown(f"- 🚩 {w}")
        else:
            st.caption("No specific warning signs were returned.")

    st.divider()
    st.markdown("#### 📝 Narrative Guidance")

    llm_stream = build_llm(
        api_key=st.session_state["openai_api_key"],
        model_name=st.session_state["model_name"],
        streaming=True,
    )
    narrative_box = st.container(border=True)
    with narrative_box:
        st.write_stream(stream_narrative(llm_stream, patient_data))

    st.divider()
    render_disclaimer_banner()


# ==========================================================================
# Main app flow
# ==========================================================================
def render_main_app() -> None:
    st.title(f"{config.APP_ICON} {config.APP_NAME}")
    st.caption(config.APP_TAGLINE)
    render_disclaimer_banner()

    render_sidebar()

    patient_data = render_intake_form()

    if patient_data is None:
        # No new submission this run — but re-display the last result if
        # one exists, so it isn't lost on unrelated sidebar interactions.
        if st.session_state.get("last_assessment") and st.session_state.get("last_patient_data"):
            render_assessment_dashboard(
                st.session_state["last_assessment"], st.session_state["last_patient_data"]
            )
        return

    # --- Client-side, pre-LLM emergency keyword safety net -------------
    matched_keywords = contains_emergency_keywords(
        patient_data["symptoms_freetext"],
        patient_data["additional_notes"],
        patient_data["symptoms_selected"],
    )
    if matched_keywords:
        st.error(config.EMERGENCY_BANNER_TEXT)
        st.caption(
            "Flagged terms detected in your description: " + ", ".join(sorted(set(matched_keywords)))
        )

    # --- Structured assessment call -------------------------------------
    with st.spinner("Analyzing your symptoms..."):
        try:
            llm = build_llm(
                api_key=st.session_state["openai_api_key"],
                model_name=st.session_state["model_name"],
                streaming=False,
            )
            raw_response = run_structured_assessment(llm, patient_data)
            parsed = safe_parse_json(raw_response)
            assessment = validate_assessment_schema(parsed)
        except Exception as exc:  # noqa: BLE001 - never let the UI hard-crash
            logger.exception("Structured assessment failed.")
            st.error(
                "Something went wrong while generating your assessment. "
                f"Details: {exc}"
            )
            return

    st.session_state["last_assessment"] = assessment
    st.session_state["last_patient_data"] = patient_data

    render_assessment_dashboard(assessment, patient_data)


# ==========================================================================
# Entrypoint
# ==========================================================================
def main() -> None:
    if not st.session_state["authenticated"]:
        render_login_page()
    else:
        render_main_app()


if __name__ == "__main__":
    main()
