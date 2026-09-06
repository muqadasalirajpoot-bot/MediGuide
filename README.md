# 🩺 MediGuide AI
https://mediguide-mbwxdbocqqq5nqkxk28ztb.streamlit.app/

**AI-Powered Medical Symptom Assessment and Patient Guidance Assistant**
*(Educational Prototype)*

> **IMPORTANT NOTICE:** MediGuide AI is an **educational prototype ONLY**.
> It is **NOT a doctor**, does **NOT provide confirmed diagnoses**, and
> **CANNOT replace professional healthcare or emergency services**. If
> you are experiencing a medical emergency, call your local emergency
> number immediately or go to the nearest emergency room.

---

## 1. Overview

MediGuide AI is a Streamlit application that demonstrates how LangChain
and an LLM can be used to build a **safety-conscious**, **non-diagnostic**
symptom intake and patient-guidance tool. It is intended as a learning
project / portfolio piece for LLM application engineering — **not** for
real-world clinical use.

Key features:

- 🔐 **API key login gate** — the app is fully inaccessible until a valid
  OpenAI API key is supplied.
- 🧠 **Structured JSON assessment** — a strict schema covering summary,
  possible (non-diagnostic) conditions, urgency level, next steps,
  doctor questions, and warning signs.
- 🌊 **Streamed narrative guidance** — a live, human-readable explanation
  rendered token-by-token with `st.write_stream()`.
- 🚦 **Color-coded urgency dashboard** — `st.info` / `st.warning` /
  `st.error` badges plus an emergency banner for red-flag cases.
- 🛡️ **Defense-in-depth safety** — both a client-side emergency-keyword
  scanner *and* a model-side emergency classifier.
- ⚡ **Toggleable caching** — switch between in-memory and persistent
  SQLite LLM response caching.
- 🌐 **Multilingual output** — request guidance in several languages.

---

## 2. Project Structure

```
mediguide_ai/
│── app.py                   # Main Streamlit UI entrypoint with login gate
│── requirements.txt         # Python dependencies
│── .env.example              # Sample environment key template
│── README.md                 # This file
└── src/
    ├── __init__.py
    ├── config.py             # App configuration, disclaimers, option lists
    ├── prompts.py             # LangChain PromptTemplates + JSON schema
    ├── chains.py               # ChatOpenAI setup, chains, streaming
    ├── cache_manager.py        # InMemoryCache / SQLiteCache toggle
    └── utils.py                 # Safe JSON parsing, emergency keyword scan
```

---

## 3. Setup

### 3.1 Prerequisites

- Python 3.10+
- An OpenAI API key with access to a chat-completions model
  (default: `gpt-4o-mini`)

### 3.2 Installation

```bash
git clone <this-repo-url> mediguide_ai
cd mediguide_ai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.3 Configuring your API key

You have two options:

**Option A — In-app login (recommended for demos):**
Just run the app (see below) and enter your API key on the Login Page.
The key is kept only in `st.session_state` for the duration of your
browser session and is never written to disk.

**Option B — Environment variable / `.env` file:**
```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```
When `OPENAI_API_KEY` is already present in the environment, the app
skips the login page automatically.

### 3.4 Running the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

---

## 4. Architecture

### 4.1 Request flow

```
User fills intake form (app.py)
        │
        ▼
Client-side emergency keyword scan (src/utils.py)
        │
        ▼
Structured assessment chain (src/chains.py -> src/prompts.py)
   prompt | ChatOpenAI  ──►  raw text response
        │
        ▼
safe_parse_json() + validate_assessment_schema() (src/utils.py)
        │
        ▼
Color-coded urgency dashboard rendered (app.py)
        │
        ▼
Narrative chain streamed live via st.write_stream() (src/chains.py)
```

### 4.2 Prompt design (`src/prompts.py`)

Two `ChatPromptTemplate`s share a common `SAFETY_SYSTEM_INSTRUCTIONS`
block (never confirm diagnoses, always evaluate for emergencies, avoid
naming specific drugs/doses, escalate ambiguity to the higher urgency
level, etc.):

1. **Structured chain** — appends strict JSON-only formatting
   instructions and the exact schema the app expects.
2. **Narrative chain** — appends instructions to produce a warm,
   plain-language explanation instead of JSON, for streaming.

### 4.3 Defensive JSON handling (`src/utils.py`)

LLMs occasionally wrap JSON in markdown fences, add stray commentary, or
omit a field. `safe_parse_json()` strips code fences and falls back to
extracting the outermost `{ ... }` substring before giving up.
`validate_assessment_schema()` then guarantees every field the dashboard
needs is present, filling safe, clearly-labeled defaults for anything
missing rather than letting the UI crash.

### 4.4 Emergency escalation (defense-in-depth)

There are **two independent layers**:

1. **Client-side keyword scan** (`contains_emergency_keywords()`) runs
   immediately on submission, before any API call, checking for phrases
   like "chest pain", "can't breathe", "unconscious", etc. This guarantees
   a warning appears even if the API call fails or is slow.
2. **Model-side classification** — the LLM itself is instructed to set
   `urgency_level: "EMERGENCY"` and to make the first recommended step
   "seek emergency care immediately" whenever red-flag symptoms are
   plausible, erring toward the higher urgency level when uncertain.

Neither layer is a substitute for the other, and neither is a substitute
for real emergency services.

---

## 5. Caching Breakdown (`src/cache_manager.py`)

MediGuide AI uses LangChain's global `set_llm_cache()` mechanism with two
selectable backends, toggled live from the sidebar:

| Backend  | Class                              | Persistence            | Best for                                   |
|----------|-------------------------------------|-------------------------|---------------------------------------------|
| `memory` | `InMemoryCache`                     | Cleared on app restart  | Local dev, quick demos                       |
| `sqlite` | `SQLiteCache(database_path=...)`    | Persists across restarts | Repeated identical test cases, cost savings |

Only one backend is active globally at a time (this mirrors LangChain's
own global-cache design). Switching backends in the sidebar calls
`configure_cache()`, which swaps the active cache cleanly via
`set_llm_cache()`. Caching keys off the exact prompt + model + params, so
it only produces cache **hits** for identical, repeated intake
submissions — it does not affect the correctness of first-time
assessments.

---

## 6. Safety & Limitations

- MediGuide AI **does not** replace a licensed medical professional.
- Outputs are **educational and non-diagnostic** by design and by prompt
  instruction — the model is explicitly told never to present a
  confirmed diagnosis.
- The app does not store patient data outside the current browser
  session; nothing is persisted to a database by default (the SQLite
  cache stores LLM prompt/response pairs for performance, not a patient
  record system).
- This project has not been validated, certified, or reviewed as a
  medical device, and must not be used for real clinical
  decision-making.

---

## 7. License

This project is provided for educational purposes as a portfolio /
learning example. Add your preferred license here before distribution.
