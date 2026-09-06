"""
MediGuide AI — src package.

This package contains the internal modules that power the MediGuide AI
educational symptom-assessment prototype:

    config.py         Application configuration, default option lists.
    prompts.py        LangChain prompt templates and JSON schema definitions.
    chains.py         ChatOpenAI setup, chain construction, streaming helpers.
    cache_manager.py  Toggleable LLM response caching (in-memory / SQLite).
    utils.py          Safe JSON parsing, string cleanup, emergency UI helpers.

IMPORTANT: MediGuide AI is an educational prototype only. It is NOT a
medical device and does NOT provide medical diagnoses. See README.md.
"""

__version__ = "1.0.0"
__app_name__ = "MediGuide AI"
