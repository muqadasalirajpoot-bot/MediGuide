"""
cache_manager.py
-----------------
Toggleable LLM response caching for MediGuide AI.

LangChain exposes a global cache via `set_llm_cache()`. This module wraps
that with two backends:

  - "memory"  -> langchain_community.cache.InMemoryCache
                 Fast, process-local, cleared on restart. Good default
                 for local development and demos.

  - "sqlite"  -> langchain_community.cache.SQLiteCache
                 Persists identical-prompt responses to a local SQLite
                 file across app restarts, reducing repeat API costs and
                 latency for repeated identical assessments (e.g., during
                 testing/demo).

Only one cache backend is active globally at a time (this mirrors how
LangChain's global cache works). Calling configure_cache() again swaps
the active backend cleanly.
"""

from __future__ import annotations

import logging

from src.config import CACHE_BACKENDS, DEFAULT_CACHE_BACKEND, SQLITE_CACHE_PATH

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Resilient import of set_llm_cache().
#
# Depending on the installed langchain / langchain-core version (and on
# some hosts, a partial/mismatched install), this function lives in one
# of a few different places:
#   - langchain.globals            (older/newer "full" langchain installs)
#   - langchain_core.globals       (newer, core-only installs)
#
# If NONE of these succeed (e.g. "langchain" failed to install on the
# deployment host and only langchain-core is present), we do NOT want to
# crash the entire app on import — caching is a nice-to-have, not a
# core feature. In that case we fall back to a no-op cache toggle and
# surface a clear warning in the sidebar instead of a hard traceback.
# --------------------------------------------------------------------------
_set_llm_cache = None
_cache_import_error: Exception | None = None

try:
    from langchain.globals import set_llm_cache as _set_llm_cache
except ImportError as exc:
    try:
        from langchain_core.globals import set_llm_cache as _set_llm_cache
    except ImportError as exc2:
        _cache_import_error = exc2
        logger.warning(
            "Could not import set_llm_cache from either langchain.globals or "
            "langchain_core.globals (%s). LLM response caching will be "
            "disabled. Check that 'langchain' is listed in requirements.txt "
            "and installed successfully.",
            exc2,
        )

try:
    from langchain_community.cache import InMemoryCache, SQLiteCache
except ImportError as exc:
    InMemoryCache = None  # type: ignore[assignment,misc]
    SQLiteCache = None  # type: ignore[assignment,misc]
    if _cache_import_error is None:
        _cache_import_error = exc
    logger.warning(
        "Could not import InMemoryCache/SQLiteCache from "
        "langchain_community.cache (%s). LLM response caching will be "
        "disabled. Check that 'langchain-community' is listed in "
        "requirements.txt and installed successfully.",
        exc,
    )

CACHING_AVAILABLE = _set_llm_cache is not None and InMemoryCache is not None

# Tracks which backend is currently active so the UI can display it and
# so repeated calls with the same backend are cheap no-ops.
_active_backend: str | None = None


def configure_cache(backend: str = DEFAULT_CACHE_BACKEND, sqlite_path: str = SQLITE_CACHE_PATH) -> str:
    """
    Configure LangChain's global LLM cache to use the requested backend.

    Args:
        backend: One of "memory" or "sqlite" (see CACHE_BACKENDS in config.py).
        sqlite_path: Filesystem path for the SQLite cache database, used
            only when backend == "sqlite".

    Returns:
        The name of the backend that was actually activated.

    Raises:
        ValueError: if an unsupported backend name is supplied.
    """
    global _active_backend

    if backend not in CACHE_BACKENDS:
        raise ValueError(
            f"Unsupported cache backend '{backend}'. Must be one of {CACHE_BACKENDS}."
        )

    if not CACHING_AVAILABLE:
        # Degrade gracefully: record the requested backend for display
        # purposes, but skip the actual set_llm_cache() call so the app
        # keeps running (just without response caching) rather than
        # crashing on a partial/mismatched langchain install.
        _active_backend = backend
        return _active_backend

    if backend == _active_backend:
        # Already configured; avoid re-instantiating unnecessarily.
        return _active_backend

    if backend == "memory":
        _set_llm_cache(InMemoryCache())
        logger.info("MediGuide AI: activated in-memory LLM cache.")
    elif backend == "sqlite":
        _set_llm_cache(SQLiteCache(database_path=sqlite_path))
        logger.info("MediGuide AI: activated SQLite LLM cache at '%s'.", sqlite_path)

    _active_backend = backend
    return _active_backend


def get_active_backend() -> str | None:
    """Return the name of the currently active cache backend, if any."""
    return _active_backend


def cache_status_label() -> str:
    """A short, human-readable label for the sidebar, e.g. for st.caption()."""
    if not CACHING_AVAILABLE:
        return "Cache: unavailable (langchain caching module not installed)"
    if _active_backend is None:
        return "Cache: not yet configured"
    pretty = {"memory": "In-Memory", "sqlite": "SQLite (persistent)"}
    return f"Cache: {pretty.get(_active_backend, _active_backend)}"
