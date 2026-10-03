"""Unit tests for Core Configuration and LLM Factory.

Run with:
    uv run pytest tests/test_llm.py
"""

import sys
from pathlib import Path

import pytest

# Ensure src is importable
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from langchain_google_genai import ChatGoogleGenerativeAI

from core.config import Settings, get_settings
from core.llm import get_llm


def test_settings_load():
    """Verify that settings can be loaded and have default values."""
    settings = get_settings()
    assert settings.gemini_model != ""
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.redis_url == "redis://localhost:6379/0"


def test_llm_initialization():
    """Verify that get_llm returns a ChatGoogleGenerativeAI instance."""
    # Test with an explicit dummy key if env is not yet populated
    test_settings = Settings(
        GEMINI_API_KEY="test_dummy_key_12345",
        GEMINI_MODEL="gemini-2.5-flash",
    )
    llm = ChatGoogleGenerativeAI(
        model=test_settings.gemini_model,
        google_api_key=test_settings.gemini_api_key,
    )
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-2.5-flash"


def test_llm_raises_error_when_key_empty(monkeypatch):
    """Verify that get_llm raises ValueError if GEMINI_API_KEY is empty."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    # Clear cached settings
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="GEMINI_API_KEY is missing"):
        get_llm()

    # Clear cache again for subsequent tests
    get_settings.cache_clear()
