from __future__ import annotations

from pathlib import Path

import pytest

from llm_intel_kb.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert not settings.llm_configured
    assert not settings.embedding_configured


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMKB_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLMKB_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLMKB_LLM_MODEL", "deepseek-chat")
    settings = Settings(_env_file=None)
    assert settings.llm_configured
    assert settings.llm_model == "deepseek-chat"


def test_empty_values_count_as_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMKB_LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("LLMKB_LLM_API_KEY", "")
    monkeypatch.setenv("LLMKB_LLM_MODEL", "deepseek-chat")
    settings = Settings(_env_file=None)
    assert not settings.llm_configured


def test_env_file_loading(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLMKB_DATA_DIR=./corpus\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.data_dir == Path("./corpus")
