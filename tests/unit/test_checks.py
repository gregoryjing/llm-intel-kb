from __future__ import annotations

import pytest

from llm_intel_kb.checks import (
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    check_chat_provider,
    check_database,
    check_embedding_provider,
    check_providers,
)
from llm_intel_kb.config import Settings
from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.client import ChatProbe, EmbeddingProbe
from llm_intel_kb.llm.providers import ProviderConfig

LOCAL_CHAT = ProviderConfig(
    name="ollama", kind="chat", base_url="http://localhost:11434/v1", model="qwen3:8b"
)
REMOTE_CHAT = ProviderConfig(
    name="deepseek", kind="chat", base_url="https://api.deepseek.com", model="deepseek-chat"
)
LOCAL_EMBED = ProviderConfig(
    name="ollama-embed",
    kind="embedding",
    base_url="http://localhost:11434/v1",
    model="nomic-embed-text",
)


def test_check_database_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(engine: object) -> dict[str, object]:
        raise OSError("connection refused")

    monkeypatch.setattr("llm_intel_kb.checks.probe_database", boom)
    results = check_database(Settings(_env_file=None))
    assert len(results) == 1
    assert results[0].status == STATUS_FAIL
    assert "OSError" in results[0].detail


def test_check_database_ok_with_pgvector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_intel_kb.checks.probe_database",
        lambda engine: {"server_version": "16.4", "pgvector_version": "0.7.0"},
    )
    results = check_database(Settings(_env_file=None))
    assert [r.status for r in results] == [STATUS_OK, STATUS_OK]


def test_check_database_ok_without_pgvector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_intel_kb.checks.probe_database",
        lambda engine: {"server_version": "16.4", "pgvector_version": None},
    )
    results = check_database(Settings(_env_file=None))
    assert results[0].status == STATUS_OK
    assert results[1].status == STATUS_FAIL


def test_check_chat_provider_ok_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "ping_chat",
        lambda provider, timeout=30: ChatProbe(model="qwen3:8b", reply="pong", latency_ms=42),
    )
    result = check_chat_provider(LOCAL_CHAT)
    assert result.status == STATUS_OK
    assert "本地" in result.detail


def test_check_chat_provider_ok_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMKB_TEST_REMOTE_KEY", "sk-test")
    remote_ready = ProviderConfig(
        name="deepseek",
        kind="chat",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        api_key="env:LLMKB_TEST_REMOTE_KEY",
    )
    monkeypatch.setattr(
        llm_client,
        "ping_chat",
        lambda provider, timeout=30: ChatProbe(model="deepseek-chat", reply="pong", latency_ms=300),
    )
    result = check_chat_provider(remote_ready)
    assert result.status == STATUS_OK
    assert "远程" in result.detail


def test_check_chat_provider_skip_without_key() -> None:
    result = check_chat_provider(REMOTE_CHAT)
    assert result.status == STATUS_SKIP
    assert "缺少 API Key" in result.detail


def test_check_chat_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(provider: object, timeout: float = 30) -> ChatProbe:
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(llm_client, "ping_chat", boom)
    result = check_chat_provider(LOCAL_CHAT)
    assert result.status == STATUS_FAIL
    assert "401" in result.detail


def test_check_embedding_provider_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_client,
        "ping_embedding",
        lambda provider, timeout=30: EmbeddingProbe(model="nomic", dimension=768, latency_ms=88),
    )
    result = check_embedding_provider(LOCAL_EMBED)
    assert result.status == STATUS_OK
    assert "768" in result.detail


def test_check_providers_mixed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_intel_kb.checks.load_provider_list",
        lambda settings: [LOCAL_CHAT, REMOTE_CHAT, LOCAL_EMBED],
    )
    monkeypatch.setattr(
        llm_client,
        "ping_chat",
        lambda provider, timeout=30: ChatProbe(model="m", reply="pong", latency_ms=1),
    )
    monkeypatch.setattr(
        llm_client,
        "ping_embedding",
        lambda provider, timeout=30: EmbeddingProbe(model="e", dimension=8, latency_ms=1),
    )
    results = check_providers(Settings(_env_file=None))
    assert [r.status for r in results] == [STATUS_OK, STATUS_SKIP, STATUS_OK]


def test_check_providers_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_intel_kb.checks.load_provider_list", lambda settings: [])
    results = check_providers(Settings(_env_file=None))
    assert len(results) == 1
    assert results[0].status == STATUS_SKIP
