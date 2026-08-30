from __future__ import annotations

import os

import pytest

from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig
from llm_intel_kb.store.db import create_db_engine, ensure_pgvector, probe_database

pytestmark = pytest.mark.integration

_DATABASE_URL = os.environ.get("LLMKB_TEST_DATABASE_URL")

_LOCAL_OLLAMA = ProviderConfig(
    name="local-ollama",
    kind="chat",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    model="qwen3-vl:8b",
)


@pytest.mark.skipif(not _DATABASE_URL, reason="需要 LLMKB_TEST_DATABASE_URL 指向真实 PostgreSQL")
class TestDatabase:
    def test_probe(self) -> None:
        engine = create_db_engine(_DATABASE_URL or "")
        try:
            info = probe_database(engine)
        finally:
            engine.dispose()
        assert info["server_version"]
        assert info["pgvector_version"]

    def test_ensure_pgvector_idempotent(self) -> None:
        engine = create_db_engine(_DATABASE_URL or "")
        try:
            first = ensure_pgvector(engine)
            second = ensure_pgvector(engine)
        finally:
            engine.dispose()
        assert first == second


@pytest.mark.skipif(os.environ.get("LLMKB_SKIP_LOCAL_LLM") == "1", reason="显式跳过本地模型连通性")
class TestLocalLLM:
    def test_chat_ping(self) -> None:
        probe = llm_client.ping_chat(_LOCAL_OLLAMA)
        assert probe.model
        assert probe.latency_ms >= 0


@pytest.mark.skipif(
    not os.environ.get("LLMKB_DEEPSEEK_API_KEY"), reason="需要 LLMKB_DEEPSEEK_API_KEY"
)
class TestRemoteLLM:
    def test_chat_ping(self) -> None:
        provider = ProviderConfig(
            name="deepseek",
            kind="chat",
            base_url="https://api.deepseek.com",
            api_key=os.environ["LLMKB_DEEPSEEK_API_KEY"],
            model="deepseek-chat",
        )
        probe = llm_client.ping_chat(provider)
        assert probe.model
