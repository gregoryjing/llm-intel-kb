from __future__ import annotations

from pathlib import Path

import pytest

from llm_intel_kb.config import Settings
from llm_intel_kb.llm.providers import ProviderConfig, load_providers


def test_provider_local_detection() -> None:
    local = ProviderConfig(
        name="ollama", kind="chat", base_url="http://localhost:11434/v1", model="qwen3:8b"
    )
    remote = ProviderConfig(
        name="deepseek", kind="chat", base_url="https://api.deepseek.com", model="deepseek-chat"
    )
    assert local.is_local
    assert not remote.is_local


def test_provider_ready_rules() -> None:
    local_no_key = ProviderConfig(
        name="ollama", kind="chat", base_url="http://127.0.0.1:11434/v1", model="m"
    )
    remote_no_key = ProviderConfig(
        name="remote", kind="chat", base_url="https://api.example.com", model="m"
    )
    remote_with_key = ProviderConfig(
        name="remote", kind="chat", base_url="https://api.example.com", model="m", api_key="sk-1"
    )
    assert local_no_key.is_ready  # 本地无需 Key
    assert not remote_no_key.is_ready
    assert remote_with_key.is_ready


def test_api_key_env_syntax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "sk-secret")
    provider = ProviderConfig(
        name="p",
        kind="chat",
        base_url="https://api.example.com",
        model="m",
        api_key="env:MY_TEST_KEY",
    )
    assert provider.resolved_api_key() == "sk-secret"

    empty_env = ProviderConfig(
        name="p",
        kind="chat",
        base_url="https://api.example.com",
        model="m",
        api_key="env:MISSING_VAR",
    )
    assert empty_env.resolved_api_key() is None


def test_kind_flags() -> None:
    chat = ProviderConfig(name="a", kind="chat", base_url="http://localhost:1", model="m")
    embedding = ProviderConfig(name="b", kind="embedding", base_url="http://localhost:1", model="m")
    both = ProviderConfig(name="c", kind="both", base_url="http://localhost:1", model="m")
    assert chat.supports_chat and not chat.supports_embedding
    assert not embedding.supports_chat and embedding.supports_embedding
    assert both.supports_chat and both.supports_embedding


def test_load_providers_from_toml(tmp_path: Path) -> None:
    toml_file = tmp_path / "providers.toml"
    toml_file.write_text(
        """
[remote]
kind = "chat"
base_url = "https://api.example.com"
api_key = "env:SOME_KEY"
model = "m1"

[local]
kind = "both"
base_url = "http://localhost:11434/v1"
api_key = "ollama"
model = "m2"
""",
        encoding="utf-8",
    )
    providers = load_providers(toml_file, Settings(_env_file=None))
    assert len(providers) == 2
    assert providers[0].name == "remote"
    assert not providers[0].is_ready
    assert providers[1].name == "local"
    assert providers[1].supports_chat and providers[1].supports_embedding


def test_load_providers_invalid_kind(tmp_path: Path) -> None:
    toml_file = tmp_path / "providers.toml"
    toml_file.write_text(
        '[bad]\nkind = "translate"\nbase_url = "http://x"\nmodel = "m"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kind 非法"):
        load_providers(toml_file, Settings(_env_file=None))


def test_load_providers_missing_fields(tmp_path: Path) -> None:
    toml_file = tmp_path / "providers.toml"
    toml_file.write_text('[bad]\nkind = "chat"\nbase_url = "http://x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="缺少 base_url 或 model"):
        load_providers(toml_file, Settings(_env_file=None))


def test_load_providers_fallback_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMKB_LLM_BASE_URL", "https://api.example.com")
    monkeypatch.setenv("LLMKB_LLM_API_KEY", "sk-1")
    monkeypatch.setenv("LLMKB_LLM_MODEL", "m1")
    providers = load_providers(None, Settings(_env_file=None))
    assert len(providers) == 1
    assert providers[0].name == "default"
    assert providers[0].supports_chat


def test_load_providers_fallback_empty() -> None:
    providers = load_providers(None, Settings(_env_file=None))
    assert providers == []


def test_anthropic_messages_url_variants() -> None:
    root = ProviderConfig(
        name="a", kind="chat", adapter="anthropic",
        base_url="https://api.example.com", model="m",
    )
    with_v1 = ProviderConfig(
        name="b", kind="chat", adapter="anthropic",
        base_url="https://api.example.com/v1", model="m",
    )
    full = ProviderConfig(
        name="c", kind="chat", adapter="anthropic",
        base_url="https://api.example.com/v1/messages", model="m",
    )
    trailing = ProviderConfig(
        name="d", kind="chat", adapter="anthropic",
        base_url="https://api.example.com/", model="m",
    )
    assert root.messages_url() == "https://api.example.com/v1/messages"
    assert with_v1.messages_url() == "https://api.example.com/v1/messages"
    assert full.messages_url() == "https://api.example.com/v1/messages"
    assert trailing.messages_url() == "https://api.example.com/v1/messages"


def test_anthropic_adapter_from_toml(tmp_path: Path) -> None:
    toml_file = tmp_path / "providers.toml"
    toml_file.write_text(
        """
[dots3]
kind = "chat"
adapter = "anthropic"
base_url = "https://api.example.com"
api_key = "env:SOME_KEY"
model = "dots3-note-prev"
skip_think = true
""",
        encoding="utf-8",
    )
    providers = load_providers(toml_file, Settings(_env_file=None))
    assert len(providers) == 1
    p = providers[0]
    assert p.is_anthropic
    assert not p.is_ollama_native
    assert p.skip_think


def test_invalid_adapter_rejected(tmp_path: Path) -> None:
    toml_file = tmp_path / "providers.toml"
    toml_file.write_text(
        '[bad]\nadapter = "grpc"\nbase_url = "http://x"\nmodel = "m"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="adapter 非法"):
        load_providers(toml_file, Settings(_env_file=None))
