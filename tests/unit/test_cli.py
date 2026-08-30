from __future__ import annotations

import pytest
from typer.testing import CliRunner

from llm_intel_kb import __version__
from llm_intel_kb.checks import CheckResult
from llm_intel_kb.cli.main import app
from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig

runner = CliRunner()

LOCAL = ProviderConfig(
    name="ollama", kind="chat", base_url="http://localhost:11434/v1", model="qwen3:8b"
)
REMOTE_READY = ProviderConfig(
    name="deepseek",
    kind="chat",
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    api_key="env:LLMKB_DEEPSEEK_API_KEY",
)


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_doctor_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_intel_kb.cli.main.run_checks",
        lambda settings: [
            CheckResult("数据库连接", "ok", "PostgreSQL 16.4"),
            CheckResult("对话模型 · ollama", "ok", "qwen3:8b · 本地"),
        ],
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "对话模型" in result.output


def test_doctor_exits_nonzero_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_intel_kb.cli.main.run_checks",
        lambda settings: [CheckResult("数据库连接", "fail", "connection refused")],
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1


def test_doctor_skip_does_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_intel_kb.cli.main.run_checks",
        lambda settings: [CheckResult("对话模型 · deepseek", "skip", "缺少 API Key")],
    )
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_compare_no_ready_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llm_intel_kb.cli.main.load_provider_list", lambda settings: [REMOTE_READY])
    monkeypatch.setenv("LLMKB_DEEPSEEK_API_KEY", "")  # 空 Key = 未就绪
    result = runner.invoke(app, ["compare", "你好"])
    assert result.exit_code == 1
    assert "没有就绪的对话模型" in result.output


def test_compare_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_intel_kb.cli.main.load_provider_list", lambda settings: [LOCAL, REMOTE_READY]
    )
    monkeypatch.setenv("LLMKB_DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setattr(
        llm_client, "chat", lambda provider, messages, timeout, **kw: f"来自{provider.name}的回答"
    )
    result = runner.invoke(app, ["compare", "什么是RAG？"])
    assert result.exit_code == 0
    assert "ollama" in result.output
    assert "deepseek" in result.output


def test_compare_partial_failure_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "llm_intel_kb.cli.main.load_provider_list", lambda settings: [LOCAL, REMOTE_READY]
    )
    monkeypatch.setenv("LLMKB_DEEPSEEK_API_KEY", "sk-test")

    def fake_chat(provider: object, messages: object, timeout: float, **kw: object) -> str:
        if getattr(provider, "name", "") == "deepseek":
            raise RuntimeError("quota exceeded")
        return "本地模型的回答"

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    result = runner.invoke(app, ["compare", "测试"])
    assert result.exit_code == 1
    assert "本地模型的回答" in result.output
