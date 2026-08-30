from __future__ import annotations

import pytest

from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig

LOCAL = ProviderConfig(
    name="local", kind="chat", base_url="http://localhost:11434/v1", model="test-model"
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeEmbeddingData:
    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingData(v) for v in vectors]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **kwargs: object) -> _FakeChatResponse:
        return _FakeChatResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors

    def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
        return _FakeEmbeddingResponse(self._vectors)


class _FakeClient:
    def __init__(self, chat_content: str, vectors: list[list[float]]) -> None:
        self.chat = _FakeChat(chat_content)
        self.embeddings = _FakeEmbeddings(vectors)


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *,
    chat_content: str = "",
    vectors: list[list[float]] | None = None,
) -> list[str]:
    """替换 OpenAI 客户端为假实现，返回捕获到的 base_url 列表."""
    captured: list[str] = []

    def factory(**kwargs: object) -> _FakeClient:
        captured.append(str(kwargs.get("base_url")))
        return _FakeClient(chat_content, vectors or [[0.1, 0.2, 0.3]])

    monkeypatch.setattr(llm_client, "OpenAI", factory)
    return captured


def test_chat_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install(monkeypatch, chat_content="你好，世界")
    reply = llm_client.chat(LOCAL, [{"role": "user", "content": "hi"}])
    assert reply == "你好，世界"
    assert captured == ["http://localhost:11434/v1"]


def test_ping_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, chat_content="pong")
    probe = llm_client.ping_chat(LOCAL)
    assert probe.model == "test-model"
    assert probe.reply == "pong"
    assert probe.latency_ms >= 0


def test_ping_chat_truncates_long_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, chat_content="p" * 100)
    probe = llm_client.ping_chat(LOCAL)
    assert len(probe.reply) == 40


def test_ping_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, vectors=[[0.1] * 8])
    probe = llm_client.ping_embedding(LOCAL)
    assert probe.dimension == 8


def test_chat_propagates_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_factory(**kwargs: object) -> object:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(llm_client, "OpenAI", broken_factory)
    with pytest.raises(RuntimeError, match="connection refused"):
        llm_client.chat(LOCAL, [{"role": "user", "content": "hi"}])


ANTHROPIC = ProviderConfig(
    name="dots3",
    kind="chat",
    adapter="anthropic",
    base_url="https://api.example.com",
    model="dots3-note-prev",
    api_key="sk-test",
    skip_think=True,
)


class _FakeAnthropicResponse:
    def __init__(self, blocks: list[dict], status_code: int = 200) -> None:
        self._data = {"content": blocks}
        self.status_code = status_code

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        pass


def _install_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    blocks: list[dict],
) -> list[dict]:
    """替换 httpx.post 为假实现，返回捕获到的请求参数（url/json/headers）."""
    captured: list[dict] = []

    def fake_post(url: str, **kwargs: object) -> _FakeAnthropicResponse:
        captured.append({"url": url, **kwargs})  # type: ignore[arg-type]
        return _FakeAnthropicResponse(blocks)

    monkeypatch.setattr(llm_client.httpx, "post", fake_post)
    return captured


def test_anthropic_chat_system_and_thinking_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_anthropic(
        monkeypatch, [{"type": "text", "text": "pong"}]
    )
    reply = llm_client.chat(
        ANTHROPIC,
        [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "hi"},
        ],
    )
    assert reply == "pong"
    req = captured[0]
    assert req["url"] == "https://api.example.com/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-test"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    body = req["json"]
    assert body["system"] == "你是助手"  # type: ignore[index]
    assert body["messages"] == [{"role": "user", "content": "hi"}]  # type: ignore[index]
    assert body["thinking"] == {"type": "disabled"}  # type: ignore[index]
    assert body["temperature"] == 0.2  # type: ignore[index]


def test_anthropic_chat_keeps_thinking_without_skip_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ProviderConfig(
        name="dots3",
        kind="chat",
        adapter="anthropic",
        base_url="https://api.example.com/v1",
        model="dots3-note-prev",
        api_key="sk-test",
    )
    captured = _install_anthropic(monkeypatch, [{"type": "text", "text": "pong"}])
    llm_client.chat(provider, [{"role": "user", "content": "hi"}])
    body = captured[0]["json"]
    assert "thinking" not in body  # type: ignore[operator]
    assert "temperature" not in body  # 思考模式要求 temperature=1，不显式传
    assert captured[0]["url"] == "https://api.example.com/v1/messages"


def test_anthropic_chat_concatenates_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = [
        {"type": "thinking", "thinking": "内部思考不应出现在正文"},
        {"type": "text", "text": "第一段。"},
        {"type": "text", "text": "第二段。"},
    ]
    _install_anthropic(monkeypatch, blocks)
    reply = llm_client.chat(ANTHROPIC, [{"role": "user", "content": "hi"}])
    assert reply == "第一段。第二段。"
