from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx2 as httpx
from openai import OpenAI

if TYPE_CHECKING:
    from llm_intel_kb.llm.providers import ProviderConfig

# 思考型模型的思考 token 预算余量（本地模型不计费，宁多勿少）
_THINKING_HEADROOM = 4096

# 预置的空思考块：模型据此跳过思考直接输出正文（qwen3 系列实测有效）
_EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class ChatProbe:
    model: str
    reply: str
    latency_ms: int


@dataclass(frozen=True)
class EmbeddingProbe:
    model: str
    dimension: int
    latency_ms: int


def _make_client(provider: ProviderConfig, timeout: float) -> OpenAI:
    # OpenAI SDK 要求 api_key 非空；本地服务不校验，占位即可
    api_key = provider.resolved_api_key() or "local-no-key"
    return OpenAI(base_url=provider.base_url, api_key=api_key, timeout=timeout)


def chat(
    provider: ProviderConfig,
    messages: list[dict[str, str]],
    timeout: float = 120.0,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """发送一次对话请求，返回回复文本.

    多路分支：adapter=anthropic 走 Messages API；adapter=ollama 走原生 /api/chat
    （对思考型模型更稳）；其余走 OpenAI 兼容接口.
    本地模型推理慢，超时下限放宽到 10 分钟.
    """
    if provider.is_anthropic:
        return _chat_anthropic(provider, messages, timeout, temperature, max_tokens)
    if provider.is_ollama_native:
        effective_timeout = max(timeout, 600.0) if provider.is_local else timeout
        return _chat_ollama_native(provider, messages, effective_timeout, temperature, max_tokens)
    client = _make_client(provider, timeout)
    response = client.chat.completions.create(
        model=provider.model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def _chat_anthropic(
    provider: ProviderConfig,
    messages: list[dict[str, str]],
    timeout: float,
    temperature: float,
    max_tokens: int,
) -> str:
    """Anthropic Messages API（/v1/messages）.

    system 消息必须放请求体的顶层 system 字段（不能混在 messages 里）。
    skip_think=True 映射为官方思考开关 thinking={"type": "disabled"}：
    dots3 等思考型模型不关思考时，短回复的 max_tokens 会被思考 token 耗尽
    导致正文为空（实测 100 token 全部花在思考上）。
    未关思考时不传 temperature——Anthropic 思考模式要求 temperature=1。
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    payload: dict[str, object] = {
        "model": provider.model,
        "max_tokens": max_tokens,
        "messages": [m for m in messages if m["role"] != "system"],
    }
    if system:
        payload["system"] = system
    if provider.skip_think:
        payload["thinking"] = {"type": "disabled"}
        payload["temperature"] = temperature
    response = httpx.post(
        provider.messages_url(),
        json=payload,
        headers={
            "x-api-key": provider.resolved_api_key() or "",
            "anthropic-version": "2023-06-01",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    blocks = response.json().get("content", [])
    return "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text").strip()


def _chat_ollama_native(
    provider: ProviderConfig,
    messages: list[dict[str, str]],
    timeout: float,
    temperature: float,
    max_tokens: int,
) -> str:
    """Ollama 原生对话接口.

    思考型模型（qwen3 系列等）即使传 think=false 也会先输出 thinking 再输出 content：
    - skip_think=True：在消息末尾预置一个空的 <think> 块，模型直接输出正文
      （快数倍，且不会因思考耗尽上下文窗口导致空回复）
    - 未开 skip_think：为思考预留 num_predict 余量、放大 num_ctx，降低空回复概率
    """
    options: dict[str, object] = {
        "temperature": temperature,
        # skip_think 已预置空思考块（模型不再输出思考 token），无需思考余量；
        # 余量会把 num_predict 推到数千 token，在显存紧张的本地 GPU 上引发
        # KV 缓存溢出→层卸载到内存→生成速度塌方 30 倍以上（8GB 卡实测）。
        "num_predict": max_tokens if provider.skip_think else max_tokens + _THINKING_HEADROOM,
    }
    if provider.num_ctx:
        options["num_ctx"] = provider.num_ctx
    payload: dict[str, object] = {
        "model": provider.model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if provider.skip_think:
        payload["messages"] = [*messages, {"role": "assistant", "content": _EMPTY_THINK_BLOCK}]
    response = httpx.post(
        f"{provider.native_base_url()}/api/chat",
        json=payload | {"think": False},
        timeout=timeout,
    )
    if response.status_code != 200:
        # 旧版 Ollama 或非思考型模型可能不认 think 参数：去掉后重试
        body = response.text[:200]
        if "think" in body.lower():
            retry = httpx.post(
                f"{provider.native_base_url()}/api/chat",
                json=payload,
                timeout=timeout,
            )
            retry.raise_for_status()
            return str(retry.json().get("message", {}).get("content", "")).strip()
        response.raise_for_status()
    return str(response.json().get("message", {}).get("content", "")).strip()


def embed(
    provider: ProviderConfig,
    texts: list[str],
    timeout: float = 60.0,
) -> list[list[float]]:
    """文本向量化，返回与输入等长的向量列表."""
    client = _make_client(provider, timeout)
    response = client.embeddings.create(model=provider.model, input=texts)
    return [item.embedding for item in response.data]


def ping_chat(provider: ProviderConfig, timeout: float = 30.0) -> ChatProbe:
    start = time.perf_counter()
    reply = chat(
        provider,
        [{"role": "user", "content": "请只回复一个词：pong"}],
        timeout,
        max_tokens=64,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    shown = reply[:40] if reply else "已连通"
    return ChatProbe(model=provider.model, reply=shown, latency_ms=latency_ms)


def ping_embedding(provider: ProviderConfig, timeout: float = 30.0) -> EmbeddingProbe:
    start = time.perf_counter()
    vectors = embed(provider, ["ping"], timeout)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return EmbeddingProbe(model=provider.model, dimension=len(vectors[0]), latency_ms=latency_ms)
