"""引用生成：检索块编号进上下文，要求模型用 [1][2] 引用，答不上就说不知道."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from llm_intel_kb.index.search import SearchHit
from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig

_SYSTEM_PROMPT = (
    "你是 AI/LLM 领域情报库的问答助手。只依据提供的资料回答；"
    "每个事实性陈述后用 [编号] 标注来源；资料不足就直说「根据现有资料无法回答」，不要编造。"
    "用中文回答，保持简洁。"
)

# 上下文字符上限：约 2400 token（中英混合），加上 1024 token 答案预算
# 恰好落在本地模型 num_ctx=3072 内——超发会触发 Ollama 截断或显存溢出降速。
_MAX_CONTEXT_CHARS = 5000


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    hits: list[SearchHit]
    provider: str
    model: str
    latency_ms: int


def build_context(hits: list[SearchHit]) -> str:
    parts: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        block = f"[{index}] {hit.chunk.title}\n来源：{hit.chunk.source_id} · {hit.chunk.url}\n{hit.chunk.text}"
        if used + len(block) > _MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    hits: list[SearchHit],
    provider: ProviderConfig,
    *,
    timeout: float = 180.0,
    max_tokens: int = 1024,
) -> AnswerResult:
    context = build_context(hits)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}\n\n资料：\n{context}"},
    ]
    start = time.perf_counter()
    reply = llm_client.chat(provider, messages, timeout, max_tokens=max_tokens, temperature=0.2)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return AnswerResult(
        answer=reply,
        hits=hits,
        provider=provider.name,
        model=provider.model,
        latency_ms=latency_ms,
    )


def extract_citations(answer: str) -> list[int]:
    """从回答文本里提取 [1][2] 形式的引用编号（去重保序）."""
    seen: list[int] = []
    for match in re.findall(r"\[(\d+)\]", answer):
        number = int(match)
        if number not in seen:
            seen.append(number)
    return seen
