"""嵌入层：从供应商配置里解析可用的 embedding 模型；无可用时返回 None（词法回退分支）."""

from __future__ import annotations

import math
from dataclasses import dataclass

from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig

EMBED_BATCH_SIZE = 32


@dataclass(frozen=True)
class EmbeddingBackend:
    """一个可用的向量化后端（远程 API 或本地服务）."""

    provider: ProviderConfig

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            vectors.extend(llm_client.embed(self.provider, batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


def resolve_embedding_backend(providers: list[ProviderConfig]) -> EmbeddingBackend | None:
    """第一个就绪的 embedding 供应商；没有则 None（调用方走纯词法分支）."""
    for provider in providers:
        if provider.supports_embedding and provider.is_ready:
            return EmbeddingBackend(provider=provider)
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度（语料规模在千级块以内，性能足够）."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
