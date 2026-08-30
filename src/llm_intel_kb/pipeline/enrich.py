from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig
from llm_intel_kb.pipeline.models import CorpusDocument, Enrichment
from llm_intel_kb.pipeline.store import CorpusStore, EnrichmentCache

_ENRICH_PROMPT = """你是一个 AI/LLM 领域情报库的编目员。请阅读下面的文档，输出一个 JSON 对象：
- summary：用中文写 2-3 句话摘要，说明这件事对 AI 应用工程师的价值
- topics：3-6 个英文主题词（小写、空格分隔的多词短语也可以），如 ["retrieval augmentation", "vllm"]

只输出 JSON，不要输出其他任何内容。

标题：{title}
正文：{content}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_CONTENT_CHARS = 6000


@dataclass
class EnrichStats:
    cached: int = 0
    enriched: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def enrich_document(doc: CorpusDocument, provider: ProviderConfig) -> Enrichment:
    """调用 LLM 为单篇文档生成中文摘要 + 英文主题词；失败抛异常."""
    truncated = doc.content[:_MAX_CONTENT_CHARS]
    prompt = _ENRICH_PROMPT.format(title=doc.title, content=truncated)
    reply = llm_client.chat(
        provider,
        [{"role": "user", "content": prompt}],
        timeout=180.0,
        max_tokens=512,
    )
    data = _parse_json_reply(reply)
    raw_topics = data.get("topics", [])
    topics_iter = raw_topics if isinstance(raw_topics, list) else []
    topics = [str(t).strip().lower() for t in topics_iter if str(t).strip()]
    summary = str(data.get("summary") or "").strip()
    if not summary:
        raise ValueError("LLM 富化结果缺少 summary 字段")
    return Enrichment(
        summary=summary,
        topics=topics[:6],
        language=doc.language,
        model=provider.model,
        enriched_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def enrich_corpus(
    store: CorpusStore,
    cache: EnrichmentCache,
    provider: ProviderConfig,
    doc_ids: list[str],
    limit: int | None = None,
) -> EnrichStats:
    """对一组文档做富化：缓存命中直接复用，未命中调 LLM 并写回缓存与文档."""
    stats = EnrichStats()
    for doc_id in doc_ids:
        doc = store.get_document(doc_id)
        if doc is None:
            stats.failed += 1
            stats.errors.append(f"{doc_id[:8]}: 文档不存在")
            continue
        cached = cache.get(doc_id)
        if cached is not None:
            if doc.enrichment is None:
                store.update_enrichment(doc_id, cached)
            stats.cached += 1
            continue
        if limit is not None and stats.enriched >= limit:
            break
        try:
            enrichment = enrich_document(doc, provider)
        except Exception as exc:
            stats.failed += 1
            stats.errors.append(f"{doc_id[:8]}: {type(exc).__name__}: {exc}")
            continue
        cache.set(doc_id, enrichment)
        store.update_enrichment(doc_id, enrichment)
        stats.enriched += 1
    return stats


def _parse_json_reply(reply: str) -> dict[str, object]:
    """容错解析：模型可能把 JSON 包在 ```json 围栏或多余文本里."""
    match = _JSON_BLOCK_RE.search(reply)
    if match is None:
        raise ValueError(f"回复中未找到 JSON：{reply[:80]}")
    parsed: object = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("富化结果不是 JSON 对象")
    return parsed
