"""混合检索：BM25 词法 + 向量语义 → RRF 融合 → 结果去 diversify（同文档限流）."""

from __future__ import annotations

from dataclasses import dataclass

from llm_intel_kb.index.chunking import Chunk
from llm_intel_kb.index.embeddings import EmbeddingBackend, cosine_similarity
from llm_intel_kb.index.lexical import LexicalIndex
from llm_intel_kb.index.store_local import IndexState

RRF_K = 60
MAX_PER_DOC = 3

# 内容质量惩罚：search_text 低于此字符数视为"标题级"文档，开始降权。
# qbitai 平均 20 字符、openai-news 平均 151 字符；有内容的文档 > 500 字符。
# 取 100 作为拐点：让绝大多数标题级源落入惩罚区，同时不误伤较短的 release note。
_QUALITY_FULL_CHARS = 100
# 标题级文档的 RRF 分数乘以此系数（0~1），降低其在 top-k 中的排位。
# 0.5 意味着标题级文档的 RRF 分减半，但不会完全消失——用户明确查"行业动态"时仍可召回。
_TITLE_QUALITY_MULTIPLIER = 0.5


def _quality_multiplier(search_text_length: int) -> float:
    """根据检索文本长度计算质量乘数.

    短文档（标题级）返回 _TITLE_QUALITY_MULTIPLIER（降权），
    长文档返回 1.0（不惩罚）。
    使用平滑过渡避免硬阈值导致的排名跳跃。
    """
    if search_text_length <= 0:
        return _TITLE_QUALITY_MULTIPLIER
    if search_text_length >= _QUALITY_FULL_CHARS:
        return 1.0
    # 线性插值：从 _TITLE_QUALITY_MULTIPLIER 渐变到 1.0
    ratio = search_text_length / _QUALITY_FULL_CHARS
    return _TITLE_QUALITY_MULTIPLIER + (1.0 - _TITLE_QUALITY_MULTIPLIER) * ratio


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    rrf_score: float
    lexical_rank: int | None  # 1-based；None = 词法未命中
    vector_rank: int | None  # 1-based；None = 向量未命中或未启用


class HybridSearcher:
    """对已加载索引做混合检索；无向量时自动退化为纯词法."""

    def __init__(self, state: IndexState, backend: EmbeddingBackend | None) -> None:
        self.state = state
        self.backend = backend
        self.lexical = LexicalIndex.build(
            {chunk.id: chunk.search_text for chunk in state.chunks}
        )
        self.has_vectors = bool(state.vectors) and backend is not None
        # 预计算每个 chunk 的内容质量乘数（基于 search_text 长度）
        self._chunk_quality: dict[str, float] = {
            chunk.id: _quality_multiplier(len(chunk.search_text)) for chunk in state.chunks
        }

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if k <= 0:
            return []
        fetch_k = max(k * 4, 20)  # 先取宽，融合后再截断
        lexical_hits = self.lexical.search(query, k=fetch_k)
        vector_hits: list[tuple[str, float]] = []
        if self.has_vectors and self.backend is not None:
            query_vector = self.backend.embed_query(query)
            scored = [
                (chunk_id, cosine_similarity(query_vector, vec))
                for chunk_id, vec in self.state.vectors.items()
            ]
            vector_hits = sorted(scored, key=lambda item: (-item[1], item[0]))[:fetch_k]

        rrf: dict[str, float] = {}
        lexical_rank_map = {chunk_id: rank for rank, (chunk_id, _) in enumerate(lexical_hits, start=1)}
        vector_rank_map = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_hits, start=1)}
        for chunk_id, rank in lexical_rank_map.items():
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for chunk_id, rank in vector_rank_map.items():
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        # 应用内容质量惩罚：标题级短文档的 RRF 分乘以一个 <1 的系数，
        # 让有实质内容的文档在同等 RRF 分下排位更高。
        if rrf:
            min_quality = min(self._chunk_quality.get(cid, 1.0) for cid in rrf)
            if min_quality < 1.0:
                for chunk_id in rrf:
                    rrf[chunk_id] *= self._chunk_quality.get(chunk_id, 1.0)

        ranked_ids = sorted(rrf, key=lambda chunk_id: (-rrf[chunk_id], chunk_id))
        hits: list[SearchHit] = []
        per_doc: dict[str, int] = {}
        for chunk_id in ranked_ids:
            chunk = self.state.chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            if per_doc.get(chunk.doc_id, 0) >= MAX_PER_DOC:
                continue
            per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
            hits.append(
                SearchHit(
                    chunk=chunk,
                    rrf_score=rrf[chunk_id],
                    lexical_rank=lexical_rank_map.get(chunk_id),
                    vector_rank=vector_rank_map.get(chunk_id),
                )
            )
            if len(hits) >= k:
                break
        return hits

    @property
    def mode(self) -> str:
        if self.has_vectors:
            return "hybrid（词法 + 向量 RRF 融合）"
        return "lexical（纯词法 BM25；配置 embedding 供应商后自动升级混合检索）"
