"""内容质量感知检索排序的单元测试：验证标题级短文档被降权但不被过滤."""

from __future__ import annotations

from llm_intel_kb.index.chunking import Chunk
from llm_intel_kb.index.search import (
    _QUALITY_FULL_CHARS,
    _TITLE_QUALITY_MULTIPLIER,
    HybridSearcher,
    _quality_multiplier,
)
from llm_intel_kb.index.store_local import IndexMeta, IndexState


def _chunk(
    cid: str,
    text: str,
    title: str = "标题",
    doc_id: str = "d1",
    source_id: str = "src",
    topics: tuple[str, ...] = (),
    summary: str | None = None,
) -> Chunk:
    return Chunk(
        id=cid,
        doc_id=doc_id,
        chunk_index=0,
        title=title,
        text=text,
        source_id=source_id,
        url="https://example.com",
        published_at=None,
        topics=topics,
        summary=summary,
    )


def _make_state(chunks: list[Chunk]) -> IndexState:
    return IndexState(
        meta=IndexMeta(
            snapshot_id="s1",
            created_at="2026-01-01T00:00:00+00:00",
            chunk_count=len(chunks),
            doc_count=len({c.doc_id for c in chunks}),
            embedding_model=None,
            embedding_dim=None,
        ),
        chunks=chunks,
    )


class TestQualityMultiplier:
    """_quality_multiplier 的边界值测试."""

    def test_zero_length_gets_title_multiplier(self) -> None:
        assert _quality_multiplier(0) == _TITLE_QUALITY_MULTIPLIER

    def test_negative_length_gets_title_multiplier(self) -> None:
        assert _quality_multiplier(-10) == _TITLE_QUALITY_MULTIPLIER

    def test_very_short_gets_title_multiplier(self) -> None:
        # 10 字符远低于 _QUALITY_FULL_CHARS(100)，应接近标题级乘数
        result = _quality_multiplier(10)
        assert result == _TITLE_QUALITY_MULTIPLIER + (1.0 - _TITLE_QUALITY_MULTIPLIER) * (10 / _QUALITY_FULL_CHARS)
        assert result < 0.6  # 明显降权

    def test_qbitai_typical_length_strongly_downweighted(self) -> None:
        # qbitai 平均 20 字符的 search_text 应被大幅降权（乘数 <= 0.6）
        result = _quality_multiplier(20)
        assert result <= 0.6
        assert result < 1.0

    def test_openai_news_typical_length_moderately_downweighted(self) -> None:
        # openai-news 平均 151 字符 >= 阈值(100)，不再降权
        result = _quality_multiplier(151)
        assert result == 1.0

    def test_medium_length_partially_downweighted(self) -> None:
        # 80 字符介于极短和阈值之间，应被部分降权
        result = _quality_multiplier(80)
        assert 0.5 < result < 1.0

    def test_full_chars_gets_no_penalty(self) -> None:
        assert _quality_multiplier(_QUALITY_FULL_CHARS) == 1.0

    def test_long_text_gets_no_penalty(self) -> None:
        assert _quality_multiplier(1000) == 1.0

    def test_monotonically_increasing(self) -> None:
        # 乘数应随长度单调不减
        lengths = [0, 10, 20, 50, 80, 99, 100, 200]
        values = [_quality_multiplier(length) for length in lengths]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]


class TestHybridSearcherQualityPenalty:
    """HybridSearcher 中质量惩罚对排名影响的测试."""

    def test_title_only_chunk_pushed_behind_content_chunk(self) -> None:
        """关键词匹配的标题级短文档应被有实质内容的长文档超越.

        模拟 g008 场景：qbitai 标题含 'agent' 但无实质内容，
        arxiv 论文有 'agent' 相关真实内容。
        """
        title_only = _chunk(
            "qbitai:0",
            text="Agent业务半年进账近5亿",  # ~10 chars of real content
            title="港股AGI第一股Agent业务半年进账近5亿",
            doc_id="qbitai_doc1",
            source_id="qbitai",
        )
        content_chunk = _chunk(
            "arxiv:0",
            text="This paper presents a novel multi-agent framework for LLM training. "
            "We propose a coordinated agent architecture where multiple AI agents collaborate "
            "to solve complex reasoning tasks. Our agent-based approach achieves state-of-the-art "
            "results on agent benchmarks and multi-agent coordination tasks.",
            title="A Multi-Agent Framework for LLM Training",
            doc_id="arxiv_doc1",
            source_id="arxiv-cs-ai",
        )
        state = _make_state([title_only, content_chunk])
        searcher = HybridSearcher(state, backend=None)

        hits = searcher.search("agent 多智能体", k=5)
        assert len(hits) == 2
        # 内容丰富的 arxiv 块应排在前面
        assert hits[0].chunk.source_id == "arxiv-cs-ai"
        assert hits[1].chunk.source_id == "qbitai"

    def test_k_equals_zero_returns_empty(self) -> None:
        chunk = _chunk("a", "vllm 推理性能优化 PagedAttention 内存管理", doc_id="d1")
        state = _make_state([chunk])
        searcher = HybridSearcher(state, backend=None)
        assert searcher.search("vllm", k=0) == []

    def test_quality_penalty_does_not_filter(self) -> None:
        """质量惩罚只降权不过滤——标题级文档仍在结果中."""
        title_only = _chunk(
            "qbitai:0",
            text="Agent大新闻",  # Very short
            title="Agent新闻",
            doc_id="qbitai_doc1",
            source_id="qbitai",
        )
        state = _make_state([title_only])
        searcher = HybridSearcher(state, backend=None)
        hits = searcher.search("agent", k=5)
        assert len(hits) == 1  # 仍然被召回
        assert hits[0].chunk.source_id == "qbitai"

    def test_no_penalty_when_all_chunks_are_quality(self) -> None:
        """所有 chunk 都 >= _QUALITY_FULL_CHARS 时，质量乘数全为 1.0，不改变排序."""
        chunk_a = _chunk("a", "x" * 200, doc_id="d1")  # 200 chars
        chunk_b = _chunk("b", "y" * 200, doc_id="d2")  # 200 chars
        state = _make_state([chunk_a, chunk_b])

        # 验证质量乘数
        assert searcher_quality(searcher := HybridSearcher(state, backend=None), "a") == 1.0
        assert searcher_quality(searcher, "b") == 1.0

    def test_rrf_score_reflects_quality_penalty(self) -> None:
        """验证存储在 SearchHit 中的 rrf_score 包含了质量惩罚."""
        title_only = _chunk(
            "qbitai:0",
            text="agent news",
            title="Agent 新闻",
            doc_id="q1",
            source_id="qbitai",
        )
        content_chunk = _chunk(
            "arxiv:0",
            text="agent " * 50,  # 约 300 字符
            title="Agent Research",
            doc_id="a1",
            source_id="arxiv",
        )
        state = _make_state([title_only, content_chunk])
        searcher = HybridSearcher(state, backend=None)

        # 质量乘数：标题级惩罚更大，内容块无惩罚
        q_title = _quality_multiplier(len(title_only.search_text))
        q_content = _quality_multiplier(len(content_chunk.search_text))
        assert q_title < q_content
        assert q_content == 1.0

        # 落到 SearchHit.rrf_score 上：内容块分数应高于被降权的标题级块
        hits = searcher.search("agent", k=5)
        scores = {hit.chunk.id: hit.rrf_score for hit in hits}
        assert scores["qbitai:0"] < scores["arxiv:0"]


def searcher_quality(searcher: HybridSearcher, chunk_id: str) -> float:
    """辅助函数：获取 searcher 中某 chunk 的质量乘数."""
    return searcher._chunk_quality.get(chunk_id, 1.0)
