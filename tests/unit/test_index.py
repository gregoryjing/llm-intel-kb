from __future__ import annotations

from llm_intel_kb.index.chunking import Chunk, chunk_document
from llm_intel_kb.index.lexical import LexicalIndex, tokenize
from llm_intel_kb.pipeline.models import CorpusDocument, Enrichment


def _doc(content: str, title: str = "标题", doc_id: str = "a" * 64) -> CorpusDocument:
    return CorpusDocument(
        id=doc_id,
        source_id="rss-a",
        source_type="rss",
        url="https://example.com/x",
        title=title,
        content=content,
        fetched_at="2026-01-01T00:00:00+00:00",
        content_chars=len(content),
        language="zh",
    )


class TestChunking:
    def test_short_doc_single_chunk(self) -> None:
        chunks = chunk_document(_doc("短文档内容"))
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_long_doc_split_with_overlap(self) -> None:
        paragraphs = [f"第{i}段。" + "内容" * 80 for i in range(20)]
        chunks = chunk_document(_doc("\n\n".join(paragraphs)))
        assert len(chunks) > 1
        # 相邻块之间应有重叠或至少覆盖全部段落
        joined = "\n\n".join(c.text for c in chunks)
        assert "第0段" in joined
        assert "第19段" in joined

    def test_enrichment_joins_search_text(self) -> None:
        doc = _doc("正文" * 100).model_copy(
            update={
                "enrichment": Enrichment(
                    summary="这是摘要",
                    topics=["rag", "retrieval"],
                    language="zh",
                    model="m",
                    enriched_at="t",
                )
            }
        )
        chunk = chunk_document(doc)[0]
        assert "这是摘要" in chunk.search_text
        assert "retrieval" in chunk.search_text


class TestTokenize:
    def test_latin_words(self) -> None:
        assert tokenize("Hello World 123") == ["hello", "world", "123"]

    def test_cjk_bigrams(self) -> None:
        tokens = tokenize("检索增强")
        assert "检" in tokens and "检索" in tokens and "索增" in tokens

    def test_mixed(self) -> None:
        tokens = tokenize("使用 RAG 技术")
        assert "rag" in tokens and "用" in tokens


class TestLexicalIndex:
    def test_ranking_by_relevance(self) -> None:
        docs = {
            "a": "vllm 推理性能优化 PagedAttention",
            "b": "今天天气不错",
            "c": "vllm 部署指南",
        }
        index = LexicalIndex.build(docs)
        results = index.search("vllm 推理", k=2)
        assert results[0][0] == "a"
        assert len(results) == 2

    def test_empty_index(self) -> None:
        index = LexicalIndex.build({})
        assert index.search("anything") == []

    def test_cjk_query(self) -> None:
        docs = {"a": "检索增强生成是重要技术", "b": "无关内容"}
        index = LexicalIndex.build(docs)
        results = index.search("检索增强")
        assert results
        assert results[0][0] == "a"


def _chunk(cid: str, text: str, doc_id: str = "d1") -> Chunk:
    return Chunk(
        id=cid,
        doc_id=doc_id,
        chunk_index=0,
        title="t",
        text=text,
        source_id="rss-a",
        url="https://example.com",
        published_at=None,
        topics=(),
        summary=None,
    )


class TestChunkSearchText:
    def test_search_text_includes_title_and_body(self) -> None:
        chunk = _chunk("c1", "正文内容")
        assert "正文内容" in chunk.search_text
