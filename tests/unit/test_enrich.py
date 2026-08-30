from __future__ import annotations

from pathlib import Path

import pytest

from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig
from llm_intel_kb.pipeline.commit import build_corpus_document
from llm_intel_kb.pipeline.enrich import _parse_json_reply, enrich_corpus
from llm_intel_kb.pipeline.models import RawDocument
from llm_intel_kb.pipeline.store import CorpusStore, EnrichmentCache

PROVIDER = ProviderConfig(
    name="test", kind="chat", base_url="http://localhost:11434/v1", model="m"
)


def _seed_corpus(tmp_path: Path, n: int = 2) -> list[str]:
    store = CorpusStore(tmp_path)
    ids = []
    for i in range(n):
        raw = RawDocument(
            source_id="s",
            source_type="rss",
            url=f"https://example.com/{i}",
            title=f"标题{i}",
            content=f"内容{i} " * 60,
            fetched_at="t",
        )
        doc = build_corpus_document(raw, 10)
        assert doc is not None
        store.save_document(doc)
        ids.append(doc.id)
    return ids


class TestParseJsonReply:
    def test_plain_json(self) -> None:
        assert _parse_json_reply('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        reply = '好的，这是结果：\n```json\n{"summary": "摘要", "topics": ["a"]}\n```'
        data = _parse_json_reply(reply)
        assert data["summary"] == "摘要"

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="未找到 JSON"):
            _parse_json_reply("完全没有 JSON 的回复")


class TestEnrichCorpus:
    def test_enrich_writes_cache_and_doc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ids = _seed_corpus(tmp_path)
        store = CorpusStore(tmp_path)
        cache = EnrichmentCache(tmp_path)

        monkeypatch.setattr(
            llm_client,
            "chat",
            lambda *a, **kw: '{"summary": "这篇讲的是测试内容", "topics": ["testing"]}',
        )
        stats = enrich_corpus(store, cache, PROVIDER, ids)

        assert stats.enriched == 2
        assert stats.failed == 0
        doc = store.get_document(ids[0])
        assert doc is not None
        assert doc.enrichment is not None
        assert doc.enrichment.topics == ["testing"]

    def test_cache_hit_avoids_llm(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ids = _seed_corpus(tmp_path)
        store = CorpusStore(tmp_path)
        cache = EnrichmentCache(tmp_path)

        calls = []

        def fake_chat(*args: object, **kwargs: object) -> str:
            calls.append(1)
            return '{"summary": "s", "topics": []}'

        monkeypatch.setattr(llm_client, "chat", fake_chat)
        enrich_corpus(store, cache, PROVIDER, ids)
        assert len(calls) == 2

        enrich_corpus(store, cache, PROVIDER, ids)
        assert len(calls) == 2  # 全部缓存命中，无新调用

    def test_llm_failure_collected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ids = _seed_corpus(tmp_path, n=1)
        store = CorpusStore(tmp_path)
        cache = EnrichmentCache(tmp_path)

        def broken_chat(*args: object, **kwargs: object) -> str:
            raise RuntimeError("boom")

        monkeypatch.setattr(llm_client, "chat", broken_chat)
        stats = enrich_corpus(store, cache, PROVIDER, ids)
        assert stats.enriched == 0
        assert stats.failed == 1
        assert "boom" in stats.errors[0]

    def test_limit_controls_spend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ids = _seed_corpus(tmp_path, n=3)
        store = CorpusStore(tmp_path)
        cache = EnrichmentCache(tmp_path)

        calls = []

        def fake_chat(*args: object, **kwargs: object) -> str:
            calls.append(1)
            return '{"summary": "s", "topics": []}'

        monkeypatch.setattr(llm_client, "chat", fake_chat)
        stats = enrich_corpus(store, cache, PROVIDER, ids, limit=1)
        assert stats.enriched == 1
        assert len(calls) == 1
