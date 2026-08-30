from __future__ import annotations

from pathlib import Path

import pytest

from llm_intel_kb.pipeline.clean import content_hash, detect_language, normalize_text, strip_html
from llm_intel_kb.pipeline.commit import build_corpus_document, commit_staged
from llm_intel_kb.pipeline.models import RawDocument, SourceSpec
from llm_intel_kb.pipeline.store import CorpusStore


def _raw(title: str, content: str, source_id: str = "rss-a") -> RawDocument:
    return RawDocument(
        source_id=source_id,
        source_type="rss",
        url=f"https://example.com/{title}",
        title=title,
        content=content,
        fetched_at="2026-01-01T00:00:00+00:00",
    )


class TestClean:
    def test_strip_html_removes_tags_and_unescapes(self) -> None:
        assert strip_html("<p>Hello &amp;   world</p>") == "Hello & world"

    def test_normalize_collapses_blank_lines(self) -> None:
        assert normalize_text("a\n\n\n\nb") == "a\n\nb"

    def test_content_hash_stable_and_sensitive(self) -> None:
        base = content_hash("t", "c")
        assert base == content_hash(" t ", " c ")
        assert base != content_hash("t", "c2")

    def test_detect_language(self) -> None:
        assert detect_language("这是一段中文文本，用于测试语言检测功能。") == "zh"
        assert detect_language("plain english text without cjk") == "en"


class TestBuildCorpusDocument:
    def test_short_content_dropped(self) -> None:
        spec_min = 100
        doc = build_corpus_document(_raw("t", "x" * 50), spec_min)
        assert doc is None

    def test_hash_is_id(self) -> None:
        doc = build_corpus_document(_raw("标题", "内容" * 100), 10)
        assert doc is not None
        assert doc.id == content_hash("标题", "内容" * 100)
        assert doc.language == "zh"


class TestCommitStaged:
    def test_first_commit_and_incremental(self, tmp_path: Path) -> None:
        store = CorpusStore(tmp_path)
        spec = SourceSpec(id="rss-a", type="rss", url="https://example.com/feed", min_chars=10)
        store.staging_write("rss-a", [_raw("a", "内容" * 50), _raw("b", "content " * 50)])

        first = commit_staged(store, [spec])
        assert first.added == 2
        assert first.snapshot.stats["total"] == 2

        # 增量复跑：同内容 → 全部合并，总量不变
        store.staging_write("rss-a", [_raw("a", "内容" * 50)])
        second = commit_staged(store, [spec])
        assert second.added == 0
        assert second.snapshot.stats["total"] == 2
        assert second.snapshot.parent == first.snapshot.id

    def test_new_doc_union_accumulates(self, tmp_path: Path) -> None:
        store = CorpusStore(tmp_path)
        spec = SourceSpec(id="rss-a", type="rss", url="https://example.com/feed", min_chars=10)
        store.staging_write("rss-a", [_raw("a", "内容" * 50)])
        commit_staged(store, [spec])
        store.staging_write("rss-a", [_raw("new", "新内容" * 50)])
        result = commit_staged(store, [spec])
        assert result.snapshot.stats["total"] == 2

    def test_unknown_source_skipped(self, tmp_path: Path) -> None:
        store = CorpusStore(tmp_path)
        store.staging_write("ghost", [_raw("a", "内容" * 50)])
        result = commit_staged(store, [])
        assert result.skipped_no_source == 1
        assert result.snapshot.stats["total"] == 0


class TestCorpusStore:
    def test_save_merges_alt_urls(self, tmp_path: Path) -> None:
        store = CorpusStore(tmp_path)
        from llm_intel_kb.pipeline.commit import build_corpus_document

        doc_a = build_corpus_document(_raw("t", "内容" * 50), 10)
        assert doc_a is not None
        store.save_document(doc_a)

        mirror = doc_a.model_copy(update={"url": "https://mirror.example.com/x"})
        status = store.save_document(mirror)
        assert status == "merged"

        loaded = store.get_document(doc_a.id)
        assert loaded is not None
        assert loaded.url == doc_a.url
        assert "https://mirror.example.com/x" in loaded.alt_urls

    def test_rollback_switches_current(self, tmp_path: Path) -> None:
        store = CorpusStore(tmp_path)
        spec = SourceSpec(id="rss-a", type="rss", url="https://example.com/feed", min_chars=10)
        store.staging_write("rss-a", [_raw("a", "内容" * 50)])
        first = commit_staged(store, [spec]).snapshot
        store.staging_write("rss-a", [_raw("b", "内容" * 50)])
        second = commit_staged(store, [spec]).snapshot
        current = store.current_snapshot()
        assert current is not None
        assert current.id == second.id

        store.set_current(first.id)
        rolled = store.current_snapshot()
        assert rolled is not None
        assert rolled.id == first.id

    def test_documents_from_snapshot_missing_raises(self, tmp_path: Path) -> None:
        store = CorpusStore(tmp_path)
        from llm_intel_kb.pipeline.models import Snapshot

        snapshot = Snapshot(id="s", created_at="t", parent=None, doc_ids=["deadbeef"], stats={})
        with pytest.raises(Exception, match="缺失"):
            store.documents_from_snapshot(snapshot)
