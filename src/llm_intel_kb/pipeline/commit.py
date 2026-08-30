from __future__ import annotations

from dataclasses import dataclass, field

from llm_intel_kb.pipeline.clean import content_hash, detect_language, normalize_text
from llm_intel_kb.pipeline.models import CorpusDocument, RawDocument, Snapshot, SourceSpec
from llm_intel_kb.pipeline.store import CorpusStore


@dataclass
class CommitResult:
    snapshot: Snapshot
    added: int = 0
    merged: int = 0
    skipped_short: int = 0
    skipped_no_source: int = 0
    dropped: list[str] = field(default_factory=list)


def build_corpus_document(raw: RawDocument, min_chars: int) -> CorpusDocument | None:
    """清洗 + 哈希 + 语种检测，产出语料文档；内容过短返回 None."""
    title = normalize_text(raw.title)
    content = normalize_text(raw.content)
    if len(content) < min_chars:
        return None
    return CorpusDocument(
        id=content_hash(title, content),
        source_id=raw.source_id,
        source_type=raw.source_type,
        url=raw.url,
        title=title,
        content=content,
        published_at=raw.published_at,
        fetched_at=raw.fetched_at,
        content_chars=len(content),
        language=detect_language(content),
    )


def commit_staged(
    store: CorpusStore,
    specs: list[SourceSpec],
) -> CommitResult:
    """把暂存区所有文档清洗入库并创建新快照.

    语义：
    - corpus 只增不删（内容寻址、不可变）
    - 新快照 = 上一快照 doc_ids ∪ 本次入库 doc_ids
    - 抓取失败（staging 缺失）的源不视为删除，文档保留
    """
    staged_docs, staged_sources = store.staging_read_all()
    min_chars_by_source = {spec.id: spec.min_chars for spec in specs}

    parent = store.current_snapshot()
    written_ids: list[str] = []
    result = CommitResult(
        snapshot=Snapshot(id="", created_at="", parent=None, doc_ids=[], stats={})
    )

    for raw in staged_docs:
        min_chars = min_chars_by_source.get(raw.source_id)
        if min_chars is None:
            # 注册表里已删除的源：跳过，不让坏数据进库
            result.skipped_no_source += 1
            continue
        doc = build_corpus_document(raw, min_chars)
        if doc is None:
            result.skipped_short += 1
            continue
        status = store.save_document(doc)
        if status == "written":
            result.added += 1
        else:
            result.merged += 1
        written_ids.append(doc.id)

    previous_ids = set(parent.doc_ids) if parent else set()
    merged_ids = sorted(previous_ids | set(written_ids))

    stats = {
        "added": result.added,
        "merged": result.merged,
        "skipped_short": result.skipped_short,
        "skipped_no_source": result.skipped_no_source,
        "total": len(merged_ids),
        "sources_fetched": len(staged_sources),
    }

    snapshot = store.create_snapshot(
        doc_ids=merged_ids,
        stats=stats,
        parent=parent.id if parent else None,
    )
    result.snapshot = snapshot
    return result


def diff_snapshots(old: Snapshot, new: Snapshot) -> dict[str, list[str]]:
    """两个快照的文档 id 差异."""
    old_ids, new_ids = set(old.doc_ids), set(new.doc_ids)
    return {
        "added": sorted(new_ids - old_ids),
        "removed": sorted(old_ids - new_ids),
        "unchanged": sorted(old_ids & new_ids),
    }
