"""本地文件索引：chunks.jsonl + vectors.json + meta.json，零外部服务依赖.

这是检索的主线实现（任何机器可跑通）；pgvector 是可选后端分支
（数据库可用时，见 store_pg.py）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from llm_intel_kb.index.chunking import Chunk
from llm_intel_kb.index.embeddings import EmbeddingBackend

_VECTOR_PRECISION = 6


@dataclass
class IndexMeta:
    snapshot_id: str
    created_at: str
    chunk_count: int
    doc_count: int
    embedding_model: str | None
    embedding_dim: int | None
    backend: str = "local"


@dataclass
class IndexState:
    """加载到内存的完整索引."""

    meta: IndexMeta
    chunks: list[Chunk]
    chunk_by_id: dict[str, Chunk] = field(default_factory=dict)
    vectors: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.chunk_by_id = {chunk.id: chunk for chunk in self.chunks}


class LocalIndexStore:
    """索引的持久化层."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.chunks_file = index_dir / "chunks.jsonl"
        self.vectors_file = index_dir / "vectors.json"
        self.meta_file = index_dir / "meta.json"

    def exists(self) -> bool:
        return self.meta_file.exists() and self.chunks_file.exists()

    def save(
        self,
        chunks: list[Chunk],
        vectors: dict[str, list[float]],
        meta: IndexMeta,
    ) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with self.chunks_file.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                payload = {
                    "id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "text": chunk.text,
                    "source_id": chunk.source_id,
                    "url": chunk.url,
                    "published_at": chunk.published_at,
                    "topics": list(chunk.topics),
                    "summary": chunk.summary,
                }
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        vectors_payload = {
            "model": meta.embedding_model,
            "dim": meta.embedding_dim,
            "vectors": {
                chunk_id: [round(v, _VECTOR_PRECISION) for v in vec]
                for chunk_id, vec in vectors.items()
            },
        }
        _atomic_write(self.vectors_file, json.dumps(vectors_payload, ensure_ascii=False))
        _atomic_write(self.meta_file, json.dumps(meta.__dict__, ensure_ascii=False))

    def load(self) -> IndexState:
        if not self.exists():
            raise FileNotFoundError("索引不存在：先执行 index build")
        meta_raw = json.loads(self.meta_file.read_text(encoding="utf-8"))
        meta = IndexMeta(**meta_raw)
        chunks: list[Chunk] = []
        with self.chunks_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                chunks.append(
                    Chunk(
                        id=item["id"],
                        doc_id=item["doc_id"],
                        chunk_index=item["chunk_index"],
                        title=item["title"],
                        text=item["text"],
                        source_id=item["source_id"],
                        url=item["url"],
                        published_at=item.get("published_at"),
                        topics=tuple(item.get("topics") or ()),
                        summary=item.get("summary"),
                    )
                )
        vectors: dict[str, list[float]] = {}
        if self.vectors_file.exists():
            vectors = json.loads(self.vectors_file.read_text(encoding="utf-8")).get("vectors", {})
        return IndexState(meta=meta, chunks=chunks, vectors=vectors)

    def build_vectors(
        self, chunks: list[Chunk], backend: EmbeddingBackend
    ) -> tuple[dict[str, list[float]], int]:
        """批量计算所有块的向量；返回 (向量表, 维度)."""
        texts = [chunk.search_text for chunk in chunks]
        vectors = backend.embed_texts(texts)
        if len(vectors) != len(chunks):
            raise RuntimeError(f"向量数量（{len(vectors)}）与块数量（{len(chunks)}）不一致")
        return dict(zip([c.id for c in chunks], vectors, strict=True)), len(vectors[0])


def new_index_meta(
    snapshot_id: str,
    chunks: list[Chunk],
    backend: EmbeddingBackend | None,
    dim: int | None,
) -> IndexMeta:
    return IndexMeta(
        snapshot_id=snapshot_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        chunk_count=len(chunks),
        doc_count=len({chunk.doc_id for chunk in chunks}),
        embedding_model=backend.model if backend else None,
        embedding_dim=dim,
    )


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, path)
