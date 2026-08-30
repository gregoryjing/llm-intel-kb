"""pgvector 索引后端：块与向量存 PostgreSQL，向量列走 pgvector 的 ANN 检索.

多路分支设计：本地文件索引（store_local.py）是主线，任何机器可跑；
本模块是可选分支 —— 数据库可用（doctor 通过）时由 index build --backend pg 启用。
表结构按快照整体重建（索引与快照一一对应，无需增量合并）。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from llm_intel_kb.index.store_local import IndexMeta, IndexState
from llm_intel_kb.store.db import create_db_engine, ensure_pgvector

if TYPE_CHECKING:

    from llm_intel_kb.index.chunking import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS index_meta (
    snapshot_id  TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    doc_count    INTEGER NOT NULL,
    embedding_model TEXT,
    embedding_dim   INTEGER
);
CREATE TABLE IF NOT EXISTS index_chunks (
    chunk_id    TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES index_meta(snapshot_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    title       TEXT NOT NULL,
    text        TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    url         TEXT NOT NULL,
    published_at TEXT,
    topics      JSONB NOT NULL DEFAULT '[]',
    summary     TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_snapshot ON index_chunks(snapshot_id);
"""

_ANN_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_vectors_ann ON index_vectors "
    "USING hnsw (embedding vector_cosine_ops)"
)


class PgIndexStore:
    """索引的 PostgreSQL + pgvector 持久化层（接口与 LocalIndexStore 对齐）."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def _engine(self) -> Engine:
        return create_db_engine(self.database_url)

    def exists(self) -> bool:
        engine = self._engine()
        try:
            with engine.connect() as conn:
                found = conn.execute(text("SELECT to_regclass('index_meta')")).scalar_one_or_none()
                if found is None:
                    return False
                row_count = int(conn.execute(text("SELECT count(*) FROM index_meta")).scalar_one())
                return row_count > 0
        finally:
            engine.dispose()

    def save(
        self,
        chunks: list[Chunk],
        vectors: dict[str, list[float]],
        meta: IndexMeta,
    ) -> None:
        engine = self._engine()
        try:
            ensure_pgvector(engine)
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE IF EXISTS index_vectors"))
                conn.execute(text("DROP TABLE IF EXISTS index_chunks"))
                conn.execute(text("DROP TABLE IF EXISTS index_meta"))
                for statement in filter(None, (s.strip() for s in _SCHEMA.split(";"))):
                    conn.execute(text(statement))
                conn.execute(
                    text(
                        "INSERT INTO index_meta (snapshot_id, created_at, chunk_count, doc_count,"
                        " embedding_model, embedding_dim) VALUES (:s, :c, :k, :d, :m, :v)"
                    ),
                    {
                        "s": meta.snapshot_id,
                        "c": meta.created_at,
                        "k": meta.chunk_count,
                        "d": meta.doc_count,
                        "m": meta.embedding_model,
                        "v": meta.embedding_dim,
                    },
                )
                conn.execute(
                    text(
                        "INSERT INTO index_chunks (chunk_id, snapshot_id, doc_id, chunk_index,"
                        " title, text, source_id, url, published_at, topics, summary)"
                        " VALUES (:i, :s, :doc, :ci, :t, :tx, :src, :u, :p, :tp, :sm)"
                    ),
                    [
                        {
                            "i": chunk.id,
                            "s": meta.snapshot_id,
                            "doc": chunk.doc_id,
                            "ci": chunk.chunk_index,
                            "t": chunk.title,
                            "tx": chunk.text,
                            "src": chunk.source_id,
                            "u": chunk.url,
                            "p": chunk.published_at,
                            "tp": json.dumps(list(chunk.topics)),
                            "sm": chunk.summary,
                        }
                        for chunk in chunks
                    ],
                )
                if vectors:
                    dim = len(next(iter(vectors.values())))
                    conn.execute(
                        text(
                            "CREATE TABLE index_vectors (chunk_id TEXT PRIMARY KEY"
                            " REFERENCES index_chunks(chunk_id) ON DELETE CASCADE,"
                            f" embedding vector({dim}))"
                        )
                    )
                    conn.execute(
                        text(
                            "INSERT INTO index_vectors (chunk_id, embedding)"
                            " VALUES (:c, :v::vector)"
                        ),
                        [
                            {
                                "c": chunk_id,
                                "v": "[" + ",".join(f"{v:.6f}" for v in vec) + "]",
                            }
                            for chunk_id, vec in vectors.items()
                        ],
                    )
                    conn.execute(text(_ANN_INDEX))
        finally:
            engine.dispose()

    def load(self) -> IndexState:
        from llm_intel_kb.index.chunking import Chunk

        engine = self._engine()
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT snapshot_id, created_at, chunk_count, doc_count,"
                        " embedding_model, embedding_dim FROM index_meta LIMIT 1"
                    )
                ).mappings().first()
                if row is None:
                    raise FileNotFoundError("pgvector 索引不存在：先执行 index build --backend pg")
                meta = IndexMeta(backend="pg", **dict(row))
                chunks: list[Chunk] = []
                for item in conn.execute(
                    text(
                        "SELECT chunk_id, doc_id, chunk_index, title, text, source_id, url,"
                        " published_at, topics, summary FROM index_chunks"
                    )
                ).mappings():
                    chunks.append(
                        Chunk(
                            id=item["chunk_id"],
                            doc_id=item["doc_id"],
                            chunk_index=item["chunk_index"],
                            title=item["title"],
                            text=item["text"],
                            source_id=item["source_id"],
                            url=item["url"],
                            published_at=item["published_at"],
                            topics=tuple(item["topics"] or ()),
                            summary=item["summary"],
                        )
                    )
                vectors: dict[str, list[float]] = {}
                for item in conn.execute(
                    text("SELECT chunk_id, embedding::text FROM index_vectors")
                ).mappings():
                    vectors[item["chunk_id"]] = [float(v) for v in item["embedding"].strip("[]").split(",")]
        finally:
            engine.dispose()
        return IndexState(meta=meta, chunks=chunks, vectors=vectors)

    def ann_search(self, query_vector: list[float], k: int = 5) -> list[tuple[str, float]]:
        """pgvector 余弦距离 ANN 检索：返回 (chunk_id, distance)，按距离升序."""
        engine = self._engine()
        literal = "[" + ",".join(f"{v:.6f}" for v in query_vector) + "]"
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT chunk_id, 1 - (embedding <=> :q::vector) AS similarity"
                        " FROM index_vectors ORDER BY embedding <=> :q::vector LIMIT :k"
                    ),
                    {"q": literal, "k": k},
                ).all()
            return [(str(chunk_id), float(similarity)) for chunk_id, similarity in rows]
        finally:
            engine.dispose()


def pg_store_from_settings() -> PgIndexStore:
    """从全局配置构建 pg 索引存储（database_url 来自 .env / 环境变量）."""
    from llm_intel_kb.config import get_settings

    return PgIndexStore(get_settings().database_url)


def probe_pg_ready() -> tuple[bool, str]:
    """探测 pg 后端是否可用（数据库 + pgvector 扩展），返回 (可用, 说明)."""
    try:
        store = pg_store_from_settings()
        engine = store._engine()
        try:
            from llm_intel_kb.store.db import probe_database

            info: dict[str, Any] = probe_database(engine)
        finally:
            engine.dispose()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:160]
    if not info.get("pgvector_version"):
        return False, "数据库可达但未安装 pgvector 扩展（运行 init-db）"
    return True, f"PostgreSQL {info['server_version']} · pgvector {info['pgvector_version']}"
