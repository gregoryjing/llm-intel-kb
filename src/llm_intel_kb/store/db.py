from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

_PGVECTOR_VERSION_SQL = text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")


def create_db_engine(database_url: str) -> Engine:
    # connect_timeout: 部分环境下连到未监听端口不会立刻拒绝而是挂起，必须显式限时
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )


def probe_database(engine: Engine) -> dict[str, Any]:
    """探测数据库连通性与 pgvector 扩展状态."""
    with engine.connect() as conn:
        server_version = conn.execute(text("SELECT version()")).scalar_one()
        pgvector_version = conn.execute(_PGVECTOR_VERSION_SQL).scalar_one_or_none()
    # "PostgreSQL 16.4 (Debian 16.4-1.pgdg120+2) ..." -> "16.4"
    short_version = str(server_version).split(" ")[1] if server_version else "unknown"
    return {
        "server_version": short_version,
        "pgvector_version": pgvector_version,
    }


def ensure_pgvector(engine: Engine) -> str:
    """安装 pgvector 扩展（幂等）并返回其版本."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    with engine.connect() as conn:
        version = conn.execute(_PGVECTOR_VERSION_SQL).scalar_one_or_none()
    if version is None:
        raise RuntimeError("pgvector 扩展创建后仍未检测到，请确认使用 pgvector 官方镜像")
    return str(version)
