"""索引加载入口：根据 meta.backend 透明切换本地文件 / pgvector 后端."""

from __future__ import annotations

import json
import os
from pathlib import Path

from llm_intel_kb.config import get_settings
from llm_intel_kb.index.store_local import IndexMeta, IndexState, LocalIndexStore
from llm_intel_kb.index.store_pg import pg_store_from_settings


def load_index_state() -> IndexState:
    """读本地 meta.json 判断后端：backend=pg 走 pgvector，其余走本地文件."""
    local = LocalIndexStore(get_settings().data_dir / "index")
    if local.meta_file.exists():
        meta_raw = json.loads(local.meta_file.read_text(encoding="utf-8"))
        if meta_raw.get("backend") == "pg":
            try:
                return pg_store_from_settings().load()
            except FileNotFoundError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"索引在 pgvector 后端但数据库不可用（{type(exc).__name__}）。"
                    "启动数据库后重试，或 index build --backend local 重建本地索引"
                ) from exc
    return local.load()


def write_pg_meta_marker(meta_path: Path, meta: IndexMeta) -> None:
    """backend=pg 时在本地写一个 meta 标记文件（块与向量本体在数据库里）."""
    payload = dict(meta.__dict__)
    payload["backend"] = "pg"
    path = Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
