from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from llm_intel_kb.pipeline.models import CorpusDocument, Enrichment, RawDocument, Snapshot


class CorpusIntegrityError(RuntimeError):
    """快照引用的文档在语料库中缺失（数据损坏）."""


class CorpusStore:
    """内容寻址的版本化语料存储.

    布局：
        data/corpus/<id前2位>/<id>.json        文档（不可变）
        data/enrichment/<id前2位>/<id>.json    LLM 富化缓存
        data/snapshots/<snapshot_id>.json      版本快照
        data/current.json                      当前版本指针
        data/staging/<source_id>.json          抓取暂存区
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.corpus_dir = data_dir / "corpus"
        self.enrichment_dir = data_dir / "enrichment"
        self.snapshots_dir = data_dir / "snapshots"
        self.staging_dir = data_dir / "staging"
        self.current_file = data_dir / "current.json"

    # ---------- 文档 ----------

    def save_document(self, doc: CorpusDocument) -> str:
        """写入文档；同 id 已存在时合并镜像地址并保留已有富化，返回 written/merged."""
        path = self._doc_path(doc.id)
        status = "written"
        if path.exists():
            existing = CorpusDocument.model_validate_json(path.read_text(encoding="utf-8"))
            extra_urls = {doc.url, *doc.alt_urls} - {existing.url} - set(existing.alt_urls)
            doc = doc.model_copy(
                update={
                    "url": existing.url,
                    "alt_urls": sorted(set(existing.alt_urls) | extra_urls),
                    "enrichment": doc.enrichment or existing.enrichment,
                }
            )
            status = "merged"
        _atomic_write_json(path, doc.model_dump_json())
        return status

    def get_document(self, doc_id: str) -> CorpusDocument | None:
        path = self._doc_path(doc_id)
        if not path.exists():
            return None
        return CorpusDocument.model_validate_json(path.read_text(encoding="utf-8"))

    def update_enrichment(self, doc_id: str, enrichment: Enrichment) -> None:
        """把富化结果写回文档（文档其余字段不变）."""
        doc = self.get_document(doc_id)
        if doc is None:
            raise FileNotFoundError(f"文档不存在：{doc_id}")
        doc = doc.model_copy(update={"enrichment": enrichment})
        _atomic_write_json(self._doc_path(doc_id), doc.model_dump_json())

    def iter_documents(self, doc_ids: list[str]) -> list[CorpusDocument]:
        """按 id 清单读文档，跳过缺失项（与 documents_from_snapshot 的严格版互补）."""
        docs: list[CorpusDocument] = []
        for doc_id in doc_ids:
            doc = self.get_document(doc_id)
            if doc is not None:
                docs.append(doc)
        return docs

    def documents_from_snapshot(self, snapshot: Snapshot) -> list[CorpusDocument]:
        docs: list[CorpusDocument] = []
        missing: list[str] = []
        for doc_id in snapshot.doc_ids:
            doc = self.get_document(doc_id)
            if doc is None:
                missing.append(doc_id)
            else:
                docs.append(doc)
        if missing:
            raise CorpusIntegrityError(
                f"快照 {snapshot.id} 引用的 {len(missing)} 篇文档缺失：{missing[:3]}..."
            )
        return docs

    # ---------- 快照与版本指针 ----------

    def current_snapshot(self) -> Snapshot | None:
        if not self.current_file.exists():
            return None
        snapshot_id = json.loads(self.current_file.read_text(encoding="utf-8"))["snapshot_id"]
        return self.get_snapshot(snapshot_id)

    def get_snapshot(self, snapshot_id: str) -> Snapshot:
        path = self.snapshots_dir / f"{snapshot_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"快照不存在：{snapshot_id}")
        return Snapshot.model_validate_json(path.read_text(encoding="utf-8"))

    def list_snapshots(self) -> list[Snapshot]:
        if not self.snapshots_dir.exists():
            return []
        snapshots = [
            Snapshot.model_validate_json(p.read_text(encoding="utf-8"))
            for p in sorted(self.snapshots_dir.glob("*.json"))
        ]
        return snapshots

    def create_snapshot(
        self, doc_ids: list[str], stats: dict[str, int], parent: str | None
    ) -> Snapshot:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_id = self._next_snapshot_id()
        snapshot = Snapshot(
            id=snapshot_id,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            parent=parent,
            doc_ids=doc_ids,
            stats=stats,
        )
        _atomic_write_json(self.snapshots_dir / f"{snapshot_id}.json", snapshot.model_dump_json())
        self.set_current(snapshot_id)
        return snapshot

    def set_current(self, snapshot_id: str) -> None:
        self.get_snapshot(snapshot_id)  # 校验存在
        self.data_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.current_file, json.dumps({"snapshot_id": snapshot_id}))

    def _next_snapshot_id(self) -> str:
        base = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        candidate = base
        suffix = 2
        while (self.snapshots_dir / f"{candidate}.json").exists():
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    # ---------- 暂存区 ----------

    def staging_write(self, source_id: str, docs: list[RawDocument]) -> None:
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([doc.model_dump() for doc in docs], ensure_ascii=False, indent=1)
        _atomic_write_json(self.staging_dir / f"{source_id}.json", payload)

    def staging_read_all(self) -> tuple[list[RawDocument], set[str]]:
        docs: list[RawDocument] = []
        if not self.staging_dir.exists():
            return docs, set()
        for path in sorted(self.staging_dir.glob("*.json")):
            for item in json.loads(path.read_text(encoding="utf-8")):
                docs.append(RawDocument.model_validate(item))
        return docs, {doc.source_id for doc in docs}

    def _doc_path(self, doc_id: str) -> Path:
        return self.corpus_dir / doc_id[:2] / f"{doc_id}.json"


class EnrichmentCache:
    """按内容哈希缓存的 LLM 富化结果：内容不变则永不重复付费."""

    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "enrichment"

    def get(self, doc_id: str) -> Enrichment | None:
        path = self._path(doc_id)
        if not path.exists():
            return None
        return Enrichment.model_validate_json(path.read_text(encoding="utf-8"))

    def set(self, doc_id: str, enrichment: Enrichment) -> None:
        path = self._path(doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, enrichment.model_dump_json())

    def _path(self, doc_id: str) -> Path:
        return self.dir / doc_id[:2] / f"{doc_id}.json"


def _atomic_write_json(path: Path, payload: str) -> None:
    """先写临时文件再原子替换，避免进程中断留下半截 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, path)
