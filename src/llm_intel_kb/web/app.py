"""Web 后端：问答 / 语料浏览 / 版本历史 / 评估报告的 HTTP API."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from llm_intel_kb.config import get_settings
from llm_intel_kb.eval.runner import EvalReport
from llm_intel_kb.index.embeddings import EmbeddingBackend, resolve_embedding_backend
from llm_intel_kb.index.loader import load_index_state
from llm_intel_kb.index.search import HybridSearcher
from llm_intel_kb.index.store_local import IndexState
from llm_intel_kb.llm.providers import ProviderConfig, load_providers
from llm_intel_kb.pipeline.store import CorpusStore
from llm_intel_kb.rag.answer import generate_answer

STATIC_DIR = Path(__file__).parent / "static"


class WebContext:
    """索引状态的进程内缓存（CLI 重建索引后调 reload）."""

    lock = threading.Lock()
    _state: IndexState | None = None

    @classmethod
    def reload(cls) -> None:
        with cls.lock:
            cls._state = None

    @classmethod
    def index_state(cls) -> IndexState:
        with cls.lock:
            if cls._state is None:
                try:
                    cls._state = load_index_state()
                except FileNotFoundError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                except RuntimeError as exc:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
            return cls._state


app = FastAPI(title="llm-intel-kb", version="0.1.0")


# ---------- 请求/响应模型 ----------


class AskRequest(BaseModel):
    question: str
    k: int = 5
    provider: str | None = None
    max_tokens: int = 2048


class SearchRequest(BaseModel):
    query: str
    k: int = 8


class RollbackRequest(BaseModel):
    snapshot_id: str


# ---------- 供应商与检索 ----------


def _providers() -> list[ProviderConfig]:
    settings = get_settings()
    return load_providers(settings.providers_file or Path("llm_providers.toml"), settings)


def _chat_provider(name: str | None) -> ProviderConfig:
    ready = [p for p in _providers() if p.supports_chat and p.is_ready]
    if not ready:
        raise HTTPException(status_code=503, detail="没有就绪的对话模型（llm_providers.toml）")
    if name:
        for provider in ready:
            if provider.name == name:
                return provider
        raise HTTPException(status_code=400, detail=f"供应商不存在或未就绪：{name}")
    return ready[0]


def _searcher() -> tuple[HybridSearcher, EmbeddingBackend | None]:
    backend = resolve_embedding_backend(_providers())
    return HybridSearcher(WebContext.index_state(), backend), backend


def _hit_payload(index: int, hit: Any) -> dict[str, Any]:
    return {
        "n": index,
        "title": hit.chunk.title,
        "source_id": hit.chunk.source_id,
        "url": hit.chunk.url,
        "excerpt": hit.chunk.text[:360],
        "lexical_rank": hit.lexical_rank,
        "vector_rank": hit.vector_rank,
        "rrf_score": round(hit.rrf_score, 5),
        "published_at": hit.chunk.published_at,
    }


# ---------- 页面 ----------


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# ---------- API ----------


@app.get("/api/status")
def status() -> dict[str, Any]:
    settings = get_settings()
    corpus = CorpusStore(settings.data_dir)
    current = corpus.current_snapshot()
    try:
        state = WebContext.index_state()
        index_info: dict[str, Any] = {
            "snapshot_id": state.meta.snapshot_id,
            "chunks": state.meta.chunk_count,
            "docs": state.meta.doc_count,
            "embedding_model": state.meta.embedding_model,
            "embedding_dim": state.meta.embedding_dim,
            "built_at": state.meta.created_at,
        }
        index_ready = True
    except HTTPException:
        index_info = {}
        index_ready = False
    providers = [
        {
            "name": p.name,
            "model": p.model,
            "kind": p.kind,
            "local": p.is_local,
            "ready": p.is_ready,
        }
        for p in _providers()
    ]
    return {
        "snapshot": {
            "id": current.id if current else None,
            "docs": current.stats.get("total", 0) if current else 0,
            "created_at": current.created_at if current else None,
        },
        "index": index_info,
        "index_ready": index_ready,
        "providers": providers,
        "data_dir": str(settings.data_dir),
    }


@app.post("/api/search")
def search(req: SearchRequest) -> dict[str, Any]:
    searcher, _ = _searcher()
    hits = searcher.search(req.query, k=req.k)
    return {
        "mode": searcher.mode,
        "hits": [_hit_payload(i, hit) for i, hit in enumerate(hits, start=1)],
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    searcher, _ = _searcher()
    hits = searcher.search(req.question, k=req.k)
    if not hits:
        raise HTTPException(status_code=404, detail="检索无命中，换个说法试试")
    provider = _chat_provider(req.provider)
    result = generate_answer(req.question, hits, provider, max_tokens=req.max_tokens)
    return {
        "answer": result.answer,
        "provider": result.provider,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "mode": searcher.mode,
        "hits": [_hit_payload(i, hit) for i, hit in enumerate(hits, start=1)],
    }


@app.get("/api/providers")
def providers() -> dict[str, Any]:
    return {
        "chat": [
            {"name": p.name, "model": p.model, "local": p.is_local}
            for p in _providers()
            if p.supports_chat and p.is_ready
        ]
    }


@app.get("/api/corpus")
def corpus_list(
    snapshot_id: str | None = None,
    q: str = "",
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    store = CorpusStore(get_settings().data_dir)
    if snapshot_id:
        try:
            snapshot = store.get_snapshot(snapshot_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        current = store.current_snapshot()
        if current is None:
            return {"docs": [], "total": 0, "snapshot_id": None}
        snapshot = current
    docs = store.iter_documents(snapshot.doc_ids)
    if q:
        needle = q.lower()
        docs = [
            d
            for d in docs
            if needle in d.title.lower() or needle in d.content.lower() or needle in d.source_id
        ]
    docs.sort(key=lambda d: d.published_at or "", reverse=True)
    total = len(docs)
    start = (page - 1) * page_size
    items = [
        {
            "id": d.id,
            "title": d.title,
            "source_id": d.source_id,
            "url": d.url,
            "chars": d.content_chars,
            "language": d.language,
            "published_at": d.published_at,
            "enriched": d.enrichment is not None,
            "summary": d.enrichment.summary if d.enrichment else None,
        }
        for d in docs[start : start + page_size]
    ]
    return {"docs": items, "total": total, "snapshot_id": snapshot.id, "page": page}


@app.get("/api/corpus/{doc_id}")
def corpus_detail(doc_id: str) -> dict[str, Any]:
    store = CorpusStore(get_settings().data_dir)
    doc = store.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_id}")
    return {
        "id": doc.id,
        "title": doc.title,
        "source_id": doc.source_id,
        "url": doc.url,
        "alt_urls": doc.alt_urls,
        "content": doc.content,
        "chars": doc.content_chars,
        "language": doc.language,
        "published_at": doc.published_at,
        "fetched_at": doc.fetched_at,
        "enrichment": doc.enrichment.model_dump() if doc.enrichment else None,
    }


@app.get("/api/snapshots")
def snapshots() -> dict[str, Any]:
    store = CorpusStore(get_settings().data_dir)
    current = store.current_snapshot()
    items = []
    for snap in store.list_snapshots():
        items.append(
            {
                "id": snap.id,
                "created_at": snap.created_at,
                "parent": snap.parent,
                "total": snap.stats.get("total", len(snap.doc_ids)),
                "added": snap.stats.get("added", 0),
                "current": current is not None and snap.id == current.id,
            }
        )
    return {"snapshots": items}


@app.post("/api/snapshots/rollback")
def rollback(req: RollbackRequest) -> dict[str, Any]:
    store = CorpusStore(get_settings().data_dir)
    known = {s.id for s in store.list_snapshots()}
    if req.snapshot_id not in known:
        raise HTTPException(status_code=404, detail=f"快照不存在：{req.snapshot_id}")
    store.set_current(req.snapshot_id)
    return {"ok": True, "current": req.snapshot_id}


@app.get("/api/eval/latest")
def eval_latest() -> dict[str, Any]:
    evals_dir = get_settings().data_dir / "evals"
    reports = sorted(evals_dir.glob("eval_*.json")) if evals_dir.exists() else []
    if not reports:
        return {"report": None}
    report = EvalReport.load(reports[-1])
    entries = [
        {
            "qa_id": e.qa_id,
            "question": e.question,
            "hit": e.hit,
            "first_hit_rank": e.first_hit_rank,
            "keyword_coverage": e.keyword_coverage,
            "judge_score": e.judge_score,
            "judge_comment": e.judge_comment,
            "citations": e.citations,
            "passed": e.passed,
        }
        for e in report.entries
    ]
    return {
        "report": {
            "snapshot_id": report.snapshot_id,
            "created_at": report.created_at,
            "retriever_mode": report.retriever_mode,
            "judge_model": report.judge_model,
            "k": report.k,
            "retrieval_hit_rate": report.retrieval_hit_rate,
            "mrr": report.mrr,
            "keyword_coverage_avg": report.keyword_coverage_avg,
            "citation_rate": report.citation_rate,
            "judge_score_avg": report.judge_score_avg,
            "pass_rate": report.pass_rate,
            "entries": entries,
        }
    }


@app.get("/api/eval/history")
def eval_history() -> dict[str, Any]:
    """历次评估报告的指标摘要（按时间升序），用于前端画指标趋势对比."""
    evals_dir = get_settings().data_dir / "evals"
    reports = sorted(evals_dir.glob("eval_*.json")) if evals_dir.exists() else []
    runs = []
    for path in reports:
        try:
            report = EvalReport.load(path)
        except Exception:
            continue
        runs.append(
            {
                "file": path.name,
                "created_at": report.created_at,
                "snapshot_id": report.snapshot_id,
                "retriever_mode": report.retriever_mode,
                "judge_model": report.judge_model,
                "retrieval_hit_rate": report.retrieval_hit_rate,
                "mrr": report.mrr,
                "pass_rate": report.pass_rate,
                "keyword_coverage_avg": report.keyword_coverage_avg,
                "citation_rate": report.citation_rate,
                "judge_score_avg": report.judge_score_avg,
                "entry_count": len(report.entries),
            }
        )
    return {"runs": runs}


@app.post("/api/index/reload")
def index_reload() -> dict[str, Any]:
    """索引文件被 CLI 重建后，让 Web 进程重新加载（无需重启服务）."""
    WebContext.reload()
    return {"ok": True, "index": WebContext.index_state().meta.snapshot_id}
