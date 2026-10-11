"""Web 后端：问答 / 语料浏览 / 版本历史 / 评估报告的 HTTP API."""

from __future__ import annotations

import concurrent.futures
import json
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from llm_intel_kb.config import get_settings
from llm_intel_kb.eval.golden import load_golden
from llm_intel_kb.eval.importers import audit_golden, import_golden_entries, parse_import_text
from llm_intel_kb.eval.runner import EvalReport
from llm_intel_kb.guard import (
    GuardConfig,
    append_guard_log,
    check_question,
    clear_guard_log,
    load_guard_config,
    read_guard_log,
    save_guard_config,
)
from llm_intel_kb.guard.domain import VALID_MODES as VALID_GUARD_MODES
from llm_intel_kb.guard.regression import (
    DEFAULT_CASES_FILE,
    cases_from_log,
    load_cases,
    merge_cases,
    run_cases,
    save_cases,
)
from llm_intel_kb.index.embeddings import EmbeddingBackend, resolve_embedding_backend
from llm_intel_kb.index.loader import load_index_state
from llm_intel_kb.index.search import HybridSearcher
from llm_intel_kb.index.store_local import IndexState
from llm_intel_kb.llm.providers import (
    ADAPTER_ANTHROPIC,
    ADAPTER_OLLAMA,
    ADAPTER_OPENAI,
    ProviderConfig,
    load_providers,
)
from llm_intel_kb.pipeline.sources import enabled_source_ids
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


class RuntimeProviders:
    """运行时供应商注册（页面添加的 API 供应商，仅当前会话有效，不写 toml）."""

    _lock = threading.Lock()
    _providers: ClassVar[list[ProviderConfig]] = []

    @classmethod
    def list(cls) -> list[ProviderConfig]:
        with cls._lock:
            return list(cls._providers)

    @classmethod
    def add(cls, provider: ProviderConfig) -> None:
        with cls._lock:
            cls._providers.append(provider)

    @classmethod
    def remove(cls, name: str) -> bool:
        with cls._lock:
            before = len(cls._providers)
            cls._providers[:] = [p for p in cls._providers if p.name != name]
            return len(cls._providers) < before


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


class AddProviderRequest(BaseModel):
    name: str
    base_url: str
    model: str
    api_key: str | None = None
    adapter: str = "openai"
    kind: str = "chat"


class CompareRequest(BaseModel):
    question: str
    k: int = 5
    providers: list[str] = []
    max_tokens: int = 1024


class GuardConfigRequest(BaseModel):
    enabled: bool = True
    scope_name: str = ""
    scope_description: str = ""
    mode: str = "blocklist"
    blocked_topics: list[str] = []
    allowed_topics: list[str] = []
    refusal_message: str = ""
    semantic_enabled: bool = False
    semantic_threshold: float = 0.30


class GuardCheckRequest(BaseModel):
    question: str
    log: bool = False


class BadcaseImportRequest(BaseModel):
    content: str = ""
    dry_run: bool = False
    golden_file: str | None = None


def _guard_block_payload(guard: Any, refusal: str) -> dict[str, Any]:
    """守卫拦截时的统一响应体：不检索、不调用生成模型."""
    return {
        "blocked": True,
        "guard": guard.to_dict(),
        "answer": refusal,
        "provider": "domain-guard",
        "model": "guard",
        "latency_ms": 0,
        "mode": "guard（检索前拦截）",
        "hits": [],
    }


# ---------- 供应商与检索 ----------


def _providers() -> list[ProviderConfig]:
    settings = get_settings()
    toml_providers = load_providers(settings.providers_file or Path("llm_providers.toml"), settings)
    return toml_providers + RuntimeProviders.list()


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


def _embedder() -> EmbeddingBackend | None:
    """守卫语义兜底用的嵌入后端；不可用时返回 None（守卫自动跳过语义层）."""
    return resolve_embedding_backend(_providers())


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
    guard_cfg = load_guard_config()
    guard = check_question(req.question, guard_cfg, _embedder())
    append_guard_log(req.question, guard, source="ask")
    if guard.blocked:
        return _guard_block_payload(guard, guard_cfg.refusal_text())
    searcher, _ = _searcher()
    hits = searcher.search(req.question, k=req.k)
    if not hits:
        raise HTTPException(status_code=404, detail="检索无命中，换个说法试试")
    provider = _chat_provider(req.provider)
    result = generate_answer(req.question, hits, provider, max_tokens=req.max_tokens)
    return {
        "blocked": False,
        "guard": guard.to_dict(),
        "answer": result.answer,
        "provider": result.provider,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "mode": searcher.mode,
        "hits": [_hit_payload(i, hit) for i, hit in enumerate(hits, start=1)],
    }


@app.get("/api/providers")
def providers() -> dict[str, Any]:
    all_providers = _providers()
    runtime_names = {p.name for p in RuntimeProviders.list()}
    return {
        "chat": [
            {"name": p.name, "model": p.model, "local": p.is_local, "runtime": p.name in runtime_names}
            for p in all_providers
            if p.supports_chat and p.is_ready
        ],
        "all": [
            {
                "name": p.name,
                "model": p.model,
                "kind": p.kind,
                "local": p.is_local,
                "ready": p.is_ready,
                "runtime": p.name in runtime_names,
            }
            for p in all_providers
        ],
    }


@app.post("/api/providers/add")
def add_provider(req: AddProviderRequest) -> dict[str, Any]:
    if req.adapter not in {ADAPTER_OPENAI, ADAPTER_OLLAMA, ADAPTER_ANTHROPIC}:
        raise HTTPException(status_code=400, detail="adapter 必须是 openai/ollama/anthropic")
    if req.kind not in {"chat", "embedding", "both"}:
        raise HTTPException(status_code=400, detail="kind 必须是 chat/embedding/both")
    existing = _providers()
    if any(p.name == req.name for p in existing):
        raise HTTPException(status_code=409, detail=f"供应商名称已存在：{req.name}")
    config = ProviderConfig(
        name=req.name,
        kind=req.kind,
        base_url=req.base_url,
        model=req.model,
        api_key=req.api_key or None,
        adapter=req.adapter,
    )
    RuntimeProviders.add(config)
    return {
        "ok": True,
        "provider": {
            "name": config.name,
            "model": config.model,
            "local": config.is_local,
            "ready": config.is_ready,
        },
    }


@app.delete("/api/providers/{name}")
def remove_provider(name: str) -> dict[str, Any]:
    if not RuntimeProviders.remove(name):
        raise HTTPException(status_code=404, detail=f"运行时供应商不存在（toml 配置的无法删除）：{name}")
    return {"ok": True}


@app.post("/api/ask/compare")
def ask_compare(req: CompareRequest) -> dict[str, Any]:
    """同一检索结果，多模型并发生成，用于左右对比不同模型回答差异。"""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    guard_cfg = load_guard_config()
    guard = check_question(req.question, guard_cfg, _embedder())
    append_guard_log(req.question, guard, source="compare")
    if guard.blocked:
        return {
            "blocked": True,
            "guard": guard.to_dict(),
            "refusal": guard_cfg.refusal_text(),
            "results": [],
            "hits": [],
            "mode": "guard（检索前拦截）",
        }
    searcher, _ = _searcher()
    hits = searcher.search(req.question, k=req.k)
    if not hits:
        raise HTTPException(status_code=404, detail="检索无命中，换个说法试试")
    ready = [p for p in _providers() if p.supports_chat and p.is_ready]
    if not ready:
        raise HTTPException(status_code=503, detail="没有就绪的对话模型")
    if req.providers:
        name_set = set(req.providers)
        selected = [p for p in ready if p.name in name_set]
        if not selected:
            raise HTTPException(status_code=400, detail="指定的模型均不可用")
    else:
        selected = ready

    def _gen(provider: ProviderConfig) -> dict[str, Any]:
        try:
            result = generate_answer(req.question, hits, provider, max_tokens=req.max_tokens)
            return {
                "answer": result.answer,
                "provider": result.provider,
                "model": result.model,
                "latency_ms": result.latency_ms,
                "error": None,
            }
        except Exception as exc:
            return {
                "answer": "",
                "provider": provider.name,
                "model": provider.model,
                "latency_ms": 0,
                "error": str(exc),
            }

    if len(selected) == 1:
        results = [_gen(selected[0])]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected), 4)) as pool:
            results = list(pool.map(_gen, selected))

    return {
        "mode": searcher.mode,
        "hits": [_hit_payload(i, hit) for i, hit in enumerate(hits, start=1)],
        "results": results,
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
            "context_evidence": e.context_evidence,
            "judge_score": e.judge_score,
            "faithfulness": e.faithfulness,
            "judge_comment": e.judge_comment,
            "citations": e.citations,
            "is_negative": e.is_negative,
            "guard_blocked": e.guard_blocked,
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
            "context_evidence_avg": report.context_evidence_avg,
            "judge_score_avg": report.judge_score_avg,
            "faithfulness_avg": report.faithfulness_avg,
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
                "context_evidence_avg": report.context_evidence_avg,
                "judge_score_avg": report.judge_score_avg,
                "faithfulness_avg": report.faithfulness_avg,
                "entry_count": len(report.entries),
            }
        )
    return {"runs": runs}


@app.post("/api/index/reload")
def index_reload() -> dict[str, Any]:
    """索引文件被 CLI 重建后，让 Web 进程重新加载（无需重启服务）."""
    WebContext.reload()
    return {"ok": True, "index": WebContext.index_state().meta.snapshot_id}


# ---------- 领域守卫（无关项排除区） ----------


@app.get("/api/guard/config")
def guard_config() -> dict[str, Any]:
    return {"config": asdict(load_guard_config())}


@app.put("/api/guard/config")
def guard_config_update(req: GuardConfigRequest) -> dict[str, Any]:
    if req.mode not in VALID_GUARD_MODES:
        raise HTTPException(status_code=400, detail="mode 必须是 blocklist/allowlist/hybrid")
    config = GuardConfig(
        enabled=req.enabled,
        scope_name=req.scope_name,
        scope_description=req.scope_description,
        mode=req.mode,
        blocked_topics=req.blocked_topics,
        allowed_topics=req.allowed_topics,
        refusal_message=req.refusal_message,
        semantic_enabled=req.semantic_enabled,
        semantic_threshold=req.semantic_threshold,
    )
    saved = save_guard_config(config)
    return {"ok": True, "config": asdict(saved)}


@app.post("/api/guard/check")
def guard_check(req: GuardCheckRequest) -> dict[str, Any]:
    """试跑守卫：判断某个问题是否会被拦截（不改动检索/生成）."""
    config = load_guard_config()
    decision = check_question(req.question, config, _embedder())
    if req.log and req.question.strip():
        append_guard_log(req.question, decision, source="check")
    return {"decision": decision.to_dict(), "refusal": config.refusal_text()}


@app.get("/api/guard/log")
def guard_log(limit: int = 100, only_blocked: bool = False) -> dict[str, Any]:
    return {"records": read_guard_log(limit=limit, only_blocked=only_blocked)}


@app.delete("/api/guard/log")
def guard_log_clear() -> dict[str, Any]:
    return {"ok": True, "cleared": clear_guard_log()}


@app.get("/api/guard/cases")
def guard_cases() -> dict[str, Any]:
    """跑守卫回归：逐条比对期望与实际结论，返回误拦 / 漏拦明细."""
    loaded = load_cases()
    config = load_guard_config()
    report = run_cases(loaded, config, _embedder())
    return {"report": report.to_dict(), "file": str(DEFAULT_CASES_FILE)}


@app.post("/api/guard/cases/sync")
def guard_cases_sync(limit: int = 200, dry_run: bool = False) -> dict[str, Any]:
    """从守卫日志生成用例草稿并合并进用例文件（期望取自日志当时的实际决策，待人工复核）."""
    records = read_guard_log(limit=limit)
    existing = load_cases()
    drafts = cases_from_log(records)
    merged = merge_cases(existing, drafts)
    added = len(merged) - len(existing)
    if added and not dry_run:
        save_cases(merged)
    return {
        "added": added,
        "total": len(merged),
        "dry_run": dry_run,
        "file": str(DEFAULT_CASES_FILE),
    }


# ---------- 金标集：badcase 导入与体检 ----------


def _resolve_golden_path(override: str | None) -> Path:
    """解析金标文件路径，限制在项目目录内（防路径穿越）."""
    base = Path.cwd().resolve()
    target = (base / (override or "golden/qa.toml")).resolve()
    if target != base and base not in target.parents:
        raise HTTPException(status_code=400, detail="金标文件路径必须位于项目目录内")
    return target


@app.get("/api/golden")
def golden_overview() -> dict[str, Any]:
    """金标集条目 + 体检报告（维度/时效分布与风险标记）."""
    golden = load_golden()
    report = audit_golden(golden, known_sources=enabled_source_ids())
    entries = [
        {
            "id": qa.id,
            "question": qa.question,
            "category": qa.category,
            "timeliness": qa.timeliness,
            "expected_sources": list(qa.expected_sources),
            "expected_keywords": list(qa.expected_keywords),
            "notes": qa.notes,
        }
        for qa in golden.entries
    ]
    return {"entries": entries, "audit": report.to_dict()}


@app.get("/api/badcases")
def badcases_list() -> dict[str, Any]:
    """列出 data/badcases/ 下的 badcase 文件及其条目数."""
    badcases_dir = get_settings().data_dir / "badcases"
    items = []
    if badcases_dir.exists():
        for path in sorted(badcases_dir.glob("*.jsonl")):
            try:
                count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            except OSError:
                count = 0
            items.append(
                {
                    "name": path.name,
                    "count": count,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds"),
                }
            )
    return {"badcases": items}


@app.get("/api/badcases/{name}")
def badcase_detail(name: str) -> dict[str, Any]:
    """读取单个 badcase 文件内容（解析为条目列表）."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="非法的文件名")
    path = get_settings().data_dir / "badcases" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在：{name}")
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"name": name, "entries": entries}


@app.post("/api/badcases/import")
def badcases_import(req: BadcaseImportRequest) -> dict[str, Any]:
    """把 badcase（或任意 JSONL/JSON 数组）导入金标集.

    请求体 content 为 JSONL 文本或 JSON 数组；dry_run=True 时只预览不落盘。
    """
    items = parse_import_text(req.content)
    if not items:
        raise HTTPException(status_code=400, detail="未解析到任何条目（需为 JSONL 或 JSON 数组）")
    target = _resolve_golden_path(req.golden_file)
    result = import_golden_entries(items, target, dry_run=req.dry_run, known_sources=enabled_source_ids())
    return {
        "ok": True,
        "dry_run": req.dry_run,
        "target": result.target,
        "id_range": result.id_range,
        "imported": [asdict(entry) for entry in result.imported],
        "skipped": result.skipped,
        "warnings": result.warnings,
        "preview_toml": result.preview_toml if req.dry_run else "",
    }
