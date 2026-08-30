"""服务面 CLI：index（建索引）/ search（纯检索）/ ask（检索 + 引用生成）."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_intel_kb.cli.shared import load_provider_list_or_exit
from llm_intel_kb.config import get_settings
from llm_intel_kb.index.chunking import chunk_documents
from llm_intel_kb.index.embeddings import EmbeddingBackend, resolve_embedding_backend
from llm_intel_kb.index.loader import load_index_state, write_pg_meta_marker
from llm_intel_kb.index.search import HybridSearcher
from llm_intel_kb.index.store_local import LocalIndexStore, new_index_meta
from llm_intel_kb.index.store_pg import pg_store_from_settings, probe_pg_ready
from llm_intel_kb.llm.providers import ProviderConfig
from llm_intel_kb.pipeline.store import CorpusStore
from llm_intel_kb.rag.answer import generate_answer

console = Console()
index_app = typer.Typer(help="服务面：索引与检索", no_args_is_help=True)


def _index_store() -> LocalIndexStore:
    return LocalIndexStore(get_settings().data_dir / "index")


def _load_searcher() -> tuple[HybridSearcher, EmbeddingBackend | None]:
    try:
        state = load_index_state()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    providers = load_provider_list_or_exit()
    backend = resolve_embedding_backend(providers)
    return HybridSearcher(state, backend), backend


@index_app.command()
def build(
    backend_choice: str = typer.Option(
        "local",
        "--backend",
        help="索引后端：local（本地文件，默认）/ pg（pgvector）/ auto（pg 可用则 pg，否则 local）",
    ),
) -> None:
    """把当前快照切块并建立检索索引（自动检测 embedding 供应商，没有则纯词法索引）."""
    if backend_choice not in ("local", "pg", "auto"):
        console.print(f"[red]非法 backend：{backend_choice}（local / pg / auto）[/red]")
        raise typer.Exit(code=1)
    use_pg = backend_choice == "pg"
    if backend_choice == "auto":
        ready, detail = probe_pg_ready()
        use_pg = ready
        state_word = "可用 · " + detail if ready else "不可用（" + detail + "），回退本地"
        console.print(f"[dim]pg 后端探测：{state_word}[/dim]")

    settings = get_settings()
    corpus = CorpusStore(settings.data_dir)
    current = corpus.current_snapshot()
    if current is None:
        console.print("[yellow]还没有任何快照：先 data fetch + data commit[/yellow]")
        raise typer.Exit(code=1)
    docs = corpus.documents_from_snapshot(current)
    chunks = chunk_documents(docs)

    providers = load_provider_list_or_exit()
    backend = resolve_embedding_backend(providers)
    vectors: dict[str, list[float]] = {}
    dim: int | None = None
    if backend is not None:
        console.print(f"向量模型：[bold]{backend.name}[/bold] · {backend.model}，共 {len(chunks)} 块待向量化…")
        vectors, dim = _index_store().build_vectors(chunks, backend)
    else:
        console.print("[yellow]未配置就绪的 embedding 供应商：建立纯词法（BM25）索引[/yellow]")
        console.print("[dim]升级混合检索：ollama pull nomic-embed-text 后在 llm_providers.toml 启用，"
                      "或配置 SiliconFlow 等远程向量 API[/dim]")

    meta = new_index_meta(current.id, chunks, backend, dim)
    if use_pg:
        console.print("[dim]写入 pgvector 后端（表 index_meta / index_chunks / index_vectors）…[/dim]")
        pg_store_from_settings().save(chunks, vectors, meta)
        write_pg_meta_marker(_index_store().meta_file, meta)
    else:
        _index_store().save(chunks, vectors, meta)
    console.print(
        Panel(
            f"快照 {meta.snapshot_id} → [bold]{meta.chunk_count}[/bold] 块（{meta.doc_count} 篇文档）\n"
            f"存储后端：{'pgvector' if use_pg else '本地文件'}\n"
            f"检索模式：{'混合（向量 ' + str(dim) + ' 维 + 词法）' if backend else '纯词法 BM25'}",
            title="[green]索引构建完成[/green]",
        )
    )
    console.print("试试：[bold]llm-intel-kb ask \"最近有哪些值得关注的 LLM 推理优化工作？\"[/bold]")


@index_app.command()
def status() -> None:
    """索引概况（对应快照、块数、检索模式）."""
    try:
        state = load_index_state()
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=1) from exc
    meta = state.meta
    vectors_desc = (
        f"{meta.embedding_model or '-'} · {meta.embedding_dim or '-'} 维"
        if meta.embedding_model
        else "无（纯词法）"
    )
    table = Table(title="索引概况")
    table.add_column("项", style="bold cyan")
    table.add_column("值")
    table.add_row("后端", "pgvector" if meta.backend == "pg" else "本地文件")
    table.add_row("对应快照", meta.snapshot_id)
    table.add_row("文档 / 块", f"{meta.doc_count} / {meta.chunk_count}")
    table.add_row("向量", vectors_desc)
    table.add_row("构建时间", meta.created_at)
    console.print(table)


@index_app.command("pg-ann")
def pg_ann(
    query: str = typer.Argument(..., help="检索词（向量化后在 pgvector 里做 ANN 查询）"),
    k: int = typer.Option(5, "--top", "-k", help="返回前 k 个块"),
) -> None:
    """pgvector 原生 ANN 检索演示（需要 --backend pg 构建的索引 + embedding 供应商）."""
    ready, detail = probe_pg_ready()
    if not ready:
        console.print(f"[red]pgvector 后端不可用：{detail}[/red]")
        console.print("[dim]启用方法：docker compose up -d 后 index build --backend pg[/dim]")
        raise typer.Exit(code=1)
    providers = load_provider_list_or_exit()
    backend = resolve_embedding_backend(providers)
    if backend is None:
        console.print("[red]未配置 embedding 供应商：pg-ann 需要向量化查询词[/red]")
        raise typer.Exit(code=1)
    query_vec = backend.embed_texts([query])[0]
    results = pg_store_from_settings().ann_search(query_vec, k=k)
    if not results:
        console.print("[yellow]向量表为空：先 index build --backend pg[/yellow]")
        raise typer.Exit(code=1)
    state = load_index_state()
    for rank, (chunk_id, similarity) in enumerate(results, start=1):
        chunk = state.chunk_by_id.get(chunk_id)
        title = chunk.title if chunk else chunk_id[:16]
        excerpt = chunk.text[:200] + ("…" if len(chunk.text) > 200 else "") if chunk else ""
        console.print(
            Panel(
                excerpt,
                title=f"[{rank}] {title[:70]}",
                subtitle=f"余弦相似度 {similarity:.4f}",
            )
        )


@index_app.command()
def search(
    query: str = typer.Argument(..., help="检索词"),
    k: int = typer.Option(5, "--top", "-k", help="返回前 k 个块"),
) -> None:
    """纯检索：显示命中的块（不调用生成模型，免费）."""
    searcher, _ = _load_searcher()
    console.print(f"[dim]检索模式：{searcher.mode}[/dim]\n")
    hits = searcher.search(query, k=k)
    if not hits:
        console.print("[yellow]没有命中：换个说法或先 index build 更新索引[/yellow]")
        raise typer.Exit(code=1)
    for index, hit in enumerate(hits, start=1):
        sources = []
        if hit.lexical_rank:
            sources.append(f"词法#{hit.lexical_rank}")
        if hit.vector_rank:
            sources.append(f"向量#{hit.vector_rank}")
        console.print(
            Panel(
                hit.chunk.text[:400] + ("…" if len(hit.chunk.text) > 400 else ""),
                title=f"[{index}] {hit.chunk.title[:70]}",
                subtitle=f"{hit.chunk.source_id} · {'/'.join(sources)} · RRF {hit.rrf_score:.4f}",
            )
        )


@index_app.command()
def ask(
    question: str = typer.Argument(..., help="问题"),
    k: int = typer.Option(5, "--top", "-k", help="检索并送入上下文的块数"),
    provider_name: str | None = typer.Option(None, "--provider", "-p", help="指定生成模型供应商"),
    max_tokens: int = typer.Option(2048, "--max-tokens", help="回答的最大 token 数"),
) -> None:
    """检索 + 引用生成：基于当前索引回答问题，回答带 [编号] 来源标注."""
    searcher, _ = _load_searcher()
    hits = searcher.search(question, k=k)
    if not hits:
        console.print("[yellow]检索无命中，无法作答：换个说法或先 index build[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"[dim]检索模式：{searcher.mode} · 命中 {len(hits)} 块[/dim]\n")
    provider = _pick_chat_provider(provider_name)
    console.print(f"[dim]生成模型：{provider.name} · {provider.model}[/dim]\n")
    result = generate_answer(question, hits, provider, max_tokens=max_tokens)
    console.print(Panel(result.answer, title="回答", subtitle=f"{result.provider} · {result.latency_ms}ms"))

    table = Table(title="引用来源")
    table.add_column("编号", justify="center")
    table.add_column("标题", overflow="fold", max_width=56)
    table.add_column("源", style="cyan")
    table.add_column("链接", overflow="fold")
    for index, hit in enumerate(hits, start=1):
        table.add_row(f"[{index}]", hit.chunk.title[:60], hit.chunk.source_id, hit.chunk.url)
    console.print(table)
    console.print("[dim]提示：search 命令可只看检索命中块（不消耗生成 token）[/dim]")


def _pick_chat_provider(name: str | None) -> ProviderConfig:
    providers = load_provider_list_or_exit()
    ready = [p for p in providers if p.supports_chat and p.is_ready]
    if not ready:
        console.print("[red]没有就绪的对话模型：请在 llm_providers.toml 配置[/red]")
        raise typer.Exit(code=1)
    if name:
        for provider in ready:
            if provider.name == name:
                return provider
        console.print(f"[red]供应商不存在或未就绪：{name}[/red]（可选：{'、'.join(p.name for p in ready)}）")
        raise typer.Exit(code=1)
    return ready[0]
