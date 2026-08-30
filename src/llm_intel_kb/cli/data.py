from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_intel_kb.cli.shared import load_provider_list_or_exit, resolve_sources_path
from llm_intel_kb.config import get_settings
from llm_intel_kb.llm.providers import ProviderConfig
from llm_intel_kb.pipeline import sources as sources_mod
from llm_intel_kb.pipeline.commit import commit_staged
from llm_intel_kb.pipeline.enrich import enrich_corpus
from llm_intel_kb.pipeline.models import CorpusDocument, Snapshot, SourceSpec
from llm_intel_kb.pipeline.store import CorpusStore, EnrichmentCache

console = Console()
data_app = typer.Typer(help="数据面：源发现 → 抓取 → 清洗入库 → 版本化 → 富化", no_args_is_help=True)


def _store() -> CorpusStore:
    return CorpusStore(get_settings().data_dir)


def _specs(source_filter: str | None) -> tuple[list[SourceSpec], list[SourceSpec]]:
    """加载启用的源；返回 (待抓取, 全部)."""
    specs = sources_mod.load_sources(resolve_sources_path())
    enabled = [s for s in specs if s.enabled]
    if source_filter:
        picked = [s for s in enabled if s.id == source_filter]
        if not picked:
            console.print(f"[red]未找到启用的数据源[/red]：{source_filter}")
            raise typer.Exit(code=1)
        return picked, specs
    return enabled, specs


@data_app.command("sources")
def list_sources() -> None:
    """列出数据源注册表."""
    specs = sources_mod.load_sources(resolve_sources_path())
    if not specs:
        console.print("[yellow]data_sources.toml 不存在或为空[/yellow]")
        raise typer.Exit(code=1)
    table = Table(title=f"数据源注册表（{len(specs)} 个，启用 {sum(s.enabled for s in specs)} 个）")
    table.add_column("ID", style="bold cyan")
    table.add_column("类型")
    table.add_column("名称")
    table.add_column("单次上限", justify="right")
    table.add_column("最短字符", justify="right")
    table.add_column("状态", justify="center")
    for spec in specs:
        status = "[green]启用[/green]" if spec.enabled else "[dim]禁用[/dim]"
        table.add_row(spec.id, spec.type, spec.title or "-", str(spec.max_items), str(spec.min_chars), status)
    console.print(table)


@data_app.command()
def fetch(
    source: str | None = typer.Option(None, "--source", "-s", help="只抓取指定 ID 的源"),
    timeout: float = typer.Option(40.0, "--timeout", help="单个源的超时秒数"),
) -> None:
    """抓取启用的数据源到暂存区（staging），不直接入库."""
    picked, _ = _specs(source)
    store = _store()

    def run(spec: SourceSpec) -> tuple[str, int, list[str]]:
        result = sources_mod.fetch_source(spec, timeout)
        store.staging_write(spec.id, result.docs)
        return spec.id, len(result.docs), result.errors

    with ThreadPoolExecutor(max_workers=min(8, len(picked))) as executor:
        outcomes = list(executor.map(run, picked))

    table = Table(title=f"抓取结果（{len(picked)} 个源）")
    table.add_column("源", style="bold cyan")
    table.add_column("文档数", justify="right")
    table.add_column("错误", overflow="fold")
    total = 0
    failed = 0
    for source_id, count, errors in outcomes:
        total += count
        failed += bool(errors)
        table.add_row(source_id, str(count), "；".join(errors) or "[green]-[/green]")
    console.print(table)
    console.print(f"暂存区共 [bold]{total}[/bold] 篇文档，下一步：[bold]data commit[/bold]")
    if failed:
        raise typer.Exit(code=1)


@data_app.command()
def status() -> None:
    """暂存区与当前数据集版本概况."""
    store = _store()
    staged, source_ids = store.staging_read_all()
    current = store.current_snapshot()

    if staged:
        table = Table(title=f"暂存区（{len(staged)} 篇，来自 {len(source_ids)} 个源，未提交）")
        table.add_column("源", style="cyan")
        table.add_column("篇数", justify="right")
        by_source: dict[str, int] = {}
        for doc in staged:
            by_source[doc.source_id] = by_source.get(doc.source_id, 0) + 1
        for source_id, count in sorted(by_source.items()):
            table.add_row(source_id, str(count))
        console.print(table)
    else:
        console.print("[dim]暂存区为空：先执行 data fetch[/dim]")

    _print_current(current)


def _print_current(current: Snapshot | None) -> None:
    if current is None:
        console.print("[dim]当前版本：无（暂无快照）[/dim]")
        return
    enriched = current.stats.get("total", len(current.doc_ids))
    console.print(
        f"当前版本：[bold]{current.id}[/bold] · {enriched} 篇 · 创建于 {current.created_at}"
        + (f" · 父版本 {current.parent}" if current.parent else "")
    )


@data_app.command()
def commit() -> None:
    """把暂存区清洗入库（内容哈希去重）并创建新快照."""
    store = _store()
    _, specs = _specs(None)
    staged, _ = store.staging_read_all()
    if not staged:
        console.print("[yellow]暂存区为空：先执行 data fetch[/yellow]")
        raise typer.Exit(code=1)
    result = commit_staged(store, specs)
    s = result.snapshot.stats
    console.print(
        Panel(
            f"新增 [green]{result.added}[/green] · 合并 {result.merged} · "
            f"过短丢弃 {result.skipped_short} · 注册表外 {result.skipped_no_source}",
            title=f"快照 [bold]{result.snapshot.id}[/bold] 创建成功",
        )
    )
    console.print(f"数据集总量：[bold]{s['total']}[/bold] 篇（上一版本：{result.snapshot.parent or '无'}）")
    console.print("下一步：[bold]data enrich[/bold] 生成摘要，或 [bold]index build[/bold] 建立检索索引")


@data_app.command()
def enrich(
    provider_name: str | None = typer.Option(None, "--provider", "-p", help="指定富化用的模型供应商"),
    limit: int | None = typer.Option(None, "--limit", "-l", help="本次最多富化 N 篇（控制花费）"),
) -> None:
    """LLM 富化当前快照中的文档：中文摘要 + 英文主题词（内容不变则命中缓存，不重复付费）."""
    store = _store()
    current = store.current_snapshot()
    if current is None:
        console.print("[yellow]还没有任何快照：先 data fetch + data commit[/yellow]")
        raise typer.Exit(code=1)
    provider = _pick_chat_provider(provider_name)
    locale = "本地" if provider.is_local else "远程"
    console.print(f"富化模型：[bold]{provider.name}[/bold] · {provider.model}（{locale}）")

    pending = []
    cache = EnrichmentCache(get_settings().data_dir)
    for doc_id in current.doc_ids:
        doc = store.get_document(doc_id)
        if doc is None:
            continue
        if doc.enrichment is None or cache.get(doc_id) is None:
            pending.append(doc_id)
    if not pending:
        console.print("[green]当前快照全部文档已富化（缓存命中）[/green]")
        return
    console.print(f"待富化 {len(pending)} 篇" + (f"，本次处理 {limit} 篇" if limit else "") + "\n")

    stats = enrich_corpus(store, cache, provider, pending, limit=limit)
    console.print(
        f"完成：[green]新富化 {stats.enriched}[/green] · 缓存命中 {stats.cached} · 失败 [red]{stats.failed}[/red]"
    )
    for error in stats.errors[:5]:
        console.print(f"  [red]{error}[/red]")
    if stats.failed and not stats.enriched:
        raise typer.Exit(code=1)


@data_app.command("list")
def list_docs(
    snapshot_id: str | None = typer.Option(None, "--snapshot", help="列出指定快照（默认当前）"),
    source_id: str | None = typer.Option(None, "--source", help="按源过滤"),
    limit: int = typer.Option(50, "--limit", help="最多显示 N 行"),
) -> None:
    """列出快照中的文档."""
    store = _store()
    snapshot = _get_snapshot(store, snapshot_id)
    docs = store.iter_documents(snapshot.doc_ids)
    if source_id:
        docs = [d for d in docs if d.source_id == source_id]
    docs.sort(key=lambda d: d.published_at or "", reverse=True)

    table = Table(title=f"快照 {snapshot.id} · {len(docs)} 篇")
    table.add_column("ID", style="dim")
    table.add_column("标题", overflow="fold", max_width=52)
    table.add_column("源", style="cyan")
    table.add_column("字符", justify="right")
    table.add_column("语")
    table.add_column("富化", justify="center")
    for doc in docs[:limit]:
        enriched = "[green]✓[/green]" if doc.enrichment else "[dim]·[/dim]"
        table.add_row(doc.id[:8], doc.title[:60], doc.source_id, str(doc.content_chars), doc.language, enriched)
    console.print(table)
    if len(docs) > limit:
        console.print(f"[dim]… 其余 {len(docs) - limit} 篇未显示（--limit 调整）[/dim]")


@data_app.command()
def inspect(
    doc_id: str = typer.Argument(..., help="文档 ID（data list 里显示的 8 位短 ID 即可）"),
) -> None:
    """查看一篇文档的完整信息."""
    store = _store()
    doc = _resolve_doc(store, doc_id)
    if doc is None:
        console.print(f"[red]文档不存在：{doc_id}[/red]")
        raise typer.Exit(code=1)
    console.print(Panel(doc.title, title=f"[bold]{doc.id}[/bold]", subtitle=f"{doc.source_id} · {doc.url}"))
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold cyan")
    meta.add_column()
    meta.add_row("发布时间", str(doc.published_at))
    meta.add_row("抓取时间", doc.fetched_at)
    meta.add_row("字符数", str(doc.content_chars))
    meta.add_row("语言", doc.language)
    meta.add_row("镜像地址", "\n".join(doc.alt_urls) or "-")
    if doc.enrichment:
        meta.add_row("摘要", doc.enrichment.summary)
        meta.add_row("主题词", ", ".join(doc.enrichment.topics) or "-")
        meta.add_row("富化模型", f"{doc.enrichment.model} @ {doc.enrichment.enriched_at}")
    else:
        meta.add_row("富化", "[dim]未富化（data enrich）[/dim]")
    console.print(meta)
    console.print(Panel(doc.content[:1500] + ("…" if len(doc.content) > 1500 else ""), title="正文预览"))


@data_app.command()
def log() -> None:
    """快照历史（数据集版本线）."""
    store = _store()
    snapshots = store.list_snapshots()
    if not snapshots:
        console.print("[yellow]还没有任何快照[/yellow]")
        raise typer.Exit(code=1)
    current = store.current_snapshot()
    table = Table(title="快照历史")
    table.add_column("快照 ID", style="bold cyan")
    table.add_column("创建时间")
    table.add_column("父版本")
    table.add_column("总量", justify="right")
    table.add_column("新增", justify="right")
    table.add_column("当前", justify="center")
    for snap in snapshots:
        is_current = "👉" if current and snap.id == current.id else ""
        table.add_row(
            snap.id,
            snap.created_at,
            snap.parent or "-",
            str(snap.stats.get("total", len(snap.doc_ids))),
            str(snap.stats.get("added", "-")),
            is_current,
        )
    console.print(table)


@data_app.command()
def rollback(
    snapshot_id: str = typer.Argument(..., help="回滚到的快照 ID（data log 查看）"),
) -> None:
    """把当前数据集版本切换到历史快照（不删除任何文档，随时可再切回）."""
    store = _store()
    snapshots = {s.id for s in store.list_snapshots()}
    if snapshot_id not in snapshots:
        console.print(f"[red]快照不存在：{snapshot_id}[/red]（data log 查看可用版本）")
        raise typer.Exit(code=1)
    current = store.current_snapshot()
    store.set_current(snapshot_id)
    console.print(f"[green]已回滚[/green] {current.id if current else '无'} → [bold]{snapshot_id}[/bold]")
    console.print("[yellow]注意：如已建立检索索引，需要 index build 重建以匹配当前版本[/yellow]")


# ---------- 内部工具 ----------


def _pick_chat_provider(name: str | None) -> ProviderConfig:
    providers = load_provider_list_or_exit(need="chat")
    ready = [p for p in providers if p.supports_chat and p.is_ready]
    if not ready:
        console.print("[red]没有就绪的对话模型：请在 llm_providers.toml 配置（本地 Ollama 或远程 API）[/red]")
        raise typer.Exit(code=1)
    if name:
        for provider in ready:
            if provider.name == name:
                return provider
        console.print(f"[red]供应商不存在或未就绪：{name}[/red]（可选：{'、'.join(p.name for p in ready)}）")
        raise typer.Exit(code=1)
    return ready[0]


def _get_snapshot(store: CorpusStore, snapshot_id: str | None) -> Snapshot:
    if snapshot_id:
        try:
            return store.get_snapshot(snapshot_id)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    current = store.current_snapshot()
    if current is None:
        console.print("[yellow]还没有任何快照：先 data fetch + data commit[/yellow]")
        raise typer.Exit(code=1)
    return current


def _resolve_doc(store: CorpusStore, short_id: str) -> CorpusDocument | None:
    """支持 8 位短 ID 前缀匹配."""
    if len(short_id) >= 64:
        return store.get_document(short_id)
    current = store.current_snapshot()
    candidates = current.doc_ids if current else []
    for doc_id in candidates:
        if doc_id.startswith(short_id):
            return store.get_document(doc_id)
    return store.get_document(short_id)
