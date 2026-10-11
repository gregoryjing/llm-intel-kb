"""守卫面 CLI：guard cases（回归）/ sync（从日志生成用例）/ show（查看配置）.

误拒 / 漏拦闭环的运维入口：
- `guard sync`  把 guard_log 里的真实提问快速转成用例草稿（source=log，待人工确认）；
- `guard cases` 逐条回归，实际结论与期望不一致（误拦 / 漏拦）即退出码 1，供 CI 门禁。
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_intel_kb.cli.shared import load_provider_list_or_exit
from llm_intel_kb.guard.domain import Embedder, load_guard_config, read_guard_log
from llm_intel_kb.guard.regression import (
    DEFAULT_CASES_FILE,
    KIND_FALSE_ACCEPT,
    KIND_FALSE_REJECT,
    RegressionReport,
    cases_from_log,
    load_cases,
    merge_cases,
    run_cases,
    save_cases,
)
from llm_intel_kb.index.embeddings import resolve_embedding_backend

console = Console()
guard_app = typer.Typer(help="领域守卫：回归用例 · 误拒/漏拦闭环", no_args_is_help=True)

_KIND_STYLE = {
    KIND_FALSE_REJECT: "[red]误拦[/red]",
    KIND_FALSE_ACCEPT: "[red]漏拦[/red]",
}


def _embedder() -> Embedder | None:
    """语义兜底用的嵌入后端；无可用供应商时返回 None（回归自动跳过语义层）."""
    try:
        providers = load_provider_list_or_exit()
    except typer.BadParameter:
        return None
    return resolve_embedding_backend(providers)


@guard_app.command()
def cases(
    cases_file: Path = typer.Option(DEFAULT_CASES_FILE, "--file", "-f", help="回归用例文件"),
    no_semantic: bool = typer.Option(False, "--no-semantic", help="跳过语义兜底（纯规则回归）"),
) -> None:
    """跑守卫回归：实际结论与期望不一致（误拦 / 漏拦）则以退出码 1 失败."""
    loaded = load_cases(cases_file)
    if not loaded:
        console.print(
            f"[yellow]没有回归用例：{cases_file}[/yellow]（先用 guard sync 从日志生成草稿）"
        )
        raise typer.Exit(code=0)

    config = load_guard_config()
    embedder = None if no_semantic else _embedder()
    if not no_semantic and embedder is None and config.semantic_enabled:
        console.print("[yellow]语义兜底已启用但无可用 embedding 后端，本次按纯规则回归[/yellow]")

    report = run_cases(loaded, config, embedder)
    _print_report(report, cases_file)
    if not report.ok:
        raise typer.Exit(code=1)


@guard_app.command()
def sync(
    cases_file: Path = typer.Option(DEFAULT_CASES_FILE, "--file", "-f", help="回归用例文件"),
    limit: int = typer.Option(200, "--limit", help="从最近多少条日志生成"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只预览不写入"),
) -> None:
    """从守卫日志生成回归用例草稿（以当时实际决策为期望，人工复核后即为回归资产）."""
    records = read_guard_log(limit=limit)
    if not records:
        console.print("[yellow]守卫日志为空：先提问 / 试跑产生决策记录[/yellow]")
        raise typer.Exit(code=0)

    existing = load_cases(cases_file)
    drafts = cases_from_log(records)
    merged = merge_cases(existing, drafts)
    added = len(merged) - len(existing)

    table = Table(title=f"守卫用例同步 · 日志 {len(records)} 条")
    table.add_column("项", style="bold cyan")
    table.add_column("值", justify="right")
    table.add_row("候选草稿", str(len(drafts)))
    table.add_row("已收录", str(len(existing)))
    table.add_row("新增", str(added))
    table.add_row("合计", str(len(merged)))
    console.print(table)

    if added == 0:
        console.print("[dim]没有新增：日志中的问题都已收录[/dim]")
        return

    if dry_run:
        console.print("[dim]dry-run：以下为新增草稿（expected_blocked 取自日志当时的实际决策）[/dim]")
        for case in merged[len(existing) :]:
            mark = "拦截" if case.expected_blocked else "放行"
            console.print(f"  [dim]- [{mark}] {case.question}[/dim]")
        return

    path = save_cases(merged, cases_file)
    console.print(f"[green]已写入 {added} 条到 {path}[/green]")
    console.print("[yellow]提示：日志决策未必正确，请人工复核 expected_blocked 后再作为门禁基线[/yellow]")


@guard_app.command()
def show() -> None:
    """查看当前守卫配置与用例规模（审计用）."""
    config = load_guard_config()
    cases_count = len(load_cases())
    table = Table(title="领域守卫配置")
    table.add_column("项", style="bold cyan")
    table.add_column("值", overflow="fold")
    table.add_row("启用", "是" if config.enabled else "否")
    table.add_row("范围", config.scope_name)
    table.add_row("模式", config.mode)
    table.add_row("禁止清单", f"{len(config.blocked_topics)} 个")
    table.add_row("允许清单", f"{len(config.allowed_topics)} 个")
    table.add_row(
        "语义兜底",
        f"已启用（阈值 {config.semantic_threshold:.2f}）" if config.semantic_enabled else "已停用",
    )
    table.add_row("回归用例", f"{cases_count} 条（{DEFAULT_CASES_FILE}）")
    console.print(table)


def _print_report(report: RegressionReport, cases_file: Path) -> None:
    table = Table(title=f"守卫回归 · {cases_file}")
    table.add_column("问题", overflow="fold", max_width=40)
    table.add_column("期望", justify="center")
    table.add_column("实际", justify="center")
    table.add_column("结论", justify="center")
    table.add_column("原因", overflow="fold", max_width=48)
    for outcome in report.outcomes:
        style = _KIND_STYLE.get(outcome.kind, "[green]通过[/green]")
        table.add_row(
            outcome.case.question[:38],
            "拦截" if outcome.case.expected_blocked else "放行",
            "拦截" if outcome.decision.blocked else "放行",
            style,
            outcome.decision.reason or outcome.case.note or "—",
        )
    console.print(table)

    summary = (
        f"通过 {report.passed}/{report.total} · "
        f"误拦 {len(report.false_rejects)} · 漏拦 {len(report.false_accepts)}"
    )
    head = "[green]回归通过[/green]" if report.ok else "[red]回归失败[/red]"
    console.print(Panel(f"{head}\n{summary}", title="结论"))