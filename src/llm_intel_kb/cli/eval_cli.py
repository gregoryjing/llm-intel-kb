"""评估面 CLI：eval run（跑评估）/ gate（回归门禁）/ baseline（设基线）/ show（看报告）."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_intel_kb.cli.shared import load_provider_list_or_exit
from llm_intel_kb.config import get_settings
from llm_intel_kb.eval.gate import load_baseline, run_gate, set_baseline
from llm_intel_kb.eval.golden import load_golden
from llm_intel_kb.eval.importers import audit_golden, import_golden_entries, parse_import_text
from llm_intel_kb.eval.runner import EvalReport, export_badcases, run_eval
from llm_intel_kb.index.embeddings import resolve_embedding_backend
from llm_intel_kb.index.loader import load_index_state
from llm_intel_kb.index.search import HybridSearcher
from llm_intel_kb.pipeline.sources import enabled_source_ids
from llm_intel_kb.pipeline.store import CorpusStore

console = Console()
eval_app = typer.Typer(help="评估面：金标评估 · 回归门禁 · badcase 回流", no_args_is_help=True)


def _evals_dir() -> Path:
    return get_settings().data_dir / "evals"


def _baseline_file() -> Path:
    return get_settings().data_dir / "evals" / "baseline.json"


def _badcases_dir() -> Path:
    return get_settings().data_dir / "badcases"


def _load_context() -> tuple[HybridSearcher, str, str]:
    """加载索引与当前快照；返回 (searcher, 快照 id, 索引对应快照 id)."""
    settings = get_settings()
    try:
        state = load_index_state()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    current = CorpusStore(settings.data_dir).current_snapshot()
    snapshot_id = current.id if current else "-"
    if state.meta.snapshot_id != snapshot_id:
        console.print(
            f"[yellow]警告：索引对应快照 {state.meta.snapshot_id}，当前快照 {snapshot_id}"
            f"（数据已变更，建议 index build 后再评估）[/yellow]"
        )
    providers = load_provider_list_or_exit()
    backend = resolve_embedding_backend(providers)
    return HybridSearcher(state, backend), snapshot_id, state.meta.snapshot_id


@eval_app.command()
def run(
    k: int = typer.Option(5, "--top", "-k", help="每条问题的检索块数"),
    no_llm: bool = typer.Option(False, "--no-llm", help="只评估检索层，跳过生成与 judge"),
    provider_name: str | None = typer.Option(None, "--provider", "-p", help="生成与 judge 用的模型"),
) -> None:
    """对金标问答集跑评估：检索指标 + 生成指标，报告存 data/evals/."""
    golden = load_golden()
    if not golden:
        console.print("[red]金标集为空：先在 data/golden/qa.toml 添加条目[/red]")
        raise typer.Exit(code=1)

    searcher, snapshot_id, index_snapshot_id = _load_context()
    chat_provider = None
    if not no_llm:
        ready = [p for p in load_provider_list_or_exit() if p.supports_chat and p.is_ready]
        if provider_name:
            chat_provider = next((p for p in ready if p.name == provider_name), None)
            if chat_provider is None:
                console.print(f"[red]供应商不存在或未就绪：{provider_name}[/red]")
                raise typer.Exit(code=1)
        else:
            chat_provider = ready[0] if ready else None
        if chat_provider is None:
            console.print("[yellow]无就绪对话模型：本次只评估检索层[/yellow]")

    console.print(
        f"评估 [bold]{len(golden)}[/bold] 条 · 检索模式 {searcher.mode}"
        + (f" · 生成 {chat_provider.name}" if chat_provider else " · 仅检索层")
    )
    report = run_eval(golden, searcher, snapshot_id, index_snapshot_id, chat_provider, k=k)
    path = report.save(_evals_dir())
    _print_report(report)

    badcase_path = export_badcases(report, golden, _badcases_dir())
    if badcase_path:
        console.print(f"\nbadcase 已导出：{badcase_path}（人工复盘后沉淀进金标集）")
    console.print(f"\n报告已保存：{path}")


@eval_app.command()
def gate(
    report_path: Path | None = typer.Option(None, "--report", help="指定报告文件（默认最新）"),
    min_hit_rate: float = typer.Option(0.7, help="检索命中率阈值"),
    min_judge: float = typer.Option(3.0, help="LLM 评分均值阈值"),
    max_regressions: int = typer.Option(0, help="允许的条目级退化数量"),
    min_context_evidence: float = typer.Option(
        0.0, help="块级证据率阈值（0 = 不启用）"
    ),
    min_faithfulness: float = typer.Option(0.0, help="忠实度均值阈值（0 = 不启用）"),
) -> None:
    """回归门禁：与基线对比，退化则退出码 1（CI 中拦截不达标的数据变更）."""
    path = report_path or _latest_report()
    if path is None:
        console.print("[red]没有评估报告：先 eval run[/red]")
        raise typer.Exit(code=1)
    report = EvalReport.load(path)
    baseline = load_baseline(_baseline_file())
    decision = run_gate(
        report,
        baseline,
        min_hit_rate=min_hit_rate,
        max_regressions=max_regressions,
        min_judge_score=min_judge,
        min_context_evidence=min_context_evidence,
        min_faithfulness=min_faithfulness,
    )
    console.print(Panel(decision.summary(), title=f"回归门禁 · 快照 {report.snapshot_id}"))
    if baseline is not None:
        console.print(
            f"[dim]基线：hit {baseline.retrieval_hit_rate:.0%} · MRR {baseline.mrr:.3f}"
            f" · judge {baseline.judge_score_avg or '-'}[/dim]"
        )
    if not decision.passed:
        raise typer.Exit(code=1)


@eval_app.command()
def baseline(
    report_path: Path | None = typer.Option(None, "--report", help="指定报告文件（默认最新）"),
) -> None:
    """把某份评估报告设为基线（后续 eval gate 与之对比）."""
    path = report_path or _latest_report()
    if path is None:
        console.print("[red]没有评估报告：先 eval run[/red]")
        raise typer.Exit(code=1)
    report = EvalReport.load(path)
    set_baseline(report, _baseline_file())
    console.print(
        f"[green]基线已设置[/green]：快照 {report.snapshot_id} · "
        f"hit {report.retrieval_hit_rate:.0%} · MRR {report.mrr:.3f} · "
        f"通过率 {report.pass_rate:.0%}"
    )


@eval_app.command()
def show(
    report_path: Path | None = typer.Option(None, "--report", help="指定报告文件（默认最新）"),
) -> None:
    """查看评估报告详情（逐条结果）."""
    path = report_path or _latest_report()
    if path is None:
        console.print("[red]没有评估报告：先 eval run[/red]")
        raise typer.Exit(code=1)
    _print_report(EvalReport.load(path), detail=True)


@eval_app.command("import-badcase")
def import_badcase(
    input_file: Path = typer.Argument(..., help="输入文件路径（JSONL 或 JSON 数组）"),
    golden_file: Path | None = typer.Option(None, "--golden", "-g", help="目标金标文件"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只预览不写入"),
) -> None:
    """从 badcase 文件或外部 JSON 批量导入新条目到金标集.

    支持两种格式：
    1. JSONL（badcase 导出格式）：每行一个 JSON
    2. JSON 数组：[{"question": "...", "expected_sources": [...], ...}, ...]
    """
    target = golden_file or Path("golden/qa.toml")
    if not input_file.exists():
        console.print(f"[red]文件不存在：{input_file}[/red]")
        raise typer.Exit(code=1)

    items = parse_import_text(input_file.read_text(encoding="utf-8"))
    if not items:
        console.print("[yellow]没有可导入的条目[/yellow]")
        raise typer.Exit(code=1)

    result = import_golden_entries(items, target, dry_run=dry_run, known_sources=enabled_source_ids())
    if not result.imported:
        console.print(f"[yellow]全部跳过（{len(result.skipped)} 条重复/无效）[/yellow]")
        for item in result.skipped[:10]:
            console.print(f"  [dim]- {item['reason']}：{item['question'][:50]}[/dim]")
        return

    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

    summary = "\n".join(
        [
            f"导入 [bold green]{len(result.imported)}[/bold green] 条（跳过 {len(result.skipped)} 条重复/无效）",
            f"目标文件：{result.target}",
            f"ID 范围：{result.id_range}",
        ]
    )
    console.print(Panel(summary, title="badcase 导入预览"))

    if dry_run:
        console.print("[dim]dry-run 模式，未写入文件[/dim]")
        console.print(result.preview_toml[:2000])
        return

    console.print(f"[green]已导入 {len(result.imported)} 条到 {result.target}[/green]")
    console.print("[dim]现在可以运行 eval run 验证新条目是否通过[/dim]")


@eval_app.command()
def audit(
    strict: bool = typer.Option(False, "--strict", help="存在硬错误（未注册源/空问题/重复 id）时以非零退出"),
) -> None:
    """金标集体检：维度 / 时效分布 + 风险标记（时效敏感、宽泛、宽源、近重复）."""
    golden = load_golden()
    if not golden:
        console.print("[red]金标集为空[/red]")
        raise typer.Exit(code=1)
    report = audit_golden(golden, known_sources=enabled_source_ids())

    table = Table(title=f"金标集体检 · 共 {report.total} 条")
    table.add_column("维度", style="bold cyan")
    table.add_column("数量", justify="right")
    table.add_column("占比", justify="right")
    for category, count in report.by_category.items():
        table.add_row(category, str(count), f"{count / report.total:.0%}")
    console.print(table)

    console.print(
        f"\n负例 [bold]{report.negative}[/bold] 条 · "
        f"时效敏感 [bold yellow]{report.time_sensitive}[/bold yellow] 条"
    )
    for key, label in (
        ("time_sensitive", "时效敏感"),
        ("broad", "宽泛问题"),
        ("wide_source", "宽源条目"),
        ("duplicate_groups", "近重复组"),
    ):
        ids = report.flags.get(key, [])
        if ids:
            console.print(f"[yellow]{label}[/yellow]（{len(ids)}）：{', '.join(ids[:12])}")

    if report.errors:
        console.print("\n[bold red]硬错误（数据集损坏，需优先修复）[/bold red]")
        for err in report.errors:
            console.print(f"  [red]✗[/red] {err}")

    console.print("\n[bold]治理建议[/bold]")
    for tip in report.suggestions:
        console.print(f"  - {tip}")

    if strict and not report.ok:
        console.print("\n[red]strict 模式：存在硬错误，门禁失败[/red]")
        raise typer.Exit(code=1)


def _latest_report() -> Path | None:
    evals_dir = _evals_dir()
    if not evals_dir.exists():
        return None
    reports = sorted(evals_dir.glob("eval_*.json"))
    return reports[-1] if reports else None


def _print_report(report: EvalReport, detail: bool = False) -> None:
    table = Table(title=f"评估报告 · 快照 {report.snapshot_id}")
    table.add_column("指标", style="bold cyan")
    table.add_column("值", justify="right")
    table.add_row("检索命中率 hit@k", f"{report.retrieval_hit_rate:.0%}")
    table.add_row("MRR", f"{report.mrr:.3f}")
    table.add_row("关键词覆盖率", f"{report.keyword_coverage_avg:.0%}")
    table.add_row("块级证据率", f"{report.context_evidence_avg:.0%}")
    table.add_row("引用率", f"{report.citation_rate:.0%}")
    judge = f"{report.judge_score_avg:.2f}" if report.judge_score_avg is not None else "-"
    table.add_row("LLM 评分", judge)
    faith = f"{report.faithfulness_avg:.2f}" if report.faithfulness_avg is not None else "-"
    table.add_row("忠实度", faith)
    table.add_row("通过率", f"{report.pass_rate:.0%}")
    if report.guard_enabled:
        table.add_row("守卫拦截", f"{report.guard_blocked_count} 条")
    console.print(table)

    if not detail:
        return
    rows = Table(title=f"逐条结果（k={report.k} · {report.retriever_mode}）")
    rows.add_column("ID", style="dim")
    rows.add_column("问题", overflow="fold", max_width=44)
    rows.add_column("hit", justify="center")
    rows.add_column("首位排名", justify="right")
    rows.add_column("覆盖", justify="right")
    rows.add_column("证据", justify="right")
    rows.add_column("评分", justify="right")
    rows.add_column("忠实", justify="right")
    rows.add_column("守卫", justify="center")
    rows.add_column("通过", justify="center")
    for entry in report.entries:
        rows.add_row(
            entry.qa_id,
            entry.question[:40],
            "[green]✓[/green]" if entry.hit else "[red]✗[/red]",
            str(entry.first_hit_rank or "-"),
            f"{entry.keyword_coverage:.0%}",
            f"{entry.context_evidence:.0%}",
            f"{entry.judge_score:.0f}" if entry.judge_score is not None else "-",
            f"{entry.faithfulness:.0f}" if entry.faithfulness is not None else "-",
            "[red]拦[/red]" if entry.guard_blocked else "-",
            "[green]✓[/green]" if entry.passed else "[red]✗[/red]",
        )
    console.print(rows)
