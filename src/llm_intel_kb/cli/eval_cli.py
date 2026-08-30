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
from llm_intel_kb.eval.runner import EvalReport, export_badcases, run_eval
from llm_intel_kb.index.embeddings import resolve_embedding_backend
from llm_intel_kb.index.loader import load_index_state
from llm_intel_kb.index.search import HybridSearcher
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
    table.add_row("引用率", f"{report.citation_rate:.0%}")
    judge = f"{report.judge_score_avg:.2f}" if report.judge_score_avg is not None else "-"
    table.add_row("LLM 评分", judge)
    table.add_row("通过率", f"{report.pass_rate:.0%}")
    console.print(table)

    if not detail:
        return
    rows = Table(title=f"逐条结果（k={report.k} · {report.retriever_mode}）")
    rows.add_column("ID", style="dim")
    rows.add_column("问题", overflow="fold", max_width=44)
    rows.add_column("hit", justify="center")
    rows.add_column("首位排名", justify="right")
    rows.add_column("覆盖", justify="right")
    rows.add_column("评分", justify="right")
    rows.add_column("通过", justify="center")
    for entry in report.entries:
        rows.add_row(
            entry.qa_id,
            entry.question[:40],
            "[green]✓[/green]" if entry.hit else "[red]✗[/red]",
            str(entry.first_hit_rank or "-"),
            f"{entry.keyword_coverage:.0%}",
            f"{entry.judge_score:.0f}" if entry.judge_score is not None else "-",
            "[green]✓[/green]" if entry.passed else "[red]✗[/red]",
        )
    console.print(rows)
