"""回归门禁：新评估和基线对比，指标退化超阈值则拦截（数据变更不能悄悄弄坏检索）."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llm_intel_kb.eval.runner import EvalReport

DEFAULT_MIN_HIT_RATE = 0.7
DEFAULT_MAX_REGRESSION = 0  # 允许从「过」退化到「不过」的条目数上限
DEFAULT_MIN_JUDGE_SCORE = 3.0


@dataclass
class GateDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    baseline_path: str | None = None

    def summary(self) -> str:
        head = "[green]通过[/green]" if self.passed else "[red]拦截[/red]"
        lines = "\n".join(f"  - {reason}" for reason in self.reasons) or "  - 无"
        return f"{head}\n{lines}"


def run_gate(
    current: EvalReport,
    baseline: EvalReport | None,
    *,
    min_hit_rate: float = DEFAULT_MIN_HIT_RATE,
    max_regressions: int = DEFAULT_MAX_REGRESSION,
    min_judge_score: float = DEFAULT_MIN_JUDGE_SCORE,
) -> GateDecision:
    reasons: list[str] = []

    if current.retrieval_hit_rate < min_hit_rate:
        reasons.append(
            f"检索命中率 {current.retrieval_hit_rate:.0%} 低于阈值 {min_hit_rate:.0%}"
        )

    if current.judge_score_avg is not None and current.judge_score_avg < min_judge_score:
        reasons.append(
            f"LLM 评分均值 {current.judge_score_avg:.2f} 低于阈值 {min_judge_score}"
        )

    if baseline is not None:
        if current.retrieval_hit_rate < baseline.retrieval_hit_rate - 0.001:
            reasons.append(
                f"检索命中率较基线退化：{baseline.retrieval_hit_rate:.0%} → "
                f"{current.retrieval_hit_rate:.0%}"
            )
        if current.mrr < baseline.mrr - 0.001:
            reasons.append(f"MRR 较基线退化：{baseline.mrr:.3f} → {current.mrr:.3f}")

        baseline_passed = {e.qa_id for e in baseline.entries if e.passed}
        current_failed = {e.qa_id for e in current.entries if not e.passed}
        regressed = sorted(baseline_passed & current_failed)
        if len(regressed) > max_regressions:
            reasons.append(
                f"{len(regressed)} 条基线通过的条目出现退化（允许 {max_regressions}）：{regressed[:5]}"
            )

    return GateDecision(passed=not reasons, reasons=reasons)


def load_baseline(baseline_file: Path) -> EvalReport | None:
    if not baseline_file.exists():
        return None
    return EvalReport.load(baseline_file)


def set_baseline(report: EvalReport, baseline_file: Path) -> None:
    """基线 = 一份完整的评估报告快照（含逐条结果，供回归对比）."""
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
