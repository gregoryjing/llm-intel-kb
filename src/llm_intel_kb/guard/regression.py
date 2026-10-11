"""守卫回归用例（误拒 / 漏拦闭环）：把观测到的守卫错误固化为可回归的用例.

闭环流程：
1. `guard_log` 记录每次守卫决策（含放行与拦截）；
2. 人工复盘标记「误拦（false reject）/ 漏拦（false accept）」；
3. 固化为用例（问题 + 期望结论）写入 `golden/guard_cases.toml`；
4. `guard cases` 跑回归：实际结论与期望不一致即失败（CI 门禁），
   防止后续调整词表 / 模式 / 语义阈值时悄悄引入退化。

用例文件与金标集一样属于「版本化的质量资产」，用 TOML 便于人工编辑与 code review。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_intel_kb.guard.domain import Embedder, GuardConfig, GuardDecision, check_question

DEFAULT_CASES_FILE = Path("golden/guard_cases.toml")

KIND_OK = "ok"
KIND_FALSE_REJECT = "false_reject"  # 误拦：本应放行，却被拦
KIND_FALSE_ACCEPT = "false_accept"  # 漏拦：本应拦截，却放行


@dataclass(frozen=True)
class GuardCase:
    question: str
    expected_blocked: bool
    note: str = ""
    source: str = "manual"  # manual（人工固化） | log（从日志引导，待确认）


@dataclass
class CaseOutcome:
    case: GuardCase
    decision: GuardDecision
    kind: str

    @property
    def ok(self) -> bool:
        return self.kind == KIND_OK

    @property
    def label(self) -> str:
        return {
            KIND_OK: "通过",
            KIND_FALSE_REJECT: "误拦",
            KIND_FALSE_ACCEPT: "漏拦",
        }[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.case.question,
            "expected_blocked": self.case.expected_blocked,
            "actual_blocked": self.decision.blocked,
            "reason": self.decision.reason,
            "strategy": self.decision.strategy,
            "kind": self.kind,
            "label": self.label,
            "note": self.case.note,
        }


@dataclass
class RegressionReport:
    outcomes: list[CaseOutcome] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def passed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.ok)

    @property
    def false_rejects(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.kind == KIND_FALSE_REJECT]

    @property
    def false_accepts(self) -> list[CaseOutcome]:
        return [o for o in self.outcomes if o.kind == KIND_FALSE_ACCEPT]

    @property
    def ok(self) -> bool:
        return not self.false_rejects and not self.false_accepts

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "false_reject": len(self.false_rejects),
            "false_accept": len(self.false_accepts),
            "ok": self.ok,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def load_cases(path: Path | None = None) -> list[GuardCase]:
    """读取回归用例文件；不存在时返回空列表."""
    target = path or DEFAULT_CASES_FILE
    if not target.exists():
        return []
    with target.open("rb") as fh:
        raw = tomllib.load(fh)
    cases: list[GuardCase] = []
    for item in raw.get("case", []):
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        cases.append(
            GuardCase(
                question=question,
                expected_blocked=bool(item.get("expected_blocked", False)),
                note=str(item.get("note", "")),
                source=str(item.get("source", "manual")),
            )
        )
    return cases


def save_cases(cases: list[GuardCase], path: Path | None = None) -> Path:
    """写入回归用例文件（全量覆盖，含表头注释）."""
    target = path or DEFAULT_CASES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 守卫回归用例（误拒 / 漏拦闭环）",
        "# 每段一条：expected_blocked = true 表示该问题应被守卫拦截，false 表示应放行。",
        "# `llm-intel-kb guard cases` 会逐条回归，实际结论与期望不一致即失败。",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                "[[case]]",
                f'question = "{_toml_escape(case.question)}"',
                f"expected_blocked = {'true' if case.expected_blocked else 'false'}",
            ]
        )
        if case.note:
            lines.append(f'note = "{_toml_escape(case.note)}"')
        lines.append(f'source = "{_toml_escape(case.source)}"')
        lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


def merge_cases(existing: list[GuardCase], new: list[GuardCase]) -> list[GuardCase]:
    """按问题文本合并（已有用例优先，避免覆盖人工确认过的期望）."""
    seen = {case.question for case in existing}
    merged = list(existing)
    for case in new:
        if case.question not in seen:
            merged.append(case)
            seen.add(case.question)
    return merged


def run_cases(
    cases: list[GuardCase],
    config: GuardConfig,
    embedder: Embedder | None = None,
) -> RegressionReport:
    """逐条回归：实际结论与期望不一致即标记为误拦 / 漏拦."""
    outcomes: list[CaseOutcome] = []
    for case in cases:
        decision = check_question(case.question, config, embedder)
        if decision.blocked == case.expected_blocked:
            kind = KIND_OK
        elif case.expected_blocked:
            kind = KIND_FALSE_ACCEPT  # 期望拦却没拦
        else:
            kind = KIND_FALSE_REJECT  # 期望放行却被拦
        outcomes.append(CaseOutcome(case=case, decision=decision, kind=kind))
    return RegressionReport(outcomes=outcomes)


def cases_from_log(records: list[dict[str, Any]]) -> list[GuardCase]:
    """从守卫日志引导候选用例：以「当时实际决策」为期望，供人工确认后再固化.

    仅作引导——日志里的决策未必正确（否则不会有误拒 / 漏拦）；把日志观测到的问题
    快速转成可编辑的用例草稿，人工复核修正 expected_blocked 后即为回归资产。
    records 为倒序（最新在前），同一问题保留最新一次决策。
    """
    seen: set[str] = set()
    candidates: list[GuardCase] = []
    for record in records:
        question = str(record.get("question", "")).strip()
        if not question or question in seen:
            continue
        seen.add(question)
        blocked = bool(record.get("blocked"))
        candidates.append(
            GuardCase(
                question=question,
                expected_blocked=blocked,
                note=f"来自 guard_log（{record.get('strategy', '-')}）",
                source="log",
            )
        )
    return candidates