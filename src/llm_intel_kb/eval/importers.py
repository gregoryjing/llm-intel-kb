"""金标集导入与体检：badcase 回流导入 + 数据集质量审计.

- 导入：解析 JSONL（badcase 导出格式）或 JSON 数组，去重、自动分配 id、追加进 golden/qa.toml。
- 体检：统计维度/时效分布，标记时效敏感、宽泛、近重复、宽源条目——把「评估集本身的质量」
  当作一等公民来治理（业界共识：评估集要像代码一样版本化、定期轮换、警惕 stale eval）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_intel_kb.eval.golden import GoldenQA, GoldenSet, load_golden

# arXiv 大类是「宽源」：一个来源涵盖大量不相关主题，命中来源 ≠ 命中正确内容，
# 因此这类条目在「来源级 hit@k」上几乎恒为命中，需要体检时单独标记。
_BROAD_SOURCE_PREFIXES = ("arxiv-",)
# 近重复判定阈值：字符 bigram Jaccard（中文无空格，bigram 比 SequenceMatcher 稳）。
# 关键是先剥离「问题模板词」——"有哪些关于…的研究""什么是…"这类共用框架会让
# 两个主题完全不同的问句（量化 vs 幻觉）相似度虚高，剥离后阈值 0.62 可干净分离。
_DUPLICATE_THRESHOLD = 0.62

# 问题模板词：仅用于近重复比对，不改变原问题文本。
_QUESTION_TEMPLATES: tuple[str, ...] = (
    "有哪些关于", "有什么关于", "的研究", "的论文", "的进展",
    "什么是", "是什么", "有什么", "如何", "怎么样", "怎么", "哪些",
    "分别", "区别", "不同", "在", "方面", "场景下", "各", "和", "与", "的",
)


@dataclass
class ImportedEntry:
    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)


@dataclass
class ImportResult:
    imported: list[ImportedEntry]
    skipped: list[dict[str, str]]
    target: str
    dry_run: bool
    preview_toml: str
    warnings: list[str] = field(default_factory=list)

    @property
    def id_range(self) -> str:
        if not self.imported:
            return "-"
        return f"{self.imported[0].id} ~ {self.imported[-1].id}"


def parse_import_text(text: str) -> list[dict[str, Any]]:
    """解析导入文本：JSON 数组或 JSONL（每行一个 JSON 对象）."""
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        return [item for item in data if isinstance(item, dict)]
    items: list[dict[str, Any]] = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            items.append(obj)
    return items


def _toml_str(value: str) -> str:
    """转义为 TOML 基本字符串字面量内部内容."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\r", " ")


def _next_id(existing: GoldenSet) -> int:
    max_num = 0
    for qa in existing.entries:
        match = re.match(r"g(\d+)", qa.id)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def import_golden_entries(
    items: list[dict[str, Any]],
    target: Path,
    *,
    dry_run: bool = False,
    known_sources: set[str] | None = None,
) -> ImportResult:
    """把导入条目写入金标集：去重（按问题文本）、自动分配 gNNN 编号、追加 TOML.

    期望来源 / 关键词 / 备注直接沿用条目字段；缺失则为空。
    known_sources 传入 data_sources.toml 中已启用的源 id 集合时，会校验
    expected_sources 是否引用未注册源，并以 warnings 提示（不阻断导入）。
    """
    existing = load_golden(target) if target.exists() else GoldenSet()
    existing_questions = {qa.question.strip() for qa in existing.entries}
    next_num = _next_id(existing)

    imported: list[ImportedEntry] = []
    skipped: list[dict[str, str]] = []
    warnings: list[str] = []
    toml_lines: list[str] = []

    for item in items:
        question = str(item.get("question", "")).strip()
        if not question:
            skipped.append({"question": "(空)", "reason": "缺少 question 字段"})
            continue
        if question in existing_questions:
            skipped.append({"question": question, "reason": "与现有条目重复"})
            continue
        existing_questions.add(question)

        sources = [str(s) for s in item.get("expected_sources", []) or []]
        keywords = [str(kw).lower() for kw in item.get("expected_keywords", []) or []]
        notes = str(item.get("notes", "") or "").strip()
        category = str(item.get("category", "") or "").strip()
        timeliness = str(item.get("timeliness", "") or "").strip()

        qid = f"g{next_num:03d}"
        next_num += 1
        imported.append(
            ImportedEntry(id=qid, question=question, expected_sources=sources, expected_keywords=keywords)
        )
        if known_sources is not None:
            unknown = [s for s in sources if s not in known_sources]
            if unknown:
                warnings.append(
                    f"{qid} 引用未注册/未启用的数据源 {unknown}：该条 hit@k 将恒为 0，"
                    "请核对 data_sources.toml。"
                )

        src_str = "[" + ", ".join(f'"{_toml_str(s)}"' for s in sources) + "]"
        kw_str = "[" + ", ".join(f'"{_toml_str(kw)}"' for kw in keywords) + "]"
        toml_lines.extend(
            [
                "",
                "[[qa]]",
                f'id = "{qid}"',
                f'question = "{_toml_str(question)}"',
                f"expected_sources = {src_str}",
                f"expected_keywords = {kw_str}",
            ]
        )
        if category:
            toml_lines.append(f'category = "{_toml_str(category)}"')
        if timeliness:
            toml_lines.append(f'timeliness = "{_toml_str(timeliness)}"')
        if notes:
            toml_lines.append(f'notes = "{_toml_str(notes)}"')

    preview = "\n".join(toml_lines)
    if imported and not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(preview + "\n")

    return ImportResult(
        imported=imported,
        skipped=skipped,
        target=str(target),
        dry_run=dry_run,
        preview_toml=preview,
        warnings=warnings,
    )


@dataclass
class AuditReport:
    """金标集体检报告.

    errors 为「硬错误」（数据本身坏了，会让指标失真）：未注册数据源、空问题、重复 id。
    flags 为「软风险」（结构性问题，需人工取舍）：时效敏感、宽泛、宽源、近重复。
    """

    total: int
    negative: int
    time_sensitive: int
    by_category: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    flags: dict[str, list[str]] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """无硬错误即视为通过（软风险不影响门禁）."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "negative": self.negative,
            "time_sensitive": self.time_sensitive,
            "by_category": self.by_category,
            "by_source": self.by_source,
            "flags": self.flags,
            "suggestions": self.suggestions,
            "errors": self.errors,
            "ok": self.ok,
        }


def _normalize_question(text: str) -> str:
    """归一化：去标点空白 + 剥离问题模板词，保留承载语义的实词."""
    cleaned = re.sub(r"[\s，。？！、,.?!；;：:\"'（）()【】\[\]]+", "", text.lower())
    for template in _QUESTION_TEMPLATES:
        cleaned = cleaned.replace(template, "")
    return cleaned


def _char_bigrams(text: str) -> set[str]:
    normalized = _normalize_question(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def _question_similarity(a: str, b: str) -> float:
    bigrams_a, bigrams_b = _char_bigrams(a), _char_bigrams(b)
    if not bigrams_a or not bigrams_b:
        return 0.0
    return len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)


def _find_duplicate_groups(entries: list[GoldenQA]) -> list[list[str]]:
    """以每条为锚点，找出与其字符 bigram 相似度超阈值的近重复组（待人工确认）."""
    groups: list[list[str]] = []
    assigned: set[str] = set()
    for i, qa in enumerate(entries):
        if qa.id in assigned:
            continue
        group = [qa.id]
        for other in entries[i + 1 :]:
            if other.id in assigned:
                continue
            if _question_similarity(qa.question, other.question) >= _DUPLICATE_THRESHOLD:
                group.append(other.id)
                assigned.add(other.id)
        if len(group) > 1:
            assigned.add(qa.id)
            groups.append(group)
    return groups


def audit_golden(golden: GoldenSet, known_sources: set[str] | None = None) -> AuditReport:
    """对金标集做质量体检：维度/时效分布 + 风险标记（含硬错误）.

    known_sources 传入 data_sources.toml 已启用源 id 时，会校验 expected_sources
    是否引用未注册/未启用源（硬错误：这类条目 hit@k 恒为 0，属数据集损坏）。
    """
    entries = golden.entries
    total = len(entries)
    if total == 0:
        return AuditReport(total=0, negative=0, time_sensitive=0)

    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for qa in entries:
        by_category[qa.category] = by_category.get(qa.category, 0) + 1
        for source in qa.expected_sources:
            by_source[source] = by_source.get(source, 0) + 1

    # 硬错误：会让指标失真的数据集损坏
    errors: list[str] = []
    seen_ids: set[str] = set()
    for qa in entries:
        if qa.id in seen_ids:
            errors.append(f"重复 id：{qa.id}")
        seen_ids.add(qa.id)
        if not qa.question.strip():
            errors.append(f"{qa.id} 问题为空")
        if known_sources is not None:
            unknown = [s for s in qa.expected_sources if s not in known_sources]
            if unknown:
                errors.append(f"{qa.id} 引用未注册/未启用数据源 {unknown}（hit@k 将恒为 0）")

    time_sensitive_ids = [qa.id for qa in entries if qa.is_time_sensitive]
    negative_ids = [qa.id for qa in entries if qa.is_negative]
    # 宽泛问题：期望关键词 <=1 且期望来源 >=3 —— 判定信号弱，接近冒烟测试
    broad_ids = [
        qa.id
        for qa in entries
        if not qa.is_negative and len(qa.expected_keywords) <= 1 and len(qa.expected_sources) >= 3
    ]
    # 宽源条目：期望来源全部是 arXiv 大类 —— 来源级 hit 几乎恒为真，证据力弱
    wide_source_ids = [
        qa.id
        for qa in entries
        if qa.expected_sources and all(s.startswith(_BROAD_SOURCE_PREFIXES) for s in qa.expected_sources)
    ]
    duplicate_groups = _find_duplicate_groups(entries)

    suggestions: list[str] = []
    if time_sensitive_ids:
        suggestions.append(
            f"{len(time_sensitive_ids)} 条为时效敏感问题（依赖语料新鲜度）："
            "建议按季度轮换 30%、或在 notes 记录时间锚点，避免 stale eval 高估系统能力。"
        )
    if broad_ids:
        suggestions.append(
            f"{len(broad_ids)} 条判定信号偏弱（宽泛开放题）：建议细化期望关键词或降级为冒烟用例。"
        )
    if wide_source_ids:
        suggestions.append(
            f"{len(wide_source_ids)} 条仅依赖 arXiv 宽源：来源级 hit@k 证据力弱，"
            "建议补充 expected_keywords 或块级校验。"
        )
    if duplicate_groups:
        suggestions.append(f"发现 {len(duplicate_groups)} 组近重复问题：建议合并去重，避免指标被同质问题拉偏。")
    if errors:
        suggestions.insert(
            0,
            f"{len(errors)} 条硬错误（未注册源/空问题/重复 id）会使指标失真，"
            "属数据集损坏，应优先修复（`eval audit --strict` 会因此失败）。",
        )
    if not suggestions:
        suggestions.append("未发现明显风险项。")

    return AuditReport(
        total=total,
        negative=len(negative_ids),
        time_sensitive=len(time_sensitive_ids),
        by_category=dict(sorted(by_category.items(), key=lambda kv: (-kv[1], kv[0]))),
        by_source=dict(sorted(by_source.items(), key=lambda kv: (-kv[1], kv[0]))),
        flags={
            "time_sensitive": time_sensitive_ids,
            "negative": negative_ids,
            "broad": broad_ids,
            "wide_source": wide_source_ids,
            "duplicate_groups": [" / ".join(group) for group in duplicate_groups],
        },
        suggestions=suggestions,
        errors=errors,
    )