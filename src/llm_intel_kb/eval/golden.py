"""金标问答集：一组「问题 + 期望来源 + 期望关键词」的人工标注，作为回归门禁的基准.

条目可标注两个治理维度（用于数据集体检与时效性治理）：
- category：评估维度（concept/version/news/multihop/cross_lingual/negative/general...）
- timeliness：stable（时间无关）或 time_sensitive（依赖语料新鲜度，会随时间陈旧化）

未显式标注时：timeliness 依据问题中的时间线索词自动推断（"最近/最新/近期"→time_sensitive），
category 默认 general——避免为历史条目批量补字段，同时让体检能识别时效风险。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GOLDEN_FILE = Path("golden/qa.toml")

# 时间线索词：出现即视为「时效敏感」问题（stale eval 风险）
TIME_SENSITIVE_MARKERS: tuple[str, ...] = (
    "最近", "最新", "近期", "新进展", "新版本", "有什么新", "recent", "latest",
)

TIMELINESS_STABLE = "stable"
TIMELINESS_SENSITIVE = "time_sensitive"


@dataclass(frozen=True)
class GoldenQA:
    id: str
    question: str
    expected_sources: tuple[str, ...] = ()  # 命中任一来源的块 = 检索命中
    expected_keywords: tuple[str, ...] = ()  # 回答中应覆盖的关键词（覆盖率为生成指标）
    notes: str = ""
    category: str = "general"  # 评估维度标签
    timeliness: str = TIMELINESS_STABLE  # 时效性标签

    @property
    def is_negative(self) -> bool:
        """负例：无期望来源且期望系统拒绝回答."""
        return not self.expected_sources

    @property
    def is_time_sensitive(self) -> bool:
        return self.timeliness == TIMELINESS_SENSITIVE


@dataclass
class GoldenSet:
    entries: list[GoldenQA] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


def infer_timeliness(question: str) -> str:
    """依据问题中的时间线索词推断时效性标签."""
    lowered = question.lower()
    if any(marker in lowered for marker in TIME_SENSITIVE_MARKERS):
        return TIMELINESS_SENSITIVE
    return TIMELINESS_STABLE


def load_golden(path: Path | None = None) -> GoldenSet:
    golden_path = path or DEFAULT_GOLDEN_FILE
    if not golden_path.exists():
        return GoldenSet()
    with golden_path.open("rb") as fh:
        raw = tomllib.load(fh)
    entries: list[GoldenQA] = []
    for item in raw.get("qa", []):
        question = str(item["question"])
        explicit_timeliness = str(item.get("timeliness", "")).strip()
        timeliness = explicit_timeliness or infer_timeliness(question)
        category = str(item.get("category", "")).strip()
        if not category:
            category = "negative" if not item.get("expected_sources") else "general"
        entries.append(
            GoldenQA(
                id=str(item["id"]),
                question=question,
                expected_sources=tuple(str(s) for s in item.get("expected_sources", [])),
                expected_keywords=tuple(str(kw).lower() for kw in item.get("expected_keywords", [])),
                notes=str(item.get("notes", "")),
                category=category,
                timeliness=timeliness,
            )
        )
    return GoldenSet(entries=entries)