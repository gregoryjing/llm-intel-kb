"""金标问答集：一组「问题 + 期望来源 + 期望关键词」的人工标注，作为回归门禁的基准."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GOLDEN_FILE = Path("golden/qa.toml")


@dataclass(frozen=True)
class GoldenQA:
    id: str
    question: str
    expected_sources: tuple[str, ...] = ()  # 命中任一来源的块 = 检索命中
    expected_keywords: tuple[str, ...] = ()  # 回答中应覆盖的关键词（覆盖率为生成指标）
    notes: str = ""


@dataclass
class GoldenSet:
    entries: list[GoldenQA] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


def load_golden(path: Path | None = None) -> GoldenSet:
    golden_path = path or DEFAULT_GOLDEN_FILE
    if not golden_path.exists():
        return GoldenSet()
    with golden_path.open("rb") as fh:
        raw = tomllib.load(fh)
    entries: list[GoldenQA] = []
    for item in raw.get("qa", []):
        entries.append(
            GoldenQA(
                id=str(item["id"]),
                question=str(item["question"]),
                expected_sources=tuple(str(s) for s in item.get("expected_sources", [])),
                expected_keywords=tuple(str(kw).lower() for kw in item.get("expected_keywords", [])),
                notes=str(item.get("notes", "")),
            )
        )
    return GoldenSet(entries=entries)
