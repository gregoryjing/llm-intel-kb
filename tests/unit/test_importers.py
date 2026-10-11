"""badcase 导入与金标集体检的单元测试."""

from __future__ import annotations

from pathlib import Path

from llm_intel_kb.eval.golden import GoldenQA, GoldenSet, load_golden
from llm_intel_kb.eval.importers import (
    audit_golden,
    import_golden_entries,
    parse_import_text,
)


class TestParseImportText:
    def test_json_array(self) -> None:
        items = parse_import_text('[{"question": "a"}, {"question": "b"}]')
        assert [i["question"] for i in items] == ["a", "b"]

    def test_jsonl(self) -> None:
        items = parse_import_text('{"question": "a"}\n\n{"question": "b"}')
        assert [i["question"] for i in items] == ["a", "b"]

    def test_malformed_lines_skipped(self) -> None:
        items = parse_import_text('{"question": "a"}\nnot json\n{"question": "b"}')
        assert [i["question"] for i in items] == ["a", "b"]

    def test_empty(self) -> None:
        assert parse_import_text("   ") == []


class TestImportGoldenEntries:
    def test_assigns_ids_and_dedupes(self, tmp_path: Path) -> None:
        target = tmp_path / "qa.toml"
        existing = GoldenSet(
            entries=[GoldenQA(id="g001", question="vLLM 最近更新了什么？")]
        )
        _write_golden(target, existing)

        items = [
            {"question": "vLLM 最近更新了什么？", "expected_sources": ["gh-vllm"]},  # 重复
            {"question": "PagedAttention 如何管理显存？", "expected_sources": ["gh-vllm"]},
        ]
        result = import_golden_entries(items, target)

        assert [e.id for e in result.imported] == ["g002"]
        assert len(result.skipped) == 1
        assert "重复" in result.skipped[0]["reason"]

        reloaded = load_golden(target)
        assert {qa.id for qa in reloaded.entries} == {"g001", "g002"}

    def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        target = tmp_path / "qa.toml"
        items = [{"question": "全新的问题？", "expected_sources": ["gh-vllm"]}]
        result = import_golden_entries(items, target, dry_run=True)

        assert result.dry_run is True
        assert result.preview_toml
        assert not target.exists()  # 未落盘

    def test_empty_question_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / "qa.toml"
        result = import_golden_entries([{"question": "  "}], target)
        assert result.imported == []
        assert result.skipped

    def test_warns_on_unregistered_source(self, tmp_path: Path) -> None:
        target = tmp_path / "qa.toml"
        items = [{"question": "某个新问题？", "expected_sources": ["gh-vllm", "gh-not-exist"]}]
        result = import_golden_entries(items, target, known_sources={"gh-vllm"})

        assert result.imported  # 仍导入（warning 不阻断）
        assert result.warnings
        assert "gh-not-exist" in result.warnings[0]

    def test_no_warning_when_sources_known(self, tmp_path: Path) -> None:
        target = tmp_path / "qa.toml"
        items = [{"question": "另一个问题？", "expected_sources": ["gh-vllm"]}]
        result = import_golden_entries(items, target, known_sources={"gh-vllm"})
        assert result.warnings == []


class TestAuditGolden:
    def test_flags_negative_and_time_sensitive(self) -> None:
        golden = GoldenSet(
            entries=[
                GoldenQA(id="g001", question="今天北京天气怎么样？", expected_sources=()),
                GoldenQA(
                    id="g002",
                    question="vLLM 最近有哪些版本更新？",
                    expected_sources=("gh-vllm",),
                    expected_keywords=("vllm",),
                    timeliness="time_sensitive",
                ),
            ]
        )
        report = audit_golden(golden)
        assert report.total == 2
        assert report.negative == 1
        assert "g001" in report.flags["negative"]
        assert "g002" in report.flags["time_sensitive"]

    def test_detects_duplicate_group(self) -> None:
        golden = GoldenSet(
            entries=[
                GoldenQA(id="g001", question="有哪些关于 RAG 的研究？"),
                GoldenQA(id="g002", question="有哪些关于 RAG 的论文？"),
            ]
        )
        report = audit_golden(golden)
        groups = report.flags.get("duplicate_groups", [])
        assert groups
        assert any("g001" in group and "g002" in group for group in groups)

    def test_unregistered_source_is_hard_error(self) -> None:
        golden = GoldenSet(
            entries=[
                GoldenQA(
                    id="g001",
                    question="某个问题？",
                    expected_sources=("gh-vllm", "gh-not-exist"),
                )
            ]
        )
        report = audit_golden(golden, known_sources={"gh-vllm"})
        assert report.ok is False
        assert any("gh-not-exist" in err for err in report.errors)

    def test_duplicate_id_is_hard_error(self) -> None:
        golden = GoldenSet(
            entries=[
                GoldenQA(id="g001", question="问题甲？", expected_sources=("gh-vllm",)),
                GoldenQA(id="g001", question="问题乙？", expected_sources=("gh-vllm",)),
            ]
        )
        report = audit_golden(golden, known_sources={"gh-vllm"})
        assert report.ok is False
        assert any("重复 id" in err for err in report.errors)

    def test_no_known_sources_skips_source_check(self) -> None:
        golden = GoldenSet(
            entries=[GoldenQA(id="g001", question="问题？", expected_sources=("whatever",))]
        )
        assert audit_golden(golden).ok is True


def _write_golden(path: Path, golden: GoldenSet) -> None:
    lines = []
    for qa in golden.entries:
        lines.append("[[qa]]")
        lines.append(f'id = "{qa.id}"')
        lines.append(f'question = "{qa.question}"')
        lines.append(f"expected_sources = {list(qa.expected_sources)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")