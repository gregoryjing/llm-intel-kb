"""守卫回归用例测试：用例读写、合并、回归判定、日志引导."""

from __future__ import annotations

from pathlib import Path

from llm_intel_kb.guard.domain import MODE_BLOCKLIST, GuardConfig
from llm_intel_kb.guard.regression import (
    KIND_FALSE_ACCEPT,
    KIND_FALSE_REJECT,
    KIND_OK,
    GuardCase,
    cases_from_log,
    load_cases,
    merge_cases,
    run_cases,
    save_cases,
)


def _cfg() -> GuardConfig:
    return GuardConfig(
        enabled=True,
        mode=MODE_BLOCKLIST,
        blocked_topics=["做菜"],
        allowed_topics=["rag"],
    )


class TestCasePersistence:
    def test_roundtrip(self, tmp_path: Path) -> None:
        cases = [
            GuardCase(question="做菜怎么做", expected_blocked=True, note="烹饪", source="manual"),
            GuardCase(question="rag 是什么", expected_blocked=False),
        ]
        path = tmp_path / "guard_cases.toml"
        save_cases(cases, path)
        loaded = load_cases(path)
        assert [c.question for c in loaded] == ["做菜怎么做", "rag 是什么"]
        assert loaded[0].expected_blocked is True
        assert loaded[0].note == "烹饪"
        assert loaded[0].source == "manual"
        assert loaded[1].expected_blocked is False

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_cases(tmp_path / "nope.toml") == []

    def test_escapes_quotes(self, tmp_path: Path) -> None:
        path = tmp_path / "c.toml"
        save_cases([GuardCase(question='什么是 "RAG"？', expected_blocked=False)], path)
        assert load_cases(path)[0].question == '什么是 "RAG"？'


class TestMerge:
    def test_existing_wins_and_new_appended(self) -> None:
        existing = [GuardCase(question="a", expected_blocked=True, note="人工")]
        new = [
            GuardCase(question="a", expected_blocked=False, note="日志", source="log"),
            GuardCase(question="b", expected_blocked=False, source="log"),
        ]
        merged = merge_cases(existing, new)
        assert [c.question for c in merged] == ["a", "b"]
        assert merged[0].note == "人工"  # 已有用例优先，不被日志草稿覆盖
        assert merged[1].source == "log"


class TestRunCases:
    def test_all_pass(self) -> None:
        cases = [
            GuardCase(question="做菜怎么做", expected_blocked=True),
            GuardCase(question="rag 是什么", expected_blocked=False),
        ]
        report = run_cases(cases, _cfg())
        assert report.ok is True
        assert report.passed == 2
        assert all(o.kind == KIND_OK for o in report.outcomes)

    def test_false_reject_detected(self) -> None:
        # 误拦：本应放行（expected_blocked=False），却被守卫拦下
        report = run_cases([GuardCase(question="做菜怎么做", expected_blocked=False)], _cfg())
        assert report.ok is False
        assert report.outcomes[0].kind == KIND_FALSE_REJECT
        assert report.outcomes[0].label == "误拦"

    def test_false_accept_detected(self) -> None:
        # 漏拦：本应拦截（expected_blocked=True），却被放行
        report = run_cases([GuardCase(question="rag 是什么", expected_blocked=True)], _cfg())
        assert report.ok is False
        assert report.outcomes[0].kind == KIND_FALSE_ACCEPT
        assert report.outcomes[0].label == "漏拦"

    def test_report_to_dict(self) -> None:
        report = run_cases([GuardCase(question="做菜怎么做", expected_blocked=True)], _cfg())
        data = report.to_dict()
        assert data["total"] == 1
        assert data["ok"] is True
        assert data["outcomes"][0]["actual_blocked"] is True


class TestCasesFromLog:
    def test_dedup_latest_wins_and_skips_blank(self) -> None:
        records = [
            {"question": "a", "blocked": True, "strategy": "blocklist"},
            {"question": "b", "blocked": False, "strategy": "blocklist"},
            {"question": "a", "blocked": False, "strategy": "blocklist"},  # 旧记录，忽略
            {"question": "  ", "blocked": True},
        ]
        cases = cases_from_log(records)
        assert [c.question for c in cases] == ["a", "b"]
        assert cases[0].expected_blocked is True  # 倒序：最新在前
        assert cases[1].expected_blocked is False
        assert all(c.source == "log" for c in cases)