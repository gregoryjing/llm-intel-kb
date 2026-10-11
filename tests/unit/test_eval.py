from __future__ import annotations

from pathlib import Path

from llm_intel_kb.eval.gate import run_gate, set_baseline
from llm_intel_kb.eval.golden import GoldenQA, GoldenSet, load_golden
from llm_intel_kb.eval.runner import (
    EntryResult,
    EvalReport,
    _parse_judge_reply,
    export_badcases,
    run_eval,
)
from llm_intel_kb.index.chunking import Chunk
from llm_intel_kb.index.search import HybridSearcher
from llm_intel_kb.index.store_local import IndexMeta, IndexState
from llm_intel_kb.llm.providers import ProviderConfig

PROVIDER = ProviderConfig(
    name="test", kind="chat", base_url="http://localhost:11434/v1", model="m"
)


def _chunk(cid: str, doc_id: str, source_id: str, text: str) -> Chunk:
    return Chunk(
        id=cid,
        doc_id=doc_id,
        chunk_index=0,
        title=f"doc {doc_id}",
        text=text,
        source_id=source_id,
        url=f"https://example.com/{doc_id}",
        published_at=None,
        topics=(),
        summary=None,
    )


def _searcher(chunks: list[Chunk]) -> HybridSearcher:
    state = IndexState(
        meta=IndexMeta(
            snapshot_id="s1",
            created_at="t",
            chunk_count=len(chunks),
            doc_count=len({c.doc_id for c in chunks}),
            embedding_model=None,
            embedding_dim=None,
        ),
        chunks=chunks,
    )
    return HybridSearcher(state, backend=None)


def _report(
    hit_rate: float,
    mrr: float,
    entries: list[EntryResult] | None = None,
    snapshot_id: str = "s1",
) -> EvalReport:
    return EvalReport(
        snapshot_id=snapshot_id,
        index_snapshot_id=snapshot_id,
        created_at="2026-01-01T00:00:00+00:00",
        retriever_mode="lexical",
        judge_model=None,
        k=5,
        retrieval_hit_rate=hit_rate,
        mrr=mrr,
        keyword_coverage_avg=0.5,
        citation_rate=0.5,
        judge_score_avg=None,
        entries=entries or [],
    )


class TestRunEval:
    def test_retrieval_metrics(self) -> None:
        chunks = [
            _chunk("c1", "d1", "gh-vllm", "vllm 性能优化"),
            _chunk("c2", "d2", "qbitai", "行业新闻"),
        ]
        golden = GoldenSet(
            entries=[
                GoldenQA(id="g1", question="vllm 性能优化", expected_sources=("gh-vllm",)),
                GoldenQA(id="g2", question="完全不相关的问题 abcdefg", expected_sources=("gh-vllm",)),
            ]
        )
        report = run_eval(golden, _searcher(chunks), "s1", "s1", None, k=2)
        assert report.retrieval_hit_rate == 0.5
        assert report.mrr == 0.5
        assert report.entries[0].hit is True
        assert report.entries[1].hit is False

    def test_no_llm_skips_answer(self) -> None:
        chunks = [_chunk("c1", "d1", "gh-vllm", "vllm")]
        golden = GoldenSet(entries=[GoldenQA(id="g1", question="vllm", expected_sources=("gh-vllm",))])
        report = run_eval(golden, _searcher(chunks), "s1", "s1", None, k=2)
        assert report.entries[0].answer is None
        assert report.entries[0].passed is True

    def test_report_roundtrip(self, tmp_path: Path) -> None:
        entries = [
            EntryResult(
                qa_id="g1",
                question="q",
                hit=True,
                first_hit_rank=1,
                retrieved_sources=["gh-vllm"],
                answer="a [1]",
                keyword_coverage=1.0,
                citations=[1],
            )
        ]
        report = _report(1.0, 1.0, entries)
        path = report.save(tmp_path)
        loaded = EvalReport.load(path)
        assert loaded.snapshot_id == "s1"
        assert loaded.entries[0].qa_id == "g1"
        assert loaded.pass_rate == 1.0


class TestGate:
    def test_passes_when_above_thresholds(self) -> None:
        decision = run_gate(_report(0.9, 0.9), None)
        assert decision.passed

    def test_fails_below_hit_rate(self) -> None:
        decision = run_gate(_report(0.5, 0.9), None)
        assert not decision.passed
        assert any("命中率" in r for r in decision.reasons)

    def test_regression_vs_baseline(self) -> None:
        baseline = _report(0.9, 0.9, snapshot_id="old")
        current = _report(0.9, 0.5)
        decision = run_gate(current, baseline)
        assert not decision.passed
        assert any("MRR" in r for r in decision.reasons)

    def test_entry_regression_detected(self) -> None:
        passing = EntryResult(
            qa_id="g1", question="q", hit=True, first_hit_rank=1,
            retrieved_sources=["s"], answer=None, keyword_coverage=0.0, citations=[],
        )
        failing = EntryResult(
            qa_id="g1", question="q", hit=False, first_hit_rank=None,
            retrieved_sources=[], answer=None, keyword_coverage=0.0, citations=[],
        )
        baseline = _report(1.0, 1.0, [passing])
        current = _report(0.0, 0.0, [failing])
        decision = run_gate(current, baseline)
        assert not decision.passed
        assert any("退化" in r for r in decision.reasons)

    def test_baseline_roundtrip(self, tmp_path: Path) -> None:
        report = _report(0.9, 0.9)
        baseline_file = tmp_path / "baseline.json"
        set_baseline(report, baseline_file)
        from llm_intel_kb.eval.gate import load_baseline

        loaded = load_baseline(baseline_file)
        assert loaded is not None
        assert loaded.retrieval_hit_rate == 0.9


class TestBadcases:
    def test_export_only_failures(self, tmp_path: Path) -> None:
        entries = [
            EntryResult(
                qa_id="g1", question="q1", hit=False, first_hit_rank=None,
                retrieved_sources=[], answer=None, keyword_coverage=0.0, citations=[],
            ),
            EntryResult(
                qa_id="g2", question="q2", hit=True, first_hit_rank=1,
                retrieved_sources=["s"], answer=None, keyword_coverage=0.0, citations=[],
            ),
        ]
        golden = GoldenSet(
            entries=[GoldenQA(id="g1", question="q1", expected_sources=("s",))]
        )
        report = _report(0.5, 0.5, entries)
        path = export_badcases(report, golden, tmp_path)
        assert path is not None
        content = path.read_text(encoding="utf-8")
        assert "q1" in content
        assert "q2" not in content

    def test_no_failures_no_file(self, tmp_path: Path) -> None:
        entries = [
            EntryResult(
                qa_id="g1", question="q", hit=True, first_hit_rank=1,
                retrieved_sources=["s"], answer=None, keyword_coverage=0.0, citations=[],
            )
        ]
        report = _report(1.0, 1.0, entries)
        assert export_badcases(report, GoldenSet(), tmp_path) is None


class TestGoldenLoading:
    def test_load_golden_file(self, tmp_path: Path) -> None:
        path = tmp_path / "qa.toml"
        path.write_text(
            '[[qa]]\nid = "g1"\nquestion = "q"\n'
            'expected_sources = ["gh-vllm"]\nexpected_keywords = ["vllm"]\n',
            encoding="utf-8",
        )
        golden = load_golden(path)
        assert len(golden) == 1
        assert golden.entries[0].expected_sources == ("gh-vllm",)
        assert golden.entries[0].expected_keywords == ("vllm",)

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert len(load_golden(tmp_path / "nope.toml")) == 0


class TestEntryPassed:
    def test_low_judge_score_fails(self) -> None:
        entry = EntryResult(
            qa_id="g1", question="q", hit=True, first_hit_rank=1,
            retrieved_sources=["s"], answer="a", keyword_coverage=0.0,
            citations=[], judge_score=2.0, judge_comment="差",
        )
        assert not entry.passed


class TestParseJudgeReply:
    def test_strict_json(self) -> None:
        assert _parse_judge_reply('{"score": 4, "comment": "不错"}') == (4.0, None, "不错")

    def test_json_with_faithfulness(self) -> None:
        reply = '{"score": 5, "faithfulness": 4, "comment": "有依据"}'
        assert _parse_judge_reply(reply) == (5.0, 4.0, "有依据")

    def test_json_wrapped_in_prose(self) -> None:
        reply = '好的，评分如下：\n{"score": 5, "comment": "优秀"}'
        assert _parse_judge_reply(reply) == (5.0, None, "优秀")

    def test_unquoted_comment_fallback(self) -> None:
        # dots3 实测输出：comment 值不带引号（非法 JSON）
        reply = '{"score": 5, "comment": 回答准确完整，引用充分。}'
        assert _parse_judge_reply(reply) == (5.0, None, "回答准确完整，引用充分。")

    def test_regex_fallback_extracts_faithfulness(self) -> None:
        reply = '评分 {"score": 4, "faithfulness": 3, "comment": 部分缺漏}'
        assert _parse_judge_reply(reply) == (4.0, 3.0, "部分缺漏")

    def test_no_score_returns_none(self) -> None:
        assert _parse_judge_reply("这个回答挺好的") == (None, None, None)

    def test_empty_reply(self) -> None:
        assert _parse_judge_reply("") == (None, None, None)
