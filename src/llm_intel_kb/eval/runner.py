"""评估运行器：对金标集逐条跑「检索 + 生成」，计算检索与生成指标.

- 检索指标（不需要任何模型）：hit@k（来源级）、MRR、关键词命中
- 生成指标：引用率、关键词覆盖率（无需模型）；
  有对话模型时追加 LLM-as-judge 评分（1-5 分 + 一句话点评）
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from llm_intel_kb.eval.golden import GoldenQA, GoldenSet
from llm_intel_kb.guard.domain import GuardConfig, check_question, load_guard_config
from llm_intel_kb.index.search import HybridSearcher
from llm_intel_kb.llm import client as llm_client
from llm_intel_kb.llm.providers import ProviderConfig
from llm_intel_kb.rag.answer import extract_citations, generate_answer

_JUDGE_PROMPT = """你是问答系统的评估员。根据问题、参考资料要点和系统回答，给两个 1-5 分：

score（回答质量）：
5 = 准确完整，有依据，引用清晰
4 = 基本准确，小瑕疵
3 = 部分回答，有明显缺漏
2 = 大部分错误或答非所问
1 = 完全错误或编造

faithfulness（忠实度，是否只依据参考资料、有无编造参考之外的内容）：
5 = 完全基于参考资料，无编造
4 = 基本基于参考，个别表述超出但无害
3 = 部分内容无参考支撑
2 = 多处编造或与参考矛盾
1 = 基本是编造

只输出 JSON：{{"score": 数字, "faithfulness": 数字, "comment": "一句话点评"}}

问题：{question}
参考资料要点（评估参考，不是标准答案）：{hints}
系统回答：{answer}"""


_REFUSAL_KEYWORDS = {"不知道", "无法", "不足", "无法回答", "超出", "不涉及"}


@dataclass
class EntryResult:
    qa_id: str
    question: str
    hit: bool
    first_hit_rank: int | None
    retrieved_sources: list[str]
    answer: str | None
    keyword_coverage: float
    citations: list[int]
    context_evidence: float = 0.0
    """块级证据率：期望关键词在被检索到的 chunk 正文中的覆盖率.
    来源级 hit 只说明「命中了正确的源」，本指标进一步验证「命中了正确的段落」——
    宽源（如仅依赖 arXiv 大类）条目来源级命中恒真，靠此指标才能暴露检索精度问题。
    """
    judge_score: float | None = None
    faithfulness: float | None = None
    judge_comment: str | None = None
    is_negative: bool = False  # 负例：expected_sources 为空且期望系统拒绝回答
    guard_blocked: bool | None = None
    """领域守卫是否拦截了该问题。None = 守卫未启用/未判定。

    评估与生产一致：守卫拦截 → 不检索、不生成（用户拿不到任何内容）。
    负例被拦截 = 正确拒绝；正例被拦截 = 守卫误拦（false reject）。
    """
    guard_reason: str = ""

    @property
    def passed(self) -> bool:
        """判断条目是否通过。

        正例（默认）：检索命中 AND LLM评分>=3.0，且未被守卫误拦。
        负例（is_negative=True）：系统正确拒绝回答 —— 守卫拦截 或 模型拒答。
            负例的通过 = 系统没有编造领域外的答案。
        """
        if self.is_negative:
            # 负例通过条件：守卫拦下，或模型自行拒绝
            if self.guard_blocked:
                return True
            if not self.hit and not self.answer:
                return True
            answer_lower = (self.answer or "").lower()
            if any(kw in answer_lower for kw in _REFUSAL_KEYWORDS):
                return True
            # LLM评分<3 → 回答质量差，说明模型没编出像样的答案；
            # 否则：检索命中了、也编出了"像样的"答案 → 负例失败（编造了领域外内容）
            return self.judge_score is not None and self.judge_score < 3.0
        # 正例：守卫误拦直接判失败（生产环境用户拿不到回答）
        if self.guard_blocked:
            return False
        if not self.hit:
            return False
        return not (self.judge_score is not None and self.judge_score < 3.0)


@dataclass
class EvalReport:
    snapshot_id: str
    index_snapshot_id: str
    created_at: str
    retriever_mode: str
    judge_model: str | None
    k: int
    retrieval_hit_rate: float
    mrr: float
    keyword_coverage_avg: float
    citation_rate: float
    judge_score_avg: float | None
    context_evidence_avg: float = 0.0
    faithfulness_avg: float | None = None
    guard_enabled: bool = False
    guard_blocked_count: int = 0
    entries: list[EntryResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.entries:
            return 0.0
        return sum(1 for e in self.entries if e.passed) / len(self.entries)

    def save(self, evals_dir: Path) -> Path:
        evals_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.created_at.replace(":", "").replace("-", "").replace("+", "_")
        path = evals_dir / f"eval_{self.snapshot_id}_{stamp}.json"
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> EvalReport:
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [EntryResult(**item) for item in data.pop("entries")]
        return cls(entries=entries, **data)


def run_eval(
    golden: GoldenSet,
    searcher: HybridSearcher,
    snapshot_id: str,
    index_snapshot_id: str,
    chat_provider: ProviderConfig | None,
    k: int = 5,
    guard_config: GuardConfig | None = None,
) -> EvalReport:
    """逐条评估；chat_provider 为 None 时跳过生成与 judge（只算检索指标）.

    guard_config 为 None 时读取 data/domain_guard.json 的当前配置——评估与生产
    一致：守卫拦截的条目不再检索/生成，从而把「守卫误拦 / 漏拦」纳入指标。
    """
    guard = guard_config if guard_config is not None else load_guard_config()
    results: list[EntryResult] = []
    for qa in golden.entries:
        result = _eval_entry(qa, searcher, chat_provider, k, guard)
        results.append(result)

    total = len(results)
    hit_rate = sum(1 for r in results if r.hit) / total if total else 0.0
    reciprocal = [
        1.0 / r.first_hit_rank
        for r in results
        if r.first_hit_rank is not None
    ]
    mrr = sum(reciprocal) / total if total else 0.0
    coverage_avg = sum(r.keyword_coverage for r in results) / total if total else 0.0
    answered = [r for r in results if r.answer]
    citation_rate = (
        sum(1 for r in answered if r.citations) / len(answered) if answered else 0.0
    )
    judged = [r for r in results if r.judge_score is not None]
    judge_avg = sum(r.judge_score or 0.0 for r in judged) / len(judged) if judged else None

    # 块级证据率：只统计「有期望关键词」的条目（无关键词的条目不可度量，不拉低均值）
    measured = [
        r.context_evidence
        for r, qa in zip(results, golden.entries, strict=True)
        if qa.expected_keywords
    ]
    context_evidence_avg = sum(measured) / len(measured) if measured else 0.0

    judged_faith = [r.faithfulness for r in results if r.faithfulness is not None]
    faithfulness_avg = (
        sum(judged_faith) / len(judged_faith) if judged_faith else None
    )
    blocked_count = sum(1 for r in results if r.guard_blocked)

    return EvalReport(
        snapshot_id=snapshot_id,
        index_snapshot_id=index_snapshot_id,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        retriever_mode=searcher.mode,
        judge_model=chat_provider.model if chat_provider else None,
        k=k,
        retrieval_hit_rate=hit_rate,
        mrr=mrr,
        keyword_coverage_avg=coverage_avg,
        citation_rate=citation_rate,
        judge_score_avg=judge_avg,
        context_evidence_avg=context_evidence_avg,
        faithfulness_avg=faithfulness_avg,
        guard_enabled=guard.enabled,
        guard_blocked_count=blocked_count,
        entries=results,
    )


def _eval_entry(
    qa: GoldenQA,
    searcher: HybridSearcher,
    chat_provider: ProviderConfig | None,
    k: int,
    guard_config: GuardConfig | None = None,
) -> EntryResult:
    is_negative = len(qa.expected_sources) == 0

    # 守卫优先：与生产一致，命中无关项则不再检索/生成。
    guard_blocked: bool | None = None
    guard_reason = ""
    if guard_config is not None and guard_config.enabled:
        decision = check_question(qa.question, guard_config)
        guard_blocked = decision.blocked
        guard_reason = decision.reason
    if guard_blocked:
        return EntryResult(
            qa_id=qa.id,
            question=qa.question,
            hit=False,
            first_hit_rank=None,
            retrieved_sources=[],
            answer=None,
            keyword_coverage=0.0,
            citations=[],
            context_evidence=0.0,
            is_negative=is_negative,
            guard_blocked=True,
            guard_reason=guard_reason,
        )

    hits = searcher.search(qa.question, k=k)
    retrieved_sources = [hit.chunk.source_id for hit in hits]

    hit = False
    first_hit_rank: int | None = None
    if qa.expected_sources:
        for rank, source_id in enumerate(retrieved_sources, start=1):
            if source_id in qa.expected_sources:
                hit = True
                first_hit_rank = rank
                break
    else:
        # 无期望来源的条目：检索命中就算 hit
        hit = bool(hits)
        first_hit_rank = 1 if hits else None

    answer: str | None = None
    citations: list[int] = []
    if chat_provider is not None and hits:
        # 生成失败（超时/模型异常）只降级该条的生成指标，检索层照常计算
        try:
            result = generate_answer(qa.question, hits, chat_provider)
            answer = result.answer
            citations = extract_citations(answer)
        except Exception as exc:
            print(f"  [跳过生成] {qa.id}: {type(exc).__name__}")

    lowered = (answer or "").lower()
    keyword_coverage = (
        sum(1 for kw in qa.expected_keywords if kw in lowered) / len(qa.expected_keywords)
        if qa.expected_keywords
        else 0.0
    )

    # 块级证据率：期望关键词是否真的出现在被检索到的 chunk 正文里
    context_text = " ".join(hit.chunk.search_text for hit in hits).lower()
    context_evidence = (
        sum(1 for kw in qa.expected_keywords if kw in context_text) / len(qa.expected_keywords)
        if qa.expected_keywords
        else 0.0
    )

    judge_score: float | None = None
    faithfulness: float | None = None
    judge_comment: str | None = None
    if chat_provider is not None and answer is not None:
        judge_score, faithfulness, judge_comment = _judge(qa, answer, context_text, chat_provider)

    return EntryResult(
        qa_id=qa.id,
        question=qa.question,
        hit=hit,
        first_hit_rank=first_hit_rank,
        retrieved_sources=retrieved_sources,
        answer=answer,
        keyword_coverage=keyword_coverage,
        citations=citations,
        context_evidence=context_evidence,
        judge_score=judge_score,
        faithfulness=faithfulness,
        judge_comment=judge_comment,
        is_negative=is_negative,
        guard_blocked=guard_blocked,
        guard_reason=guard_reason,
    )


def _judge(
    qa: GoldenQA, answer: str, context_text: str, provider: ProviderConfig
) -> tuple[float | None, float | None, str | None]:
    hints = "、".join([*qa.expected_keywords, *qa.expected_sources]) or "（无）"
    context_snippet = context_text[:1500] if context_text.strip() else "（无检索上下文）"
    prompt = _JUDGE_PROMPT.format(
        question=qa.question,
        hints=hints,
        answer=answer[:2000],
    )
    prompt = f"{prompt}\n\n检索上下文（判断忠实度用）：{context_snippet}"
    try:
        reply = llm_client.chat(
            provider,
            [{"role": "user", "content": prompt}],
            timeout=120.0,
            max_tokens=256,
        )
        return _parse_judge_reply(reply)
    except Exception:
        return None, None, None


def _parse_judge_reply(reply: str) -> tuple[float | None, float | None, str | None]:
    """解析 judge 回复：(质量分, 忠实度分, 点评)；严格 JSON 优先，失败则正则兜底.

    不同模型的 JSON 遵守度不一（dots3 实测会输出 comment 值不带引号的
    非法 JSON），严格解析会让这类模型的评分整批丢失。
    """
    try:
        data = json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
        score = float(data["score"])
        faith = float(data["faithfulness"]) if data.get("faithfulness") is not None else None
        return score, faith, str(data.get("comment", ""))
    except (ValueError, KeyError, TypeError):
        pass
    score_match = re.search(r'"score"\s*:\s*"?([\d.]+)"?', reply)
    if not score_match:
        return None, None, None
    faith_match = re.search(r'"faithfulness"\s*:\s*"?([\d.]+)"?', reply)
    faithfulness = float(faith_match.group(1)) if faith_match else None
    comment_match = re.search(r'"comment"\s*:\s*"?(.*?)"?\s*\}', reply, re.DOTALL)
    comment = comment_match.group(1).strip() if comment_match else ""
    return float(score_match.group(1)), faithfulness, comment


def export_badcases(
    report: EvalReport,
    golden: GoldenSet,
    badcases_dir: Path,
) -> Path | None:
    """把未通过的条目导出为 badcase 文件，供人工复盘后沉淀为新金标条目."""
    failed = [entry for entry in report.entries if not entry.passed]
    if not failed:
        return None
    questions = {qa.id: qa for qa in golden.entries}
    badcases_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.replace(":", "").replace("-", "")
    path = badcases_dir / f"badcases_{report.snapshot_id}_{stamp}.jsonl"
    lines = []
    for entry in failed:
        qa = questions.get(entry.qa_id)
        if entry.is_negative:
            review_hint = (
                "守卫漏拦（false accept）：领域外问题未被守卫拦截、且模型给出了答案——"
                "补充 blocked_topics，或开启语义兜底并上调阈值。"
            )
        elif entry.guard_blocked:
            review_hint = (
                "守卫误拦（false reject）：该条本应可答，却被守卫拦截——"
                "检查 blocked_topics 是否过宽，或在 allowed_topics 补充领域词豁免。"
            )
        elif entry.hit and entry.context_evidence < 0.5:
            review_hint = (
                "来源命中但块级证据率低：命中了正确的源、却没召回含答案的段落——"
                "考虑调整分块 / 提升向量权重 / 换用更具体的 expected_keywords。"
            )
        elif entry.faithfulness is not None and entry.faithfulness < 3.0:
            review_hint = "忠实度偏低：回答超出检索上下文（疑似编造），检查提示词约束与引用要求。"
        else:
            review_hint = "人工复盘：是检索没找到、还是回答质量差？修正后追加到 golden/qa.toml"
        lines.append(
            json.dumps(
                {
                    "qa_id": entry.qa_id,
                    "question": entry.question,
                    "expected_sources": list(qa.expected_sources) if qa else [],
                    "expected_keywords": list(qa.expected_keywords) if qa else [],
                    "retrieved_sources": entry.retrieved_sources,
                    "context_evidence": entry.context_evidence,
                    "answer_excerpt": (entry.answer or "")[:500],
                    "judge_score": entry.judge_score,
                    "faithfulness": entry.faithfulness,
                    "judge_comment": entry.judge_comment,
                    "guard_blocked": entry.guard_blocked,
                    "guard_reason": entry.guard_reason,
                    "review_hint": review_hint,
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
