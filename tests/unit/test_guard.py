"""领域守卫单元测试：拦截判定、模式差异、领域词豁免、日志读写."""

from __future__ import annotations

from pathlib import Path

from llm_intel_kb.guard.domain import (
    MODE_ALLOWLIST,
    MODE_BLOCKLIST,
    MODE_HYBRID,
    GuardConfig,
    check_question,
)


def _cfg(**kwargs: object) -> GuardConfig:
    base = {
        "enabled": True,
        "scope_name": "AI/LLM 科研情报",
        "mode": MODE_BLOCKLIST,
        "blocked_topics": ["做菜", "游戏", "天气"],
        "allowed_topics": ["ai", "强化学习", "rag"],
    }
    base.update(kwargs)
    return GuardConfig(**base)  # type: ignore[arg-type]


class TestBlocklistMode:
    def test_off_domain_blocked(self) -> None:
        assert check_question("今天做菜放多少盐？", _cfg()).blocked is True
        assert check_question("游戏怎么上分", _cfg()).blocked is True
        assert check_question("今天天气如何", _cfg()).blocked is True

    def test_in_domain_passes(self) -> None:
        assert check_question("RAG 检索增强生成有什么新进展", _cfg()).blocked is False
        assert check_question("vLLM 最近更新了什么", _cfg()).blocked is False

    def test_domain_rescue_overrides_broad_blockword(self) -> None:
        """含领域信号的正当问题不应被宽泛禁止词误拦."""
        decision = check_question("用强化学习训练游戏智能体有哪些论文", _cfg())
        assert decision.blocked is False
        assert "豁免" in decision.reason

    def test_disabled_never_blocks(self) -> None:
        assert check_question("今天做菜放多少盐", _cfg(enabled=False)).blocked is False


class TestAllowAndHybridModes:
    def test_allowlist_blocks_without_domain_term(self) -> None:
        cfg = _cfg(mode=MODE_ALLOWLIST)
        assert check_question("帮我写首诗", cfg).blocked is True
        assert check_question("RAG 是什么", cfg).blocked is False

    def test_hybrid_blocks_both_ways(self) -> None:
        cfg = _cfg(mode=MODE_HYBRID)
        # 命中禁止词 → 拦
        assert check_question("游戏怎么玩", cfg).blocked is True
        # 未命中允许词（即使不命中禁止词）→ 拦
        assert check_question("帮我写首诗", cfg).blocked is True
        # 命中允许词且不命中禁止词 → 放行
        assert check_question("RAG 是什么", cfg).blocked is False


class TestConfigNormalization:
    def test_blank_scope_and_refusal_fall_back(self) -> None:
        cfg = _cfg(scope_name="  ", refusal_message="").normalized()
        assert cfg.scope_name
        assert cfg.refusal_message

    def test_invalid_mode_falls_back_to_blocklist(self) -> None:
        assert _cfg(mode="nonsense").normalized().mode == MODE_BLOCKLIST

    def test_refusal_text_substitutes_scope(self) -> None:
        cfg = _cfg(refusal_message="超出{scope}范围")
        assert "AI/LLM 科研情报" in cfg.refusal_text()


class TestAsciiWordBoundary:
    """ASCII 关键词必须整词匹配：短词 `ai` 不能命中 explain / domain / train 等."""

    def test_ai_does_not_match_substrings(self) -> None:
        cfg = _cfg(mode=MODE_ALLOWLIST, allowed_topics=["ai"])
        # explain / domain / train / maintain 均含 "ai" 子串，但都不是领域信号
        assert check_question("please explain the domain and train the model", cfg).blocked is True
        assert check_question("how to maintain a database", cfg).blocked is True

    def test_ai_matches_whole_word_and_plural(self) -> None:
        cfg = _cfg(mode=MODE_ALLOWLIST, allowed_topics=["ai"])
        assert check_question("what is AI", cfg).blocked is False
        assert check_question("AI 是什么", cfg).blocked is False

    def test_cjk_topic_still_substring_matches(self) -> None:
        cfg = _cfg(mode=MODE_ALLOWLIST, allowed_topics=["强化学习"])
        assert check_question("强化学习有哪些应用", cfg).blocked is False


class _FakeEmbedder:
    """按问题返回固定向量：用于验证语义兜底的相似度分支（领域质心固定为 [1, 0]）."""

    def __init__(self, query_vector: list[float]) -> None:
        self._query_vector = query_vector

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return "fake-embed"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._query_vector


_SEMANTIC_CFG: dict[str, object] = {
    "blocked_topics": ["做菜"],
    "allowed_topics": ["rag"],
    "semantic_threshold": 0.30,
}


class TestSemanticFallback:
    """规则无法判断时，用「与领域质心的向量相似度」兜底."""

    def test_disabled_keeps_rule_decision(self) -> None:
        cfg = _cfg(semantic_enabled=False, **_SEMANTIC_CFG)
        decision = check_question("中午吃什么好", cfg, _FakeEmbedder([0.0, 1.0]))
        assert decision.blocked is False

    def test_below_threshold_blocks(self) -> None:
        cfg = _cfg(semantic_enabled=True, **_SEMANTIC_CFG)
        decision = check_question("中午吃什么好", cfg, _FakeEmbedder([0.0, 1.0]))
        assert decision.blocked is True
        assert decision.strategy == "blocklist+semantic"
        assert decision.score == 0.0

    def test_above_threshold_passes(self) -> None:
        cfg = _cfg(semantic_enabled=True, **_SEMANTIC_CFG)
        decision = check_question("中午吃什么好", cfg, _FakeEmbedder([1.0, 0.0]))
        assert decision.blocked is False
        assert decision.score == 1.0

    def test_no_embedder_skips_semantic(self) -> None:
        cfg = _cfg(semantic_enabled=True, **_SEMANTIC_CFG)
        assert check_question("中午吃什么好", cfg, None).blocked is False

    def test_rule_match_not_overridden_by_semantic(self) -> None:
        """已命中领域词 → 规则已明确放行，语义不再覆盖（即使向量判为领域外）."""
        cfg = _cfg(semantic_enabled=True, **_SEMANTIC_CFG)
        decision = check_question("RAG 检索增强生成是什么", cfg, _FakeEmbedder([0.0, 1.0]))
        assert decision.blocked is False
        assert decision.score is None

    def test_rule_block_not_overridden_by_semantic(self) -> None:
        """规则已拦截 → 保持拦截，语义不参与."""
        cfg = _cfg(semantic_enabled=True, **_SEMANTIC_CFG)
        decision = check_question("做菜放多少盐", cfg, _FakeEmbedder([1.0, 0.0]))
        assert decision.blocked is True
        assert decision.strategy == MODE_BLOCKLIST


def test_guard_log_roundtrip(tmp_path: Path, monkeypatch) -> None:
    """拦截决策可落盘并按倒序读回、可按 blocked 过滤."""
    from llm_intel_kb.guard import domain as guard

    class _Settings:
        data_dir = tmp_path

    monkeypatch.setattr(guard, "get_settings", lambda: _Settings())

    cfg = _cfg()
    guard.append_guard_log("今天做菜放多少盐", check_question("今天做菜放多少盐", cfg))
    guard.append_guard_log("RAG 是什么", check_question("RAG 是什么", cfg))

    records = guard.read_guard_log()
    assert len(records) == 2
    assert records[0]["question"] == "RAG 是什么"  # 最新在前
    blocked_only = guard.read_guard_log(only_blocked=True)
    assert len(blocked_only) == 1
    assert blocked_only[0]["blocked"] is True
    assert guard.clear_guard_log() == 2
    assert guard.read_guard_log() == []