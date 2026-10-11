"""领域守卫：检索前的领域范围判定与无关项拦截."""

from llm_intel_kb.guard.domain import (
    MODE_ALLOWLIST,
    MODE_BLOCKLIST,
    MODE_HYBRID,
    Embedder,
    GuardConfig,
    GuardDecision,
    append_guard_log,
    check_question,
    clear_guard_log,
    load_guard_config,
    read_guard_log,
    save_guard_config,
    semantic_similarity,
)

__all__ = [
    "MODE_ALLOWLIST",
    "MODE_BLOCKLIST",
    "MODE_HYBRID",
    "Embedder",
    "GuardConfig",
    "GuardDecision",
    "append_guard_log",
    "check_question",
    "clear_guard_log",
    "load_guard_config",
    "read_guard_log",
    "save_guard_config",
    "semantic_similarity",
]