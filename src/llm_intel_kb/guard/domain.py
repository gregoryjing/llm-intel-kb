"""领域守卫（无关项排除区）：在检索前判断问题是否属于知识库领域，领域外直接拦截.

设计参考业界实践：
- NVIDIA NeMo Guardrails 的 Topic Control：用「允许 / 禁止主题」清单限制对话范围。
- GQR-Bench（Guarded Query Routing）：在「领域内放行 vs 领域外拒答」的路由任务上，
  关键词/轻量分类器相比 LLM-as-guard 有明显延迟优势，且决策可解释。
- 范围判断与内容安全审核应当分离：本模块只回答「知识库有没有相关知识可答」（范围），
  不负责「内容是否安全」（后者属内容审核）。

因此这里采用纯规则的词表匹配：零依赖、离线可用、决策可解释、可审计。
配置持久化到 data/domain_guard.json，拦截决策落盘 data/guard_log.jsonl 供误拒分析。
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from llm_intel_kb.config import get_settings
from llm_intel_kb.index.embeddings import cosine_similarity

MODE_BLOCKLIST = "blocklist"  # 命中禁止清单才拦；同时命中领域词则豁免（默认，最宽松）
MODE_ALLOWLIST = "allowlist"  # 只放行命中允许清单的问题（最严格）
MODE_HYBRID = "hybrid"  # 命中禁止清单、或未命中允许清单 → 拦截（严格）
VALID_MODES = {MODE_BLOCKLIST, MODE_ALLOWLIST, MODE_HYBRID}

# 默认「无关项」词表：覆盖生活服务类领域外问题（与金标集负例 g047~g050/g098~g100/g109 对齐）
DEFAULT_BLOCKED_TOPICS: tuple[str, ...] = (
    # 生活 / 烹饪
    "做菜", "菜谱", "做饭", "炒菜", "番茄炒蛋", "食谱", "烘焙", "美食",
    "家常菜", "下厨", "点菜", "菜市场",
    # 游戏 / 娱乐
    "游戏", "网游", "手游", "电竞", "通关", "攻略",
    "王者荣耀", "原神", "英雄联盟", "绝地求生", "吃鸡", "上分", "开黑", "抽卡", "氪金",
    # 体育
    "足球", "篮球", "世界杯", "球队", "比赛",
    # 天气
    "天气", "气温", "降雨", "台风",
    # 出行 / 票务
    "高铁", "火车票", "机票", "航班", "公交", "地铁",
    # 证件 / 政务
    "护照", "签证", "身份证", "社保", "公积金",
    # 旅游
    "旅游", "景点", "酒店",
    # 职场文书
    "辞职信", "简历", "求职信", "请假条",
    # 娱乐八卦（"情感"单独出现会误伤 NLP 的「情感分析」，故限定为具体诉求）
    "明星", "八卦", "星座", "恋爱", "情感咨询", "情感问题",
    # 其他生活服务
    "彩票", "点外卖",
)

# 默认「领域内」词表：AI/LLM 科研情报范围（用于 allowlist / hybrid 模式）
DEFAULT_ALLOWED_TOPICS: tuple[str, ...] = (
    "ai", "人工智能", "llm", "大模型", "语言模型", "模型", "机器学习", "深度学习", "神经网络",
    "transformer", "注意力", "论文", "研究", "arxiv", "综述", "预印本",
    "rag", "检索", "向量", "embedding", "嵌入", "索引", "重排", "rerank", "混合检索",
    "agent", "智能体", "工具调用", "langchain", "llamaindex", "langgraph",
    "vllm", "ollama", "transformers", "huggingface", "推理", "部署", "显存", "量化", "lora",
    "微调", "训练", "蒸馏", "提示", "prompt", "思维链", "上下文", "token", "moe", "专家",
    "对齐", "rlhf", "dpo", "幻觉", "多模态", "评估", "基准", "数据集", "越狱", "红队",
    "gpu", "芯片", "aigc",
)

DEFAULT_REFUSAL = "该问题超出本知识库的领域范围（{scope}），已作为无关项拦截。请提出该领域内的问题。"

_CONFIG_NAME = "domain_guard.json"
_LOG_NAME = "guard_log.jsonl"
_MAX_LOG_LINES = 500

_log_lock = threading.Lock()

# 语义兜底的领域质心缓存：按「后端 + 词表」缓存，避免每次提问都重算（embedding 是远程调用）
_centroid_lock = threading.Lock()
_centroid_cache: dict[tuple[str, str, tuple[str, ...], str], list[float]] = {}


class Embedder(Protocol):
    """语义兜底所需的最小嵌入接口（由 index.embeddings.EmbeddingBackend 实现）."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class GuardConfig:
    """领域守卫配置（可经 Web / CLI 编辑，持久化到 data/domain_guard.json）."""

    enabled: bool = True
    scope_name: str = "AI/LLM 科研情报"
    scope_description: str = (
        "本知识库聚合 arXiv 论文、AI 资讯与开源框架发布日志，仅回答 AI/LLM 领域的问题。"
    )
    mode: str = MODE_BLOCKLIST
    blocked_topics: list[str] = field(default_factory=lambda: list(DEFAULT_BLOCKED_TOPICS))
    allowed_topics: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_TOPICS))
    refusal_message: str = DEFAULT_REFUSAL
    # 语义兜底：规则词表抓不到同义改写的领域外问题（如「中午吃什么好」不含"做菜"）。
    # 用「与领域质心的向量相似度」作为第二道闸，仅在规则无法判断时生效（默认关闭，
    # 因为需要可用的 embedding 后端；开启后离线/无向量服务时自动跳过）。
    semantic_enabled: bool = False
    semantic_threshold: float = 0.30

    def normalized(self) -> GuardConfig:
        mode = self.mode if self.mode in VALID_MODES else MODE_BLOCKLIST
        threshold = min(max(float(self.semantic_threshold), 0.0), 1.0)
        return GuardConfig(
            enabled=bool(self.enabled),
            scope_name=self.scope_name.strip() or "未命名范围",
            scope_description=self.scope_description.strip(),
            mode=mode,
            blocked_topics=_dedupe(self.blocked_topics),
            allowed_topics=_dedupe(self.allowed_topics),
            refusal_message=self.refusal_message.strip() or DEFAULT_REFUSAL,
            semantic_enabled=bool(self.semantic_enabled),
            semantic_threshold=threshold,
        )

    def refusal_text(self) -> str:
        return self.refusal_message.replace("{scope}", self.scope_name)


@dataclass
class GuardDecision:
    """一次范围判定的结果."""

    blocked: bool
    reason: str
    matched: list[str]
    strategy: str
    scope: str
    score: float | None = None  # 语义兜底的相似度（仅在启用且实际计算时有值）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        cleaned = str(item).strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


# 纯 ASCII 关键词按「整词」匹配：`ai` 用子串匹配会误命中 explain / domain / train 等，
# 从而在 allowlist / hybrid 模式放行领域外问题、或在 blocklist 模式错误豁免。
_ASCII_WORD_RE = re.compile(r"[a-z0-9]+")
_ASCII_TOPIC_RE = re.compile(r"^[a-z0-9]+$")


def _match_topics(text: str, topics: list[str]) -> list[str]:
    """在文本中匹配关键词：ASCII 词按整词（含简单复数），含 CJK 的词按子串.

    中英混合语料下两种匹配各取所长：CJK 无词边界、子串即可；ASCII 需整词，
    否则短词（ai / rag / moe）会命中大量无关英文单词。
    """
    tokens = set(_ASCII_WORD_RE.findall(text))
    matched: list[str] = []
    for topic in topics:
        lowered = topic.lower()
        if _ASCII_TOPIC_RE.match(lowered):
            if lowered in tokens or f"{lowered}s" in tokens:
                matched.append(topic)
        elif lowered in text:
            matched.append(topic)
    return matched


def _config_path() -> Path:
    return get_settings().data_dir / _CONFIG_NAME


def _log_path() -> Path:
    return get_settings().data_dir / _LOG_NAME


def load_guard_config() -> GuardConfig:
    """读取守卫配置；文件不存在时返回默认配置（不写盘）."""
    path = _config_path()
    if not path.exists():
        return GuardConfig()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return GuardConfig()
    if not isinstance(raw, dict):
        return GuardConfig()
    defaults = GuardConfig()
    return GuardConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        scope_name=str(raw.get("scope_name", defaults.scope_name)),
        scope_description=str(raw.get("scope_description", defaults.scope_description)),
        mode=str(raw.get("mode", defaults.mode)),
        blocked_topics=_as_str_list(raw.get("blocked_topics"), list(DEFAULT_BLOCKED_TOPICS)),
        allowed_topics=_as_str_list(raw.get("allowed_topics"), list(DEFAULT_ALLOWED_TOPICS)),
        refusal_message=str(raw.get("refusal_message", defaults.refusal_message)),
        semantic_enabled=bool(raw.get("semantic_enabled", defaults.semantic_enabled)),
        semantic_threshold=float(raw.get("semantic_threshold", defaults.semantic_threshold)),
    ).normalized()


def _as_str_list(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(fallback)
    return [str(item) for item in value]


def save_guard_config(config: GuardConfig) -> GuardConfig:
    """写入守卫配置（自动规整），返回落盘后的配置."""
    normalized = config.normalized()
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(normalized), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return normalized


def check_question(
    question: str,
    config: GuardConfig | None = None,
    embedder: Embedder | None = None,
) -> GuardDecision:
    """判断问题是否属于知识库领域；返回是否拦截及原因.

    - blocklist：命中禁止清单 → 拦截
    - allowlist：未命中允许清单 → 拦截
    - hybrid：命中禁止清单，或未命中允许清单 → 拦截
    - semantic（可选）：规则无法判断时，用「与领域质心的向量相似度」兜底拦截。

    embedder 为 None 或语义未启用时，语义兜底自动跳过（纯规则判定）。
    """
    cfg = (config or load_guard_config()).normalized()
    if not cfg.enabled:
        return GuardDecision(False, "", [], "disabled", cfg.scope_name)

    text = question.strip().lower()
    if not text:
        return GuardDecision(False, "", [], cfg.mode, cfg.scope_name)

    decision = _rule_decision(text, cfg)
    return _apply_semantic(question, cfg, decision, embedder)


def _rule_decision(text: str, cfg: GuardConfig) -> GuardDecision:
    """纯规则判定（词表匹配），不含语义兜底."""
    blocked_hits = _match_topics(text, cfg.blocked_topics)
    allowed_hits = _match_topics(text, cfg.allowed_topics)

    if cfg.mode == MODE_ALLOWLIST:
        if not allowed_hits:
            return GuardDecision(
                True,
                "未命中任何领域内关键词（允许清单模式）",
                [],
                cfg.mode,
                cfg.scope_name,
            )
        return GuardDecision(False, "", allowed_hits, cfg.mode, cfg.scope_name)

    if cfg.mode == MODE_HYBRID:
        if blocked_hits:
            return GuardDecision(
                True, f"命中无关项关键词：{'、'.join(blocked_hits)}", blocked_hits, cfg.mode, cfg.scope_name
            )
        if cfg.allowed_topics and not allowed_hits:
            return GuardDecision(
                True,
                "未命中任何领域内关键词（混合模式）",
                [],
                cfg.mode,
                cfg.scope_name,
            )
        return GuardDecision(False, "", allowed_hits, cfg.mode, cfg.scope_name)

    # blocklist（默认）：命中禁止词才拦；但若同时命中领域词，则领域词优先（豁免放行）。
    # 这样「游戏AI的强化学习」「用强化学习打游戏」这类含领域信号的正当问题不会被
    # 「游戏」这类宽泛禁止词误拦，而「游戏怎么上分」因无任何领域信号仍被拦。
    if blocked_hits:
        if allowed_hits:
            return GuardDecision(
                False,
                f"命中无关词（{'、'.join(blocked_hits)}）但同时命中领域词（{'、'.join(allowed_hits)}），豁免放行",
                blocked_hits,
                cfg.mode,
                cfg.scope_name,
            )
        return GuardDecision(
            True, f"命中无关项关键词：{'、'.join(blocked_hits)}", blocked_hits, cfg.mode, cfg.scope_name
        )
    return GuardDecision(False, "", allowed_hits, cfg.mode, cfg.scope_name)


def _apply_semantic(
    question: str,
    cfg: GuardConfig,
    decision: GuardDecision,
    embedder: Embedder | None,
) -> GuardDecision:
    """规则无法判断时（既未命中禁止词、也未命中领域词），用语义相似度兜底.

    只在规则「不明确」时生效：已拦截、或已明确命中领域词的判定，语义不再覆盖，
    以保证决策可解释、结果确定（规则优先，语义补充）。
    """
    if decision.blocked or not cfg.semantic_enabled or embedder is None or decision.matched:
        return decision
    similarity = semantic_similarity(question, cfg, embedder)
    if similarity is None:
        return decision
    if similarity < cfg.semantic_threshold:
        return GuardDecision(
            True,
            f"语义兜底：与领域质心相似度 {similarity:.2f} < 阈值 {cfg.semantic_threshold:.2f}（疑似领域外）",
            [],
            f"{cfg.mode}+semantic",
            cfg.scope_name,
            similarity,
        )
    return GuardDecision(
        False, "", [], f"{cfg.mode}+semantic", cfg.scope_name, similarity
    )


def _domain_prototypes(cfg: GuardConfig) -> list[str]:
    """构造领域原型文本：范围描述 + 领域词分组拼接（比单词嵌入更稳定）."""
    prototypes: list[str] = []
    if cfg.scope_description:
        prototypes.append(cfg.scope_description)
    topics = cfg.allowed_topics
    for start in range(0, len(topics), 10):
        prototypes.append("、".join(topics[start : start + 10]))
    return prototypes or ["AI/LLM 科研情报"]


def _domain_centroid(cfg: GuardConfig, embedder: Embedder) -> list[float] | None:
    """领域质心（原型向量的均值），按「后端 + 词表」缓存，避免每问重复请求."""
    key = (embedder.name, embedder.model, tuple(cfg.allowed_topics), cfg.scope_description)
    with _centroid_lock:
        cached = _centroid_cache.get(key)
    if cached is not None:
        return cached
    try:
        vectors = embedder.embed_texts(_domain_prototypes(cfg))
    except Exception:
        return None
    if not vectors:
        return None
    dim = len(vectors[0])
    centroid = [sum(vec[i] for vec in vectors) / len(vectors) for i in range(dim)]
    with _centroid_lock:
        _centroid_cache[key] = centroid
    return centroid


def semantic_similarity(
    question: str, cfg: GuardConfig, embedder: Embedder
) -> float | None:
    """问题与领域质心的余弦相似度；embedding 不可用时返回 None（跳过兜底）."""
    centroid = _domain_centroid(cfg, embedder)
    if centroid is None:
        return None
    try:
        vector = embedder.embed_query(question)
    except Exception:
        return None
    return cosine_similarity(vector, centroid)


def append_guard_log(question: str, decision: GuardDecision, source: str = "ask") -> None:
    """追加一条拦截决策日志（含放行与拦截），供误拒 / 漏拦分析."""
    if decision.strategy == "disabled":
        return
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": question.strip()[:500],
        "blocked": decision.blocked,
        "reason": decision.reason,
        "matched": decision.matched,
        "strategy": decision.strategy,
        "scope": decision.scope,
        "source": source,
    }
    path = _log_path()
    with _log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        _trim_log(path)


def _trim_log(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= _MAX_LOG_LINES:
        return
    path.write_text("\n".join(lines[-_MAX_LOG_LINES:]) + "\n", encoding="utf-8")


def read_guard_log(limit: int = 100, only_blocked: bool = False) -> list[dict[str, Any]]:
    """读取最近的守卫决策日志（倒序，最新在前）."""
    path = _log_path()
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if only_blocked and not item.get("blocked"):
            continue
        records.append(item)
    records.reverse()
    return records[:limit]


def clear_guard_log() -> int:
    """清空守卫日志，返回清空的条数."""
    path = _log_path()
    if not path.exists():
        return 0
    count = len(path.read_text(encoding="utf-8").splitlines())
    path.write_text("", encoding="utf-8")
    return count