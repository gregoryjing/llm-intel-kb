from __future__ import annotations

from pydantic import BaseModel, Field


class SourceSpec(BaseModel):
    """数据源注册表中的一条源配置（data_sources.toml 的一个段）."""

    id: str
    type: str = Field(description="rss | github-releases")
    url: str | None = Field(default=None, description="rss 类型的订阅地址")
    repo: str | None = Field(default=None, description="github-releases 类型的 owner/repo")
    title: str | None = Field(default=None, description="人类可读名称")
    max_items: int = Field(default=15, description="单次抓取条数上限")
    min_chars: int = Field(default=200, description="清洗后内容最短字符数，低于则丢弃")
    enabled: bool = Field(default=True)


class RawDocument(BaseModel):
    """抓取到的原始文档（未清洗、未去重）."""

    source_id: str
    source_type: str
    url: str
    title: str
    content: str
    published_at: str | None = None
    fetched_at: str


class Enrichment(BaseModel):
    """LLM 富化产物：中文摘要 + 英文主题词."""

    summary: str
    topics: list[str]
    language: str
    model: str
    enriched_at: str


class CorpusDocument(BaseModel):
    """语料库中的正式文档：以内容哈希为 id，不可变、可回溯."""

    id: str
    source_id: str
    source_type: str
    url: str
    alt_urls: list[str] = Field(default_factory=list, description="同内容跨源镜像地址")
    title: str
    content: str
    published_at: str | None = None
    fetched_at: str
    content_chars: int
    language: str
    enrichment: Enrichment | None = None


class Snapshot(BaseModel):
    """数据集版本快照：一份文档 id 清单 + 变更统计."""

    id: str
    created_at: str
    parent: str | None
    doc_ids: list[str]
    stats: dict[str, int]
