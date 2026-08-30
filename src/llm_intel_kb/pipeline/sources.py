from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import feedparser
import httpx2 as httpx

from llm_intel_kb.pipeline.clean import normalize_text, strip_html
from llm_intel_kb.pipeline.models import RawDocument, SourceSpec

DEFAULT_SOURCES_FILE = Path("data_sources.toml")
VALID_SOURCE_TYPES = {"rss", "github-releases"}
_USER_AGENT = "llm-intel-kb/0.1 (personal research pipeline)"


@dataclass
class FetchResult:
    docs: list[RawDocument] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class StagingData:
    docs: list[RawDocument]
    source_ids: set[str]


def load_sources(path: Path | None) -> list[SourceSpec]:
    """读取数据源注册表；文件不存在时返回空列表."""
    source_path = path or DEFAULT_SOURCES_FILE
    if not source_path.exists():
        return []
    with source_path.open("rb") as fh:
        raw = tomllib.load(fh)
    specs: list[SourceSpec] = []
    for section_name, section in raw.items():
        if not isinstance(section, dict):
            continue
        spec = _spec_from_section(section_name, section)
        _validate_spec(spec)
        specs.append(spec)
    return specs


def _spec_from_section(section_name: str, section: dict[str, object]) -> SourceSpec:
    def opt_str(key: str) -> str | None:
        value = section.get(key)
        return value if isinstance(value, str) else None

    def opt_int(key: str, default: int) -> int:
        value = section.get(key)
        return value if isinstance(value, int) else default

    type_value = section.get("type")
    return SourceSpec(
        id=section_name,
        type=type_value if isinstance(type_value, str) else "",
        url=opt_str("url"),
        repo=opt_str("repo"),
        title=opt_str("title"),
        max_items=opt_int("max_items", 15),
        min_chars=opt_int("min_chars", 200),
        enabled=bool(section.get("enabled", True)),
    )


def _validate_spec(spec: SourceSpec) -> None:
    if spec.type not in VALID_SOURCE_TYPES:
        raise ValueError(f"数据源 {spec.id} 的 type 非法：{spec.type}（应为 rss / github-releases）")
    if spec.type == "rss" and not spec.url:
        raise ValueError(f"数据源 {spec.id} 类型为 rss 但缺少 url")
    if spec.type == "github-releases" and not spec.repo:
        raise ValueError(f"数据源 {spec.id} 类型为 github-releases 但缺少 repo")


def fetch_source(spec: SourceSpec, timeout: float = 30.0) -> FetchResult:
    """抓取单个数据源；网络错误不抛出，收集到 errors 里."""
    try:
        if spec.type == "rss":
            return _fetch_rss(spec, timeout)
        if spec.type == "github-releases":
            return _fetch_github_releases(spec, timeout)
        return FetchResult(errors=[f"未知源类型：{spec.type}"])
    except Exception as exc:
        return FetchResult(errors=[f"{type(exc).__name__}: {exc}"])


def _fetch_rss(spec: SourceSpec, timeout: float) -> FetchResult:
    assert spec.url is not None
    response = httpx.get(
        spec.url,
        headers={"User-Agent": _USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
    )
    response.raise_for_status()
    feed = feedparser.parse(response.text)
    docs: list[RawDocument] = []
    now = _now_iso()
    for entry in feed.entries[: spec.max_items]:
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("link") or "").strip()
        if not (title and url):
            continue
        raw_content = str(entry.get("summary") or entry.get("description") or "")
        docs.append(
            RawDocument(
                source_id=spec.id,
                source_type="rss",
                url=url,
                title=title,
                content=strip_html(raw_content),
                published_at=entry.get("published") or entry.get("updated"),
                fetched_at=now,
            )
        )
    return FetchResult(docs=docs)


def _fetch_github_releases(spec: SourceSpec, timeout: float) -> FetchResult:
    assert spec.repo is not None
    api_url = f"https://api.github.com/repos/{spec.repo}/releases"
    response = httpx.get(
        api_url,
        params={"per_page": spec.max_items},
        headers={"User-Agent": _USER_AGENT, "Accept": "application/vnd.github+json"},
        timeout=timeout,
        follow_redirects=True,
    )
    response.raise_for_status()
    docs: list[RawDocument] = []
    now = _now_iso()
    for item in response.json():
        body = normalize_text(str(item.get("body") or ""))
        tag = str(item.get("tag_name") or "").strip()
        html_url = str(item.get("html_url") or "").strip()
        if not (tag and html_url):
            continue
        docs.append(
            RawDocument(
                source_id=spec.id,
                source_type="github-releases",
                url=html_url,
                title=f"{spec.repo} {tag}",
                content=body,
                published_at=item.get("published_at"),
                fetched_at=now,
            )
        )
    return FetchResult(docs=docs)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
