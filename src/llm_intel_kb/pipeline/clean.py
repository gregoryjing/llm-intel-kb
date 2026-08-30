from __future__ import annotations

import hashlib
import html
import re

_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def strip_html(text: str) -> str:
    """去 HTML 标签、反转义实体、规整空白."""
    without_tags = _TAG_RE.sub(" ", text)
    unescaped = html.unescape(without_tags)
    return normalize_text(unescaped)


def normalize_text(text: str) -> str:
    """规整空白：行内多空格合一、行首尾去空、三个以上换行压缩为两个."""
    lines = [line.strip() for line in text.splitlines()]
    joined = "\n".join(lines)
    joined = _SPACES_RE.sub(" ", joined)
    return _BLANK_LINES_RE.sub("\n\n", joined).strip()


def content_hash(title: str, content: str) -> str:
    """文档身份 = 标题 + 正文的 SHA-256：同内容天然同 id，改一字则换新版本."""
    material = f"{title.strip()}\n\n{content.strip()}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    """粗粒度语言检测：CJK 字符占比 >= 15% 判为中文."""
    if not text:
        return "unknown"
    cjk_count = len(_CJK_RE.findall(text))
    return "zh" if cjk_count / len(text) >= 0.15 else "en"
