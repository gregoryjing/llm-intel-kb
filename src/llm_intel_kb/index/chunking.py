"""结构感知切块：按段落边界聚合，标题作为块前缀，相邻块带重叠."""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm_intel_kb.pipeline.models import CorpusDocument

TARGET_CHARS = 900
OVERLAP_CHARS = 150
MIN_CHARS = 120

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    chunk_index: int
    title: str
    text: str
    source_id: str
    url: str
    published_at: str | None
    topics: tuple[str, ...]
    summary: str | None

    @property
    def search_text(self) -> str:
        """用于检索的文本：标题 + 主题词 + 摘要 + 正文块."""
        parts = [self.title]
        if self.topics:
            parts.append(" ".join(self.topics))
        if self.summary:
            parts.append(self.summary)
        parts.append(self.text)
        return "\n".join(parts)


def flatten_markdown_tables(text: str) -> str:
    """把 Markdown 表格转成自然语言行，避免切块时表格被切碎.

    输入（表格）：
        | 模型 | MMLU | HumanEval |
        |------|------|-----------|
        | GPT-4o | 88.7 | 90.2 |
        | Qwen3 | 86.1 | 85.0 |

    输出（自然语言行）：
        表格：模型 / MMLU / HumanEval
        GPT-4o：MMLU 88.7，HumanEval 90.2
        Qwen3：MMLU 86.1，HumanEval 85.0

    这样切块后每一行都是完整的"属性-值"对，检索时不会因为切到半张表而丢失行列对应关系。
    """
    lines = text.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 检测表格行：以 | 开头和结尾，且不是分隔行
        if line.strip().startswith("|") and line.strip().endswith("|"):
            # 收集整个表格
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            # 解析表格
            rows = []
            for tl in table_lines:
                # 去掉首尾的 |
                cells = [c.strip() for c in tl.strip("|").split("|")]
                rows.append(cells)
            if len(rows) >= 2:
                # 检测分隔行（第二行全是 --- 或 :---: 之类）
                is_header_sep = all(
                    re.match(r"^[:\-\s]+$", cell) for cell in rows[1]
                )
                if is_header_sep:
                    headers = rows[0]
                    data_rows = rows[2:] if len(rows) > 2 else []
                    # 表格标题行
                    result.append("表格：" + " / ".join(headers))
                    for dr in data_rows:
                        # 每行转成 "第一列：列2 值2，列3 值3"
                        if dr:
                            first = dr[0] if dr else ""
                            rest = []
                            for j in range(1, min(len(headers), len(dr))):
                                rest.append(f"{headers[j]} {dr[j]}")
                            if rest:
                                result.append(f"{first}：{'，'.join(rest)}")
                            else:
                                result.append(first)
                    continue
            # 不是标准表格（无分隔行），原样保留
            for tl in table_lines:
                result.append(tl)
        else:
            result.append(line)
            i += 1
    return "\n".join(result)


def chunk_document(doc: CorpusDocument) -> list[Chunk]:
    """把一篇文档切成若干块：段落聚合 + 句子级重叠，短文成单块."""
    processed = flatten_markdown_tables(doc.content)
    paragraphs = [p for p in processed.split("\n\n") if p.strip()]
    blocks: list[str] = []
    buffer: list[str] = []
    buffer_chars = 0
    for para in paragraphs:
        para_len = len(para)
        if buffer_chars + para_len > TARGET_CHARS and buffer:
            blocks.append("\n\n".join(buffer))
            # 重叠：保留上一块结尾的句子，保持语义连续
            tail = _tail_text(blocks[-1], OVERLAP_CHARS)
            buffer = [tail] if tail else []
            buffer_chars = len(tail)
        buffer.append(para)
        buffer_chars += para_len
    if buffer:
        blocks.append("\n\n".join(buffer))

    chunks: list[Chunk] = []
    topics = tuple(doc.enrichment.topics) if doc.enrichment else ()
    summary = doc.enrichment.summary if doc.enrichment else None
    for index, block in enumerate(blocks):
        if len(block) < MIN_CHARS and len(blocks) > 1:
            continue  # 过短的尾块并入不了就丢弃（信息量太低）
        chunks.append(
            Chunk(
                id=f"{doc.id}:{index}",
                doc_id=doc.id,
                chunk_index=index,
                title=doc.title,
                text=block,
                source_id=doc.source_id,
                url=doc.url,
                published_at=doc.published_at,
                topics=topics,
                summary=summary,
            )
        )
    # 单块过短也保留（短文档本身就有价值，如 release note）
    if not chunks:
        chunks.append(
            Chunk(
                id=f"{doc.id}:0",
                doc_id=doc.id,
                chunk_index=0,
                title=doc.title,
                text=doc.content,
                source_id=doc.source_id,
                url=doc.url,
                published_at=doc.published_at,
                topics=topics,
                summary=summary,
            )
        )
    return chunks


def chunk_documents(docs: list[CorpusDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


def _tail_text(text: str, max_chars: int) -> str:
    """取文本结尾的完整句子（最多 max_chars），作为下一块的重叠前缀."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    cut = min(
        [pos for pos in (tail.find(". "), tail.find("。"), tail.find("! "), tail.find("? ")) if pos >= 0],
        default=-1,
    )
    return tail[cut + 1 :].strip() if cut >= 0 else tail
