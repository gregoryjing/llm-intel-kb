"""纯 Python BM25 词法检索：中英混合分词（CJK 二元 + 拉丁词），无外部依赖."""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from dataclasses import dataclass

_LATIN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """小写化后切词：拉丁字母/数字连续段为词，CJK 逐字 + 相邻二元."""
    lowered = text.lower()
    tokens = _LATIN_RE.findall(lowered)
    cjk_chars = _CJK_RE.findall(lowered)
    tokens.extend(cjk_chars)
    tokens.extend(a + b for a, b in itertools.pairwise(cjk_chars))
    return tokens


@dataclass
class LexicalIndex:
    """BM25 倒排索引（构建后只读）."""

    doc_tokens: dict[str, list[str]]  # chunk_id -> tokens
    doc_lengths: dict[str, int]
    df: Counter[str]  # term -> 出现过的 chunk 数
    avg_length: float

    @classmethod
    def build(cls, docs: dict[str, str]) -> LexicalIndex:
        doc_tokens: dict[str, list[str]] = {}
        doc_lengths: dict[str, int] = {}
        df: Counter[str] = Counter()
        for chunk_id, text in docs.items():
            tokens = tokenize(text)
            doc_tokens[chunk_id] = tokens
            doc_lengths[chunk_id] = len(tokens)
            for term in set(tokens):
                df[term] += 1
        avg = sum(doc_lengths.values()) / len(doc_lengths) if doc_lengths else 0.0
        return cls(doc_tokens=doc_tokens, doc_lengths=doc_lengths, df=df, avg_length=avg)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """返回 (chunk_id, bm25 分数) 前 k 个."""
        if not self.doc_tokens:
            return []
        query_terms = tokenize(query)
        scores: dict[str, float] = {}
        for term in query_terms:
            postings = self._postings(term)
            if not postings:
                continue
            idf = math.log(1 + (len(self.doc_tokens) - len(postings) + 0.5) / (len(postings) + 0.5))
            for chunk_id, tf in postings.items():
                length_norm = 1 - B + B * (self.doc_lengths[chunk_id] / self.avg_length if self.avg_length else 1.0)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (tf * (K1 + 1)) / (tf + K1 * length_norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def _postings(self, term: str) -> dict[str, int]:
        return {
            chunk_id: tokens.count(term)
            for chunk_id, tokens in self.doc_tokens.items()
            if term in tokens
        }
