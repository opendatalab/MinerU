"""FTS5 manager and CJK tokenization."""

from __future__ import annotations

from typing import cast

try:
    import jieba
except ImportError:
    jieba = None  # type: ignore[assignment]

from ...types import Tier
from ..rows import FtsContentSearchRow, FtsFilenameSearchRow
from .db import DatabaseManager

FTS_SEP = ""


# ── tokenization ───────────────────────────────────────────────────


def tokenize_for_index(text: str) -> str:
    """Tokenize text for FTS storage.  Uses jieba + Unit Separator."""
    if not text or jieba is None:
        return text
    return FTS_SEP.join(jieba.cut(text))


def tokenize_for_query(text: str) -> str:
    """Tokenize query for FTS search.  Uses jieba + space."""
    if not text or jieba is None:
        return text
    return " ".join(jieba.cut(text))


def strip_sep(text: str) -> str:
    """Remove FTS separator from snippet output."""
    return text.replace(FTS_SEP, "")


# ── FTS manager ────────────────────────────────────────────────────


class FTSManager:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    # ── fts_contents ────────────────────────────────────────────

    async def replace(
        self,
        *,
        sha256: str,
        tier: Tier | None,
        text: str,
        title: str,
        author: str,
    ) -> None:
        """Insert or replace content for a doc.  Caller is responsible
        for tier-gating (only call if new tier >= current)."""
        tokenized = tokenize_for_index(text)
        await self.db.execute_atomic(
            [
                ("DELETE FROM fts_contents WHERE sha256=?", (sha256,)),
                (
                    "INSERT INTO fts_contents (sha256, tier, text, title, author) VALUES (?, ?, ?, ?, ?)",
                    (sha256, tier, tokenized, title or "", author or ""),
                ),
            ]
        )

    async def search(self, query: str, limit: int = 200) -> list[FtsContentSearchRow]:
        tokens = _sanitize_query_tokens(tokenize_for_query(query))
        if not tokens:
            return []
        fts_query = " ".join(tokens)
        try:
            return cast(
                list[FtsContentSearchRow],
                await self.db.fetchall(
                    "SELECT sha256, title, author, tier, "
                    "snippet(fts_contents, 2, '<mark>', '</mark>', '...', 40) AS snippet, rank "
                    "FROM fts_contents WHERE fts_contents MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (fts_query, limit),
                ),
            )
        except Exception:
            return []

    async def get_tier(self, sha256: str) -> Tier | None:
        row = await self.db.fetchone("SELECT tier FROM fts_contents WHERE sha256=?", (sha256,))
        return cast(Tier | None, row["tier"]) if row else None

    async def delete(self, sha256: str) -> None:
        await self.db.execute("DELETE FROM fts_contents WHERE sha256=?", (sha256,))

    # ── fts_filenames ────────────────────────────────────────────

    async def upsert_filename(self, file_id: int, filename: str, ext: str) -> None:
        await self.db.execute_atomic(
            [
                ("DELETE FROM fts_filenames WHERE file_id=?", (file_id,)),
                (
                    "INSERT INTO fts_filenames (file_id, filename, ext) VALUES (?, ?, ?)",
                    (file_id, tokenize_for_index(filename), ext),
                ),
            ]
        )

    async def search_filenames(self, query: str, limit: int | None = 50) -> list[FtsFilenameSearchRow]:
        """仅对文件名查询的最后一个词默认启用前缀匹配；None 获取完整候选快照。"""
        tokens = _sanitize_query_tokens(tokenize_for_query(query))
        if not tokens:
            return []
        if not query.rstrip().endswith("*"):
            tokens[-1] += "*"
        fts_query = " ".join(tokens)
        sql = (
            "SELECT file_id, ext, "
            "snippet(fts_filenames, 1, '<mark>', '</mark>', '...', 40) AS snippet "
            "FROM fts_filenames WHERE fts_filenames MATCH ? ORDER BY rank, file_id"
        )
        params: tuple[str | int, ...] = (fts_query,)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        try:
            return cast(
                list[FtsFilenameSearchRow],
                await self.db.fetchall(sql, params),
            )
        except Exception:
            return []

    async def delete_filename(self, file_id: int) -> None:
        await self.db.execute("DELETE FROM fts_filenames WHERE file_id=?", (file_id,))


# ── helpers ────────────────────────────────────────────────────────


def _sanitize_query_tokens(tokenized: str) -> list[str]:
    """Remove special characters and FTS5 operators from query tokens."""
    safe: list[str] = []
    for t in tokenized.split():
        t = t.strip().replace('"', "").replace(".", "").replace("-", "")
        if t and t.upper() not in ("AND", "OR", "NOT", "NEAR"):
            safe.append(t)
    return safe


MAX_FTS_CHARS = 30_000
FTS_HEAD_HALF = 15_000


def truncate_head_tail(text: str, max_chars: int = MAX_FTS_CHARS, half: int = FTS_HEAD_HALF) -> str:
    """超长正文仅保留首尾内容，沿用原有 FTS 长度与省略标记。"""
    if len(text) <= max_chars:
        return text
    return text[:half] + "\n...\n" + text[-half:]
