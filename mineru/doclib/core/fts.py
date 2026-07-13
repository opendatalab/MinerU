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
        tier: Tier,
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
        return cast(Tier, row["tier"]) if row else None

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

    async def search_filenames(self, query: str, limit: int = 50) -> list[FtsFilenameSearchRow]:
        tokens = _sanitize_query_tokens(tokenize_for_query(query))
        if not tokens:
            return []
        fts_query = " ".join(tokens)
        try:
            return cast(
                list[FtsFilenameSearchRow],
                await self.db.fetchall(
                    "SELECT file_id, ext, "
                    "snippet(fts_filenames, 1, '<mark>', '</mark>', '...', 40) AS snippet "
                    "FROM fts_filenames WHERE fts_filenames MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (fts_query, limit),
                ),
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
