"""Parse service: file ingestion, parse request/acquire/process."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from ...errors import MineruError
from ...parser.base import ParseResult
from ...schema.middle_json import MIDDLE_JSON_SCHEMA_VERSION
from ...types import TIER_ORDER, PageInfo, Tier
from ...utils.image_payload import parse_image_data_uri
from ..constants import ALLOWED_EXTENSIONS, TEXT_EXTENSIONS
from ..core.db import DatabaseManager
from ..core.file_io import FileStat, compute_sha256, extract_metadata, get_file_stat
from ..core.fts import FTSManager
from ..rows import FileRow, PageCountRow, ParseBatchRow, ParseRow, Sha256Row, ShortIdRow, WatchTargetRow
from ..types import (
    FILE_STATUS_ACTIVE,
    FILE_STATUS_DELETED,
    FILE_STATUS_UNREACHABLE,
    PARSE_STATUS_DONE,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PARSING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_SUPERSEDED,
    RULE_TYPE_PARSING_RULE,
    WATCH_STATUS_UNREACHABLE,
    FileInfo,
    ParseResponse,
)
from .config_svc import ConfigService


def _now_ms() -> int:
    return int(time.time() * 1000)


class ParseFailure(MineruError):
    """Raised when a parse cannot be completed.  Carries an error code for the parses row."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)
        self.message = message


MAX_FTS_CHARS = 30_000
FTS_HEAD_HALF = 15_000

DOC_TYPE_BY_EXT = {
    "md": "markdown",
    "markdown": "markdown",
    "htm": "html",
    "html": "html",
}


FileRefreshStatus = Literal["known", "new", "changed", "missing", "deleted", "unreachable", "unsupported", "error"]


@dataclass(frozen=True)
class FileRefreshResult:
    file: FileInfo | None
    status: FileRefreshStatus

    @property
    def needs_ingest(self) -> bool:
        return self.file is not None and self.file.sha256 is None


# ── range helpers ──────────────────────────────────────────────────


def parse_page_range_set(page_range: str) -> set[int]:
    """Parse a page_range string like '1~5,10~15' into 1-based page numbers."""
    page_numbers: set[int] = set()
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "~" in part:
            a, b = part.split("~", 1)
            page_numbers.update(range(int(a.strip()), int(b.strip()) + 1))
        else:
            page_numbers.add(int(part))
    return page_numbers


def filter_pages_by_user_range(pages: list[PageInfo], page_range: str) -> list[PageInfo]:
    """Filter 0-based PageInfo objects by a user-facing 1-based page range."""
    requested_page_numbers = parse_page_range_set(page_range)
    return [page for page in pages if page.page_idx + 1 in requested_page_numbers]


def page_range_covered(request_page_range: str, done_batches: list[ParseRow]) -> bool:
    """Check whether request_page_range is fully covered by done batches.
    Batches are sorted by done_at DESC so newer batches take precedence."""
    needed_page_numbers = parse_page_range_set(request_page_range)
    covered_page_numbers: set[int] = set()
    for b in sorted(done_batches, key=lambda r: r.get("done_at") or 0, reverse=True):
        covered_page_numbers |= parse_page_range_set(b["page_range"])
        if needed_page_numbers <= covered_page_numbers:
            return True
    return False


def page_range_uncovered(request_page_range: str, done_batches: list[ParseRow]) -> set[int]:
    """Return the subset of request_page_range NOT covered by any done batch."""
    needed_page_numbers = parse_page_range_set(request_page_range)
    covered_page_numbers: set[int] = set()
    for b in done_batches:
        covered_page_numbers |= parse_page_range_set(b["page_range"])
    return needed_page_numbers - covered_page_numbers


def _page_numbers_to_range_str(page_numbers: set[int]) -> str:
    """Convert a set of page numbers (1-based) to a compact range string."""
    if not page_numbers:
        return ""
    sorted_page_numbers = sorted(page_numbers)
    ranges: list[str] = []
    start = sorted_page_numbers[0]
    end = start
    for page_no in sorted_page_numbers[1:]:
        if page_no == end + 1:
            end = page_no
        else:
            ranges.append(f"{start}~{end}" if start != end else str(start))
            start = page_no
            end = page_no
    ranges.append(f"{start}~{end}" if start != end else str(start))
    return ",".join(ranges)


def expand_page_range(page_range: str | None, page_count: int) -> str:
    """Expand shorthand page_range like 'all' or negative ranges like '-5~-1'
    into positive page ranges.  Returns '1~{page_count}' for None/empty."""
    if not page_range or page_range.strip() == "all":
        return f"1~{page_count}"
    result: list[str] = []
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "~" in part:
            a, b = part.split("~", 1)
            a, b = a.strip(), b.strip()
            start = int(a)
            end = int(b)
            # negative index: -5 means page_count-5+1
            if start < 0:
                start = page_count + start + 1
            if end < 0:
                end = page_count + end + 1
            result.append(f"{start}~{end}")
        else:
            p = int(part.strip())
            if p < 0:
                p = page_count + p + 1
            result.append(str(p))
    return ",".join(result)


def default_parse_range(page_count: int | None) -> str:
    """Return the default page range for active reading."""
    if page_count and page_count > 0:
        return f"1~{min(page_count, 10)}"
    return "1"


async def ensure_doc_record(
    db: DatabaseManager,
    *,
    sha256: str,
    size_bytes: int,
    file_type: str,
    page_count: int | None,
    title: str | None,
    author: str | None,
    subject: str | None,
    keywords: str | None,
    error_code: str | None,
    error_msg: str | None,
    first_seen_at: int,
    updated_at: int,
) -> None:
    existing = cast(ShortIdRow | None, await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (sha256,)))
    if existing is not None:
        return

    for length in range(7, len(sha256) + 1):
        short_id = sha256[:length]
        await db.execute(
            "INSERT OR IGNORE INTO docs (sha256, short_id, size_bytes, file_type, page_count, "
            "title, author, subject, keywords, error_code, error_msg, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sha256,
                short_id,
                size_bytes,
                file_type,
                page_count,
                title,
                author,
                subject,
                keywords,
                error_code,
                error_msg,
                first_seen_at,
                updated_at,
            ),
        )
        inserted = cast(ShortIdRow | None, await db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (sha256,)))
        if inserted is not None:
            return
        conflict = cast(Sha256Row | None, await db.fetchone("SELECT sha256 FROM docs WHERE short_id=?", (short_id,)))
        if conflict is not None and conflict["sha256"] != sha256:
            continue
        break

    raise RuntimeError(f"Failed to allocate unique short_id for document {sha256}")


# ── text truncation ────────────────────────────────────────────────


def truncate_head_tail(text: str, max_chars: int = MAX_FTS_CHARS, half: int = FTS_HEAD_HALF) -> str:
    if len(text) <= max_chars:
        return text
    return text[:half] + "\n...\n" + text[-half:]


# ── ParseService ───────────────────────────────────────────────────


class ParseService:
    def __init__(
        self,
        db: DatabaseManager,
        fts: FTSManager,
        config_svc: ConfigService,
        data_dir: str,
        *,
        parse_lock_timeout_sec: int,
    ) -> None:
        self.db = db
        self.fts = fts
        self.config_svc = config_svc
        self.data_dir = os.path.expanduser(data_dir)
        self.parse_lock_timeout_ms = parse_lock_timeout_sec * 1000

    # ── discovery / ingestion (shared) ──────────────────────

    async def refresh_file(
        self,
        path: str,
        watch_id: int | None = None,
        *,
        ensure_ingested: bool = False,
    ) -> FileRefreshResult:
        """Refresh one source path against the files table.

        This method owns file stat, DB row lookup, new/changed/known/missing/deleted
        classification, and optional synchronous ingest.
        """
        ext = Path(path).suffix.lower().lstrip(".")
        existing = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))
        if ext not in ALLOWED_EXTENSIONS:
            return FileRefreshResult(file=_file_info(existing), status="unsupported")

        try:
            stat = await get_file_stat(path)
        except FileNotFoundError:
            return await self._refresh_missing_file(path, existing)
        except PermissionError as exc:
            return await self._refresh_stat_error(existing, "file_permission_denied", str(exc))
        except OSError as exc:
            return await self._refresh_stat_error(existing, "stat_failed", str(exc))

        result = await self._refresh_existing_file_with_stat(path, ext, stat, existing, watch_id)
        if ensure_ingested and result.needs_ingest:
            row = await self.ingest_file(path, watch_id=watch_id)
            return FileRefreshResult(file=_file_info(row), status=result.status)
        return result

    async def _refresh_existing_file_with_stat(
        self,
        path: str,
        ext: str,
        stat: FileStat,
        existing: FileRow | None,
        watch_id: int | None,
    ) -> FileRefreshResult:
        now = _now_ms()
        filename = Path(path).name

        if existing and existing["status"] == FILE_STATUS_ACTIVE:
            unchanged = existing["mtime_ms"] == stat.mtime_ms and existing["size_bytes"] == stat.size_bytes
            if unchanged:
                return FileRefreshResult(file=_file_info(existing), status="known")

            await self.db.execute(
                "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, "
                "sha256=NULL, watch_id=COALESCE(?, watch_id), locked_at=NULL, error_code=NULL, error_msg=NULL, updated_at=? "
                "WHERE id=?",
                (
                    filename,
                    ext,
                    stat.size_bytes,
                    stat.mtime_ms,
                    watch_id,
                    now,
                    existing["id"],
                ),
            )
            row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
            return FileRefreshResult(file=_file_info(row), status="changed")

        if existing:
            await self.db.execute(
                "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, "
                "sha256=NULL, watch_id=COALESCE(?, watch_id), status=?, locked_at=NULL, error_code=NULL, error_msg=NULL, "
                "deleted_at=NULL, updated_at=? WHERE id=?",
                (
                    filename,
                    ext,
                    stat.size_bytes,
                    stat.mtime_ms,
                    watch_id,
                    FILE_STATUS_ACTIVE,
                    now,
                    existing["id"],
                ),
            )
            row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
            return FileRefreshResult(file=_file_info(row), status="changed")

        file_id = await self.db.execute_insert(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, "
            "watch_id, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                filename,
                ext,
                stat.size_bytes,
                stat.mtime_ms,
                watch_id,
                now,
                now,
            ),
        )
        row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (file_id,)))
        return FileRefreshResult(file=_file_info(row), status="new")

    async def _refresh_missing_file(self, path: str, existing: FileRow | None) -> FileRefreshResult:
        if existing is None:
            return FileRefreshResult(file=None, status="missing")

        now = _now_ms()
        file_status = FILE_STATUS_DELETED
        deleted_at = now
        if existing["status"] != FILE_STATUS_DELETED and await self._is_missing_due_to_unreachable_watch(existing):
            file_status = FILE_STATUS_UNREACHABLE
            deleted_at = None

        await self.db.execute(
            "UPDATE files SET status=?, locked_at=NULL, error_code=NULL, error_msg=NULL, deleted_at=?, updated_at=? WHERE id=?",
            (file_status, deleted_at, now, existing["id"]),
        )
        row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
        refresh_status: FileRefreshStatus = "unreachable" if file_status == FILE_STATUS_UNREACHABLE else "deleted"
        return FileRefreshResult(file=_file_info(row), status=refresh_status)

    async def _is_missing_due_to_unreachable_watch(self, file_row: FileRow) -> bool:
        watch_id = file_row.get("watch_id")
        if watch_id is None:
            return False
        watch = cast(WatchTargetRow | None, await self.db.fetchone("SELECT * FROM watches WHERE id=?", (watch_id,)))
        if not watch or not watch.get("removable"):
            return False
        if watch["status"] == WATCH_STATUS_UNREACHABLE:
            return True
        try:
            await get_file_stat(watch["path"])
        except OSError:
            if self.config_svc is not None:
                await self.config_svc.update_watch_status(watch_id, WATCH_STATUS_UNREACHABLE)
            return True
        return False

    async def _refresh_stat_error(self, existing: FileRow | None, error_code: str, error_msg: str) -> FileRefreshResult:
        if existing is None:
            return FileRefreshResult(file=None, status="error")

        now = _now_ms()
        await self.db.execute(
            "UPDATE files SET sha256=NULL, locked_at=NULL, error_code=?, error_msg=?, updated_at=? WHERE id=?",
            (error_code, error_msg[:500], now, existing["id"]),
        )
        row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],)))
        return FileRefreshResult(file=_file_info(row), status="error")

    async def ensure_ingested(self, path: str, watch_id: int | None = None) -> FileRow | None:
        """Synchronously discover and ingest a source path when needed."""
        refreshed = await self.refresh_file(path, watch_id=watch_id, ensure_ingested=True)
        if refreshed.file is None:
            return None
        return cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))

    async def ingest_file(self, path: str, watch_id: int | None = None) -> FileRow | None:
        """Ingest a discovered file: SHA-256 + metadata + trigger default parse."""
        ext = Path(path).suffix.lower().lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            return None

        stat = await get_file_stat(path)
        sha256 = await compute_sha256(path)
        now = _now_ms()
        filename = Path(path).name

        existing_path = cast(
            FileRow | None,
            await self.db.fetchone("SELECT * FROM files WHERE path=? AND status=?", (path, FILE_STATUS_ACTIVE)),
        )
        if existing_path and existing_path["sha256"] == sha256:
            return existing_path

        # check if same content (sha256) is already tracked by another path
        existing_sha = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE sha256=?", (sha256,)))
        if existing_sha:
            # same content, possibly a new or changed path — bind this file row
            # to the existing doc without duplicating docs/parses.
            if existing_path:
                await self.db.execute(
                    "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, "
                    "sha256=?, watch_id=COALESCE(?, watch_id), locked_at=NULL, error_code=NULL, error_msg=NULL, updated_at=? "
                    "WHERE id=?",
                    (
                        filename,
                        ext,
                        stat.size_bytes,
                        stat.mtime_ms,
                        sha256,
                        watch_id,
                        now,
                        existing_path["id"],
                    ),
                )
            else:
                await self.db.execute(
                    "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, "
                    "sha256, watch_id, first_seen_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        path,
                        filename,
                        ext,
                        stat.size_bytes,
                        stat.mtime_ms,
                        sha256,
                        watch_id,
                        now,
                        now,
                    ),
                )
            file_row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))
            if file_row:
                await self.fts.upsert_filename(file_row["id"], Path(path).stem, ext)
            return file_row

        # brand new document
        metadata_error_code = None
        metadata_error_msg = None
        try:
            metadata = await extract_metadata(path)
        except Exception as exc:
            metadata_error_code = "metadata_failed"
            metadata_error_msg = str(exc)[:500] or "Failed to extract document metadata"
            metadata = {
                "page_count": None,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_image_based": 0,
            }

        # Office: no page_count → default to 1
        page_count = metadata["page_count"]
        if page_count is None:
            if ext in ("docx",):
                page_count = 1  # reflow docs
            elif ext in ("pptx", "xlsx"):
                page_count = 1  # will be updated by metadata extraction

        now = _now_ms()
        await self.db.execute(
            "INSERT OR IGNORE INTO files (path, filename, ext, size_bytes, mtime_ms, "
            "watch_id, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                filename,
                ext,
                stat.size_bytes,
                stat.mtime_ms,
                watch_id,
                now,
                now,
            ),
        )

        await ensure_doc_record(
            self.db,
            sha256=sha256,
            size_bytes=stat.size_bytes,
            file_type=_file_type_from_ext(ext),
            page_count=page_count,
            title=metadata["title"],
            author=metadata["author"],
            subject=metadata["subject"],
            keywords=metadata["keywords"],
            error_code=metadata_error_code,
            error_msg=metadata_error_msg,
            first_seen_at=now,
            updated_at=now,
        )

        # FTS filename index
        file_row = await self.db.fetchone("SELECT id FROM files WHERE path=?", (path,))
        if file_row:
            await self.fts.upsert_filename(
                file_row["id"],
                Path(path).stem,
                ext,
            )

        if ext in TEXT_EXTENSIONS:
            await self.db.execute(
                "UPDATE files SET sha256=?, locked_at=NULL, updated_at=? WHERE path=?",
                (sha256, now, path),
            )
            text = await asyncio.to_thread(Path(path).read_text, encoding="utf-8", errors="replace")
            await self.fts.replace(
                sha256=sha256,
                tier="flash",
                text=truncate_head_tail(text),
                title=metadata["title"] or "",
                author=metadata["author"] or "",
                filename=filename,
            )
            return cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))

        # determine tier and page_range for initial parse
        tier: Tier = "flash"
        initial_page_range = default_parse_range(page_count)

        # check parsing-rules
        matched = await self.config_svc.match_rules(path, RULE_TYPE_PARSING_RULE)
        if matched:
            rule = matched[0]
            tier = rule.get("tier") or tier
            rule_page_range = rule.get("page_range")
            if rule_page_range:
                initial_page_range = expand_page_range(rule_page_range, page_count or 1)

        # insert parse batch
        parse_page_range = expand_page_range(initial_page_range, page_count or 1)
        await self.db.execute(
            "INSERT INTO parses (sha256, tier, page_range, status, priority, created_at, updated_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
            (sha256, tier, parse_page_range, PARSE_STATUS_PENDING, now, now),
        )

        await self.db.execute(
            "UPDATE files SET sha256=?, locked_at=NULL, updated_at=? WHERE path=?",
            (sha256, now, path),
        )

        return cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))

    # ── parse request ───────────────────────────────────────────

    async def request_parse(
        self,
        path: str,
        *,
        tier: Tier | None = None,
        page_range: str | None = None,
        force: bool = False,
        remote: bool = False,
    ) -> ParseResponse:
        """Handle a parse request from CLI.  Returns info for status polling."""
        # ensure the path is current before trusting files.sha256
        file_row = await self.ensure_ingested(path)
        if file_row is None:
            return _failed_response(tier or "flash", page_range or "", "File could not be ingested.")

        sha256 = file_row["sha256"]
        if sha256 is None:
            return _failed_response(tier or "flash", page_range or "", "File could not be ingested.")
        doc = cast(PageCountRow | None, await self.db.fetchone("SELECT page_count FROM docs WHERE sha256=?", (sha256,)))
        page_count = doc["page_count"] if doc else 1
        privacy = "remote" if remote else "local"

        # ── resolve tier (before cache check) ──
        requested_tier = tier or _resolve_default_tier(remote)
        ext = file_row["ext"]
        if ext in TEXT_EXTENSIONS:
            return _text_response(sha256)
        if ext in ("docx", "pptx", "xlsx"):
            requested_tier = "flash"

        # ── expand page range ──
        default_page_range = default_parse_range(page_count)
        requested_page_range_input = page_range or default_page_range
        request_page_range = expand_page_range(requested_page_range_input, page_count or 1)
        needed_page_numbers = parse_page_range_set(request_page_range)

        # ── step 1: remove page numbers covered by valid done batches ──
        if not force:
            done_batches = cast(
                list[ParseRow],
                await self.db.fetchall(
                    "SELECT * FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
                    (sha256, requested_tier, PARSE_STATUS_DONE),
                ),
            )
            for batch in done_batches:
                if not _json_file_exists_by_batch(self.data_dir, sha256, requested_tier, batch):
                    continue  # JSON gone → cache invalid
                covered_page_numbers = parse_page_range_set(batch["page_range"])
                needed_page_numbers -= covered_page_numbers

            if not needed_page_numbers:
                return _done_response(sha256, requested_tier, request_page_range)

        # ── step 2: remove page numbers covered by pending/parsing batches ──
        reused_parse_ids: list[int] = []
        active_batches = cast(
            list[ParseRow],
            await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN (?, ?)",
                (sha256, requested_tier, PARSE_STATUS_PENDING, PARSE_STATUS_PARSING),
            ),
        )
        if active_batches:
            active_covered_page_numbers: set[int] = set()
            for batch in active_batches:
                covered_page_numbers = parse_page_range_set(batch["page_range"])
                if needed_page_numbers & covered_page_numbers:
                    reused_parse_ids.append(batch["id"])
                active_covered_page_numbers |= covered_page_numbers
            needed_page_numbers -= active_covered_page_numbers

            # bump priority for reused in-progress batches
            now = _now_ms()
            for batch in active_batches:
                if batch["id"] in reused_parse_ids and batch["priority"] < 1:
                    await self.db.execute(
                        "UPDATE parses SET priority=1, updated_at=? WHERE id=?",
                        (now, batch["id"]),
                    )

            if not needed_page_numbers:
                return ParseResponse(
                    sha256=sha256,
                    tier=requested_tier,
                    page_range=request_page_range,
                    status=PARSE_STATUS_PENDING,
                    cache_hit=False,
                    wait_parse_ids=reused_parse_ids,
                    created_parse_ids=[],
                    reused_parse_ids=reused_parse_ids,
                    tip="Pages already queued. Priority bumped.",
                )

        # ── step 3: enqueue remaining uncovered page numbers ──
        uncovered_page_range = _page_numbers_to_range_str(needed_page_numbers)
        now = _now_ms()
        parse_id = await self.db.execute_insert(
            "INSERT INTO parses (sha256, tier, page_range, status, privacy, priority, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (sha256, requested_tier, uncovered_page_range, PARSE_STATUS_PENDING, privacy, now, now),
        )
        created_parse_ids = [parse_id]
        return ParseResponse(
            sha256=sha256,
            tier=requested_tier,
            page_range=request_page_range,
            status=PARSE_STATUS_PENDING,
            cache_hit=False,
            wait_parse_ids=reused_parse_ids + created_parse_ids,
            created_parse_ids=created_parse_ids,
            reused_parse_ids=reused_parse_ids,
        )

    # ── worker ──────────────────────────────────────────────────

    async def get_queue_length(self) -> int:
        timeout_ms = _now_ms() - self.parse_lock_timeout_ms
        row = await self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM parses WHERE status=? AND (locked_at IS NULL OR locked_at < ?)",
            (PARSE_STATUS_PENDING, timeout_ms),
        )
        return row["cnt"] if row else 0

    async def acquire_task(self) -> ParseRow | None:
        now = _now_ms()
        timeout = now - self.parse_lock_timeout_ms
        return cast(
            ParseRow | None,
            await self.db.fetchone(
                "UPDATE parses SET locked_at=?, status=? "
                "WHERE id = ("
                "  SELECT id FROM parses WHERE status=? "
                "  AND (locked_at IS NULL OR locked_at < ?) "
                "  ORDER BY priority DESC, created_at ASC LIMIT 1"
                ") RETURNING *",
                (now, PARSE_STATUS_PARSING, PARSE_STATUS_PENDING, timeout),
            ),
        )

    async def process_doc(self, task: ParseRow) -> bool:
        """Execute parse for a batch.  Returns True on success."""
        sha256 = task["sha256"]
        tier = task["tier"]
        page_range = task["page_range"]
        privacy = task.get("privacy", "local")

        # guard
        current = await self.db.fetchone("SELECT status FROM parses WHERE id=?", (task["id"],))
        if current is None or current["status"] != PARSE_STATUS_PARSING:
            return False

        # find the file
        file_row = cast(
            FileRow | None,
            await self.db.fetchone(
                "SELECT * FROM files WHERE sha256=? AND status=? LIMIT 1",
                (sha256, FILE_STATUS_ACTIVE),
            ),
        )
        if file_row is None:
            await self._fail_task(task["id"], "no_accessible_file", "No active file found for this document")
            return False

        output_dir = os.path.join(self.data_dir, "parsed", sha256[:2], sha256, tier)

        # route parse based on tier
        via = "local"
        try:
            if tier == "flash":
                result = await self._parse_via_local(file_row, tier, page_range)
            else:
                result, via = await self._parse_via_api(file_row, tier, page_range, privacy)
        except ParseFailure as exc:
            await self._fail_task(task["id"], exc.code, exc.message)
            return False
        except Exception as exc:
            await self._fail_task(task["id"], "parse_failed", str(exc)[:500])
            return False

        new_pages = result.to_dict(skip_defaults=True)["pages"]
        if not new_pages:
            await self._fail_task(task["id"], "parse_empty", "Parse completed but returned no pages")
            return False

        # save per-batch JSON (markdown is generated on read from /docs/{sha256}/content)
        done_at_ms = _now_ms()
        json_path = os.path.join(output_dir, _safe_filename(page_range, done_at_ms))
        try:
            os.makedirs(output_dir, exist_ok=True)
            _write_cached_image_sidecars(parse_image_sidecar_dir(self.data_dir, sha256, tier), result.images())
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"schema_version": MIDDLE_JSON_SCHEMA_VERSION, "pages": new_pages}, f, ensure_ascii=False, indent=4)
        except Exception as exc:
            await self._fail_task(task["id"], "parse_json_write_failed", str(exc)[:500])
            return False

        # generate markdown for FTS (not persisted, only for search indexing)
        md_text = ""
        md = result.markdown(add_markers=False) if hasattr(result, "markdown") else ""
        md_text += md + "\n"

        # update fts_contents (tier-gated)
        await self._maybe_update_fts(sha256, tier, md_text, file_row)

        # update docs metadata (tier-gated)
        await self._maybe_update_docs_meta(sha256, tier)

        # mark done
        await self.db.execute(
            "UPDATE parses SET status=?, done_at=?, locked_at=NULL, via=?, updated_at=? WHERE id=?",
            (PARSE_STATUS_DONE, done_at_ms, via, done_at_ms, task["id"]),
        )
        return True

    # ── parse routing helpers ─────────────────────────────────────

    async def _parse_via_local(
        self,
        file_row: FileRow,
        tier: Tier,
        page_range: str,
    ) -> ParseResult:
        """Parse via local library call."""
        from ...parser import parse

        result = await asyncio.to_thread(
            parse,
            file_row["path"],
            tier=tier,
            page_range=page_range,
        )
        return result

    async def _parse_via_api(
        self,
        file_row: FileRow,
        tier: Tier,
        page_range: str,
        privacy: str,
    ) -> tuple[ParseResult, str]:
        """Parse via HTTP API (local or remote parse-server). Returns (results, via)."""
        from ...parser.api_client import MinerUApiParser

        base_url, api_key, via = await self._resolve_api_target(privacy, tier)

        # resolve and validate tier against server capabilities
        resolved_tier = self._resolve_tier(tier, via)

        parser = MinerUApiParser(
            api_url=base_url,
            api_key=api_key,
            tier=resolved_tier,
        )
        result = await parser.parse_async(file_row["path"], page_range=page_range)
        _remap_api_result_pages_to_page_range(result, page_range)
        return result, via

    @staticmethod
    def _resolve_tier(tier: Tier, via: str) -> Tier:
        """Resolve and validate tier against parse-server capabilities.

        - validate specified tier against supported_tiers
        - raise tier_mismatch if unsupported
        """
        from ..background.parse_server_health import get_health

        health = get_health()
        if via == "remote":
            supported = health.remote_supported_tiers
        else:
            supported = health.local_supported_tiers

        if not supported:
            raise ParseFailure("engine_unavailable", "Parse-server health status unknown, cannot validate tier")

        if tier not in supported:
            raise ParseFailure("tier_mismatch", f"Tier '{tier}' not supported by parse-server. Available: {supported}")
        return tier

    async def _resolve_api_target(self, privacy: str, tier: Tier) -> tuple[str, str | None, str]:
        """Resolve the api target URL based on privacy and config.

        Returns (base_url, api_key, via_label).
        """
        from ..background.parse_server_health import get_health

        health = get_health()

        if privacy == "remote":
            url = cast(str, await self.config_svc.get("parse_server.remote.url"))
            api_key = os.environ.get("MINERU_API_KEY") or (await self.config_svc.get("parse_server.remote.api_key"))
            if not health.remote_healthy:
                # try fallback to local
                local_mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
                if local_mode != "disabled" and health.local_healthy:
                    local_url = _local_parse_server_url(local_mode, health)
                    if local_url:
                        return local_url, api_key, "local"
                raise ParseFailure("parse_server_unavailable", "Remote parse-server unavailable and no local fallback")
            return url, api_key, "remote"

        # privacy == local
        local_mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
        if local_mode == "disabled":
            raise ParseFailure("no_engine", "Local parse-server is disabled. Use --tier flash or --remote.")

        local_url = _local_parse_server_url(local_mode, health)
        if local_url is None:
            raise ParseFailure("engine_unavailable", "Local parse-server URL not configured.")

        if not health.local_healthy:
            raise ParseFailure("engine_unavailable", "Local parse-server is not ready. Please wait or check server status.")

        api_key = os.environ.get("MINERU_API_KEY") or (await self.config_svc.get("parse_server.local.self_hosted_api_key"))
        return local_url, api_key, "local"

    async def _fail_task(self, task_id: int, code: str, message: str) -> None:
        now = _now_ms()
        await self.db.execute(
            "UPDATE parses SET status=?, error_code=?, error_msg=?, locked_at=NULL, updated_at=? WHERE id=?",
            (PARSE_STATUS_FAILED, code, message, now, task_id),
        )

    async def get_parse_record(self, parse_id: int) -> dict | None:
        row = cast(ParseRow | None, await self.db.fetchone("SELECT * FROM parses WHERE id=?", (parse_id,)))
        return _parse_record_response(row) if row else None

    async def list_parse_records(
        self,
        *,
        ids: list[int] | None = None,
        sha256: str | None = None,
        tier: Tier | None = None,
        status: list[str] | None = None,
        page_range: str | None = None,
        include_superseded: bool = False,
    ) -> dict:
        if ids:
            placeholders = ",".join("?" * len(ids))
            rows = cast(
                list[ParseRow], await self.db.fetchall(f"SELECT * FROM parses WHERE id IN ({placeholders})", tuple(ids))
            )
            rows_by_id = {row["id"]: row for row in rows}
            ordered_rows = [rows_by_id[parse_id] for parse_id in ids if parse_id in rows_by_id]
            return {"parses": [_parse_record_response(row) for row in ordered_rows]}

        clauses: list[str] = []
        params: list[object] = []
        if sha256:
            clauses.append("sha256=?")
            params.append(sha256)
        if tier:
            clauses.append("tier=?")
            params.append(tier)
        if status:
            placeholders = ",".join("?" * len(status))
            clauses.append(f"status IN ({placeholders})")
            params.extend(status)
        elif not include_superseded:
            clauses.append("status!=?")
            params.append(PARSE_STATUS_SUPERSEDED)

        sql = "SELECT * FROM parses"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        rows = cast(list[ParseRow], await self.db.fetchall(sql, tuple(params)))
        result: dict[str, object] = {"parses": [_parse_record_response(row) for row in rows]}
        if sha256 and tier and page_range:
            result["coverage"] = _parse_coverage(page_range, rows)
        return result

    async def invalidate(self, sha256: str, tier: Tier | None = None) -> int:
        """Mark done parses as superseded.  Returns count of affected rows."""
        now = _now_ms()
        sql = "UPDATE parses SET status=?, updated_at=? WHERE sha256=? AND status=?"
        params = [PARSE_STATUS_SUPERSEDED, now, sha256, PARSE_STATUS_DONE]
        if tier:
            sql += " AND tier=?"
            params.append(tier)
        cursor = await self.db.execute(sql, tuple(params))
        if cursor.rowcount:
            await self._rebuild_fts_after_invalidate(sha256)
        return cursor.rowcount

    # ── internal helpers ────────────────────────────────────────

    async def _maybe_update_fts(self, sha256: str, tier: Tier, text: str, file_row: FileRow) -> None:
        existing_tier = await self.fts.get_tier(sha256)
        if existing_tier and TIER_ORDER.get(existing_tier, -1) >= TIER_ORDER.get(tier, -1):
            return  # current FTS data is from a higher or equal tier

        text = truncate_head_tail(text)
        await self.fts.replace(
            sha256=sha256,
            tier=tier,
            text=text,
            title=file_row.get("title") or "",
            author=file_row.get("author") or "",
            filename=file_row["filename"],
        )

    async def _maybe_update_docs_meta(self, sha256: str, tier: Tier) -> None:
        doc = await self.db.fetchone("SELECT meta_tier FROM docs WHERE sha256=?", (sha256,))
        if doc is None:
            return
        existing_tier = doc.get("meta_tier")
        now = _now_ms()
        if existing_tier and TIER_ORDER.get(existing_tier, -1) >= TIER_ORDER.get(tier, -1):
            await self.db.execute(
                "UPDATE docs SET error_code=NULL, error_msg=NULL, updated_at=? WHERE sha256=?",
                (now, sha256),
            )
            return

        # for now just update meta_tier; full metadata update comes when
        # engine provides richer output
        await self.db.execute(
            "UPDATE docs SET meta_tier=?, error_code=NULL, error_msg=NULL, updated_at=? WHERE sha256=?",
            (tier, now, sha256),
        )

    async def _rebuild_fts_after_invalidate(self, sha256: str) -> None:
        file_row = cast(
            FileRow | None,
            await self.db.fetchone(
                "SELECT * FROM files WHERE sha256=? AND status=? LIMIT 1",
                (sha256, FILE_STATUS_ACTIVE),
            ),
        )
        if file_row is None:
            await self.fts.delete(sha256)
            return

        rows = cast(
            list[ParseRow],
            await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND status=? ORDER BY done_at DESC",
                (sha256, PARSE_STATUS_DONE),
            ),
        )
        tier_set: set[Tier] = {row["tier"] for row in rows}
        tiers = sorted(tier_set, key=lambda item: TIER_ORDER.get(item, -1), reverse=True)
        for tier in tiers:
            tier_rows = [row for row in rows if row["tier"] == tier]
            pages = load_pages_from_done_batches(self.data_dir, sha256, tier, tier_rows)
            if not pages:
                continue

            from ...render import render_markdown

            text = truncate_head_tail(render_markdown(pages, add_markers=False))
            await self.fts.replace(
                sha256=sha256,
                tier=tier,
                text=text,
                title=file_row.get("title") or "",
                author=file_row.get("author") or "",
                filename=file_row["filename"],
            )
            return

        await self.fts.delete(sha256)


# ── tier resolution helpers ────────────────────────────────────────


def _resolve_default_tier(remote: bool = False) -> Tier:
    """Pick the best available tier from health check.  ``pro`` > ``standard`` > ``flash``."""
    from ..background.parse_server_health import get_health

    health = get_health()
    supported = health.remote_supported_tiers if remote else health.local_supported_tiers
    for candidate in ("pro", "standard"):
        if candidate in supported:
            return candidate
    raise ParseFailure(
        "quality_tier_unavailable",
        "No standard or pro engine available. Start a parse-server or use --tier flash for text-only preview.",
    )


def _json_file_exists_by_batch(data_dir: str, sha256: str, tier: Tier, batch: ParseRow) -> bool:
    """Check that the JSON result file for a parses batch row actually exists on disk."""
    json_path = parse_batch_json_path(data_dir, sha256, tier, batch["page_range"], batch["done_at"])
    return os.path.isfile(json_path)


def parse_batch_json_path(data_dir: str, sha256: str, tier: Tier, page_range: str, done_at: int | None = 0) -> str:
    """Return the persisted Middle JSON path for one parse batch."""
    filename = _safe_filename(page_range, done_at or 0)
    return os.path.join(os.path.expanduser(data_dir), "parsed", sha256[:2], sha256, tier, filename)


def parse_image_sidecar_dir(data_dir: str, sha256: str, tier: Tier) -> str:
    """返回 doclib 解析缓存图片 sidecar 目录，供同一文档同一 tier 的多个 batch 共用。"""
    return os.path.join(os.path.expanduser(data_dir), "parsed", sha256[:2], sha256, tier, "images")


def _safe_sidecar_path(image_dir: str, image_path: str) -> Path | None:
    """把 middle_json 中的相对 image_path 安全映射到 doclib 缓存图片目录。"""
    if not image_path:
        return None
    raw_path = PurePosixPath(image_path)
    if raw_path.is_absolute() or any(part in {"", ".", ".."} for part in raw_path.parts):
        return None
    return Path(image_dir).joinpath(*raw_path.parts)


def _write_cached_image_sidecars(image_dir: str, images: dict[str, bytes]) -> None:
    """把 ParseResult.images() 的图片字节写入 doclib 缓存，保证后续 markdown 链接可访问。"""
    for image_path, image_bytes in images.items():
        sidecar_path = _safe_sidecar_path(image_dir, image_path)
        if sidecar_path is None:
            continue
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_bytes(image_bytes)


def _restore_missing_image_sidecars(page: PageInfo, image_dir: str) -> None:
    """读取历史缓存时先找 image_path 文件，缺失时再从 image_base64 补写同名 sidecar。"""
    for block_list in (page.preproc_blocks, page.para_blocks, page.discarded_blocks):
        for block in block_list:
            for span in block.all_spans():
                sidecar_path = _safe_sidecar_path(image_dir, span.image_path)
                if sidecar_path is None or sidecar_path.is_file() or not span.image_base64:
                    continue
                parsed = parse_image_data_uri(span.image_base64)
                if parsed is None:
                    continue
                try:
                    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                    sidecar_path.write_bytes(parsed[0])
                except OSError:
                    continue


def load_pages_from_done_batches(data_dir: str, sha256: str, tier: Tier, done_rows: Sequence[ParseBatchRow]) -> list[PageInfo]:
    """Load valid done JSON batches and keep the newest page for duplicate page_idx values."""
    pages_by_page_idx: dict[int, PageInfo] = {}
    image_dir = parse_image_sidecar_dir(data_dir, sha256, tier)
    for row in reversed(done_rows):
        fpath = parse_batch_json_path(data_dir, sha256, tier, row["page_range"], row["done_at"])
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            for raw in data.get("pages", []):
                page = PageInfo.from_dict(raw)
                _restore_missing_image_sidecars(page, image_dir)
                pages_by_page_idx[page.page_idx] = page
        except Exception:
            pass
    return [pages_by_page_idx[page_idx] for page_idx in sorted(pages_by_page_idx)]


def _done_response(sha256: str, tier: Tier, page_range: str) -> ParseResponse:
    return ParseResponse(
        sha256=sha256,
        tier=tier,
        page_range=page_range,
        status=PARSE_STATUS_DONE,
        cache_hit=True,
        wait_parse_ids=[],
        created_parse_ids=[],
        reused_parse_ids=[],
        tip="Cached. Use --force to re-parse.",
    )


def _text_response(sha256: str) -> ParseResponse:
    return ParseResponse(
        sha256=sha256,
        tier="flash",
        page_range="1",
        status=PARSE_STATUS_DONE,
        cache_hit=False,
        wait_parse_ids=[],
        created_parse_ids=[],
        reused_parse_ids=[],
        tip="Plain text files do not require parsing.",
    )


def _failed_response(tier: Tier, page_range: str, tip: str) -> ParseResponse:
    return ParseResponse(
        sha256="",
        tier=tier,
        page_range=page_range,
        status=PARSE_STATUS_FAILED,
        cache_hit=False,
        wait_parse_ids=[],
        created_parse_ids=[],
        reused_parse_ids=[],
        tip=tip,
    )


def _parse_record_response(row: ParseRow) -> dict:
    error = None
    if row.get("error_code") or row.get("error_msg"):
        error = {"code": row.get("error_code"), "message": row.get("error_msg")}
    return {
        "id": row["id"],
        "sha256": row["sha256"],
        "tier": row["tier"],
        "page_range": row["page_range"],
        "status": row["status"],
        "done_at": row.get("done_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "error": error,
    }


def _remap_api_result_pages_to_page_range(result: ParseResult, page_range: str) -> None:
    """Restore API-backed partial parse pages to original document page indices."""
    if not result.pages:
        return
    requested_page_numbers = sorted(parse_page_range_set(page_range))
    actual_page_numbers = [page.page_idx + 1 for page in result.pages]
    if actual_page_numbers == requested_page_numbers:
        return
    if len(requested_page_numbers) != len(result.pages):
        raise ParseFailure(
            "parse_page_remap_failed",
            f"Parse result page count does not match requested page_range: requested={page_range}, returned={actual_page_numbers}",
        )
    for page, page_no in zip(result.pages, requested_page_numbers, strict=True):
        page.page_idx = page_no - 1


def _parse_coverage(request_page_range: str, rows: list[ParseRow]) -> dict:
    requested_page_numbers = parse_page_range_set(request_page_range)
    done_page_numbers: set[int] = set()
    active_page_numbers: set[int] = set()
    for row in rows:
        row_page_numbers = parse_page_range_set(row["page_range"]) & requested_page_numbers
        if row["status"] == PARSE_STATUS_DONE:
            done_page_numbers |= row_page_numbers
        elif row["status"] in (PARSE_STATUS_PENDING, PARSE_STATUS_PARSING):
            active_page_numbers |= row_page_numbers
    active_page_numbers -= done_page_numbers
    missing_page_numbers = requested_page_numbers - done_page_numbers - active_page_numbers
    return {
        "done_page_range": _page_numbers_to_range_str(done_page_numbers),
        "active_page_range": _page_numbers_to_range_str(active_page_numbers),
        "missing_page_range": _page_numbers_to_range_str(missing_page_numbers),
    }


def _local_parse_server_url(mode: str, health: object) -> str | None:
    """Resolve local parse-server URL from mode and health state."""
    if mode == "managed":
        return "http://127.0.0.1:15981"
    if mode == "self_hosted":
        return getattr(health, "self_hosted_url", None)
    return None


def _file_info(row: FileRow | None) -> FileInfo | None:
    return FileInfo.model_validate(row) if row is not None else None


def _safe_filename(page_range: str, done_at: int) -> str:
    """Convert a page_range string + done_at to a filename."""
    return f"{page_range}_{done_at}.json" if done_at else page_range


def _file_type_from_ext(ext: str) -> str:
    return DOC_TYPE_BY_EXT.get(ext, ext or "unknown")
