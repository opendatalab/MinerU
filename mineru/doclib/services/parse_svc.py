"""Parse service: file ingestion, parse request/acquire/process."""

from __future__ import annotations

import asyncio
import json as _json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ...errors import MineruError
from ...types import PageInfo, TIER_ORDER, Tier
from ..constants import ALLOWED_EXTENSIONS, TEXT_EXTENSIONS
from ..core.db import DatabaseManager
from ..core.file_io import compute_sha256, extract_metadata, get_file_stat
from ..core.fts import FTSManager
from ..types import (
    PARSE_STATUS_DONE,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PARSING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_SUPERSEDED,
    RULE_TYPE_PARSING_RULE,
    SCAN_STATUS_ACTIVE,
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


@dataclass(frozen=True)
class DiscoverResult:
    file: dict | None
    changed: bool
    needs_ingest: bool
    unsupported: bool = False


# ── range helpers ──────────────────────────────────────────────────


def parse_range_set(s: str) -> set[int]:
    """Parse a pages range string like '1~5,10~15' into a set of integers."""
    pages: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "~" in part:
            a, b = part.split("~", 1)
            pages.update(range(int(a.strip()), int(b.strip()) + 1))
        else:
            pages.add(int(part))
    return pages


def filter_pages_by_user_range(pages: list[PageInfo], requested_pages: str) -> list[PageInfo]:
    """Filter 0-based PageInfo objects by a user-facing 1-based page range."""
    requested = parse_range_set(requested_pages)
    return [page for page in pages if page.page_idx + 1 in requested]


def pages_covered(request_pages: str, done_batches: list[dict]) -> bool:
    """Check whether request_pages is fully covered by done batches.
    Batches are sorted by done_at DESC so newer batches take precedence."""
    needed = parse_range_set(request_pages)
    covered: set[int] = set()
    for b in sorted(done_batches, key=lambda r: r.get("done_at") or 0, reverse=True):
        covered |= parse_range_set(b["pages"])
        if needed <= covered:
            return True
    return False


def pages_uncovered(request_pages: str, done_batches: list[dict]) -> set[int]:
    """Return the subset of request_pages NOT covered by any done batch."""
    needed = parse_range_set(request_pages)
    covered: set[int] = set()
    for b in done_batches:
        covered |= parse_range_set(b["pages"])
    return needed - covered


def _pages_set_to_str(pages: set[int]) -> str:
    """Convert a set of page numbers (1-based) to a compact range string."""
    if not pages:
        return ""
    sorted_pages = sorted(pages)
    ranges: list[str] = []
    start = sorted_pages[0]
    end = start
    for p in sorted_pages[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append(f"{start}~{end}" if start != end else str(start))
            start = p
            end = p
    ranges.append(f"{start}~{end}" if start != end else str(start))
    return ",".join(ranges)


def expand_pages(pages: str | None, page_count: int) -> str:
    """Expand shorthand pages like 'all' or negative ranges like '-5~-1'
    into positive page ranges.  Returns '1~{page_count}' for None/empty."""
    if not pages or pages.strip() == "all":
        return f"1~{page_count}"
    result: list[str] = []
    for part in pages.split(","):
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
    ) -> None:
        self.db = db
        self.fts = fts
        self.config_svc = config_svc
        self.data_dir = os.path.expanduser(data_dir)

    # ── discovery / ingestion (shared) ──────────────────────

    async def discover_file(self, path: str, watch_id: int | None = None) -> DiscoverResult:
        """Discover a path and mark it for ingest if stat changed.

        This is the lightweight path used by watch and by synchronous path
        operations before they trust ``files.sha256``.
        """
        ext = Path(path).suffix.lower().lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            return DiscoverResult(file=None, changed=False, needs_ingest=False, unsupported=True)

        stat = await get_file_stat(path)
        now = _now_ms()
        filename = Path(path).name

        existing = await self.db.fetchone("SELECT * FROM files WHERE path=? AND scan_status=?", (path, SCAN_STATUS_ACTIVE))
        if existing:
            unchanged = existing["mtime_ms"] == stat["mtime_ms"] and existing["size_bytes"] == stat["size_bytes"]
            if unchanged:
                return DiscoverResult(file=existing, changed=False, needs_ingest=existing["sha256"] is None)

            await self.db.execute(
                "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, birthtime_ms=?, "
                "sha256=NULL, watch_id=COALESCE(?, watch_id), locked_at=NULL, error_code=NULL, error_msg=NULL, updated_at=? "
                "WHERE id=?",
                (
                    filename,
                    ext,
                    stat["size_bytes"],
                    stat["mtime_ms"],
                    stat["birthtime_ms"],
                    watch_id,
                    now,
                    existing["id"],
                ),
            )
            row = await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing["id"],))
            return DiscoverResult(file=row, changed=True, needs_ingest=True)

        existing_any_status = await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,))
        if existing_any_status:
            await self.db.execute(
                "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, birthtime_ms=?, "
                "sha256=NULL, watch_id=?, scan_status=?, locked_at=NULL, error_code=NULL, error_msg=NULL, "
                "deleted_at=NULL, updated_at=? WHERE id=?",
                (
                    filename,
                    ext,
                    stat["size_bytes"],
                    stat["mtime_ms"],
                    stat["birthtime_ms"],
                    watch_id,
                    SCAN_STATUS_ACTIVE,
                    now,
                    existing_any_status["id"],
                ),
            )
            row = await self.db.fetchone("SELECT * FROM files WHERE id=?", (existing_any_status["id"],))
            return DiscoverResult(file=row, changed=True, needs_ingest=True)

        file_id = await self.db.execute_insert(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, "
            "birthtime_ms, watch_id, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                filename,
                ext,
                stat["size_bytes"],
                stat["mtime_ms"],
                stat["birthtime_ms"],
                watch_id,
                now,
                now,
            ),
        )
        row = await self.db.fetchone("SELECT * FROM files WHERE id=?", (file_id,))
        return DiscoverResult(file=row, changed=True, needs_ingest=True)

    async def ensure_ingested(self, path: str, watch_id: int | None = None) -> dict | None:
        """Synchronously discover and ingest a source path when needed."""
        discovered = await self.discover_file(path, watch_id=watch_id)
        if discovered.file is None:
            return None
        if not discovered.needs_ingest:
            return discovered.file
        return await self.ingest_file(path, watch_id=watch_id)

    async def ingest_file(self, path: str, watch_id: int | None = None) -> dict | None:
        """Ingest a discovered file: SHA-256 + metadata + trigger default parse."""
        ext = Path(path).suffix.lower().lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            return None

        stat = await get_file_stat(path)
        sha256 = await compute_sha256(path)
        now = _now_ms()
        filename = Path(path).name

        existing_path = await self.db.fetchone("SELECT * FROM files WHERE path=? AND scan_status=?", (path, SCAN_STATUS_ACTIVE))
        if existing_path and existing_path["sha256"] == sha256:
            return existing_path

        # check if same content (sha256) is already tracked by another path
        existing_sha = await self.db.fetchone("SELECT * FROM files WHERE sha256=?", (sha256,))
        if existing_sha:
            # same content, possibly a new or changed path — bind this file row
            # to the existing doc without duplicating docs/parses.
            if existing_path:
                await self.db.execute(
                    "UPDATE files SET filename=?, ext=?, size_bytes=?, mtime_ms=?, birthtime_ms=?, "
                    "sha256=?, watch_id=COALESCE(?, watch_id), locked_at=NULL, error_code=NULL, error_msg=NULL, updated_at=? "
                    "WHERE id=?",
                    (
                        filename,
                        ext,
                        stat["size_bytes"],
                        stat["mtime_ms"],
                        stat["birthtime_ms"],
                        sha256,
                        watch_id,
                        now,
                        existing_path["id"],
                    ),
                )
            else:
                await self.db.execute(
                    "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, "
                    "birthtime_ms, sha256, watch_id, first_seen_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        path,
                        filename,
                        ext,
                        stat["size_bytes"],
                        stat["mtime_ms"],
                        stat["birthtime_ms"],
                        sha256,
                        watch_id,
                        now,
                        now,
                    ),
                )
            file_row = await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,))
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
                "mime_type": None,
                "page_count": None,
                "title": None,
                "author": None,
                "subject": None,
                "keywords": None,
                "is_scanned": 0,
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
            "birthtime_ms, watch_id, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                path,
                filename,
                ext,
                stat["size_bytes"],
                stat["mtime_ms"],
                stat["birthtime_ms"],
                watch_id,
                now,
                now,
            ),
        )

        await self.db.execute(
            "INSERT OR IGNORE INTO docs (sha256, size_bytes, mime_type, page_count, "
            "title, author, subject, keywords, error_code, error_msg, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sha256,
                stat["size_bytes"],
                metadata["mime_type"],
                page_count,
                metadata["title"],
                metadata["author"],
                metadata["subject"],
                metadata["keywords"],
                metadata_error_code,
                metadata_error_msg,
                now,
                now,
            ),
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
            return await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,))

        # determine tier and pages for initial parse
        tier: Tier = "flash"
        if page_count and page_count > 10:
            initial_pages = "1~5,-5~-1"
        elif page_count:
            initial_pages = f"1~{page_count}"
        else:
            initial_pages = "1~5"

        # check parsing-rules
        matched = await self.config_svc.match_rules(path, RULE_TYPE_PARSING_RULE)
        if matched:
            rule = matched[0]
            tier = rule.get("tier") or tier
            rule_pages = rule.get("pages")
            if rule_pages:
                initial_pages = expand_pages(rule_pages, page_count or 1)

        # insert parse batch
        pages_str = expand_pages(initial_pages, page_count or 1)
        await self.db.execute(
            "INSERT INTO parses (sha256, tier, pages, status, priority, created_at, updated_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
            (sha256, tier, pages_str, PARSE_STATUS_PENDING, now, now),
        )

        await self.db.execute(
            "UPDATE files SET sha256=?, locked_at=NULL, updated_at=? WHERE path=?",
            (sha256, now, path),
        )

        return await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,))

    # ── parse request ───────────────────────────────────────────

    async def request_parse(
        self,
        path: str,
        *,
        tier: Tier | None = None,
        pages: str | None = None,
        force: bool = False,
        remote: bool = False,
    ) -> dict:
        """Handle a parse request from CLI.  Returns info for status polling."""
        # ensure the path is current before trusting files.sha256
        file_row = await self.ensure_ingested(path)
        if file_row is None:
            return {"sha256": "", "tier": tier or "", "status": "error", "tip": "File could not be ingested."}

        sha256 = file_row["sha256"]
        doc = await self.db.fetchone("SELECT page_count FROM docs WHERE sha256=?", (sha256,))
        page_count = doc["page_count"] if doc else 1
        privacy = "remote" if remote else "local"

        # ── resolve tier (before cache check) ──
        requested_tier = tier or _resolve_default_tier(remote)
        ext = file_row["ext"]
        if ext in TEXT_EXTENSIONS:
            return _text_response(sha256)
        if ext in ("docx", "pptx", "xlsx"):
            requested_tier = "flash"

        # ── expand pages ──
        default_pages = "1~5,-5~-1" if page_count and page_count > 10 else f"1~{page_count}"
        raw_pages = pages or default_pages
        request_pages_str = expand_pages(raw_pages, page_count or 1)
        needed = parse_range_set(request_pages_str)

        # ── step 1: remove pages covered by valid done batches ──
        if not force:
            done_batches = await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
                (sha256, requested_tier, PARSE_STATUS_DONE),
            )
            for batch in done_batches:
                if not _json_file_exists_by_batch(self.data_dir, sha256, requested_tier, batch):
                    continue  # JSON gone → cache invalid
                covered = parse_range_set(batch["pages"])
                needed -= covered

            if not needed:
                return _done_response(sha256, requested_tier, request_pages_str)

        # ── step 2: remove pages covered by pending/parsing batches ──
        reused_parse_ids: list[int] = []
        active_batches = await self.db.fetchall(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN (?, ?)",
            (sha256, requested_tier, PARSE_STATUS_PENDING, PARSE_STATUS_PARSING),
        )
        if active_batches:
            active_covered: set[int] = set()
            for batch in active_batches:
                covered = parse_range_set(batch["pages"])
                if needed & covered:
                    reused_parse_ids.append(batch["id"])
                active_covered |= covered
            needed -= active_covered

            # bump priority for reused in-progress batches
            now = _now_ms()
            for batch in active_batches:
                if batch["id"] in reused_parse_ids and batch["priority"] < 1:
                    await self.db.execute(
                        "UPDATE parses SET priority=1, updated_at=? WHERE id=?",
                        (now, batch["id"]),
                    )

            if not needed:
                return {
                    "sha256": sha256,
                    "tier": requested_tier,
                    "pages": request_pages_str,
                    "status": PARSE_STATUS_PENDING,
                    "cache_hit": False,
                    "wait_parse_ids": reused_parse_ids,
                    "created_parse_ids": [],
                    "reused_parse_ids": reused_parse_ids,
                    "tip": "Pages already queued. Priority bumped.",
                }

        # ── step 3: enqueue remaining uncovered pages ──
        uncovered_str = _pages_set_to_str(needed)
        now = _now_ms()
        parse_id = await self.db.execute_insert(
            "INSERT INTO parses (sha256, tier, pages, status, privacy, priority, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (sha256, requested_tier, uncovered_str, PARSE_STATUS_PENDING, privacy, now, now),
        )
        created_parse_ids = [parse_id]
        return {
            "sha256": sha256,
            "tier": requested_tier,
            "pages": request_pages_str,
            "status": PARSE_STATUS_PENDING,
            "cache_hit": False,
            "wait_parse_ids": reused_parse_ids + created_parse_ids,
            "created_parse_ids": created_parse_ids,
            "reused_parse_ids": reused_parse_ids,
        }

    # ── worker ──────────────────────────────────────────────────

    async def get_queue_length(self) -> int:
        timeout_ms = _now_ms() - 30 * 60 * 1000  # 30min lock timeout
        row = await self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM parses WHERE status=? AND (locked_at IS NULL OR locked_at < ?)",
            (PARSE_STATUS_PENDING, timeout_ms),
        )
        return row["cnt"] if row else 0

    async def acquire_task(self) -> dict | None:
        now = _now_ms()
        timeout = now - 30 * 60 * 1000  # 30min lock timeout
        return await self.db.fetchone(
            "UPDATE parses SET locked_at=?, status=? "
            "WHERE id = ("
            "  SELECT id FROM parses WHERE status=? "
            "  AND (locked_at IS NULL OR locked_at < ?) "
            "  ORDER BY priority DESC, created_at ASC LIMIT 1"
            ") RETURNING *",
            (now, PARSE_STATUS_PARSING, PARSE_STATUS_PENDING, timeout),
        )

    async def process_doc(self, task: dict) -> bool:
        """Execute parse for a batch.  Returns True on success."""
        sha256 = task["sha256"]
        tier = task["tier"]
        pages = task["pages"]
        privacy = task.get("privacy", "local")

        # guard
        current = await self.db.fetchone("SELECT status FROM parses WHERE id=?", (task["id"],))
        if current is None or current["status"] != PARSE_STATUS_PARSING:
            return False

        # find the file
        file_row = await self.db.fetchone(
            "SELECT * FROM files WHERE sha256=? AND scan_status=? LIMIT 1",
            (sha256, SCAN_STATUS_ACTIVE),
        )
        if file_row is None:
            await self._fail_task(task["id"], "no_accessible_file", "No active file found for this document")
            return False

        output_dir = os.path.join(self.data_dir, "parsed", sha256[:2], sha256, tier)
        os.makedirs(output_dir, exist_ok=True)

        # route parse based on tier
        via = "local"
        try:
            if tier == "flash":
                results = await self._parse_via_local(file_row, tier, pages)
            else:
                results, via = await self._parse_via_api(file_row, tier, pages, privacy)
        except ParseFailure as exc:
            await self._fail_task(task["id"], exc.code, exc.message)
            return False
        except Exception as exc:
            await self._fail_task(task["id"], "parse_failed", str(exc)[:500])
            return False

        # save per-batch JSON (markdown is generated on read from /docs/{sha256}/content)
        done_at_ms = _now_ms()
        new_pages: list[dict] = []
        if results:
            import json as _json

            pages_key = _safe_filename(task["pages"])
            json_path = os.path.join(output_dir, f"{pages_key}_{done_at_ms}.json")
            os.makedirs(output_dir, exist_ok=True)
            for r in results:
                if hasattr(r, "to_dict"):
                    new_pages.extend(r.to_dict().get("pages", []))
            if not new_pages:
                await self._fail_task(task["id"], "parse_empty", "Parse completed but returned no pages")
                return False
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump({"pages": new_pages}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        else:
            await self._fail_task(task["id"], "parse_empty", "Parse completed but returned no results")
            return False

        # generate markdown for FTS (not persisted, only for search indexing)
        md_text = ""
        for result in results:
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

    async def _parse_via_local(self, file_row: dict, tier: Tier, pages: str) -> list:
        """Parse via local library call."""
        from ...parser import parse

        result = await asyncio.to_thread(
            parse,
            file_row["path"],
            tier=tier,
            page_range=pages,
        )
        return [result]

    async def _parse_via_api(
        self,
        file_row: dict,
        tier: Tier,
        pages: str,
        privacy: str,
    ) -> tuple[list, str]:
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
        result = await parser.parse_async(file_row["path"], page_range=pages)
        return [result], via

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
            url = (await self.config_svc.get("parse_server.remote.url")) or "https://mineru.net/api"
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
        row = await self.db.fetchone("SELECT * FROM parses WHERE id=?", (parse_id,))
        return _parse_record_response(row) if row else None

    async def list_parse_records(
        self,
        *,
        ids: list[int] | None = None,
        sha256: str | None = None,
        tier: Tier | None = None,
        status: list[str] | None = None,
        pages: str | None = None,
        include_superseded: bool = False,
    ) -> dict:
        if ids:
            placeholders = ",".join("?" * len(ids))
            rows = await self.db.fetchall(f"SELECT * FROM parses WHERE id IN ({placeholders})", tuple(ids))
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
        rows = await self.db.fetchall(sql, tuple(params))
        result = {"parses": [_parse_record_response(row) for row in rows]}
        if sha256 and tier and pages:
            result["coverage"] = _parse_coverage(pages, rows)
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

    async def _maybe_update_fts(self, sha256: str, tier: Tier, text: str, file_row: dict) -> None:
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
        file_row = await self.db.fetchone(
            "SELECT * FROM files WHERE sha256=? AND scan_status=? LIMIT 1",
            (sha256, SCAN_STATUS_ACTIVE),
        )
        if file_row is None:
            await self.fts.delete(sha256)
            return

        rows = await self.db.fetchall(
            "SELECT * FROM parses WHERE sha256=? AND status=? ORDER BY done_at DESC",
            (sha256, PARSE_STATUS_DONE),
        )
        tiers = sorted({row["tier"] for row in rows}, key=lambda t: TIER_ORDER.get(t, -1), reverse=True)
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


def _json_file_exists_by_batch(data_dir: str, sha256: str, tier: Tier, batch: dict) -> bool:
    """Check that the JSON result file for a parses batch row actually exists on disk."""
    json_path = parse_batch_json_path(data_dir, sha256, tier, batch["pages"], batch["done_at"])
    return os.path.isfile(json_path)


def parse_batch_json_path(data_dir: str, sha256: str, tier: Tier, pages: str, done_at: int | None = 0) -> str:
    """Return the persisted Middle JSON path for one parse batch."""
    key = _safe_filename(pages, done_at or 0)
    return os.path.join(os.path.expanduser(data_dir), "parsed", sha256[:2], sha256, tier, f"{key}.json")


def load_pages_from_done_batches(data_dir: str, sha256: str, tier: Tier, done_rows: list[dict]) -> list[PageInfo]:
    """Load valid done JSON batches and keep the newest page for duplicate page_idx values."""
    pages_by_idx: dict[int, PageInfo] = {}
    for row in reversed(done_rows):
        fpath = parse_batch_json_path(data_dir, sha256, tier, row["pages"], row["done_at"])
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = _json.load(f)
            for raw in data.get("pages", []):
                page = PageInfo.from_dict(raw)
                pages_by_idx[page.page_idx] = page
        except Exception:
            pass
    return [pages_by_idx[idx] for idx in sorted(pages_by_idx)]


def _done_response(sha256: str, tier: Tier, pages: str) -> dict:
    return {
        "sha256": sha256,
        "tier": tier,
        "pages": pages,
        "status": PARSE_STATUS_DONE,
        "cache_hit": True,
        "wait_parse_ids": [],
        "created_parse_ids": [],
        "reused_parse_ids": [],
        "tip": "Cached. Use --force to re-parse.",
    }


def _text_response(sha256: str) -> dict:
    return {
        "sha256": sha256,
        "tier": "flash",
        "pages": "1",
        "status": PARSE_STATUS_DONE,
        "cache_hit": False,
        "wait_parse_ids": [],
        "created_parse_ids": [],
        "reused_parse_ids": [],
        "tip": "Plain text files do not require parsing.",
    }


def _parse_record_response(row: dict) -> dict:
    error = None
    if row.get("error_code") or row.get("error_msg"):
        error = {"code": row.get("error_code"), "message": row.get("error_msg")}
    return {
        "id": row["id"],
        "sha256": row["sha256"],
        "tier": row["tier"],
        "pages": row["pages"],
        "status": row["status"],
        "done_at": row.get("done_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "error": error,
    }


def _parse_coverage(request_pages: str, rows: list[dict]) -> dict:
    requested = parse_range_set(request_pages)
    done_pages: set[int] = set()
    active_pages: set[int] = set()
    for row in rows:
        row_pages = parse_range_set(row["pages"]) & requested
        if row["status"] == PARSE_STATUS_DONE:
            done_pages |= row_pages
        elif row["status"] in (PARSE_STATUS_PENDING, PARSE_STATUS_PARSING):
            active_pages |= row_pages
    active_pages -= done_pages
    missing_pages = requested - done_pages - active_pages
    return {
        "done_pages": _pages_set_to_str(done_pages),
        "active_pages": _pages_set_to_str(active_pages),
        "missing_pages": _pages_set_to_str(missing_pages),
    }


def _local_parse_server_url(mode: str, health: object) -> str | None:
    """Resolve local parse-server URL from mode and health state."""
    if mode == "managed":
        return "http://127.0.0.1:15981"
    if mode == "self_hosted":
        return getattr(health, "self_hosted_url", None)
    return None


def _safe_filename(s: str, done_at: int = 0) -> str:
    """Convert a pages range string + done_at to a filename."""
    return f"{s}_{done_at}" if done_at else s
