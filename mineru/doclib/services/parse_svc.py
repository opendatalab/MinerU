"""Parse service: file ingestion, parse request/acquire/process."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from mineru.constants import TIER_ORDER, ParseStatus, Tier
from mineru.errors import MineruError

from ..core.db import DatabaseManager
from ..core.file_io import compute_sha256, extract_metadata, get_file_stat
from ..core.fts import FTSManager
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
        data_dir: str = "~/MinerU",
    ) -> None:
        self.db = db
        self.fts = fts
        self.config_svc = config_svc
        self.data_dir = os.path.expanduser(data_dir)

    # ── ingestion (shared) ──────────────────────────────────

    async def ingest_file(self, path: str, watch_id: int | None = None) -> dict | None:
        """Ingest a file: SHA-256 + metadata + trigger default parse."""
        stat = await get_file_stat(path)
        sha256 = await compute_sha256(path)
        now = _now_ms()

        # check if this path is already ingested
        existing_path = await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,))
        if existing_path and existing_path["sha256"]:
            return existing_path

        # check if same content (sha256) is already tracked by another path
        existing_sha = await self.db.fetchone("SELECT * FROM files WHERE sha256=?", (sha256,))
        if existing_sha:
            # same file, different path — just add this path pointing to the same doc
            await self.db.execute("DELETE FROM files WHERE path=?", (path,))
            await self.db.execute(
                "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, "
                "birthtime_ms, sha256, watch_id, first_seen_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    path,
                    Path(path).name,
                    Path(path).suffix.lstrip(".").lower(),
                    stat["size_bytes"],
                    stat["mtime_ms"],
                    stat["birthtime_ms"],
                    sha256,
                    watch_id,
                    now,
                    now,
                ),
            )
            return await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,))

        # brand new document
        metadata = await extract_metadata(path)

        # Office: no page_count → default to 1
        page_count = metadata["page_count"]
        if page_count is None:
            ext = Path(path).suffix.lower().lstrip(".")
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
                Path(path).name,
                Path(path).suffix.lstrip(".").lower(),
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
            "title, author, subject, keywords, is_encrypted, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sha256,
                stat["size_bytes"],
                metadata["mime_type"],
                page_count,
                metadata["title"],
                metadata["author"],
                metadata["subject"],
                metadata["keywords"],
                metadata["is_encrypted"],
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
                Path(path).suffix.lstrip(".").lower(),
            )

        # determine tier and pages for initial parse
        tier = Tier.FLASH
        if page_count and page_count > 10:
            initial_pages = "1~5,-5~-1"
        elif page_count:
            initial_pages = f"1~{page_count}"
        else:
            initial_pages = "1~5"

        # check parsing-rules
        matched = await self.config_svc.match_rules(path, "parsing_rule")
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
            (sha256, tier, pages_str, ParseStatus.PENDING, now, now),
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
        tier: str | None = None,
        pages: str | None = None,
        force: bool = False,
        remote: bool = False,
        remote_url: str | None = None,
    ) -> dict:
        """Handle a parse request from CLI.  Returns info for status polling."""
        # ensure file is ingested
        file_row = await self.db.fetchone("SELECT * FROM files WHERE path=? AND scan_status='active'", (path,))
        if file_row is None or file_row["sha256"] is None:
            file_row = await self.ingest_file(path)
        if file_row is None:
            return {"sha256": "", "tier": tier or "", "status": "error", "tip": "File could not be ingested."}

        sha256 = file_row["sha256"]
        doc = await self.db.fetchone("SELECT page_count FROM docs WHERE sha256=?", (sha256,))
        page_count = doc["page_count"] if doc else 1
        privacy = "remote" if remote else "local"

        # ── resolve tier (before cache check) ──
        requested_tier = tier or _resolve_default_tier(remote)
        ext = file_row["ext"]
        if ext in ("docx", "pptx", "xlsx", "doc", "xls", "ppt"):
            requested_tier = Tier.FLASH

        # ── expand pages ──
        default_pages = "1~5,-5~-1" if page_count and page_count > 10 else f"1~{page_count}"
        raw_pages = pages or default_pages
        request_pages_str = expand_pages(raw_pages, page_count or 1)
        needed = parse_range_set(request_pages_str)

        # ── step 1: remove pages covered by valid done batches ──
        if not force:
            done_batches = await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND tier=? AND status='done' ORDER BY done_at DESC",
                (sha256, requested_tier),
            )
            for batch in done_batches:
                if not _json_file_exists_by_batch(sha256, requested_tier, batch):
                    continue  # JSON gone → cache invalid
                covered = parse_range_set(batch["pages"])
                needed -= covered

            if not needed:
                return _done_response(sha256, requested_tier, request_pages_str)

        # ── step 2: remove pages covered by pending/parsing batches ──
        active_batches = await self.db.fetchall(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN ('pending', 'parsing')",
            (sha256, requested_tier),
        )
        if active_batches:
            active_covered: set[int] = set()
            for batch in active_batches:
                active_covered |= parse_range_set(batch["pages"])
            needed -= active_covered

            if not needed:
                # all remaining pages covered by in-progress batches → bump priority
                now = _now_ms()
                for batch in active_batches:
                    if batch["priority"] < 1:
                        await self.db.execute(
                            "UPDATE parses SET priority=1, updated_at=? WHERE id=?",
                            (now, batch["id"]),
                        )
                return {
                    "sha256": sha256,
                    "tier": requested_tier,
                    "pages": request_pages_str,
                    "status": ParseStatus.PENDING,
                    "tip": "Pages already queued. Priority bumped.",
                }

        # ── step 3: enqueue remaining uncovered pages ──
        uncovered_str = _pages_set_to_str(needed)
        now = _now_ms()
        await self.db.execute(
            "INSERT INTO parses (sha256, tier, pages, status, privacy, remote_url, priority, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
            (sha256, requested_tier, uncovered_str, ParseStatus.PENDING, privacy, remote_url, now, now),
        )
        return {
            "sha256": sha256,
            "tier": requested_tier,
            "pages": uncovered_str,
            "status": ParseStatus.PENDING,
        }

    # ── worker ──────────────────────────────────────────────────

    async def get_queue_length(self) -> int:
        timeout_ms = _now_ms() - 30 * 60 * 1000  # 30min lock timeout
        row = await self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM parses WHERE status=? AND (locked_at IS NULL OR locked_at < ?)",
            (ParseStatus.PENDING, timeout_ms),
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
            (now, ParseStatus.PARSING, ParseStatus.PENDING, timeout),
        )

    async def process_doc(self, task: dict) -> bool:
        """Execute parse for a batch.  Returns True on success."""
        sha256 = task["sha256"]
        tier = task["tier"]
        pages = task["pages"]
        privacy = task.get("privacy", "local")
        remote_url = task.get("remote_url")

        # guard
        current = await self.db.fetchone("SELECT status FROM parses WHERE id=?", (task["id"],))
        if current is None or current["status"] != ParseStatus.PARSING:
            return False

        # find the file
        file_row = await self.db.fetchone(
            "SELECT * FROM files WHERE sha256=? AND scan_status='active' LIMIT 1",
            (sha256,),
        )
        if file_row is None:
            await self._fail_task(task["id"], "no_accessible_file", "No active file found for this document")
            return False

        output_dir = os.path.join(self.data_dir, "parsed", sha256[:2], sha256, tier)
        os.makedirs(output_dir, exist_ok=True)

        # route parse based on tier
        via = "local"
        try:
            if tier == Tier.FLASH:
                results = await self._parse_via_local(file_row, tier, pages)
            else:
                results, via = await self._parse_via_api(file_row, tier, pages, privacy, remote_url)
        except ParseFailure as exc:
            await self._fail_task(task["id"], exc.code, exc.message)
            return False
        except Exception as exc:
            await self._fail_task(task["id"], "parse_failed", str(exc)[:500])
            return False

        # save per-batch JSON (markdown is generated on read from /parse/content)
        done_at_ms = _now_ms()
        if results:
            import json as _json

            pages_key = _safe_filename(task["pages"])
            json_path = os.path.join(output_dir, f"{pages_key}_{done_at_ms}.json")
            os.makedirs(output_dir, exist_ok=True)
            new_pages: list[dict] = []
            for r in results:
                if hasattr(r, "to_dict"):
                    new_pages.extend(r.to_dict().get("pages", []))
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump({"pages": new_pages}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

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
            (ParseStatus.DONE, done_at_ms, via, done_at_ms, task["id"]),
        )
        return True

    # ── parse routing helpers ─────────────────────────────────────

    async def _parse_via_local(self, file_row: dict, tier: str, pages: str) -> list:
        """Parse via local library call."""
        from mineru.parser import parse

        result = await asyncio.to_thread(
            parse,
            file_row["path"],
            backend=_tier_to_backend(tier),
            page_range=pages,
        )
        return [result]

    async def _parse_via_api(
        self,
        file_row: dict,
        tier: str,
        pages: str,
        privacy: str,
        remote_url: str | None,
    ) -> tuple[list, str]:
        """Parse via HTTP API (local or remote parse-server). Returns (results, via)."""
        from mineru.parser.api_client import MinerUApiParser

        base_url, api_key, via = await self._resolve_api_target(privacy, remote_url, tier)

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
    def _resolve_tier(tier: str, via: str) -> str:
        """Resolve and validate tier against parse-server capabilities.

        - ``auto`` → pick best available tier from health check (pro > standard)
        - specified tier → validate against supported_tiers, raise tier_mismatch if unsupported
        """
        from mineru.doclib.background.parse_server_health import get_health

        health = get_health()
        if via == "remote":
            supported = health.remote_supported_tiers
        else:
            supported = health.local_supported_tiers

        if not supported:
            raise ParseFailure("engine_unavailable", "Parse-server health status unknown, cannot validate tier")

        if tier == "auto":
            tier_order = ["pro", "standard"]
            for t in tier_order:
                if t in supported:
                    return t
            raise ParseFailure("tier_mismatch", f"No supported tier found. Available: {supported}")

        if tier not in supported:
            raise ParseFailure("tier_mismatch", f"Tier '{tier}' not supported by parse-server. Available: {supported}")
        return tier

    async def _resolve_api_target(self, privacy: str, remote_url: str | None, tier: str) -> tuple[str, str | None, str]:
        """Resolve the api target URL based on privacy and config.

        Returns (base_url, api_key, via_label).
        """
        from mineru.doclib.background.parse_server_health import get_health

        health = get_health()

        if privacy == "remote":
            url = remote_url or (await self.config_svc.get("parse_server.remote.url")) or "https://mineru.net/api"
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
            (ParseStatus.FAILED, code, message, now, task_id),
        )

    async def get_parse_status(self, sha256: str, tier: str) -> dict | None:
        """Get aggregate status for a doc+tier.  Returns latest batch info."""
        done = await self.db.fetchone(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status='done' ORDER BY done_at DESC LIMIT 1",
            (sha256, tier),
        )
        if done:
            # still have active batches? → not fully done yet
            active_count = await self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM parses WHERE sha256=? AND tier=? AND status IN ('pending', 'parsing')",
                (sha256, tier),
            )
            if active_count and active_count["cnt"] == 0:
                return {"sha256": sha256, "tier": tier, "status": "done", "pages": "TBD"}

        active = await self.db.fetchone(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN ('pending', 'parsing') "
            "ORDER BY created_at DESC LIMIT 1",
            (sha256, tier),
        )
        if active:
            return {
                "sha256": sha256,
                "tier": tier,
                "status": active["status"],
                "pages": active["pages"],
            }

        failed = await self.db.fetchone(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status='failed' ORDER BY created_at DESC LIMIT 1",
            (sha256, tier),
        )
        if failed:
            return {
                "sha256": sha256,
                "tier": tier,
                "status": "failed",
                "error": {"code": failed["error_code"], "message": failed["error_msg"]},
            }

        return None

    async def invalidate(self, sha256: str, tier: str | None = None) -> int:
        """Mark done parses as superseded.  Returns count of affected rows."""
        now = _now_ms()
        sql = "UPDATE parses SET status='superseded', updated_at=? WHERE sha256=? AND status='done'"
        params = [now, sha256]
        if tier:
            sql += " AND tier=?"
            params.append(tier)
        cursor = await self.db.execute(sql, tuple(params))
        return cursor.rowcount

    # ── internal helpers ────────────────────────────────────────

    async def _maybe_update_fts(self, sha256: str, tier: str, text: str, file_row: dict) -> None:
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

    async def _maybe_update_docs_meta(self, sha256: str, tier: str) -> None:
        doc = await self.db.fetchone("SELECT meta_tier FROM docs WHERE sha256=?", (sha256,))
        if doc is None:
            return
        existing_tier = doc.get("meta_tier")
        if existing_tier and TIER_ORDER.get(existing_tier, -1) >= TIER_ORDER.get(tier, -1):
            return

        # for now just update meta_tier; full metadata update comes when
        # engine provides richer output
        now = _now_ms()
        await self.db.execute(
            "UPDATE docs SET meta_tier=?, updated_at=? WHERE sha256=?",
            (tier, now, sha256),
        )


# ── tier → backend mapping ─────────────────────────────────────────


def _resolve_default_tier(remote: bool = False) -> str:
    """Pick the best available tier from health check.  ``pro`` > ``standard`` > ``flash``."""
    from mineru.doclib.background.parse_server_health import get_health

    health = get_health()
    supported = health.remote_supported_tiers if remote else health.local_supported_tiers
    for candidate in (Tier.PRO, Tier.STANDARD):
        if candidate in supported:
            return candidate
    raise ParseFailure(
        "no_engine", "No standard or pro engine available. Start a parse-server or use --tier flash for text-only preview."
    )


def _json_file_exists_by_batch(sha256: str, tier: str, batch: dict) -> bool:
    """Check that the JSON result file for a parses batch row actually exists on disk."""
    data_dir = os.path.expanduser("~/MinerU")
    key = _safe_filename(batch["pages"], batch["done_at"])
    json_path = os.path.join(data_dir, "parsed", sha256[:2], sha256, tier, f"{key}.json")
    return os.path.isfile(json_path)


def _tier_to_backend(tier: str) -> str:
    mapping = {
        Tier.FLASH: "flash",
        Tier.STANDARD: "pipeline",
        Tier.PRO: "hybrid-auto-engine",
    }
    return mapping.get(tier, "pipeline")


def _done_response(sha256: str, tier: str, pages: str) -> dict:
    return {
        "sha256": sha256,
        "tier": tier,
        "pages": pages,
        "status": "done",
        "tip": "Cached. Use --force to re-parse.",
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
