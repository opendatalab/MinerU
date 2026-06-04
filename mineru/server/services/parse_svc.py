"""Parse service: file registration, parse request/acquire/process."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from mineru.constants import TIER_ORDER, ParseStatus, Tier

from ..core.file_io import compute_sha256, extract_metadata, get_file_stat


def _now_ms() -> int:
    return int(time.time() * 1000)


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
    def __init__(self, db, fts, config_svc, data_dir: str = "~/MinerU") -> None:
        self.db = db
        self.fts = fts
        self.config_svc = config_svc
        self.data_dir = os.path.expanduser(data_dir)

    # ── registration (shared) ──────────────────────────────────

    async def register_file(self, path: str, watch_id: int | None = None) -> dict | None:
        """Register a file: SHA-256 + metadata + trigger default parse."""
        stat = await get_file_stat(path)
        sha256 = await compute_sha256(path)
        now = _now_ms()

        # check if this path is already registered
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

    async def request_parse(self, path: str, *, tier: str | None = None, pages: str | None = None, force: bool = False) -> dict:
        """Handle a parse request from CLI.  Returns info for status polling."""
        # ensure file is registered
        file_row = await self.db.fetchone("SELECT * FROM files WHERE path=? AND scan_status='active'", (path,))
        if file_row is None or file_row["sha256"] is None:
            file_row = await self.register_file(path)
        if file_row is None:
            return {"sha256": "", "tier": tier or "", "status": "error", "tip": "File could not be registered."}

        sha256 = file_row["sha256"]
        doc = await self.db.fetchone("SELECT page_count FROM docs WHERE sha256=?", (sha256,))
        page_count = doc["page_count"] if doc else 1

        requested_tier = tier or Tier.FLASH

        # Office documents: always flash, ignore requested tier
        ext = file_row["ext"]
        if ext in ("docx", "pptx", "xlsx", "doc", "xls", "ppt"):
            requested_tier = Tier.FLASH

        # expand pages
        default_pages = f"1~5,-5~-1" if page_count and page_count > 10 else f"1~{page_count}"
        raw_pages = pages or default_pages
        pages_str = expand_pages(raw_pages, page_count or 1)

        # check cache
        if not force:
            done_batches = await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND tier=? AND status='done'",
                (sha256, requested_tier),
            )
            if done_batches:
                if pages_covered(pages_str, done_batches):
                    return {
                        "sha256": sha256,
                        "tier": requested_tier,
                        "pages": pages_str,
                        "status": "done",
                        "tip": "Cached. Use --force to re-parse.",
                    }
                # only parse pages not yet covered
                uncovered = pages_uncovered(pages_str, done_batches)
                if not uncovered:
                    return {
                        "sha256": sha256,
                        "tier": requested_tier,
                        "pages": pages_str,
                        "status": "done",
                        "tip": "Cached.",
                    }
                pages_str = _pages_set_to_str(uncovered)

        # check if there's already a pending/parsing batch with overlapping pages
        active = await self.db.fetchone(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN ('pending', 'parsing') "
            "ORDER BY created_at DESC LIMIT 1",
            (sha256, requested_tier),
        )
        if active:
            return {
                "sha256": sha256,
                "tier": requested_tier,
                "pages": pages_str,
                "status": active["status"],
                "tip": "Parse already in progress. Check status with: mineru info <path>",
            }

        # insert new batch
        now = _now_ms()
        await self.db.execute(
            "INSERT INTO parses (sha256, tier, pages, status, priority, created_at, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?)",
            (sha256, requested_tier, pages_str, ParseStatus.PENDING, now, now),
        )

        return {
            "sha256": sha256,
            "tier": requested_tier,
            "pages": pages_str,
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
            await self.db.execute(
                "UPDATE parses SET status=?, error_code=?, error_msg=?, locked_at=NULL WHERE id=?",
                (ParseStatus.FAILED, "no_accessible_file", "No active file found for this document", task["id"]),
            )
            return False

        output_dir = os.path.join(self.data_dir, "parsed", sha256[:2], sha256, tier)
        os.makedirs(output_dir, exist_ok=True)

        # call mineru.parser
        try:
            from mineru.parser import parse
        except ImportError:
            await self.db.execute(
                "UPDATE parses SET status=?, error_code=?, error_msg=?, locked_at=NULL WHERE id=?",
                (ParseStatus.FAILED, "no_engine", "mineru.parser module not available", task["id"]),
            )
            return False

        try:
            # parse each contiguous range sequentially, collect results
            results: list = []
            ranges = _split_ranges(pages, file_row, sha256)
            for start, end in ranges:
                result = await asyncio.to_thread(
                    parse,
                    file_row["path"],
                    backend=_tier_to_backend(tier),
                    start_page_id=start,
                    end_page_id=end,
                    output_dir=output_dir,
                )
                results.append(result)
        except Exception as exc:
            await self.db.execute(
                "UPDATE parses SET status=?, error_code=?, error_msg=?, locked_at=NULL WHERE id=?",
                (ParseStatus.FAILED, "parse_failed", str(exc)[:500], task["id"]),
            )
            return False

        # save per-batch JSON (each parses row → one JSON file)
        md_text = ""
        for result in results:
            md = result.markdown() if hasattr(result, "markdown") else ""
            md_text += md + "\n"

        if results:
            import json as _json

            pages_key = _safe_filename(task["pages"])
            json_path = os.path.join(output_dir, f"{pages_key}.json")
            new_pages: list[dict] = []
            for r in results:
                if hasattr(r, "to_dict"):
                    new_pages.extend(r.to_dict().get("pdf_info", []))

            os.makedirs(output_dir, exist_ok=True)
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump({"pdf_info": new_pages}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # update fts_contents (tier-gated)
        await self._maybe_update_fts(sha256, tier, md_text, file_row)

        # update docs metadata (tier-gated)
        await self._maybe_update_docs_meta(sha256, tier)

        # mark done
        now = _now_ms()
        await self.db.execute(
            "UPDATE parses SET status=?, done_at=?, locked_at=NULL, output_path=?, updated_at=? WHERE id=?",
            (ParseStatus.DONE, now, output_dir, now, task["id"]),
        )
        return True

    async def get_parse_status(self, sha256: str, tier: str) -> dict | None:
        """Get aggregate status for a doc+tier.  Returns latest batch info."""
        done = await self.db.fetchone(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status='done' ORDER BY done_at DESC LIMIT 1",
            (sha256, tier),
        )
        if done:
            return {"sha256": sha256, "tier": tier, "status": "done", "pages": "TBD"}

        active = await self.db.fetchone(
            "SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN ('pending', 'parsing') "
            "ORDER BY created_at DESC LIMIT 1",
            (sha256, tier),
        )
        if active:
            return {"sha256": sha256, "tier": tier, "status": active["status"], "pages": active["pages"]}

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


def _tier_to_backend(tier: str) -> str:
    mapping = {
        Tier.FLASH: "flash",
        Tier.STANDARD: "pipeline",
        Tier.PRO: "hybrid-auto-engine",
    }
    return mapping.get(tier, "pipeline")


def _split_ranges(pages: str, file_row: dict, sha256: str) -> list[tuple[int, int | None]]:
    """Split a pages range string into contiguous (start, end) tuples
    suitable for the current parse() API.  start is 0-based, end is inclusive."""
    page_set = parse_range_set(pages)
    if not page_set:
        return [(0, None)]

    sorted_pages = sorted(page_set)
    ranges: list[tuple[int, int | None]] = []
    start = sorted_pages[0]
    end = start

    for p in sorted_pages[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append((start - 1, end - 1))  # convert to 0-based
            start = p
            end = p
    ranges.append((start - 1, end - 1))
    return ranges


def _safe_filename(s: str) -> str:
    """Convert a pages range string to a safe filename — keep ~ and , as-is."""
    return s  # pages string is already safe: "1~5,43~47"
