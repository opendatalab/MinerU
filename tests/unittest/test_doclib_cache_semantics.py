from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from mineru.doclib.background.compaction import Compaction
from mineru.doclib.background.parse_server_health import api_server_args_for_tier
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services import parse_svc as parse_svc_module
from mineru.doclib.services.parse_svc import (
    ParseService,
    filter_pages_by_user_range,
    load_pages_from_done_batches,
    parse_batch_json_path,
)
from mineru.parser import backend_for_tier, resolve_tier_and_backend
from mineru.types import PageInfo, Tier


class _Cursor:
    def __init__(self, rowcount: int, lastrowid: int | None = None) -> None:
        self.rowcount = rowcount
        self.lastrowid = lastrowid


class _FakeDB:
    def __init__(
        self,
        *,
        parses: list[dict[str, Any]],
        file_row: dict[str, Any] | None,
        doc_row: dict[str, Any] | None = None,
    ) -> None:
        self.parses = parses
        self.file_row = file_row
        self.doc_row = doc_row
        self.updated_priorities: list[int] = []

    async def execute(self, sql: str, params: tuple[Any, ...]) -> _Cursor:
        if sql.startswith("UPDATE parses SET status=?, updated_at=?"):
            status, _, sha256, done_status, *rest = params
            tier = rest[0] if rest else None
            rowcount = 0
            for row in self.parses:
                if row["sha256"] != sha256 or row["status"] != done_status:
                    continue
                if tier and row["tier"] != tier:
                    continue
                row["status"] = status
                rowcount += 1
            return _Cursor(rowcount)
        if sql.startswith("UPDATE parses SET priority=1"):
            parse_id = params[1]
            self.updated_priorities.append(parse_id)
            for row in self.parses:
                if row["id"] == parse_id:
                    row["priority"] = 1
                    return _Cursor(1)
        if sql.startswith("UPDATE parses SET status=?, error_code=?, error_msg=?"):
            status, error_code, error_msg, updated_at, parse_id = params
            for row in self.parses:
                if row["id"] == parse_id:
                    row["status"] = status
                    row["error_code"] = error_code
                    row["error_msg"] = error_msg
                    row["locked_at"] = None
                    row["updated_at"] = updated_at
                    return _Cursor(1)
        if sql.startswith("UPDATE parses SET status=?, done_at=?"):
            status, done_at, via, updated_at, parse_id = params
            for row in self.parses:
                if row["id"] == parse_id:
                    row["status"] = status
                    row["done_at"] = done_at
                    row["via"] = via
                    row["locked_at"] = None
                    row["updated_at"] = updated_at
                    return _Cursor(1)
        return _Cursor(0)

    async def execute_insert(self, sql: str, params: tuple[Any, ...]) -> int:
        if sql.startswith("INSERT INTO parses"):
            parse_id = max((row.get("id", 0) for row in self.parses), default=0) + 1
            sha256, tier, pages, status, privacy, created_at, updated_at = params
            self.parses.append(
                {
                    "id": parse_id,
                    "sha256": sha256,
                    "tier": tier,
                    "pages": pages,
                    "status": status,
                    "privacy": privacy,
                    "priority": 1,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "done_at": None,
                }
            )
            return parse_id
        return 0

    async def fetchone(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        if sql.startswith("SELECT * FROM files WHERE path="):
            path = params[0]
            if self.file_row and self.file_row["path"] == path and self.file_row["scan_status"] == "active":
                return self.file_row
        if sql.startswith("SELECT page_count FROM docs WHERE sha256="):
            sha256 = params[0]
            if self.doc_row and self.doc_row["sha256"] == sha256:
                return self.doc_row
        if sql.startswith("SELECT * FROM files WHERE sha256="):
            sha256 = params[0]
            if self.file_row and self.file_row["sha256"] == sha256 and self.file_row["scan_status"] == "active":
                return self.file_row
        if sql.startswith("SELECT status FROM parses WHERE id=") or sql.startswith("SELECT * FROM parses WHERE id="):
            parse_id = params[0]
            for row in self.parses:
                if row["id"] == parse_id:
                    return row
        return None

    async def fetchall(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if sql.startswith("SELECT * FROM parses WHERE id IN"):
            ids = set(params)
            return [row for row in self.parses if row["id"] in ids]
        if sql.startswith("SELECT * FROM parses WHERE sha256=? AND tier=? AND status=?"):
            sha256, tier, status = params
            rows = [
                row
                for row in self.parses
                if row["sha256"] == sha256 and row["tier"] == tier and row["status"] == status
            ]
            return sorted(rows, key=lambda row: row["done_at"] or 0, reverse=True)
        if sql.startswith("SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN (?, ?)"):
            sha256, tier, *statuses = params
            return [
                row
                for row in self.parses
                if row["sha256"] == sha256 and row["tier"] == tier and row["status"] in statuses
            ]
        if sql.startswith("SELECT * FROM parses WHERE sha256=? AND status=?"):
            sha256, status = params
            rows = [row for row in self.parses if row["sha256"] == sha256 and row["status"] == status]
            return sorted(rows, key=lambda row: row["done_at"], reverse=True)
        return []


class _FakeFTS:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.replaced: list[dict[str, Any]] = []

    async def delete(self, sha256: str) -> None:
        self.deleted.append(sha256)

    async def replace(self, **kwargs: Any) -> None:
        self.replaced.append(kwargs)


def _write_batch(data_dir: Path, sha256: str, tier: Tier, pages: str, done_at: int, json_pages: list[dict]) -> None:
    path = Path(parse_batch_json_path(str(data_dir), sha256, tier, pages, done_at))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pages": json_pages}), encoding="utf-8")


def test_load_pages_from_done_batches_keeps_newest_page_idx(tmp_path: Path) -> None:
    sha256 = "a" * 64
    tier = "standard"
    older_page = {"page_idx": 1, "page_size": [100, 100], "para_blocks": []}
    older_duplicate = {"page_idx": 2, "page_size": [100, 100], "para_blocks": []}
    newer_duplicate = {"page_idx": 2, "page_size": [200, 200], "para_blocks": []}

    _write_batch(tmp_path, sha256, tier, "1~2", 1000, [older_page, older_duplicate])
    _write_batch(tmp_path, sha256, tier, "2", 2000, [newer_duplicate])

    done_rows = [
        {"pages": "2", "done_at": 2000},
        {"pages": "1~2", "done_at": 1000},
    ]

    pages = load_pages_from_done_batches(str(tmp_path), sha256, tier, done_rows)

    assert [page.page_idx for page in pages] == [1, 2]
    assert pages[0].page_size == (100, 100)
    assert pages[1].page_size == (200, 200)


def test_parser_tier_backend_mapping_is_parser_layer_only() -> None:
    assert backend_for_tier("flash") == "flash"
    assert backend_for_tier("standard") == "pipeline"
    assert backend_for_tier("pro") == "hybrid-auto-engine"
    assert resolve_tier_and_backend(tier=None) == ("pro", "hybrid-auto-engine")
    assert resolve_tier_and_backend(tier="pro", backend="vlm-auto-engine") == ("pro", "vlm-auto-engine")


def test_managed_api_server_args_use_tier_for_process_start() -> None:
    assert api_server_args_for_tier("standard") == ["--tier", "standard", "--port", "15981"]
    assert api_server_args_for_tier("pro") == ["--tier", "pro", "--port", "15981"]


def test_compaction_uses_configured_data_dir(tmp_path: Path) -> None:
    sha256 = "b" * 64
    tier = "standard"
    older_page = {"page_idx": 1, "content": "old"}
    older_duplicate = {"page_idx": 2, "content": "old"}
    newer_duplicate = {"page_idx": 2, "content": "new"}

    _write_batch(tmp_path, sha256, tier, "1~2", 1000, [older_page, older_duplicate])
    _write_batch(tmp_path, sha256, tier, "2", 2000, [newer_duplicate])

    compaction = Compaction(db=None, interval_sec=600, data_dir=str(tmp_path))
    done_rows = [
        {"pages": "2", "done_at": 2000},
        {"pages": "1~2", "done_at": 1000},
    ]

    asyncio.run(compaction._compact_json(sha256, tier, ["1~2"], done_rows, 2000))

    compacted_path = Path(parse_batch_json_path(str(tmp_path), sha256, tier, "1~2", 2000))
    compacted = json.loads(compacted_path.read_text(encoding="utf-8"))

    assert compacted["pages"] == [older_page, newer_duplicate]
    assert sorted(path.name for path in compacted_path.parent.glob("*.json")) == ["1~2_2000.json"]


def test_invalidate_deletes_fts_when_no_done_batches_remain(tmp_path: Path) -> None:
    sha256 = "c" * 64
    parses = [{"sha256": sha256, "tier": "standard", "pages": "1", "status": "done", "done_at": 1000}]
    db = _FakeDB(parses=parses, file_row={"sha256": sha256, "scan_status": "active", "filename": "doc.pdf"})
    fts = _FakeFTS()
    service = ParseService(db=db, fts=fts, config_svc=None, data_dir=str(tmp_path))

    count = asyncio.run(service.invalidate(sha256, "standard"))

    assert count == 1
    assert parses[0]["status"] == "superseded"
    assert fts.deleted == [sha256]
    assert fts.replaced == []


def test_invalidate_rebuilds_fts_from_highest_remaining_done_tier(tmp_path: Path) -> None:
    sha256 = "d" * 64
    _write_batch(tmp_path, sha256, "flash", "1", 1000, [{"page_idx": 1, "page_size": [100, 100], "para_blocks": []}])
    _write_batch(tmp_path, sha256, "standard", "1", 2000, [{"page_idx": 1, "page_size": [200, 200], "para_blocks": []}])
    parses = [
        {"sha256": sha256, "tier": "flash", "pages": "1", "status": "done", "done_at": 1000},
        {"sha256": sha256, "tier": "standard", "pages": "1", "status": "done", "done_at": 2000},
    ]
    db = _FakeDB(parses=parses, file_row={"sha256": sha256, "scan_status": "active", "filename": "doc.pdf"})
    fts = _FakeFTS()
    service = ParseService(db=db, fts=fts, config_svc=None, data_dir=str(tmp_path))

    count = asyncio.run(service.invalidate(sha256, "standard"))

    assert count == 1
    assert fts.deleted == []
    assert len(fts.replaced) == 1
    assert fts.replaced[0]["sha256"] == sha256
    assert fts.replaced[0]["tier"] == "flash"


def test_force_request_reuses_active_and_creates_only_uncovered_parse(tmp_path: Path) -> None:
    sha256 = "e" * 64
    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    stat = source.stat()
    path = str(source)
    parses = [
        {
            "id": 10,
            "sha256": sha256,
            "tier": "standard",
            "pages": "1~5",
            "status": "done",
            "priority": 0,
            "done_at": 1000,
            "created_at": 900,
        },
        {
            "id": 11,
            "sha256": sha256,
            "tier": "standard",
            "pages": "6~8",
            "status": "pending",
            "priority": 0,
            "done_at": None,
            "created_at": 1100,
        },
    ]
    db = _FakeDB(
        parses=parses,
        file_row={
            "path": path,
            "sha256": sha256,
            "scan_status": "active",
            "ext": "pdf",
            "mtime_ms": int(stat.st_mtime * 1000),
            "size_bytes": stat.st_size,
        },
        doc_row={"sha256": sha256, "page_count": 10},
    )
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path))

    result = asyncio.run(service.request_parse(path, tier="standard", pages="1~10", force=True))

    assert result["wait_parse_ids"] == [11, 12]
    assert result["reused_parse_ids"] == [11]
    assert result["created_parse_ids"] == [12]
    assert result["pages"] == "1~10"
    assert result["status"] == "pending"
    assert result["cache_hit"] is False
    assert db.updated_priorities == [11]
    assert parses[-1]["pages"] == "1~5,9~10"


def test_list_parse_records_by_ids_returns_precise_status(tmp_path: Path) -> None:
    parses = [
        {"id": 1, "sha256": "f" * 64, "tier": "standard", "pages": "1~5", "status": "done", "done_at": 1000},
        {"id": 2, "sha256": "f" * 64, "tier": "standard", "pages": "6~10", "status": "failed", "error_code": "parse_failed", "error_msg": "boom"},
    ]
    db = _FakeDB(parses=parses, file_row=None)
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path))

    result = asyncio.run(service.list_parse_records(ids=[2, 1]))

    assert result["parses"] == [
        {
            "id": 2,
            "sha256": "f" * 64,
            "tier": "standard",
            "pages": "6~10",
            "status": "failed",
            "done_at": None,
            "created_at": None,
            "updated_at": None,
            "error": {"code": "parse_failed", "message": "boom"},
        },
        {
            "id": 1,
            "sha256": "f" * 64,
            "tier": "standard",
            "pages": "1~5",
            "status": "done",
            "done_at": 1000,
            "created_at": None,
            "updated_at": None,
            "error": None,
        },
    ]


def test_filter_pages_by_user_range_uses_one_based_request_pages() -> None:
    pages = [PageInfo(page_idx=0), PageInfo(page_idx=1), PageInfo(page_idx=2)]

    selected = filter_pages_by_user_range(pages, "1")

    assert [page.page_idx for page in selected] == [0]


def test_ensure_ingested_rebinds_changed_text_file_to_new_sha(tmp_path: Path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "mineru.db"))
        await db.initialize()
        service = ParseService(db=db, fts=FTSManager(db), config_svc=None, data_dir=str(tmp_path / "data"))
        source = tmp_path / "note.txt"

        source.write_text("old", encoding="utf-8")
        first = await service.ensure_ingested(str(source))

        assert first is not None
        assert first["sha256"] == hashlib.sha256(b"old").hexdigest()

        source.write_text("new content", encoding="utf-8")
        discovered = await service.discover_file(str(source))
        changed_row = await db.fetchone("SELECT * FROM files WHERE path=?", (str(source),))

        assert discovered.changed is True
        assert discovered.needs_ingest is True
        assert changed_row is not None
        assert changed_row["sha256"] is None

        second = await service.ensure_ingested(str(source))

        assert second is not None
        assert second["sha256"] == hashlib.sha256(b"new content").hexdigest()
        fts_row = await db.fetchone("SELECT sha256, tier, filename FROM fts_contents WHERE sha256=?", (second["sha256"],))
        assert fts_row == {"sha256": second["sha256"], "tier": "flash", "filename": "note.txt"}

    asyncio.run(_run())


def test_ingest_records_doc_error_when_metadata_extraction_fails(tmp_path: Path, monkeypatch) -> None:
    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list:
            return []

    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "mineru.db"))
        await db.initialize()
        service = ParseService(db=db, fts=FTSManager(db), config_svc=_NoRulesConfig(), data_dir=str(tmp_path / "data"))
        source = tmp_path / "broken.pdf"
        source.write_bytes(b"%PDF-1.4\nnot actually a valid pdf")

        async def _fail_metadata(path: str) -> dict[str, Any]:
            raise RuntimeError("metadata boom")

        monkeypatch.setattr(parse_svc_module, "extract_metadata", _fail_metadata)

        row = await service.ingest_file(str(source))

        doc = await db.fetchone("SELECT error_code, error_msg FROM docs WHERE sha256=?", (row["sha256"],))
        assert doc is not None
        assert doc["error_code"] == "metadata_failed"
        assert doc["error_msg"] == "metadata boom"

    asyncio.run(_run())


def test_process_doc_marks_empty_page_result_failed(tmp_path: Path) -> None:
    sha256 = "a" * 64
    task = {
        "id": 1,
        "sha256": sha256,
        "tier": "flash",
        "pages": "1",
        "status": "parsing",
        "privacy": "local",
    }
    parses = [
        {
            **task,
            "error_code": None,
            "error_msg": None,
            "done_at": None,
            "locked_at": 123,
            "updated_at": 123,
        }
    ]
    db = _FakeDB(
        parses=parses,
        file_row={
            "path": "/tmp/doc.pdf",
            "sha256": sha256,
            "scan_status": "active",
            "filename": "doc.pdf",
            "title": "",
            "author": "",
        },
    )
    service = ParseService(db=db, fts=_FakeFTS(), config_svc=None, data_dir=str(tmp_path))

    class _EmptyResult:
        def to_dict(self) -> dict[str, list]:
            return {"pages": []}

        def markdown(self, *, add_markers: bool = False) -> str:
            return ""

    async def _empty_parse(file_row: dict, tier: Tier, pages: str) -> list:
        return [_EmptyResult()]

    service._parse_via_local = _empty_parse  # type: ignore[method-assign]

    success = asyncio.run(service.process_doc(task))

    assert success is False
    assert parses[0]["status"] == "failed"
    assert parses[0]["error_code"] == "parse_empty"
    assert list(tmp_path.rglob("*.json")) == []
