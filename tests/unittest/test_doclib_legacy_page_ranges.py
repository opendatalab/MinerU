from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from mineru.doclib.background import compaction as compaction_module
from mineru.doclib.background.compaction import Compaction
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.server import DoclibServer, _parse_info, _tier_parse_info
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.parse_svc import (
    ParseService,
    _parse_record_response,
    load_pages_from_done_batches,
    parse_batch_json_path,
)
from mineru.parser.base import MIDDLE_JSON_SCHEMA_VERSION
from mineru.version import __version__

_SHA256 = "a" * 64


def _cache_path(root: Path, page_range: str, done_at: int) -> Path:
    """按数据库原始范围定位批次文件，保留旧半角波浪号文件名。"""
    return Path(parse_batch_json_path(str(root), _SHA256, "flash", page_range, done_at))


async def _add_result(
    db: DatabaseManager, root: Path, page_range: str, done_at: int, page_numbers: Iterable[int], label: str
) -> int:
    """写入真实历史或新格式批次及可辨识正文，构造混合缓存。"""
    path = _cache_path(root, page_range, done_at)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MIDDLE_JSON_SCHEMA_VERSION,
        "is_full_document": False,
        "file_suffix": "pdf",
        "effort": "flash",
        "parse_mode": "txt",
        "mineru_version": __version__,
        "pages": [
            {
                "page_idx": page_no - 1,
                "blocks": [
                    {
                        "type": "text",
                        "index": 0,
                        "bbox": [0, 0, 1, 1],
                        "content": [{"type": "text", "content": f"{label} page {page_no}"}],
                    }
                ],
            }
            for page_no in page_numbers
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return await db.execute_insert(
        "INSERT INTO parses (sha256, tier, page_range, status, privacy, done_at, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (_SHA256, "flash", page_range, "done", "local", done_at, done_at, done_at),
    )


@asynccontextmanager
async def _legacy_store(root: Path) -> AsyncIterator[tuple[DatabaseManager, ParseService, DoclibServer]]:
    """建立已入库的源文件元数据、旧名称缓存及真实 SQLite 服务，退出时关闭连接。"""
    source = root / "document.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    stat = source.stat()
    db = DatabaseManager(str(root / "doclib.db"))
    await db.initialize()
    try:
        await db.execute(
            "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (_SHA256, "aaaaaaa", stat.st_size, "pdf", 10, 1000, 1000),
        )
        await db.execute(
            "INSERT INTO files (path, filename, ext, size_bytes, mtime_ms, sha256, status, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(source), source.name, "pdf", stat.st_size, int(stat.st_mtime * 1000), _SHA256, "active", 1000, 1000),
        )
        await _add_result(db, root, "1~5", 1000, range(1, 6), "old")
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=ConfigService(db), data_dir=str(root), parse_lock_timeout_sec=1800
        )
        server = DoclibServer(SimpleNamespace(db=db, parse_svc=service, data_dir=str(root)))
        yield db, service, server
    finally:
        await db.close()


def test_old_cache_hits_exports_and_serializes_without_mutating_storage(tmp_path: Path) -> None:
    """旧结果正常命中、导出和查询，返回新格式而文件与数据库始终保留旧名称。"""

    async def verify() -> None:
        """读取各个结果出口，确认规范化不破坏原始缓存定位。"""
        async with _legacy_store(tmp_path) as (db, service, server):
            old_path = _cache_path(tmp_path, "1~5", 1000)
            original_bytes = old_path.read_bytes()
            row = await db.fetchone("SELECT * FROM parses WHERE sha256=?", (_SHA256,))
            original_row = dict(row)
            result = await service.request_parse(str(tmp_path / "document.pdf"), tier="flash", page_range="1-5")
            assert result.cache_hit and result.status == "done"
            assert result.page_range == "1-5"
            assert result.created_parse_ids == []
            listing = await server.list_parses(doc_ref="aaaaaaa", tier="flash", page_range="1-10")
            assert listing.parses[0].page_range == "1-5"
            assert listing.coverage.done_page_range == "1-5"
            assert listing.coverage.missing_page_range == "6-10"
            assert (await server.get_parse(row["id"])).page_range == "1-5"
            assert (await service.get_parse_record(row["id"]))["page_range"] == "1-5"
            assert (await service.list_parse_records(sha256=_SHA256))["parses"][0]["page_range"] == "1-5"
            public_row = dict(row, short_id="aaaaaaa")
            assert _parse_info(public_row).page_range == "1-5"
            assert _tier_parse_info(public_row).page_range == "1-5"
            assert _parse_record_response(public_row)["page_range"] == "1-5"
            assert public_row["page_range"] == "1~5"
            exported = await server._render_doc_content(
                _SHA256, tier="flash", page_range="1-5", format="markdown", no_marker=True
            )
            assert "old page 1" in exported and "old page 5" in exported
            content = await server.get_doc_content("aaaaaaa", tier="flash", page_range="1-5")
            assert content.request_scope.page_range == "1-5"
            assert "old page 3" in content.content
            assert row == original_row
            assert (await db.fetchone("SELECT * FROM parses WHERE id=?", (row["id"],)))["page_range"] == "1~5"
            assert old_path.read_bytes() == original_bytes
            assert not _cache_path(tmp_path, "1-5", 1000).exists()

    asyncio.run(verify())


def test_old_cache_only_schedules_uncovered_pages(tmp_path: Path) -> None:
    """旧 1~5 批次覆盖前五页，新请求只创建 6-10，重复请求复用新任务。"""

    async def verify() -> None:
        """检查真实任务入队和原记录内容。"""
        async with _legacy_store(tmp_path) as (db, service, _server):
            result = await service.request_parse(str(tmp_path / "document.pdf"), tier="flash", page_range="1-10")
            assert not result.cache_hit and len(result.created_parse_ids) == 1
            rows = await db.fetchall("SELECT page_range, status FROM parses ORDER BY id")
            assert rows == [{"page_range": "1~5", "status": "done"}, {"page_range": "6-10", "status": "pending"}]
            repeated = await service.request_parse(str(tmp_path / "document.pdf"), tier="flash", page_range="1-10")
            assert repeated.reused_parse_ids == result.created_parse_ids
            assert repeated.created_parse_ids == []
            assert _cache_path(tmp_path, "1~5", 1000).is_file()

    asyncio.run(verify())


def test_mixed_cache_compaction_keeps_latest_pages_and_writes_new_ranges(tmp_path: Path) -> None:
    """新旧缓存合并时重复页取最新正文，并生成连字符记录与文件名。"""

    async def verify() -> None:
        """压缩前后均核对页序与重复页正文。"""
        async with _legacy_store(tmp_path) as (db, _service, server):
            await _add_result(db, tmp_path, "4-7", 2000, range(4, 8), "new")
            rows = await db.fetchall("SELECT * FROM parses ORDER BY done_at DESC")
            pages = load_pages_from_done_batches(str(tmp_path), _SHA256, "flash", rows)
            assert [page.page_idx for page in pages] == list(range(7))
            assert pages[3].blocks[0].content[0].content == "new page 4"
            assert pages[4].blocks[0].content[0].content == "new page 5"
            compaction = Compaction(db=db, interval_sec=600, data_dir=str(tmp_path))
            assert await compaction._compact_doc_tier(_SHA256, "flash") == 1
            assert await db.fetchall("SELECT page_range, done_at FROM parses") == [{"page_range": "1-7", "done_at": 2000}]
            path = _cache_path(tmp_path, "1-7", 2000)
            assert sorted(p.name for p in path.parent.glob("*.json")) == [path.name]
            exported = await server._render_doc_content(
                _SHA256, tier="flash", page_range="1-7", format="markdown", no_marker=True
            )
            assert "old page 3" in exported and "new page 4" in exported
            assert "old page 4" not in exported

    asyncio.run(verify())


@pytest.mark.parametrize("failure", ["missing_source", "write_failure"])
def test_failed_mixed_cache_compaction_preserves_old_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """源文件缺失或新文件提升失败时保留旧记录与现存缓存。"""

    async def verify() -> None:
        """在真实缓存与数据库上触发失败并检查无破坏性替换。"""
        async with _legacy_store(tmp_path) as (db, _service, _server):
            await _add_result(db, tmp_path, "4-7", 2000, range(4, 8), "new")
            old_path = _cache_path(tmp_path, "1~5", 1000)
            new_path = _cache_path(tmp_path, "4-7", 2000)
            if failure == "missing_source":
                new_path.unlink()
            else:

                def fail_replace(source: str, target: str) -> None:
                    """模拟压缩结果提升到目标文件时失败。"""
                    raise OSError("simulated cache write failure")

                monkeypatch.setattr(compaction_module.os, "replace", fail_replace)
            records = await db.fetchall("SELECT * FROM parses ORDER BY id")
            files = {path.name: path.read_bytes() for path in old_path.parent.iterdir()}
            compaction = Compaction(db=db, interval_sec=600, data_dir=str(tmp_path))
            assert await compaction._compact_doc_tier(_SHA256, "flash") == 0
            assert await db.fetchall("SELECT * FROM parses ORDER BY id") == records
            assert {path.name: path.read_bytes() for path in old_path.parent.iterdir()} == files

    asyncio.run(verify())
