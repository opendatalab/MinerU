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
from mineru.errors import MineruError

_SHA256 = "a" * 64


def _convert_fixture_family(path: Path, family: str) -> None:
    """仅在测试建库阶段生成不同历史格式，正文及来源记录保持可对照。"""
    if family == "current":
        return
    source = json.loads(path.read_text())
    metadata = source["metadata"]
    product = source["extensions"]["mineru"]
    effort = {"flash": "flash", "basic": "medium", "standard": "high", "advanced": "xhigh"}[product["tier"]]
    if family == "v2":
        output = {
            "schema_version": "2.0",
            "pages": source["pages"],
            "is_full_document": source["is_full_document"],
            "file_suffix": metadata["file_suffix"],
            "mineru_version": metadata["producer"]["version"],
            "effort": effort,
            "parse_mode": product["parse_mode"],
        }
    else:
        pages = []
        for page in source["pages"]:
            blocks = []
            for block in page["blocks"]:
                assert block["type"] == "text"
                bbox = [coordinate * 100 for coordinate in block["bbox"]]
                text = "".join(span["content"] for span in block["content"])
                blocks.append(
                    {"type": "text", "bbox": bbox, "lines": [{"bbox": bbox, "spans": [{"type": "text", "content": text}]}]}
                )
            pages.append(
                {"page_idx": page["page_idx"], "page_size": [100, 100], "preproc_blocks": blocks, "discarded_blocks": []}
            )
        output = {
            "_version_name": metadata["producer"]["version"],
            "_effort": effort,
            "_ocr_enable": product["parse_mode"] == "ocr",
        }
        if family == "345":
            output["pdf_info"] = pages
        else:
            assert family == "v1"
            output.update(schema_version="1.0", pages=pages)
    path.write_text(json.dumps(output))


@pytest.mark.parametrize("family", ["345", "v1", "v2"])
def test_historical_cache_rebuilds_fts_and_default_tier(tmp_path: Path, family: str) -> None:
    """历史缓存参与全文索引重建和默认档位选择，读取不修改源批次。"""

    async def verify() -> None:
        """用真实 FTS5 和数据库查询核对所有历史格式的消费链。"""
        async with _legacy_store(tmp_path, family=family) as (db, service, server):
            path = _cache_path(tmp_path, "1~5", 1000)
            standard_path = Path(parse_batch_json_path(str(tmp_path), _SHA256, "standard", "1~5", 1000))
            standard_path.parent.mkdir(parents=True, exist_ok=True)
            standard_path.write_bytes(path.read_bytes())
            await db.execute("UPDATE parses SET tier='standard'")
            before = standard_path.read_bytes()
            assert await server._default_read_tier(_SHA256) == "standard"
            await service._rebuild_fts_after_invalidate(_SHA256)
            assert await service.fts.search("old")
            assert await service.fts.get_tier(_SHA256) == "standard"
            assert standard_path.read_bytes() == before

    asyncio.run(verify())


@pytest.mark.parametrize("legacy_family", ["345", "v1"])
def test_all_supported_formats_compact_to_current_protocol(tmp_path: Path, legacy_family: str) -> None:
    """混合三种协议的兼容批次合并后只输出新协议，重复页保留最新正文。"""

    async def verify() -> None:
        """使用非零起始页模拟旧抽页结果，统一后保持同一份来源及整本语义。"""
        async with _legacy_store(tmp_path) as (db, _service, server):
            await db.execute("DELETE FROM parses")
            _cache_path(tmp_path, "1~5", 1000).unlink()
            for family, page_range, done_at, numbers, label in [
                (legacy_family, "3-4", 1000, [3, 4], "old"),
                ("v2", "4-5", 2000, [4, 5], "intermediate"),
                ("current", "5-6", 3000, [5, 6], "latest"),
            ]:
                await _add_result(db, tmp_path, page_range, done_at, numbers, label)
                _convert_fixture_family(_cache_path(tmp_path, page_range, done_at), family)
            compaction = Compaction(db=db, interval_sec=600, data_dir=str(tmp_path))
            assert await compaction._compact_doc_tier(_SHA256, "flash") == 2
            path = _cache_path(tmp_path, "3-6", 3000)
            payload = json.loads(path.read_text())
            assert payload["schema"] == "docvortex.middle"
            assert payload["schema_version"] == "2.0"
            assert "file_suffix" not in payload and "effort" not in payload
            assert payload["metadata"]["producer"]["version"] == __version__
            assert [page["page_idx"] for page in payload["pages"]] == [2, 3, 4, 5]
            content = await server._render_doc_content(
                _SHA256, tier="flash", page_range="3-6", format="markdown", no_marker=True
            )
            assert "old page 3" in content and "intermediate page 4" in content and "latest page 5" in content
            assert "old page 4" not in content and "intermediate page 5" not in content
            assert [item.name for item in path.parent.glob("*.json")] == [path.name]

    asyncio.run(verify())


@pytest.mark.parametrize("old_version", [None, "1.0"])
def test_old_protocol_cache_is_not_available_or_compacted(tmp_path: Path, old_version: str | None) -> None:
    """旧协议不命中、不参与覆盖和压缩，读取旧页明确要求重新解析。"""

    async def verify() -> None:
        """在真实数据库中混存新旧批次，验证读写和缓存调度边界。"""
        async with _legacy_store(tmp_path) as (db, service, server):
            old_path = _cache_path(tmp_path, "1~5", 1000)
            payload = json.loads(old_path.read_text())
            if old_version is None:
                payload.pop("schema")
            else:
                payload["schema_version"] = old_version
            old_path.write_text(json.dumps(payload))
            await _add_result(db, tmp_path, "4-7", 2000, range(4, 8), "new")
            rows = await db.fetchall("SELECT * FROM parses ORDER BY done_at DESC")
            before_files = {path.name: path.read_bytes() for path in old_path.parent.iterdir()}
            compaction = Compaction(db=db, interval_sec=600, data_dir=str(tmp_path))
            assert await compaction._compact_doc_tier(_SHA256, "flash") == 0
            assert {path.name: path.read_bytes() for path in old_path.parent.iterdir()} == before_files
            with pytest.raises(MineruError, match="reparse"):
                load_pages_from_done_batches(str(tmp_path), _SHA256, "flash", rows)
            current = await server._render_doc_content(
                _SHA256, tier="flash", page_range="4-7", format="markdown", no_marker=True
            )
            assert "new page 4" in current and "old page" not in current
            with pytest.raises(MineruError, match="reparse"):
                await server._render_doc_content(_SHA256, tier="flash", page_range="1-3", format="markdown", no_marker=True)
            listing = await server.list_parses(doc_ref="aaaaaaa", tier="flash", page_range="1-7")
            assert listing.coverage.done_page_range == "4-7"
            assert listing.coverage.missing_page_range == "1-3"
            result = await service.request_parse(str(tmp_path / "document.pdf"), tier="flash", page_range="1-7")
            assert not result.cache_hit and len(result.created_parse_ids) == 1
            queued = await db.fetchone("SELECT page_range FROM parses WHERE id=?", (result.created_parse_ids[0],))
            assert queued["page_range"] == "1-3"
            assert old_path.read_bytes() == before_files[old_path.name]

    asyncio.run(verify())


@pytest.mark.parametrize("field", ["metadata", "extensions", "is_full_document", "extension_value_type"])
def test_compaction_preserves_batches_with_conflicting_envelopes(tmp_path: Path, field: str) -> None:
    """不同来源、扩展或整本语义的批次不得被合并为虚假来源的文档。"""

    async def verify() -> None:
        """逐项注入头部冲突并确认源文件和数据库完全保留。"""
        async with _legacy_store(tmp_path) as (db, _service, _server):
            await _add_result(db, tmp_path, "4-7", 2000, range(4, 8), "new")
            path = _cache_path(tmp_path, "4-7", 2000)
            payload = json.loads(path.read_text())
            if field == "metadata":
                payload[field]["producer"]["version"] = "different"
            elif field == "extensions":
                payload[field]["application"] = {"source": "different"}
            elif field == "extension_value_type":
                old_path = _cache_path(tmp_path, "1~5", 1000)
                old_payload = json.loads(old_path.read_text())
                old_payload["extensions"]["application"] = {"value": True}
                old_path.write_text(json.dumps(old_payload))
                payload["extensions"]["application"] = {"value": 1}
            else:
                payload[field] = True
            path.write_text(json.dumps(payload))
            rows = await db.fetchall("SELECT * FROM parses ORDER BY id")
            files = {item.name: item.read_bytes() for item in path.parent.iterdir()}
            compaction = Compaction(db=db, interval_sec=600, data_dir=str(tmp_path))
            assert await compaction._compact_doc_tier(_SHA256, "flash") == 0
            assert await db.fetchall("SELECT * FROM parses ORDER BY id") == rows
            assert {item.name: item.read_bytes() for item in path.parent.iterdir()} == files

    asyncio.run(verify())


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
        "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": __version__}},
        "schema": "docvortex.middle",
        "extensions": {"mineru": {"tier": "flash", "parse_mode": "txt"}},
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
async def _legacy_store(
    root: Path, *, family: str = "current"
) -> AsyncIterator[tuple[DatabaseManager, ParseService, DoclibServer]]:
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
        _convert_fixture_family(_cache_path(root, "1~5", 1000), family)
        service = ParseService(
            db=db, fts=FTSManager(db), config_svc=ConfigService(db), data_dir=str(root), parse_lock_timeout_sec=1800
        )
        server = DoclibServer(SimpleNamespace(db=db, parse_svc=service, data_dir=str(root)))
        yield db, service, server
    finally:
        await db.close()


@pytest.mark.parametrize("family", ["current", "345", "v1", "v2"])
def test_old_cache_hits_exports_and_serializes_without_mutating_storage(tmp_path: Path, family: str) -> None:
    """旧结果正常命中、导出和查询，返回新格式而文件与数据库始终保留旧名称。"""

    async def verify() -> None:
        """读取各个结果出口，确认规范化不破坏原始缓存定位。"""
        async with _legacy_store(tmp_path, family=family) as (db, service, server):
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


@pytest.mark.parametrize("family", ["current", "345", "v1", "v2"])
def test_old_cache_only_schedules_uncovered_pages(tmp_path: Path, family: str) -> None:
    """旧 1~5 批次覆盖前五页，新请求只创建 6-10，重复请求复用新任务。"""

    async def verify() -> None:
        """检查真实任务入队和原记录内容。"""
        async with _legacy_store(tmp_path, family=family) as (db, service, _server):
            result = await service.request_parse(str(tmp_path / "document.pdf"), tier="flash", page_range="1-10")
            assert not result.cache_hit and len(result.created_parse_ids) == 1
            rows = await db.fetchall("SELECT page_range, status FROM parses ORDER BY id")
            assert rows == [{"page_range": "1~5", "status": "done"}, {"page_range": "6-10", "status": "pending"}]
            repeated = await service.request_parse(str(tmp_path / "document.pdf"), tier="flash", page_range="1-10")
            assert repeated.reused_parse_ids == result.created_parse_ids
            assert repeated.created_parse_ids == []
            assert _cache_path(tmp_path, "1~5", 1000).is_file()

    asyncio.run(verify())


@pytest.mark.parametrize("family", ["current", "v2"])
def test_mixed_cache_compaction_keeps_latest_pages_and_writes_new_ranges(tmp_path: Path, family: str) -> None:
    """新旧缓存合并时重复页取最新正文，并生成连字符记录与文件名。"""

    async def verify() -> None:
        """压缩前后均核对页序与重复页正文。"""
        async with _legacy_store(tmp_path, family=family) as (db, _service, server):
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
