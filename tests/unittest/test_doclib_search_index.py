"""验证无图片索引、真实服务写入以及旧库可恢复迁移，关联 Issue #5776。"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from docvortex.options import LatexDelimitersConfig
from fastapi.testclient import TestClient

from doclib_test_utils import DOCLIB_AUTH_HEADERS, authenticated_doclib_config, create_parse_services
from mineru.cli.commands.search import _render_search_results
from mineru.config import Config
from mineru.doclib import app as app_module
from mineru.doclib.core import db as db_module
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager, strip_sep, tokenize_for_index
from mineru.doclib.core.fts_content import migrate_fts_contents, render_fts_text
from mineru.doclib.core.parse_artifacts import parse_batch_json_path
from mineru.doclib.rows import FileRow
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.parse_execution_svc import ParseExecutionService
from mineru.doclib.services.search_svc import SearchService
from mineru.doclib.types import SearchResponse, SearchResult
from mineru.parser.base import ParseResult
from mineru.types import MiddleJson, Tier

_SHA = "a" * 64
_TEXT_SHA = "b" * 64
_DELIMITERS = LatexDelimitersConfig()
_URI = "data:image/png;base64," + "A" * 40000


def _result(text: str = "needle", page_idx: int = 0) -> ParseResult:
    """构造含超预算图片、图注、公式与图表识别内容的真实协议对象。"""
    blocks = [
        {"type": "text", "content": [{"type": "text", "content": text}]},
        {
            "type": "image",
            "content": [
                {"type": "image_body", "content": "recognized", "image_base64": _URI},
                {"type": "image_caption", "content": [{"type": "text", "content": "caption"}]},
            ],
        },
        {"type": "text", "content": [{"type": "text", "content": "middlekeyword"}]},
        {"type": "table", "content": [{"type": "table_body", "content": "", "image_base64": _URI}]},
        {"type": "equation", "content": "", "image_base64": _URI},
        {"type": "chart", "content": [{"type": "chart_body", "content": "charttext", "image_base64": _URI}]},
        {"type": "text", "content": [{"type": "text", "content": "tailkeyword"}]},
    ]
    for index, block in enumerate(blocks):
        block["index"] = index
        if block["type"] in {"image", "table", "chart"}:
            for child in block["content"]:
                child["index"] = index
    return ParseResult.from_dict(
        {
            "schema": "docvortex.middle",
            "schema_version": "2.0",
            "is_full_document": True,
            "metadata": {"file_suffix": "docx", "producer": {"name": "mineru", "version": "4.2.2"}},
            "pages": [{"page_idx": page_idx, "blocks": blocks}],
        }
    )


async def _seed_doc(db: DatabaseManager, sha: str = _SHA) -> None:
    """只建立文档记录，特意不创建源文件记录以覆盖孤立缓存。"""
    await db.execute(
        "INSERT OR IGNORE INTO docs (sha256, short_id, size_bytes, first_seen_at, updated_at) VALUES (?, ?, 1, 1, 1)",
        (sha, sha[:7]),
    )


async def _cache(db: DatabaseManager, root: Path, result: ParseResult, tier: Tier, timestamp: int) -> Path:
    """写入真实解析批次协议和对应完成记录，供生产缓存加载器读取。"""
    page_range = str(result.pages[0].page_idx + 1)
    path = Path(parse_batch_json_path(str(root), _SHA, tier, page_range, timestamp))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json())
    await db.execute(
        "INSERT INTO parses (sha256,tier,page_range,status,done_at,created_at,updated_at) VALUES (?,?,?,'done',?,?,?)",
        (_SHA, tier, page_range, timestamp, timestamp, timestamp),
    )
    return path


async def _old_database(root: Path, monkeypatch: pytest.MonkeyPatch) -> DatabaseManager:
    """创建 v2 旧库并写入污染索引；文本直读中的字面 data URI 用作误清理反例。"""
    db = DatabaseManager(str(root / "library.sqlite"))
    with monkeypatch.context() as patch:
        patch.setattr(db_module, "SCHEMA_VERSION", 2)
        await db.initialize()
    await _seed_doc(db)
    await _seed_doc(db, _TEXT_SHA)
    for sha, tier, text in [(_SHA, "standard", f"needle ![]({_URI})"), (_TEXT_SHA, None, "literal data:image/png;base64")]:
        await db.execute(
            "INSERT INTO fts_contents (sha256,tier,text,title,author) VALUES (?,?,?,'','')",
            (sha, tier, tokenize_for_index(text)),
        )
    return db


def test_new_parse_and_cached_rebuild_share_image_free_search_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """执行真实任务收尾与搜索服务，确认图片不挤占正文预算且重建得到相同文本。"""
    result = _result()
    before = result.to_json()

    async def parse_result(self: ParseExecutionService, file_row: FileRow, tier: Tier, page_range: str) -> ParseResult:
        """仅替换模型推理，保留真实 ParseResult、缓存写出、索引和任务生命周期。"""
        return result

    monkeypatch.setattr(ParseExecutionService, "_parse_via_local", parse_result)

    async def scenario() -> None:
        """连接真实 SQLite、解析执行器、搜索服务和 CLI 文本格式化。"""
        db = DatabaseManager(str(tmp_path / "library.sqlite"))
        await db.initialize()
        fts = FTSManager(db)
        services = create_parse_services(db, fts, ConfigService(db), str(tmp_path), parse_lock_timeout_sec=60)
        source = tmp_path / "sample.html"
        source.write_text("<html><body>placeholder</body></html>")
        request = await services.queue.request_parse(str(source), tier="flash")
        task = await services.queue.acquire_task()
        assert task is not None and await services.execution.process_doc(task)
        first = await db.fetchone("SELECT text FROM fts_contents WHERE sha256=?", (request.sha256,))
        assert strip_sep(first["text"]) == render_fts_text(result.middle_json, latex_delimiters=_DELIMITERS)
        search = SearchService(db, fts)
        for word in ("needle", "recognized", "caption", "middlekeyword", "charttext", "tailkeyword"):
            rows, total = await search.search(word)
            assert total == 1
            response = SearchResponse(query=word, total=total, results=[SearchResult(**row) for row in rows])
            assert "base64" not in response.model_dump_json()
            assert "data:image" not in _render_search_results(response)
        for word in ("base64", "png"):
            assert await fts.search(word) == []
        await services.queue._rebuild_fts_after_invalidate(request.sha256)
        assert await db.fetchone("SELECT text FROM fts_contents WHERE sha256=?", (request.sha256,)) == first
        assert result.to_json() == before
        assert _URI in result.markdown()

    asyncio.run(scenario())


def test_issue5776_attachment_keeps_images_in_export_but_not_search() -> None:
    """用原始 PDF 的真实原生解析验证载荷隔离，不依赖 OCR 模型或 Windows 设备。"""
    from mineru.parser import parse

    source = Path(__file__).resolve().parents[1] / "fixtures" / "issue5776" / "source.pdf"
    result = parse(source, tier="flash", ocr_mode="txt", image_analysis=False)
    exported = result.markdown()
    assert "data:image" in exported
    indexed = render_fts_text(result.middle_json, latex_delimiters=_DELIMITERS)
    assert "测试" in indexed
    assert "data:image" not in indexed and "base64" not in indexed
    assert result.markdown() == exported


def test_search_text_preserves_cross_page_continuation(tmp_path: Path) -> None:
    """去除图片不能改变已有跨页正文拼接；缓存重建也必须使用同一整本渲染规则。"""
    middle = MiddleJson(
        metadata={"file_suffix": "docx", "producer": {"name": "test", "version": "test"}},
        is_full_document=True,
        pages=[
            {"page_idx": 0, "blocks": [{"type": "text", "index": 0, "content": [{"type": "text", "content": "跨页"}]}]},
            {
                "page_idx": 1,
                "blocks": [
                    {
                        "type": "text",
                        "index": 0,
                        "continues_prev": True,
                        "content": [{"type": "text", "content": "关键词"}],
                    }
                ],
            },
        ],
    )
    expected = render_fts_text(middle, latex_delimiters=_DELIMITERS)
    assert expected == "跨页关键词"

    async def scenario() -> None:
        """将跨页正文拆成两个真实缓存批次，确认合并后仍能恢复原本的检索文本。"""
        db = DatabaseManager(str(tmp_path / "library.sqlite"))
        await db.initialize()
        await _seed_doc(db)
        for index, page in enumerate(middle.pages):
            await _cache(db, tmp_path, ParseResult(middle.model_copy(update={"pages": [page]})), "flash", index + 1)
        services = create_parse_services(db, FTSManager(db), ConfigService(db), str(tmp_path), parse_lock_timeout_sec=60)
        await services.queue._rebuild_fts_after_invalidate(_SHA)
        row = await db.fetchone("SELECT text FROM fts_contents WHERE sha256=?", (_SHA,))
        assert strip_sep(row["text"]) == expected

    asyncio.run(scenario())


def test_upgrade_rebuilds_orphan_cache_with_latest_pages_and_keeps_literal_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """迁移从最高档位选最新重复页，缺失源文件也可恢复，且不会处理纯文本的字面 URI。"""

    async def scenario() -> None:
        """创建旧索引、混合档位和重叠批次，验证事务进度与重复运行。"""
        db = await _old_database(tmp_path, monkeypatch)
        await _cache(db, tmp_path, _result("lowerword"), "flash", 1)
        await _cache(db, tmp_path, _result("obsoleteword"), "standard", 2)
        newest = await _cache(db, tmp_path, _result("newestword"), "standard", 3)
        await _cache(db, tmp_path, _result("secondpage", 1), "standard", 4)
        original = newest.read_bytes()
        await db.initialize()
        fts = FTSManager(db)
        assert await db.fetchall("SELECT sha256 FROM fts_rebuild_queue") == [{"sha256": _SHA}]
        assert await fts.search("needle") == []
        await migrate_fts_contents(db, fts, data_dir=str(tmp_path), latex_delimiters=_DELIMITERS)
        assert await db.fetchall("SELECT * FROM fts_rebuild_queue") == []
        for word in ("newestword", "secondpage", "caption", "middlekeyword"):
            assert await fts.search(word)
        for word in ("obsoleteword", "lowerword"):
            assert await fts.search(word) == []
        assert [row["sha256"] for row in await fts.search("base64")] == [_TEXT_SHA]
        snapshot = await db.fetchall("SELECT * FROM fts_contents ORDER BY sha256")
        await db.initialize()
        await migrate_fts_contents(db, fts, data_dir=str(tmp_path), latex_delimiters=_DELIMITERS)
        assert await db.fetchall("SELECT * FROM fts_contents ORDER BY sha256") == snapshot
        assert newest.read_bytes() == original

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_upgrade_records_failed_cache_and_retries_without_exposing_old_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    """缺失或损坏缓存保留待修复标记与原因，恢复后可继续迁移而不丢正文。"""

    async def scenario() -> None:
        """保留一页正常缓存，确认迁移不会把部分读取伪报为整份索引恢复。"""
        db = await _old_database(tmp_path, monkeypatch)
        await _cache(db, tmp_path, _result("firstpage"), "standard", 1)
        broken = await _cache(db, tmp_path, _result("repairedword", 1), "standard", 2)
        original = broken.read_text()
        if failure == "missing":
            broken.unlink()
        else:
            broken.write_text("{broken")
        await db.initialize()
        fts = FTSManager(db)
        old_row = await db.fetchone("SELECT text FROM fts_contents WHERE sha256=?", (_SHA,))
        await migrate_fts_contents(db, fts, data_dir=str(tmp_path), latex_delimiters=_DELIMITERS)
        assert (await db.fetchone("SELECT error FROM fts_rebuild_queue WHERE sha256=?", (_SHA,)))["error"]
        assert "rebuild deferred" in caplog.text
        assert await fts.search("needle") == []
        assert await db.fetchone("SELECT text FROM fts_contents WHERE sha256=?", (_SHA,)) == old_row
        broken.write_text(original)
        await migrate_fts_contents(db, fts, data_dir=str(tmp_path), latex_delimiters=_DELIMITERS)
        assert await fts.search("repairedword")
        assert await db.fetchall("SELECT * FROM fts_rebuild_queue") == []

    asyncio.run(scenario())


def test_index_replacement_and_migration_acknowledgement_are_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟清除迁移标记时的数据库故障，旧索引和标记必须同时回滚。"""

    async def scenario() -> None:
        """使用真实 SQLite 触发器中断事务，随后验证可重试恢复。"""
        db = await _old_database(tmp_path, monkeypatch)
        await _cache(db, tmp_path, _result(), "standard", 1)
        await db.initialize()
        old_row = await db.fetchone("SELECT text FROM fts_contents WHERE sha256=?", (_SHA,))
        await db.execute(
            "CREATE TRIGGER fail_ack BEFORE DELETE ON fts_rebuild_queue BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        fts = FTSManager(db)
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            await migrate_fts_contents(db, fts, data_dir=str(tmp_path), latex_delimiters=_DELIMITERS)
        assert await db.fetchone("SELECT text FROM fts_contents WHERE sha256=?", (_SHA,)) == old_row
        assert await db.fetchone("SELECT sha256 FROM fts_rebuild_queue WHERE sha256=?", (_SHA,))
        await db.execute("DROP TRIGGER fail_ack")
        await migrate_fts_contents(db, fts, data_dir=str(tmp_path), latex_delimiters=_DELIMITERS)
        assert await db.fetchall("SELECT * FROM fts_rebuild_queue") == []

    asyncio.run(scenario())


def test_doclib_startup_migrates_before_serving_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """用真实应用启动与 HTTP 搜索验证升级钩子，而非只测试迁移函数。"""
    monkeypatch.setenv("MINERU_HOME", str(tmp_path / "home"))

    async def seed() -> None:
        """准备服务即将打开的 v2 数据库和解析缓存。"""
        db = await _old_database(tmp_path, monkeypatch)
        await _cache(db, tmp_path, _result(), "standard", 1)

    asyncio.run(seed())
    config = Config()
    config.doclib.data_dir = str(tmp_path)
    config.doclib.sqlite.path = str(tmp_path / "library.sqlite")
    app = app_module.create_app(authenticated_doclib_config(config))
    with TestClient(app, headers=DOCLIB_AUTH_HEADERS) as client:
        response = client.get("/api/v1/search", params={"query": "middlekeyword"})
        assert response.status_code == 200
        assert response.json()["total"] == 1
        assert "base64" not in response.text
        assert client.portal.call(app.state.doclib_state.db.fetchall, "SELECT * FROM fts_rebuild_queue") == []
