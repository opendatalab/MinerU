"""验证文件名前缀、完整候选分页及 HTTP/SDK/CLI 的参数一致性。"""

from __future__ import annotations

from doclib_test_utils import create_doclib_server

from mineru.doclib.services.ingest_svc import FileRefreshResult

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from mineru.cli.commands import search as search_commands
from mineru.cli.main import app
from mineru.doclib.client import DoclibClient
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager, tokenize_for_index

from mineru.doclib.services.search_svc import SearchService
from mineru.doclib.types import FindResponse
from mineru.errors import InvalidRequestError


async def _filename_library(tmp_path: Path, filenames: list[str]) -> tuple[DatabaseManager, FTSManager, SearchService]:
    """一次事务建立真实 FTS 与文件记录，路径不同的同名文件保留独立身份。"""
    db = DatabaseManager(str(tmp_path / "library.sqlite"))
    await db.initialize()
    statements: list[tuple[str, tuple[Any, ...]]] = []
    for file_id, filename in enumerate(filenames, start=1):
        ext = Path(filename).suffix.lstrip(".")
        statements.extend(
            [
                (
                    "INSERT INTO files (id, path, filename, ext, size_bytes, mtime_ms, first_seen_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (file_id, str(tmp_path / str(file_id) / filename), filename, ext, 10, 1, 1, 1),
                ),
                (
                    "INSERT INTO fts_filenames (file_id, filename, ext) VALUES (?, ?, ?)",
                    (file_id, tokenize_for_index(filename), ext),
                ),
            ]
        )
    await db.execute_atomic(statements)
    fts = FTSManager(db)
    return db, fts, SearchService(db, fts)


def test_filename_prefix_is_applied_only_to_final_token(tmp_path: Path) -> None:
    """前缀查询覆盖中文和多词，显式星号等价且正文索引不自动补前缀。"""

    async def run() -> None:
        """在真实 SQLite FTS 上比较隐式与显式前缀查询。"""
        db, fts, _ = await _filename_library(
            tmp_path,
            ["bench1.pdf", "bench5.pdf", "annual report2026.pdf", "项目报告2026.pdf", "annualized report2026.pdf"],
        )
        try:
            await fts.replace(sha256="f" * 64, tier="standard", text="benchmark content", title="", author="")
            for query in ("bench", "bench*", "bench*  "):
                assert [row["file_id"] for row in await fts.search_filenames(query)] == [1, 2]
            assert [row["file_id"] for row in await fts.search_filenames("annual rep")] == [3]
            assert [row["file_id"] for row in await fts.search_filenames("项目报告")] == [4]
            assert [row["file_id"] for row in await fts.search_filenames("bench1")] == [1]
            assert await fts.search_filenames("   ") == []
            assert await fts.search("bench") == []
            assert len(await fts.search("benchmark")) == 1
        finally:
            await db.close()

    asyncio.run(run())


def test_find_paginates_all_candidates_in_stable_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """六百余条同分候选跨过旧上限与元信息批次，每页无遗漏且总数不随 limit 变化。"""

    async def run() -> None:
        """记录元信息查询批次，同时遍历真实搜索的所有分页。"""
        db, _, service = await _filename_library(tmp_path, ["report.pdf"] * 601)
        batch_sizes: list[int] = []
        fetchall = db.fetchall

        async def record_fetchall(sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
            """只观察元信息批大小，不替换实际 SQL 结果。"""
            if "WHERE f.id IN" in sql:
                batch_sizes.append(sum(isinstance(value, int) for value in params or ()))
            return await fetchall(sql, params)

        monkeypatch.setattr(db, "fetchall", record_fetchall)
        try:
            paths: list[str] = []
            for offset in range(0, 601, 97):
                rows, total = await service.search_filenames("report", limit=97, offset=offset)
                assert total == 601
                paths.extend(row["paths"][0] for row in rows)
            assert paths == [str(tmp_path / str(index) / "report.pdf") for index in range(1, 602)]
            assert max(batch_sizes) == 256
            assert await service.search_filenames("report", limit=50, offset=601) == ([], 601)
            assert await service.search_filenames("missing", limit=50, offset=10) == ([], 0)
        finally:
            await db.close()

    asyncio.run(run())


def test_find_filters_and_refreshes_before_pagination(tmp_path: Path) -> None:
    """删除前面的索引记录不会使后续批次漏项，扩展名与有效文件数均先于分页生效。"""

    async def run() -> None:
        """刷新期间删除 FTS 行，验证分页依赖最初的有序候选快照。"""
        filenames = ["report.pdf"] * 270 + ["report.docx"] * 10
        db, fts, service = await _filename_library(tmp_path, filenames)
        refreshed_ids: list[int] = []

        async def refresh(path: str) -> FileRefreshResult:
            """把前三个路径标记为删除，同时移除其 FTS 行。"""
            file_id = int(Path(path).parent.name)
            refreshed_ids.append(file_id)
            if file_id <= 3:
                await db.execute("UPDATE files SET status='deleted' WHERE id=?", (file_id,))
                await fts.delete_filename(file_id)
                return FileRefreshResult(file=None, status="deleted")
            return FileRefreshResult(file=None, status="known")

        try:
            rows, total = await service.search_filenames("report", ext=".PDF", limit=10, offset=260, refresh_file=refresh)
            assert total == 267
            assert [int(Path(row["paths"][0]).parent.name) for row in rows] == list(range(264, 271))
            assert refreshed_ids == list(range(1, 271))
            rows, total = await service.search_filenames("report", ext="pdf", limit=2)
            assert total == 267
            assert [int(Path(row["paths"][0]).parent.name) for row in rows] == [4, 5]
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(("limit", "offset", "param"), [(0, 0, "limit"), (-1, 0, "limit"), (50, -1, "offset")])
def test_find_rejects_invalid_pagination(tmp_path: Path, limit: int, offset: int, param: str) -> None:
    """非法参数在查询前返回明确的产品校验错误。"""

    async def run() -> None:
        """空索引也必须校验分页参数。"""
        db, _, service = await _filename_library(tmp_path, [])
        try:
            with pytest.raises(InvalidRequestError) as error:
                await service.search_filenames("report", limit=limit, offset=offset)
            assert error.value.param == param
        finally:
            await db.close()

    asyncio.run(run())


def test_find_http_exposes_offset_and_validates_pagination(tmp_path: Path) -> None:
    """HTTP 参数、默认值、总数和错误结构与服务层一致。"""

    async def run() -> None:
        """通过实际 ASGI 路由请求文件名查询。"""
        db, _, service = await _filename_library(tmp_path, ["report.pdf"] * 4)

        async def refresh(path: str) -> FileRefreshResult:
            """保留已有有效记录，让接口走正常刷新边界。"""
            return FileRefreshResult(file=None, status="known")

        server = create_doclib_server(
            SimpleNamespace(db=db, search_svc=service, ingest_svc=SimpleNamespace(refresh_file=refresh))
        )
        try:
            parameters = server.app.openapi()["paths"]["/api/v1/find"]["get"]["parameters"]
            assert next(item for item in parameters if item["name"] == "offset")["schema"]["default"] == 0
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                response = await client.get("/api/v1/find", params={"query": "report", "limit": 2, "offset": 2})
                assert response.status_code == 200
                assert response.json()["total"] == 4
                assert [int(Path(row["paths"][0]).parent.name) for row in response.json()["results"]] == [3, 4]
                invalid = await client.get("/api/v1/find", params={"query": "report", "offset": -1})
                assert invalid.status_code == 400
                assert invalid.json()["error"]["param"] == "offset"
        finally:
            await db.close()

    asyncio.run(run())


def test_find_sdk_transmits_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """公共同步 SDK 在请求中传递分页字段，而非只改变函数签名。"""
    calls: list[dict[str, Any]] = []

    def send_request(self: DoclibClient, method: str, path: str, *, params: dict[str, Any], json_data: Any) -> httpx.Response:
        """捕获实际 HTTP 参数，返回合法的文件名搜索响应。"""
        calls.append(params)
        return httpx.Response(
            200, request=httpx.Request(method, f"http://test{path}"), json={"results": [], "total": 601, "query": "report"}
        )

    monkeypatch.setattr(DoclibClient, "_send_request", send_request)
    client = DoclibClient(base_url="http://test")
    try:
        assert client.find("report", ext="pdf", limit=10, offset=200).total == 601
        assert calls == [{"query": "report", "ext": "pdf", "limit": 10, "offset": 200}]
    finally:
        client.close()


def test_find_cli_passes_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 Typer 命令把 --offset 传递到 SDK，并保留默认 limit。"""
    calls: list[dict[str, Any]] = []

    class Client:
        """只替换外部连接，保留完整的 CLI 参数解析。"""

        def __init__(self, *, timeout: int) -> None:
            """验证查询命令沿用原有超时设置。"""
            assert timeout == 10

        def find(self, query: str, **kwargs: Any) -> FindResponse:
            """记录 CLI 提交的分页参数。"""
            calls.append({"query": query, **kwargs})
            return FindResponse(results=[], total=300, query=query)

    monkeypatch.setattr(search_commands, "DoclibClient", Client)
    result = CliRunner().invoke(app, ["find", "report", "--ext", "pdf", "--offset", "200", "--json"])
    assert result.exit_code == 0
    assert calls == [{"query": "report", "ext": "pdf", "limit": 50, "offset": 200}]
