"""验证共享页选择语法、单点约束以及范围内可直接执行的续读请求。"""

from __future__ import annotations

from doclib_test_utils import create_doclib_server

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from mineru.cli.commands import read as read_commands
from mineru.cli.main import app
from mineru.doclib.client import DoclibClient
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.locators import parse_content_cursor
from mineru.doclib.server import DoclibServer
from mineru.doclib.services.content_read_svc import ContentReadService, _ReadPlan
from mineru.doclib.core.parse_artifacts import parse_batch_json_path
from mineru.doclib.types import ContentAsset, ContentNextRequest, ContentRequestScope, DocContentResponse
from mineru.errors import InvalidRequestError, NotFoundError
from mineru.parser.page_range import format_page_range
from mineru.types import MiddleJson, PageInfo, TextBlock

_SHORT_ID = "aaaaaaa"
_SHA256 = "a" * 64
_DOC_REF = f"doc:{_SHORT_ID}/tier:flash"


def _page(page_idx: int, *texts: str) -> PageInfo:
    """用稳定块号构造带几何信息的缓存页面。"""
    return PageInfo(
        page_idx=page_idx,
        blocks=[
            TextBlock(
                type="text",
                index=index,
                bbox=(0.0, 0.0, 0.1, 0.1),
                content=[{"type": "text", "content": text}],
            )
            for index, text in enumerate(texts)
        ],
    )


async def _cached_library(
    tmp_path: Path, *, page_count: int | None = 12, pages: list[PageInfo] | None = None
) -> tuple[DatabaseManager, DoclibServer]:
    """写入真实新版缓存；总页数与已缓存页范围可以独立设置。"""
    db = DatabaseManager(str(tmp_path / "library.sqlite"))
    await db.initialize()
    await db.execute(
        "INSERT INTO docs (sha256, short_id, size_bytes, file_type, page_count, first_seen_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_SHA256, _SHORT_ID, 12, "pdf", page_count, 1, 1),
    )
    if pages is None:
        pages = [_page(index, f"PAGE{index + 1:02}") for index in range(page_count or 1)]
    page_range = format_page_range(page.page_idx + 1 for page in pages)
    await db.execute(
        "INSERT INTO parses (sha256, tier, page_range, status, done_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_SHA256, "flash", page_range, "done", 1, 1, 1),
    )
    middle = MiddleJson(
        pages=pages,
        is_full_document=page_count == len(pages),
        metadata={"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
        extensions={"mineru": {"tier": "flash", "parse_mode": "txt"}},
    )
    cache_path = Path(parse_batch_json_path(str(tmp_path), _SHA256, "flash", page_range, 1))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(middle.to_dict(), ensure_ascii=False), encoding="utf-8")
    return db, create_doclib_server(SimpleNamespace(db=db, data_dir=str(tmp_path)))


@pytest.mark.parametrize(
    ("selection", "expected_range", "expected_pages"),
    [
        ("r1", "12", [12]),
        ("r2", "11", [11]),
        ("1-5", "1-5", [1, 2, 3, 4, 5]),
        ("r3-r1", "10-12", [10, 11, 12]),
        ("1,3,r1", "1,3,12", [1, 3, 12]),
        ("all", "1-12", list(range(1, 13))),
        ("1-99,3", "1-12", list(range(1, 13))),
        ("3,1,3", "1,3", [1, 3]),
    ],
)
def test_read_page_selection_is_resolved_and_canonical(
    tmp_path: Path, selection: str, expected_range: str, expected_pages: list[int]
) -> None:
    """所有页选择共享排序、去重和裁剪规则，返回引用均使用实际页码。"""

    async def run() -> None:
        """通过真实缓存读取检查请求范围和实际输出页面。"""
        db, server = await _cached_library(tmp_path)
        try:
            result = await server.read_content(f"{_DOC_REF}/page:{selection}", no_marker=True)
            assert result.request_scope.page_range == expected_range
            assert result.request_scope.locator == f"{_DOC_REF}/page:{expected_range}"
            assert result.content.split("\n\n") == [f"PAGE{page_no:02}" for page_no in expected_pages]
            if len(expected_pages) > 1 or selection == "all":
                assert result.next_request is None
            elif expected_pages[-1] < 12:
                assert result.next_request is not None
                assert result.next_request.locator == f"{_DOC_REF}/page:{expected_pages[-1] + 1}"
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize("selection", ["0", "r0", "last", "3-1", "-1", "1~3", "1,,3", " "])
def test_read_rejects_invalid_page_selection(tmp_path: Path, selection: str) -> None:
    """定位器不引入 last、旧分隔符或与输入页范围不同的非法语法。"""

    async def run() -> None:
        """验证非法定位在读取缓存前返回产品错误。"""
        db, server = await _cached_library(tmp_path)
        try:
            with pytest.raises(InvalidRequestError):
                await server.read_content(f"{_DOC_REF}/page:{selection}")
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize("selection", ["r1", "r3-r1", "all"])
def test_relative_selection_requires_known_total(tmp_path: Path, selection: str) -> None:
    """部分缓存不能用于猜测最后一页或完整文档的范围。"""

    async def run() -> None:
        """建立只有第一页的缓存，但文档总页数未知。"""
        db, server = await _cached_library(tmp_path, page_count=None)
        try:
            with pytest.raises(InvalidRequestError) as error:
                await server.read_content(f"{_DOC_REF}/page:{selection}")
            assert error.value.param == "locator"
            assert "page count is unavailable" in error.value.message
        finally:
            await db.close()

    asyncio.run(run())


def test_single_relative_page_supports_blocks_chars_and_context(tmp_path: Path) -> None:
    """倒数单页仍能定位块、字符以及邻页，返回稳定的绝对引用。"""

    async def run() -> None:
        """检查单点扩展与既有块读取行为一致。"""
        db, server = await _cached_library(tmp_path)
        try:
            block = await server.read_content(f"{_DOC_REF}/page:r1/block:1", no_marker=True)
            assert block.content == "PAGE12"
            assert block.request_scope.locator == f"{_DOC_REF}/page:12/block:1"
            char = await server.read_content(f"{_DOC_REF}/page:r1/block:1/char:4", no_marker=True)
            assert char.content == "12"
            assert char.request_scope.after == f"{_DOC_REF}/page:12/block:1/char:4"
            context = await server.read_content(f"{_DOC_REF}/page:r1", context=1, no_marker=True)
            assert context.request_scope.page_range == "11-12"
            assert context.content == "PAGE11\n\nPAGE12"
            singleton = await server.read_content(f"{_DOC_REF}/page:12-99/block:1", no_marker=True)
            assert singleton.request_scope.locator == f"{_DOC_REF}/page:12/block:1"
            assert singleton.content == "PAGE12"
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("suffix", "options", "param"),
    [
        ("/block:1", {}, "locator"),
        ("/block:1/char:0", {}, "locator"),
        ("", {"format": "image"}, "locator"),
        ("", {"context": 1}, "context"),
    ],
)
def test_multi_page_selection_rejects_single_target_options(
    tmp_path: Path, suffix: str, options: dict[str, Any], param: str
) -> None:
    """多页范围不能同时表示一个块、字符、图片或带上下文的单点目标。"""

    async def run() -> None:
        """检查单点约束在加载缓存及生成图片之前生效。"""
        db, server = await _cached_library(tmp_path)
        try:
            with pytest.raises(InvalidRequestError) as error:
                await server.read_content(f"{_DOC_REF}/page:1,3{suffix}", **options)
            assert error.value.param == param
        finally:
            await db.close()

    asyncio.run(run())


def test_relative_page_image_uses_resolved_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """图片读取把倒数页解析为实际页码后再交给原有素材边界。"""

    async def run() -> None:
        """替换实际裁图，检查素材调用获得的页号与响应引用。"""
        db, server = await _cached_library(tmp_path)
        captured: list[int] = []

        async def render_asset(self: ContentReadService, plan: _ReadPlan, page: PageInfo) -> ContentAsset:
            """记录素材调用目标，不初始化 PDFium。"""
            assert plan.target is not None
            captured.append(plan.target.page_no)
            return ContentAsset(path=str(tmp_path / "page.png"), mime_type="image/png")

        monkeypatch.setattr(ContentReadService, "_render_source_image_asset", render_asset)
        try:
            result = await server.read_content(f"{_DOC_REF}/page:r1", format="image")
            assert captured == [12]
            assert result.request_scope.locator == f"{_DOC_REF}/page:12"
            assert result.content_ranges[0].page_range == "12"
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("limit", "texts"),
    [
        (20, [("A" * 20,), ("FORBIDDEN",), ("Z" * 20,)]),
        (35, [("A" * 20, "B" * 20), ("FORBIDDEN",), ("Z" * 20,)]),
        (7, [("ABCDEFGHIJKLMNO",), ("FORBIDDEN",), ("xyz0123456789",)]),
    ],
)
@pytest.mark.parametrize("selection", ["1,3", "1-3"])
@pytest.mark.parametrize("no_marker", [True, False])
def test_range_continuation_preserves_pages_blocks_and_chars(
    tmp_path: Path, limit: int, texts: list[tuple[str, ...]], selection: str, no_marker: bool
) -> None:
    """逐次执行返回的续读请求，验证页间、块间和字符截断均不漏、不重、不扩页。"""

    async def run() -> None:
        """通过真实缓存和游标循环重建选定页面的全部正文。"""
        pages = [_page(index, *items) for index, items in enumerate(texts)]
        db, server = await _cached_library(tmp_path, page_count=4, pages=pages)
        try:
            locator = f"{_DOC_REF}/page:{selection}"
            after = None
            chunks: list[str] = []
            for _ in range(30):
                result = await server.read_content(locator, after=after, limit=limit, no_marker=no_marker)
                assert result.request_scope.page_range == selection
                if selection == "1,3":
                    assert "FORBIDDEN" not in result.content
                body = re.sub(r"<!-- page [0-9]+(?: of [0-9]+)? -->\n*", "", result.content)
                chunks.append(body.replace("\n\n", ""))
                if result.next_request is None:
                    break
                assert result.truncated
                assert result.next_request.locator == locator
                assert result.next_request.after is not None
                cursor = parse_content_cursor(result.next_request.after)
                assert cursor.page_no in ({1, 3} if selection == "1,3" else {1, 2, 3})
                assert result.next_request.after != after
                after = result.next_request.after
            else:
                pytest.fail("范围续读未能在有限次数内完成")
            selected_texts = texts[0] + texts[2] if selection == "1,3" else texts[0] + texts[1] + texts[2]
            assert "".join(chunks) == "".join(selected_texts)
            assert not result.truncated
        finally:
            await db.close()

    asyncio.run(run())


def test_range_reads_only_cached_selected_pages(tmp_path: Path) -> None:
    """部分缓存只输出选中的已缓存页面，不推测或引入其他页。"""

    async def run() -> None:
        """选择跨过未缓存页，末尾仍按本次范围结束。"""
        db, server = await _cached_library(tmp_path, pages=[_page(0, "FIRST"), _page(2, "THIRD")])
        try:
            result = await server.read_content(f"{_DOC_REF}/page:1-3", no_marker=True)
            assert result.content == "FIRST\n\nTHIRD"
            assert result.request_scope.page_range == "1-3"
            assert result.next_request is None
            with pytest.raises(NotFoundError):
                await server.read_content(f"{_DOC_REF}/page:2")
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(("selection", "page_count"), [("1-1", 4), ("1,1", 4), ("all", 1)])
def test_single_page_range_continuation_stays_bounded(tmp_path: Path, selection: str, page_count: int) -> None:
    """范围规范化为一个数字后，携带 after 的续读仍须在该范围内结束。"""

    async def run() -> None:
        """保留后续缓存页，确保单页范围不会退回旧的单点继续推荐行为。"""
        pages = [_page(0, "abcdefghijklmnop")]
        if page_count > 1:
            pages.append(_page(1, "OUTSIDE"))
        db, server = await _cached_library(tmp_path, page_count=page_count, pages=pages)
        try:
            locator = f"{_DOC_REF}/page:{selection}"
            after = None
            chunks: list[str] = []
            for _ in range(10):
                result = await server.read_content(locator, after=after, limit=3, no_marker=True)
                chunks.append(result.content)
                assert result.request_scope.page_range == "1"
                if result.next_request is None:
                    break
                assert result.next_request.locator == f"{_DOC_REF}/page:1"
                locator = result.next_request.locator
                after = result.next_request.after
                assert after is not None
            else:
                pytest.fail("单页范围续读未完成")
            assert "".join(chunks) == "abcdefghijklmnop"
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "after",
    ["doc:bbbbbbb/tier:flash/page:1/block:1", f"doc:{_SHORT_ID}/tier:basic/page:1/block:1", f"{_DOC_REF}/page:2/block:1"],
)
def test_range_continuation_validates_cursor_scope(tmp_path: Path, after: str) -> None:
    """续读游标必须属于当前文档、档位和选定页面。"""

    async def run() -> None:
        """非法游标不能绕过范围选择。"""
        db, server = await _cached_library(tmp_path)
        try:
            with pytest.raises(InvalidRequestError) as error:
                await server.read_content(f"{_DOC_REF}/page:1,3", after=after)
            assert error.value.param == "after"
        finally:
            await db.close()

    asyncio.run(run())


def test_range_http_continuation_round_trip(tmp_path: Path) -> None:
    """HTTP 接口返回的 locator/after 可以直接再次提交并完整读完范围。"""

    async def run() -> None:
        """经真实 ASGI 路由执行两页非连续选择的字符续读。"""
        db, server = await _cached_library(tmp_path, page_count=4)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
                params: dict[str, Any] = {"locator": f"{_DOC_REF}/page:1,3", "limit": 4, "no_marker": True}
                chunks: list[str] = []
                for _ in range(10):
                    response = await client.get("/api/v1/content", params=params)
                    assert response.status_code == 200
                    data = response.json()
                    chunks.append(data["content"].replace("\n\n", ""))
                    if data["next_request"] is None:
                        break
                    params.update(locator=data["next_request"]["locator"], after=data["next_request"]["after"])
                else:
                    pytest.fail("HTTP 范围续读未完成")
                assert "".join(chunks) == "PAGE01PAGE03"
        finally:
            await db.close()

    asyncio.run(run())


def test_read_sdk_transmits_range_after(monkeypatch: pytest.MonkeyPatch) -> None:
    """同步 SDK 同时传递范围定位器及绝对续读游标。"""
    captured: list[dict[str, Any]] = []
    locator = f"{_DOC_REF}/page:1,3"
    after = f"{_DOC_REF}/page:1/block:1/char:4"

    def send_request(self: DoclibClient, method: str, path: str, *, params: dict[str, Any], json_data: Any) -> httpx.Response:
        """保留 SDK 构造参数的路径，只替换外部 HTTP 发送。"""
        captured.append(params)
        response = DocContentResponse(
            sha256=_SHA256, short_id=_SHORT_ID, tier="flash", content="text", request_scope=ContentRequestScope(locator=locator)
        )
        return httpx.Response(200, request=httpx.Request(method, f"http://test{path}"), json=response.model_dump(mode="json"))

    monkeypatch.setattr(DoclibClient, "_send_request", send_request)
    client = DoclibClient(base_url="http://test")
    try:
        client.read_content(locator, after=after)
        assert captured[0]["locator"] == locator
        assert captured[0]["after"] == after
    finally:
        client.close()


def test_read_cli_passes_and_renders_range_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 接收 --after，并在下一次命令中同时保留范围和字符游标。"""
    locator = f"{_DOC_REF}/page:1,3"
    after = f"{_DOC_REF}/page:1/block:1/char:4"
    next_after = f"{_DOC_REF}/page:1/block:1/char:8"
    calls: list[dict[str, Any]] = []

    class Client:
        """用记录器替换外部服务连接。"""

        def __init__(self, *, timeout: int) -> None:
            """内容读取沿用原有一分钟超时。"""
            assert timeout == 60

        def read_content(self, value: str, **kwargs: Any) -> DocContentResponse:
            """返回仍在范围内部的字符续读请求。"""
            calls.append({"locator": value, **kwargs})
            return DocContentResponse(
                sha256=_SHA256,
                short_id=_SHORT_ID,
                tier="flash",
                content="text",
                request_scope=ContentRequestScope(locator=value, after=after),
                truncated=True,
                next_request=ContentNextRequest(locator=value, after=next_after),
            )

    monkeypatch.setattr(read_commands, "DoclibClient", Client)
    result = CliRunner().invoke(app, ["read", locator, "--after", after])
    assert result.exit_code == 0
    assert calls[0]["after"] == after
    assert f"<!-- Next: mineru read {locator} --after {next_after} -->" in result.output
