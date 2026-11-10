"""验证单点续读不漏块，并保留单块读取、范围读取及正常翻页语义。"""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path

import httpx
import pytest
from doclib_test_utils import DOCLIB_AUTH_HEADERS
from test_doclib_page_selection import _DOC_REF, _cached_library, _page

from mineru.cli.commands.read import _read_next_marker
from mineru.doclib.server import DoclibServer
from mineru.doclib.types import DocContentResponse
from mineru.errors import InvalidRequestError


def _body(content: str) -> str:
    """忽略页标记和块间空白，保留所有测试字符用于查重和漏读检查。"""
    content = re.sub(r"<!-- page [0-9]+(?: of [0-9]+)? -->", "", content)
    return re.sub(r"\s+", "", content)


async def _follow(
    server: DoclibServer,
    locator: str,
    *,
    limit: int,
    no_marker: bool = True,
    cli_markers: bool = False,
) -> list[DocContentResponse]:
    """只使用服务返回的请求推进，不自行推测下一个块或下一页。"""
    results: list[DocContentResponse] = []
    after = None
    visited: set[tuple[str, str | None]] = set()
    for _ in range(100):
        state = (locator, after)
        assert state not in visited, state
        visited.add(state)
        result = await server.read_content(
            locator,
            after=after,
            limit=30000 if cli_markers and results else limit,
            no_marker=no_marker,
        )
        results.append(result)
        if result.next_request is None:
            return results
        if cli_markers:
            marker = _read_next_marker(result.next_request)
            assert marker is not None
            command = shlex.split(marker.removeprefix("<!-- Next: ").removesuffix(" -->"))
            assert command[:2] == ["mineru", "read"]
            locator = command[2]
            after = command[command.index("--after") + 1] if "--after" in command else None
        else:
            assert result.next_request.locator is not None
            locator = result.next_request.locator
            after = result.next_request.after
    pytest.fail("Locator continuation did not terminate")


@pytest.mark.parametrize("suffix", ["", "/page:1"])
@pytest.mark.parametrize("limit", [3, 8, 21])
@pytest.mark.parametrize("no_marker", [True, False])
def test_locator_continuation_preserves_all_blocks(tmp_path: Path, suffix: str, limit: int, no_marker: bool) -> None:
    """字符内、块间及页间截断后，原样续读必须不重、不漏地到达下一页。"""

    async def run() -> None:
        db, server = await _cached_library(
            tmp_path, page_count=2, pages=[_page(0, "ALPHA", "BRAVO", "CHARLIE"), _page(1, "DELTA")]
        )
        try:
            results = await _follow(server, _DOC_REF + suffix, limit=limit, no_marker=no_marker)
            assert "".join(_body(result.content) for result in results) == "ALPHABRAVOCHARLIEDELTA"
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "suffix,expected",
    [
        ("/block:2", "BRAVODELTA"),
        ("/block:2/char:0", "BRAVODELTA"),
        ("/block:2/char:2", "AVODELTA"),
    ],
)
@pytest.mark.parametrize("limit", [3, 30000])
def test_direct_block_continuation_does_not_expand_to_neighbours(
    tmp_path: Path, suffix: str, expected: str, limit: int
) -> None:
    """直接选择一个块时保留块内字符续读，不把同页相邻块纳入读取范围。"""

    async def run() -> None:
        db, server = await _cached_library(
            tmp_path, page_count=2, pages=[_page(0, "ALPHA", "BRAVO", "CHARLIE"), _page(1, "DELTA")]
        )
        try:
            results = await _follow(server, f"{_DOC_REF}/page:1{suffix}", limit=limit)
            assert "".join(_body(result.content) for result in results) == expected
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "suffix,expected,next_page",
    [
        ("/block:1", "BRAVOCHARLIE", 2),
        ("/block:1/char:2", "PHABRAVOCHARLIE", 2),
        ("", "DELTA", 3),
    ],
)
def test_document_after_preserves_page_turning(tmp_path: Path, suffix: str, expected: str, next_page: int) -> None:
    """文档续读恢复游标所在页；页游标表示已完成该页，读完后保留下一页建议。"""

    async def run() -> None:
        db, server = await _cached_library(
            tmp_path,
            page_count=3,
            pages=[_page(0, "ALPHA", "BRAVO", "CHARLIE"), _page(1, "DELTA"), _page(2, "LAST")],
        )
        try:
            result = await server.read_content(_DOC_REF, after=f"{_DOC_REF}/page:1{suffix}", no_marker=True)
            assert _body(result.content) == expected
            assert result.next_request is not None
            assert result.next_request.locator == f"{_DOC_REF}/page:{next_page}"
            assert result.next_request.after is None
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "after",
    [
        "doc:bbbbbbb/tier:flash/page:1/block:1",
        "doc:aaaaaaa/tier:standard/page:1/block:1",
    ],
)
def test_document_after_rejects_other_documents_and_tiers(tmp_path: Path, after: str) -> None:
    """文档级续读仍必须拒绝跨文档和跨档位游标。"""

    async def run() -> None:
        db, server = await _cached_library(tmp_path)
        try:
            with pytest.raises(InvalidRequestError) as error:
                await server.read_content(_DOC_REF, after=after)
            assert error.value.param == "after"
        finally:
            await db.close()

    asyncio.run(run())


def test_locator_continuation_skips_empty_blocks_and_index_gaps(tmp_path: Path) -> None:
    """使用实际消费游标，不假定块号连续或每个块都可渲染。"""

    async def run() -> None:
        page = _page(0, "ALPHA", " ", "CHARLIE")
        page.blocks[-1].index = 7
        db, server = await _cached_library(tmp_path, page_count=2, pages=[page, _page(1, "DELTA")])
        try:
            results = await _follow(server, f"{_DOC_REF}/page:1", limit=5)
            assert "".join(_body(result.content) for result in results) == "ALPHACHARLIEDELTA"
        finally:
            await db.close()

    asyncio.run(run())


@pytest.mark.parametrize("limit", [5, 30000])
def test_document_continuation_reaches_pages_after_default_window(tmp_path: Path, limit: int) -> None:
    """默认十页窗口之后仍能读到后续页面，且每页只出现一次。"""

    async def run() -> None:
        db, server = await _cached_library(tmp_path)
        try:
            results = await _follow(server, _DOC_REF, limit=limit)
            assert "".join(_body(result.content) for result in results) == "".join(
                f"PAGE{number:02}" for number in range(1, 13)
            )
        finally:
            await db.close()

    asyncio.run(run())


def test_following_cli_markers_with_default_budget_preserves_content(tmp_path: Path) -> None:
    """CLI 续读命令不携带 --limit；恢复默认预算后仍不能遗漏同页剩余块。"""

    async def run() -> None:
        db, server = await _cached_library(
            tmp_path, page_count=2, pages=[_page(0, "ALPHA", "BRAVO", "CHARLIE"), _page(1, "DELTA")]
        )
        try:
            results = await _follow(server, f"{_DOC_REF}/page:1", limit=8, no_marker=False, cli_markers=True)
            assert "".join(_body(result.content) for result in results) == "ALPHABRAVOCHARLIEDELTA"
        finally:
            await db.close()

    asyncio.run(run())


def test_locator_continuation_http_round_trip(tmp_path: Path) -> None:
    """真实路由返回的续读请求可直接回传，不依赖调用者推测正确起点。"""

    async def run() -> None:
        db, server = await _cached_library(
            tmp_path, page_count=2, pages=[_page(0, "ALPHA", "BRAVO", "CHARLIE"), _page(1, "DELTA")]
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app), base_url="http://test", headers=DOCLIB_AUTH_HEADERS
            ) as client:
                locator = f"{_DOC_REF}/page:1"
                after = None
                chunks: list[str] = []
                for _ in range(20):
                    response = await client.get(
                        "/api/v1/content",
                        params={
                            "locator": locator,
                            "limit": 8,
                            "no_marker": True,
                            **({"after": after} if after is not None else {}),
                        },
                    )
                    assert response.status_code == 200, response.text
                    data = response.json()
                    chunks.append(_body(data["content"]))
                    if data["next_request"] is None:
                        break
                    locator = data["next_request"]["locator"]
                    after = data["next_request"]["after"]
                else:
                    pytest.fail("HTTP continuation did not terminate")
                assert "".join(chunks) == "ALPHABRAVOCHARLIEDELTA"
        finally:
            await db.close()

    asyncio.run(run())
