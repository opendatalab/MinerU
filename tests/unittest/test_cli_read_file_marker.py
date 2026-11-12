"""验证文件保存只恢复已有续读提示，不改变正文、普通输出或其他格式。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_doclib_page_selection import _DOC_REF, _cached_library, _page
from typer.testing import CliRunner

from mineru.cli.commands import read
from mineru.cli.main import app
from mineru.doclib.types import ContentAsset, ContentNextRequest, ContentRequestScope, DocContentResponse

_LOCATOR = "doc:ab12cd3/tier:flash/page:all"
_BODY = "# 原文标题\n\n正文 **加粗** 与 `代码`。\n<!-- page 1 -->\n"
_REQUESTS = {
    "none": None,
    "next_page": ContentNextRequest(locator="doc:ab12cd3/tier:flash/page:2"),
    "block": ContentNextRequest(locator=_LOCATOR, after="doc:ab12cd3/tier:flash/page:1/block:2"),
    "character": ContentNextRequest(locator=_LOCATOR, after="doc:ab12cd3/tier:flash/page:1/block:2/char:5"),
    "no_locator": ContentNextRequest(page_range="2"),
}
runner = CliRunner()


def _response(*, body: str = _BODY, request: str = "block") -> DocContentResponse:
    return DocContentResponse(
        sha256="a" * 64,
        short_id="ab12cd3",
        tier="flash",
        content=body,
        request_scope=ContentRequestScope(locator=_LOCATOR),
        truncated=request in {"block", "character"},
        next_request=_REQUESTS[request],
    )


def _use_response(monkeypatch: pytest.MonkeyPatch, response: DocContentResponse) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            calls.append({"locator": locator, **kwargs})
            return response

    monkeypatch.setattr(read, "DoclibClient", Client)
    return calls


def _expected_marker(request: ContentNextRequest | None, *, no_marker: bool) -> str:
    if no_marker or request is None or not request.locator:
        return ""
    after = f" --after {request.after}" if request.after else ""
    return f"<!-- Next: mineru read {request.locator}{after} -->"


def _record_file(path: Path, expected: bytes, record_property: Callable[[str, object], None]) -> None:
    actual = path.read_bytes()
    record_property("saved_sha256", hashlib.sha256(actual).hexdigest())
    record_property("saved_size", len(actual))
    assert actual == expected


def _file_text_bytes(text: str) -> bytes:
    """Match Path.write_text's existing platform-native newline handling."""
    return text.replace("\n", os.linesep).encode("utf-8")


@pytest.mark.parametrize("output_mode", ["stdout", "dash", "file"])
@pytest.mark.parametrize("no_marker", [False, True])
@pytest.mark.parametrize("request_kind", list(_REQUESTS))
def test_read_output_marker_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
    output_mode: str,
    no_marker: bool,
    request_kind: str,
) -> None:
    response = _response(request=request_kind)
    calls = _use_response(monkeypatch, response)
    output = tmp_path / "nested" / "part.md"
    args = ["read", _LOCATOR]
    if output_mode != "stdout":
        args += ["-o", "-" if output_mode == "dash" else str(output)]
    if no_marker:
        args.append("--no-marker")
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert calls == [{"locator": _LOCATOR, "context": 0, "limit": 30000, "format": "markdown", "no_marker": no_marker}]
    record_property("stdout", result.stdout)
    if output_mode == "file":
        _record_file(output, _file_text_bytes(_BODY), record_property)
        expected = f"Written to {output.resolve()}"
    else:
        assert not output.exists()
        expected = _BODY
    marker = _expected_marker(response.next_request, no_marker=no_marker)
    if marker:
        expected = expected.rstrip("\n") + "\n\n" + marker
    assert result.stdout == expected + "\n"


@pytest.mark.parametrize("output_mode", ["stdout", "dash", "file"])
@pytest.mark.parametrize("no_marker", [False, True])
def test_json_output_keeps_metadata_without_text_notices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
    output_mode: str,
    no_marker: bool,
) -> None:
    response = _response()
    _use_response(monkeypatch, response)
    output = tmp_path / "part.md"
    args = ["read", _LOCATOR, "--json"]
    if output_mode != "stdout":
        args += ["-o", "-" if output_mode == "dash" else str(output)]
    if no_marker:
        args.append("--no-marker")
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    expected = response.model_dump(mode="json")
    if output_mode == "file":
        _record_file(output, _file_text_bytes(_BODY), record_property)
        expected.update(content=None, output={"status": "written", "path": str(output.resolve())})
    assert json.loads(result.stdout) == expected
    assert "<!-- Next:" not in result.stdout
    assert "Written to" not in result.stdout


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("no_marker", [False, True])
def test_image_file_output_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
    json_mode: bool,
    no_marker: bool,
) -> None:
    asset = tmp_path / "asset.png"
    asset.write_bytes(b"existing-image-asset")
    output = tmp_path / "image.png"
    response = _response(body="").model_copy(
        update={"format": "image", "asset": ContentAsset(path=str(asset), mime_type="image/png")}
    )
    _use_response(monkeypatch, response)
    args = ["read", _LOCATOR, "-f", "image", "-o", str(output)]
    if json_mode:
        args.append("--json")
    if no_marker:
        args.append("--no-marker")
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    _record_file(output, asset.read_bytes(), record_property)
    assert result.stderr == ""
    if json_mode:
        expected = response.model_dump(mode="json")
        expected.update(content=None, output={"status": "written", "path": str(output.resolve())})
        assert json.loads(result.stdout) == expected
    else:
        assert result.stdout == f"Written to {output.resolve()}\n"


@pytest.mark.parametrize("body", ["", "没有换行", "一行\n", "两行\n\n", "标题：αβ\n\n|甲|乙|\n|--|--|\n|1|2|\n"])
def test_saved_body_replaces_old_content_without_adding_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record_property: Callable[[str, object], None],
    body: str,
) -> None:
    _use_response(monkeypatch, _response(body=body))
    output = tmp_path / "part.md"
    output.write_text("old content that must be replaced", encoding="utf-8")
    result = runner.invoke(app, ["read", _LOCATOR, "-o", str(output)])
    assert result.exit_code == 0, result.output
    _record_file(output, _file_text_bytes(body), record_property)
    assert result.stderr == ""
    marker = _expected_marker(_REQUESTS["block"], no_marker=False)
    assert result.stdout == f"Written to {output.resolve()}\n\n{marker}\n"


def test_failed_write_does_not_report_success_or_continuation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_response(monkeypatch, _response())
    result = runner.invoke(app, ["read", _LOCATOR, "-o", str(tmp_path)])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Written to" not in result.output
    assert "<!-- Next:" not in result.output


def test_saved_file_continuation_reads_real_cached_content_without_loss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    """真实缓存产生续读位置；仅替换 HTTP 传输，CLI 和服务读取逻辑均执行原实现。"""
    cache = tmp_path / "cache"
    cache.mkdir()
    db, server = asyncio.run(
        _cached_library(cache, page_count=2, pages=[_page(0, "A" * 12000, "B" * 12000, "C" * 12000), _page(1, "D" * 10)])
    )
    responses: list[DocContentResponse] = []

    class Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def read_content(self, locator: str, **kwargs: Any) -> DocContentResponse:
            response = asyncio.run(server.read_content(locator, **kwargs))
            responses.append(response)
            return response

    monkeypatch.setattr(read, "DoclibClient", Client)
    first_output = tmp_path / "part-01.md"
    second_output = tmp_path / "part-02.md"
    try:
        first = runner.invoke(app, ["read", f"{_DOC_REF}/page:all", "-o", str(first_output)])
        assert first.exit_code == 0, first.output
        assert responses[0].truncated
        assert responses[0].next_request is not None
        _record_file(first_output, _file_text_bytes(responses[0].content), record_property)
        record_property("stdout", first.stdout)
        marker = re.search(r"<!-- Next: (.*?) -->", first.stdout)
        assert marker is not None, first.stdout
        command = shlex.split(marker.group(1))
        assert command[:2] == ["mineru", "read"]
        assert "-o" not in command and "--output" not in command
        second = runner.invoke(app, [*command[1:], "-o", str(second_output)])
        assert second.exit_code == 0, second.output
        assert responses[1].next_request is None
        assert second_output.read_bytes() == _file_text_bytes(responses[1].content)
        assert first_output.read_bytes() == _file_text_bytes(responses[0].content)
        assert "<!-- Next:" not in second.stdout
        combined = first_output.read_text(encoding="utf-8") + second_output.read_text(encoding="utf-8")
        combined = re.sub(r"<!--.*?-->|\s+", "", combined)
        assert combined == "A" * 12000 + "B" * 12000 + "C" * 12000 + "D" * 10
    finally:
        asyncio.run(db.close())
