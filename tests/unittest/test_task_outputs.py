"""验证 Tasks 的全部格式组合、单文件完整性和图片所有权。"""

from __future__ import annotations

import base64
import io
import itertools
import json
import re
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from test_task_api import HTML, _submit, _terminal, _upload, service as task_service

from mineru.parser import api_server
from mineru.parser.base import ParseResult
from mineru.types import (
    EquationBlock,
    ImageBlock,
    ImageBodyBlock,
    MiddleJson,
    PageInfo,
    TableBlock,
    TableBodyBlock,
    TextBlock,
    TextSpan,
)

service = task_service

COMBINATIONS = [
    list(values) for size in range(1, 5) for values in itertools.combinations(("default", "docx", "html", "latex"), size)
]


def _illustrated_result() -> tuple[ParseResult, bytes]:
    """构造同时包含正文图、表格内图片和公式的严格语义结果。"""
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), (20, 80, 150)).save(buffer, format="PNG")
    png = buffer.getvalue()
    uri = "data:image/png;base64," + base64.b64encode(png).decode()
    middle = MiddleJson(
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[
                    TextBlock(type="text", index=0, content=[TextSpan(type="text", content="Export content")]),
                    ImageBlock(
                        type="image",
                        index=1,
                        content=[ImageBodyBlock(type="image_body", index=1, content="Figure", image_base64=uri)],
                    ),
                    TableBlock(
                        type="table",
                        index=2,
                        content=[
                            TableBodyBlock(
                                type="table_body",
                                index=2,
                                content=f'<table><tr><td>Cell</td><td><img src="{uri}"></td></tr></table>',
                            )
                        ],
                    ),
                    EquationBlock(type="equation", index=3, content=r"E=mc^2"),
                ],
            )
        ],
        is_full_document=True,
        metadata={"file_suffix": "docx", "producer": {"name": "mineru", "version": "test"}},
        extensions={"mineru": {"tier": "flash", "parse_mode": "txt"}},
    )
    return ParseResult(middle_json=middle), png


def _check_html(content: bytes, png: bytes) -> None:
    """确认正文和富文本图片均内嵌，并保留文字、表格和公式内容。"""
    soup = BeautifulSoup(content, "html.parser")
    images = soup.find_all("img")
    assert len(images) >= 2
    assert all(image["src"].startswith("data:image/") for image in images)
    assert all(base64.b64decode(image["src"].split(",", 1)[1]) == png for image in images)
    assert "Export content" in soup.get_text() and "Cell" in soup.get_text()
    assert "E=mc^2" in str(soup)


def _check_docx(content: bytes, png: bytes) -> None:
    """检查真正的 OOXML 图片及关系，不把外链或只有 ZIP 外壳当作 DOCX。"""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        media = [name for name in names if name.startswith("word/media/")]
        assert media and all(archive.read(name) == png for name in media)
        assert b"Export content" in archive.read("word/document.xml")
        assert b"Cell" in archive.read("word/document.xml")
        assert b'TargetMode="External"' not in archive.read("word/_rels/document.xml.rels")


@pytest.mark.parametrize("formats", COMBINATIONS, ids=lambda values: "+".join(values))
def test_all_task_output_combinations(
    service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch, formats: list[str]
) -> None:
    """四种格式的十五种组合都只解析一次，产物完整并共享外置图片目录。"""
    client, _ = service
    result, png = _illustrated_result()
    original = deepcopy(result.middle_json)
    calls = 0

    async def parse_one(*args: Any, **kwargs: Any) -> ParseResult:
        """仅隔离输入分析，真实渲染、保存、Files 下载和 Router 转发全部保留。"""
        nonlocal calls
        calls += 1
        return result

    monkeypatch.setattr(api_server, "parse_async", parse_one)
    task = _submit(client, [("source.html", HTML)], output_formats=formats)
    _upload(client, task, [("source.html", HTML)])
    finished = _terminal(client, task)
    assert finished["status"] == "completed", finished
    assert calls == 1 and result.middle_json == original
    downloaded = client.get(finished["files"][0].get("result_url"))
    assert downloaded.status_code == 200
    if formats == ["html"]:
        assert downloaded.headers["content-type"].startswith("text/html")
        _check_html(downloaded.content, png)
        return
    if formats == ["docx"]:
        assert downloaded.headers["content-type"] == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        _check_docx(downloaded.content, png)
        return
    assert downloaded.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(downloaded.content)) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    assert not any(name.endswith(".zip") for name in entries)
    assert "manifest.json" not in entries and "model_output.json" not in entries
    expected = set()
    if "default" in formats:
        expected.update({"markdown.md", "middle_json.json", "structured_content.json"})
    for fmt, name in (("html", "document.html"), ("docx", "document.docx"), ("latex", "document.tex")):
        if fmt in formats:
            expected.add(name)
    assert {name for name in entries if not name.startswith("images/")} == expected
    if "default" in formats or "latex" in formats:
        pictures = {name for name in entries if name.startswith("images/") and name != "images/"}
        assert pictures and all(entries[name] == png for name in pictures)
        if "default" in formats:
            middle = json.loads(entries["middle_json.json"])
            serialized = json.dumps(middle)
            assert "data:image/" not in serialized and "image_base64" not in serialized
            assert "Export content" in entries["markdown.md"].decode()
            for name in re.findall(r'"image_path": "([^"]+)"', serialized):
                assert name in pictures
        if "latex" in formats:
            tex = entries["document.tex"].decode()
            assert "E=mc^2" in tex
            assert r"\includegraphics" in tex
            assert any(name in tex for name in pictures)
    else:
        assert not any(name.startswith("images/") for name in entries)
    if "html" in formats:
        _check_html(entries["document.html"], png)
    if "docx" in formats:
        _check_docx(entries["document.docx"], png)


@pytest.mark.parametrize("formats", [["docx"], ["html"], ["default"], ["latex"]])
def test_task_formats_use_only_service_authentication(tmp_path: Path, formats: list[str]) -> None:
    """启用 Key 时四种格式都需要服务认证，通过认证后都可执行。"""
    app = api_server.create_app(upload_dir=str(tmp_path), tiers=("flash",), api_key="secret")
    with TestClient(app) as client:
        body = {"files": [{"file_name": "a.html", "bytes": len(HTML)}], "output_formats": formats}
        assert client.post("/v1/tasks", json=body).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        task = _submit(client, [("a.html", HTML)], output_formats=formats, headers=headers)
        _upload(client, task, [("a.html", HTML)], headers=headers)
        assert _terminal(client, task, headers=headers)["status"] == "completed"


def test_duplicate_formats_and_missing_images(service: tuple[TestClient, FastAPI], monkeypatch: pytest.MonkeyPatch) -> None:
    """重复格式归一化，缺失素材不能输出伪成功的多格式文件。"""
    client, _ = service
    task = _submit(client, [("a.html", HTML)], output_formats=["html", "html"])
    assert task["output_formats"] == ["html"]
    _upload(client, task, [("a.html", HTML)])
    completed = _terminal(client, task)
    assert client.get(completed["files"][0].get("result_url")).headers["content-type"].startswith("text/html")
    broken, _ = _illustrated_result()
    block = broken.middle_json.pages[0].blocks[1].content[0]
    block.image_base64 = None
    block.image_path = "images/missing.png"

    async def missing_asset(*args: Any, **kwargs: Any) -> ParseResult:
        """保留悬空素材引用，验证真实导出预检会阻止发布。"""
        return broken

    monkeypatch.setattr(api_server, "parse_async", missing_asset)
    task = _submit(client, [("a.html", HTML)], output_formats=["default", "html"])
    _upload(client, task, [("a.html", HTML)])
    failed = _terminal(client, task)
    assert failed["status"] == "failed"
    assert failed["files"][0].get("result_url") is None
    assert failed["files"][0].get("error")["code"] == "result_packaging_failed"
