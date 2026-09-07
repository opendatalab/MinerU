"""HTML 预览的 HTTP 图片、独立文本下载包及物化语义树回归。"""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, unquote

import gradio as gr
import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from mineru.kit.gradio.app import _download_handler, _gradio_public_base_url, build_gradio_app
from mineru.kit.gradio.artifacts import persist_parse_result, render_download, render_html_preview
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.parser.base import ParseResult
from mineru.types import BlockType, EquationBlock, TableBlock, TableBodyBlock
from test_kit_gradio import _middle_json


def _image_bytes(color: str) -> bytes:
    """生成可真实加载的彩色 PNG，以区分同页不同图片。"""
    output = io.BytesIO()
    Image.new("RGB", (24, 16), color).save(output, "PNG")
    return output.getvalue()


@pytest.mark.parametrize("image_source", ["inline", "sidecar"])
def test_materialized_html_and_independent_archives(tmp_path: Path, image_source: str) -> None:
    """正文、表格图片和公式保持语义，HTTP 与 ZIP 各自使用正确的图片地址。"""
    source = tmp_path / "中文 报告.docx"
    source.write_bytes(b"source")
    middle = _middle_json(file_suffix="docx")
    image_body = middle.pages[0].blocks[1].content[0]
    image_body.image_url = "https://example.test/fallback.png"
    red = _image_bytes("red")
    blue = _image_bytes("blue")
    if image_source == "inline":
        image_body.image_base64 = "data:image/png;base64," + base64.b64encode(red).decode()
        table_image = "data:image/png;base64," + base64.b64encode(blue).decode()
    else:
        (tmp_path / "图 1.png").write_bytes(red)
        (tmp_path / "图 2.png").write_bytes(blue)
        image_body.image_path = "图 1.png"
        table_image = quote("图 2.png")
    middle.pages[0].blocks.extend(
        [
            TableBlock(
                type=BlockType.TABLE,
                index=2,
                content=[
                    TableBodyBlock(
                        type=BlockType.TABLE_BODY,
                        index=2,
                        content=f'<table><tr><td><img src="{table_image}"></td></tr></table>',
                    ),
                ],
            ),
            EquationBlock(type=BlockType.EQUATION, index=3, content="x^2 + y^2 = z^2"),
        ]
    )
    original = middle.model_dump_json()
    artifacts = persist_parse_result(ParseResult(middle_json=middle), source, output_root=tmp_path / "输出 空格", page_range="")
    assert middle.model_dump_json() == original
    assert len(list((artifacts.root / "images").iterdir())) == 2
    assert "base64," not in artifacts.middle_json_path.read_text()
    # 后续导出只能读取物化素材，不再需要源文件或源目录中的图片。
    source.unlink()
    artifacts.source_path.unlink()
    if image_source == "sidecar":
        (tmp_path / "图 1.png").unlink()
        (tmp_path / "图 2.png").unlink()

    base = "https://demo.example.test/mineru"
    preview = render_html_preview(artifacts, public_base_url=base)
    frame = BeautifulSoup(preview, "html.parser").iframe
    html_document = frame["srcdoc"]
    assert html_document.lower().startswith("<!doctype html>")
    assert "<style>" in html_document and "MathJax-script" in html_document
    assert "allow-scripts" in frame["sandbox"] and "allow-same-origin" not in frame["sandbox"]
    assert "data:image/" not in html_document
    urls = [image["src"] for image in BeautifulSoup(html_document, "html.parser").find_all("img")]
    assert len(urls) == 2
    for url in urls:
        assert url.startswith(f"{base}/gradio_api/file=") and " " not in url
        path = Path(unquote(url.split("/gradio_api/file=", 1)[1]))
        assert path.read_bytes() in {red, blue}

    for format_name, extension in (("markdown", "md"), ("json", "json"), ("latex", "tex")):
        path = Path(render_download(artifacts.as_state(), format_name, allowed_root=tmp_path))
        assert path.name == f"{artifacts.stem}_{format_name}.zip"
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            document = archive.read(f"{artifacts.stem}.{extension}").decode()
            assert "base64," not in document and "gradio_api/file=" not in document
            assert all(name.startswith("images/") or name == f"{artifacts.stem}.{extension}" for name in names)
            references = set(re.findall(r"images/[a-f0-9]+\.png", document))
            assert references and references.issubset(names)
            assert {archive.read(name) for name in references} == {red, blue}
            if format_name == "json":
                structured = json.loads(document)
                assert "pages" in structured and "schema_version" not in structured
                assert structured["pages"][0]["blocks"][1]["image_source"] in names
        modified = path.stat().st_mtime_ns
        assert render_download(artifacts.as_state(), format_name) == str(path)
        assert path.stat().st_mtime_ns == modified

    # HTML 是一个文件，每次按当前请求站点重建链接。
    for public_base in (base, "http://localhost:17866"):
        path = Path(render_download(artifacts.as_state(), "html", public_base_url=public_base))
        assert path.suffix == ".html"
        content = path.read_text()
        assert "data:image/" not in content
        assert all(
            img["src"].startswith(public_base + "/gradio_api/file=")
            for img in BeautifulSoup(content, "html.parser").find_all("img")
        )
    with pytest.raises(ValueError, match="Unsupported download format"):
        render_download(artifacts.as_state(), "zip")


@pytest.mark.parametrize("format_name,extension", [("markdown", "md"), ("json", "json")])
def test_text_download_without_images(tmp_path: Path, format_name: str, extension: str) -> None:
    """无图文档仍下载独立 ZIP，不添加其他格式或内部产物。"""
    source = tmp_path / "text.docx"
    source.write_bytes(b"source")
    result = ParseResult(middle_json=_middle_json(with_image=False, file_suffix="docx"))
    artifacts = persist_parse_result(result, source, output_root=tmp_path / "output", page_range="")
    with zipfile.ZipFile(render_download(artifacts.as_state(), format_name)) as archive:
        assert archive.namelist() == [f"text.{extension}"]


@pytest.mark.parametrize(
    "headers,root_path,expected",
    [
        ({"host": "localhost:17866"}, "", "http://localhost:17866"),
        ({"host": "localhost:17866"}, "/mineru", "http://localhost:17866/mineru"),
        (
            {"host": "internal", "x-forwarded-host": "demo.test, proxy", "x-forwarded-proto": "https, http"},
            "/mineru",
            "https://demo.test/mineru",
        ),
        ({"host": "internal"}, "https://shared.test/app/", "https://shared.test/app"),
    ],
)
def test_html_public_url_uses_request_context(headers: dict[str, str], root_path: str, expected: str) -> None:
    """验证当前端口、反向代理和子路径都进入 HTML 图片公开地址。"""
    request = Request(
        {
            "type": "http",
            "scheme": "http",
            "server": ("internal", 80),
            "path": "/queue/join",
            "root_path": root_path,
            "headers": [(key.encode(), value.encode()) for key, value in headers.items()],
        }
    )
    assert _gradio_public_base_url(gr.Request(request=request)) == expected


def test_html_images_are_served_by_gradio(tmp_path: Path) -> None:
    """通过真实 Gradio 文件路由读取预览中的 PNG，同时检查下载请求没有新增用户输入。"""
    source = tmp_path / "report.docx"
    source.write_bytes(b"source")
    middle = _middle_json(file_suffix="docx")
    data = _image_bytes("green")
    middle.pages[0].blocks[1].content[0].image_base64 = "data:image/png;base64," + base64.b64encode(data).decode()
    artifacts = persist_parse_result(ParseResult(middle_json=middle), source, output_root=tmp_path / "output", page_range="")
    cap = V1ServerCapabilities("http://unused.test", ("flash",), ("zip",), ("file_id",))
    demo = build_gradio_app(SimpleNamespace(), cap, output_root=tmp_path / "output", enable_example=False)
    demo.allowed_paths = [str(tmp_path / "output")]
    app = gr.routes.App.create_app(demo)
    preview = render_html_preview(artifacts, public_base_url="http://testserver")
    content = BeautifulSoup(preview, "html.parser").iframe["srcdoc"]
    url = BeautifulSoup(content, "html.parser").img["src"]
    with TestClient(app) as client:
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content == data
    output = next(component for component in demo.blocks.values() if "mineru-markdown-output" in (component.elem_classes or []))
    assert isinstance(output, gr.HTML)
    handlers = [fn for fn in demo.fns.values() if fn.name == "handler"]
    assert len(handlers) == 7 and all(len(fn.inputs) == 2 for fn in handlers)
    handler = _download_handler("html", tmp_path / "output")
    path, receipt = handler(
        artifacts.as_state(),
        json.dumps({"run_id": artifacts.root.name}),
        SimpleNamespace(headers={"host": "current.test:9999"}),
    )
    assert not json.loads(receipt)["error"]
    assert "http://current.test:9999/gradio_api/file=" in Path(path).read_text()


def test_archive_failure_does_not_poison_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """图片读取中途失败不能生成有效缓存，恢复后再次下载应得到完整压缩包。"""
    source = tmp_path / "report.docx"
    source.write_bytes(b"source")
    middle = _middle_json(file_suffix="docx")
    middle.pages[0].blocks[1].content[0].image_base64 = (
        "data:image/png;base64," + base64.b64encode(_image_bytes("red")).decode()
    )
    artifacts = persist_parse_result(ParseResult(middle_json=middle), source, output_root=tmp_path / "output", page_range="")
    write = zipfile.ZipFile.write

    def fail_image(archive: zipfile.ZipFile, filename: str | Path, arcname: str | None = None) -> None:
        """模拟正文已写入、图片开始写入时发生磁盘读取错误。"""
        if arcname and arcname.startswith("images/"):
            raise OSError("image read failed")
        write(archive, filename, arcname)

    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile, "write", fail_image)
        with pytest.raises(OSError, match="image read failed"):
            render_download(artifacts.as_state(), "markdown")
    assert not list(artifacts.downloads_dir.iterdir())
    with zipfile.ZipFile(render_download(artifacts.as_state(), "markdown")) as archive:
        assert len(archive.namelist()) == 2
