"""HTML 预览的 HTTP 图片、独立文本下载包及物化语义树回归。"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import zipfile
from dataclasses import replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from urllib.parse import quote, unquote

import gradio as gr
import pytest
from _epub_test_utils import build_epub_fixture
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

from mineru.backend.analyze import doc_analyze
from mineru.kit.gradio import artifacts as artifacts_module
from mineru.kit.gradio.app import _download_handler, _gradio_public_base_url, build_gradio_app
from mineru.kit.gradio.artifacts import _prepare_preview_links, persist_parse_result, render_download, render_html_preview
from mineru.kit.gradio.client import V1ServerCapabilities
from mineru.parser.base import ParseResult
from mineru.types import BlockType, ChartBlock, ChartBodyBlock, EquationBlock, TableBlock, TableBodyBlock
from test_kit_gradio import _middle_json


def _image_bytes(color: str) -> bytes:
    """生成可真实加载的彩色 PNG，以区分同页不同图片。"""
    output = io.BytesIO()
    Image.new("RGB", (24, 16), color).save(output, "PNG")
    return output.getvalue()


@pytest.mark.parametrize("entrypoint", ["preview", "download"])
@pytest.mark.parametrize(
    "platform_root",
    [
        PurePosixPath("/tmp/输出 文件/gradio/授权书"),
        PureWindowsPath("C:/Users/测试用户/输出 文件/gradio/授权书"),
        PureWindowsPath("//fileserver/共享目录/输出 文件/gradio/授权书"),
    ],
    ids=["posix", "windows-drive", "windows-unc"],
)
def test_platform_asset_urls_preserve_seal_images_and_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform_root: PurePosixPath | PureWindowsPath,
    entrypoint: str,
) -> None:
    """跨平台资源根保留印章、普通图片、图表和表内图片，预览及下载均不会被路径校验过滤。"""
    middle = _middle_json(file_suffix="docx")
    seal = middle.pages[0].blocks[1]
    seal.sub_type = "seal"
    seal.content[0].content = "上海人工智能创新中心"
    seal.content[0].image_path = "images/印章 1.png"
    photo = seal.model_copy(deep=True, update={"index": 2, "sub_type": None})
    photo.content[0].index = 2
    middle.pages[0].blocks.extend(
        [
            photo,
            ChartBlock(
                type="chart",
                index=3,
                content=[ChartBodyBlock(type="chart_body", index=3, content="", image_path="images/chart.png")],
            ),
            TableBlock(
                type="table",
                index=4,
                content=[
                    TableBodyBlock(
                        type="table_body", index=4, content='<table><tr><td><img src="images/cell.png"></td></tr></table>'
                    )
                ],
            ),
        ]
    )
    artifacts = artifacts_module.create_run_artifacts(tmp_path / "授权书.docx", tmp_path)
    artifacts.middle_json_path.write_text(ParseResult(middle_json=middle).to_json(), encoding="utf-8")
    original_render_html = artifacts_module._render_html

    def render_with_platform_root(artifacts: artifacts_module.RunArtifacts, *, public_base_url: str) -> str:
        """文件读写与边界校验仍在本机执行，仅在实际 URL 构造处注入目标平台路径。"""
        return original_render_html(replace(artifacts, root=platform_root), public_base_url=public_base_url)

    monkeypatch.setattr(artifacts_module, "_render_html", render_with_platform_root)
    public_base = "https://demo.example.test/mineru"
    if entrypoint == "preview":
        frame = BeautifulSoup(render_html_preview(artifacts, public_base_url=public_base), "html.parser").iframe
        document = frame["srcdoc"]
    else:
        path = render_download(artifacts.as_state(), "html", allowed_root=tmp_path, public_base_url=public_base)
        document = Path(path).read_text(encoding="utf-8")
    soup = BeautifulSoup(document, "html.parser")
    images = soup.find_all("img")
    assert len(images) == 4
    expected_base = f"{public_base}/gradio_api/file={quote(platform_root.as_posix(), safe='/:')}/images/"
    assert all(image["src"].startswith(expected_base) for image in images)
    assert all("%5c" not in image["src"].lower() and " " not in image["src"] for image in images)
    details = soup.find("summary", string="seal").parent
    assert details.name == "details" and "open" not in details.attrs
    assert "上海人工智能创新中心" in details.get_text()
    assert details.find_previous_sibling("img") is not None


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
    assert {path.name for path in (artifacts.root / "images").iterdir()} == {"page_0_image_1.png", "page_0_table_image_2_1.png"}
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
            references = set(re.findall(r"images/page_\d+_[a-z_]+_\d+(?:_\d+)?\.png", document))
            assert references == {"images/page_0_image_1.png", "images/page_0_table_image_2_1.png"}
            assert references.issubset(names)
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


@pytest.mark.parametrize("public_base", ["http://localhost:17866", "https://demo.example.test/mineru"])
def test_epub_preview_anchors_stay_in_srcdoc_without_changing_download(tmp_path: Path, public_base: str) -> None:
    """EPUB 跨章和返回链接使用预览自身的基准地址，独立 HTML 下载保留普通锚点语义。"""
    source = tmp_path / "book.epub"
    source.write_bytes(build_epub_fixture())
    middle, _ = doc_analyze(source.read_bytes(), effort="flash", file_suffix="epub")
    artifacts = persist_parse_result(ParseResult(middle_json=middle), source, output_root=tmp_path, page_range="")

    frame = BeautifulSoup(render_html_preview(artifacts, public_base_url=public_base), "html.parser").iframe
    preview = BeautifulSoup(frame["srcdoc"], "html.parser")
    assert [base["href"] for base in preview.find_all("base")] == ["about:srcdoc"]
    assert preview.base.parent is preview.head
    assert set(frame["sandbox"]) == {"allow-scripts", "allow-popups", "allow-popups-to-escape-sandbox"}
    for label, heading in (("chapter two", "Section Two"), ("Back", "Chapter One")):
        link = preview.find("a", string=label)
        assert link["href"].startswith("#")
        target = preview.find(id=unquote(link["href"][1:]))
        assert target is not None and target.get_text(strip=True) == heading

    download_path = Path(render_download(artifacts.as_state(), "html", public_base_url=public_base))
    download = BeautifulSoup(download_path.read_text(encoding="utf-8"), "html.parser")
    assert download.find("base") is None
    assert "about:srcdoc" not in str(download)
    assert [link["href"] for link in preview.find_all("a", href=True)] == [
        link["href"] for link in download.find_all("a", href=True)
    ]
    assert preview.body == download.body
    assert preview.find("script", id="MathJax-script")["src"].startswith("https://")
    assert preview.find_all("img")
    assert all(image["src"].startswith(f"{public_base}/gradio_api/file=") for image in preview.find_all("img"))


@pytest.mark.parametrize(
    "href,external",
    [
        ("https://doi.org/10.1000/example", True),
        ("http://example.test/paper#references", True),
        ("HTTPS://example.test/?a=1&b=2#section", True),
        ("#chapter-one", False),
        ("#", False),
        ("", False),
        ("mailto:author@example.test", False),
        ("tel:+12345678", False),
        ("chapter.html#section", False),
        ("/papers/one", False),
        ("https://[invalid", False),
    ],
)
def test_preview_link_targets_preserve_document_links(href: str, external: bool) -> None:
    """按链接协议区分网页与文内跳转，覆盖表格链接并保留已有 rel 和其他标签内容。"""
    document = (
        '<html><head><style>a { color: blue; }</style></head><body>'
        f'<table><tr><td><a href="{html.escape(href, quote=True)}" target="_self" rel="nofollow noopener">'
        '<em>Reference</em></a></td></tr></table><a id="chapter-one">Chapter One</a>'
        '<script>const example = "<a href=\'https://example.test\'>";</script></body></html>'
    )
    original = BeautifulSoup(document, "html.parser")
    prepared = BeautifulSoup(_prepare_preview_links(document), "html.parser")
    link = prepared.find("a", href=True)
    assert link["href"] == href
    assert link["target"] == ("_blank" if external else "_self")
    assert link["rel"] == (["nofollow", "noopener", "noreferrer"] if external else ["nofollow", "noopener"])
    assert link.em.get_text() == "Reference"
    assert prepared.find("a", id="chapter-one").attrs == {"id": "chapter-one"}
    assert prepared.script == original.script and prepared.style == original.style
    assert _prepare_preview_links(str(prepared)) == str(prepared)


def test_external_preview_links_do_not_change_html_download(tmp_path: Path) -> None:
    """真实 renderer 产生的自动链接仅在预览中新增 target，下载继续使用独立 HTML 的默认行为。"""
    source = tmp_path / "references.docx"
    source.write_bytes(b"source")
    middle = _middle_json(with_image=False, file_suffix="docx")
    middle.pages[0].blocks[0].content[0].content = "See https://example.test/paper#references"
    artifacts = persist_parse_result(ParseResult(middle_json=middle), source, output_root=tmp_path, page_range="")
    base = "https://demo.example.test/mineru"
    frame = BeautifulSoup(render_html_preview(artifacts, public_base_url=base), "html.parser").iframe
    preview = BeautifulSoup(frame["srcdoc"], "html.parser")
    download_path = Path(render_download(artifacts.as_state(), "html", public_base_url=base))
    download = BeautifulSoup(download_path.read_text(encoding="utf-8"), "html.parser")
    assert preview.a["target"] == "_blank"
    assert set(preview.a["rel"]) == {"noopener", "noreferrer"}
    assert preview.a["href"] == download.a["href"] == "https://example.test/paper#references"
    assert "target" not in download.a.attrs
    assert download.find("base") is None
    assert "allow-popups-to-escape-sandbox" in frame["sandbox"]
    assert "allow-same-origin" not in frame["sandbox"]


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
