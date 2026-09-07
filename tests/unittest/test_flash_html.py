from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import ZipFile

import httpx
import pytest
from _span_test_utils import inline_text
from docvortex.analyzers.native import HtmlModel
from docvortex.analyzers.native.html import HtmlResourceLimitError
from docvortex.analyzers.native.html import converter as html_converter_module
from docvortex.export.middle import export_middle_json
from fastapi.testclient import TestClient
from PIL import Image

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.parser import ParseResult, api_server, parse, parse_async
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.render.docx import render_docx
from mineru.render.html import render_html
from mineru.render.markdown import render_markdown
from mineru.render.structured_content import render_structured_content
from mineru.types import BlockType, ImageBlock, ImageBodyBlock, MiddleJson


def _all_raw_blocks(model_pages: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """按页展开 raw model-list，方便断言 HTML 映射结果。"""
    return [block for page in model_pages for block in page]


def _image_body(middle: MiddleJson) -> ImageBodyBlock:
    """返回文档中首个严格图片 body。"""
    image = next(block for block in middle.pages[0].blocks if isinstance(block, ImageBlock))
    return next(child for child in image.content if isinstance(child, ImageBodyBlock))


def test_html_doc_analyze_projects_static_semantics_and_renderers() -> None:
    """验证 HTML 主链路保留正文结构、行内语义、公式、图片 URL 与固定元数据。"""
    payload = b"""<!doctype html>
    <html><head><title>Demo - Example</title><meta property="og:site_name" content="Example">
    <style>.hidden { display:none } .strong { font-weight:700 }</style></head>
    <body><nav>menu links</nav><main><article>
      <h1 id="top">Demo</h1>
      <p>Hello <span class="strong">world</span>, <code>a`b</code>,
         <a href="#top">back</a>.</p>
      <p class="hidden">secret</p><script>alert(1)</script>
      <ol start="3" reversed><li value="9">Three</li><li>Four</li></ol>
      <table><caption>Data</caption><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
      <pre><code class="language-python">print(1)</code></pre>
      <script type="math/tex; mode=display">x^2</script>
      <img src="https://cdn.example.com/a.png" alt="Remote image">
    </article></main></body></html>"""

    middle, model = doc_analyze(payload, effort="xhigh", parse_mode="ocr", file_suffix="html")
    async_middle, async_model = asyncio.run(aio_doc_analyze(payload, effort="medium", parse_mode="auto", file_suffix="html"))

    assert middle.model_dump() == async_middle.model_dump()
    assert model.pages == async_model.pages
    assert middle.file_suffix == model.file_suffix == "html"
    assert middle.extensions["mineru"]["effort"] == model.extensions["mineru"]["effort"] == "flash"
    assert middle.extensions["mineru"]["parse_mode"] == model.extensions["mineru"]["parse_mode"] == "txt"
    assert middle.is_full_document is True
    assert [page.page_idx for page in middle.pages] == [0]
    assert all(block.bbox is None for block in middle.pages[0].blocks)
    title = next(block for block in middle.pages[0].blocks if block.type == BlockType.DOC_TITLE)
    assert title.anchor == "html-39fc7010518f54fa3fa9"  # type: ignore[union-attr]

    raw_blocks = _all_raw_blocks(model.pages)
    raw_types = {block["type"] for block in raw_blocks}
    assert {
        BlockType.DOC_TITLE,
        BlockType.TEXT,
        BlockType.LIST,
        BlockType.TABLE,
        BlockType.CODE,
        BlockType.EQUATION,
        BlockType.IMAGE,
    } <= raw_types
    assert "secret" not in str(raw_blocks)
    assert "alert(1)" not in str(raw_blocks)
    assert any(block.get("guess_lang") == "python" for block in raw_blocks)
    assert _image_body(middle).image_url == "https://cdn.example.com/a.png"
    assert ParseResult.from_dict(ParseResult(middle_json=middle).to_dict()).middle_json == middle

    markdown = render_markdown(middle)
    assert "# Demo" in markdown
    assert "**world**" in markdown
    assert "``a`b``" in markdown
    assert "3. Three" in markdown and "4. Four" in markdown
    assert "```python" in markdown and "x^2" in markdown
    assert "https://cdn.example.com/a.png" in markdown
    assert "<table" in render_html(middle)
    assert render_structured_content(middle)["file_suffix"] == "html"
    docx = render_docx(middle)
    assert docx.startswith(b"PK")
    with ZipFile(BytesIO(docx)) as archive:
        relationships = archive.read("word/_rels/document.xml.rels").decode()
        assert "https://cdn.example.com/a.png" in relationships


def test_html_parse_server_url_preserves_http_declared_charset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 URL HTML 使用 HTTP Content-Type 声明编码而不依赖文档内 meta。"""
    expected = "日本語テスト"
    url = "https://example.com/sample.html"
    response = httpx.Response(
        200,
        content=f"<html><body><p>{expected}</p></body></html>".encode("shift_jis"),
        headers={"Content-Type": "text/html; charset=shift_jis"},
        request=httpx.Request("GET", url),
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = response
    monkeypatch.setattr(api_server, "httpx", SimpleNamespace(AsyncClient=lambda **_: client))
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "url", "url": url}}],
            "tier": "standard",
            "output_formats": ["middle_json"],
        }
    )
    record = api_server.JobStore().create(request, file_store)

    asyncio.run(
        api_server._run_job(
            record,
            request,
            file_store,
            image_analysis=True,
        )
    )

    parsed_file = record.files[0]
    assert parsed_file.status == "completed"
    assert parsed_file.output_files is not None and parsed_file.output_files.middle_json is not None
    middle_record = file_store.get_file(parsed_file.output_files.middle_json.file_id)
    assert middle_record.sha256sum is not None
    middle_payload = json.loads(file_store.read_blob(middle_record.sha256sum))
    assert inline_text(middle_payload["pages"][0]["blocks"][0]["content"]) == expected


@pytest.mark.parametrize(
    "url",
    ["https://example.com/article", "https://example.com/"],
    ids=["path-without-extension", "trailing-slash"],
)
def test_html_parse_server_url_accepts_extensionless_text_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    """验证无扩展名 URL 可按 HTTP text/html 响应进入 HTML Flash 路由。"""
    expected = "Extensionless HTML"
    response = httpx.Response(
        200,
        content=f"<html><body><p>{expected}</p></body></html>".encode(),
        headers={"Content-Type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", url),
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = response
    monkeypatch.setattr(api_server, "httpx", SimpleNamespace(AsyncClient=lambda **_: client))
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "url", "url": url}}],
            "tier": "standard",
            "output_formats": ["middle_json"],
        }
    )
    record = api_server.JobStore().create(request, file_store)

    asyncio.run(
        api_server._run_job(
            record,
            request,
            file_store,
            image_analysis=True,
        )
    )

    parsed_file = record.files[0]
    assert parsed_file.status == "completed"
    assert parsed_file.output_files is not None and parsed_file.output_files.middle_json is not None
    middle_record = file_store.get_file(parsed_file.output_files.middle_json.file_id)
    assert middle_record.sha256sum is not None
    middle_payload = json.loads(file_store.read_blob(middle_record.sha256sum))
    assert inline_text(middle_payload["pages"][0]["blocks"][0]["content"]) == expected


def test_html_parse_server_flash_only_admits_extensionless_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Flash-only server 在下载识别类型前不会拒绝无扩展名 URL。"""

    async def fake_run_job(*args: object, **kwargs: object) -> None:
        """跳过后台解析，仅验证 admission 已选择 Flash tier。"""

    monkeypatch.setattr(api_server, "_run_job", fake_run_job)
    app = api_server.create_app(upload_dir=str(tmp_path / "api"), tier="flash")

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/jobs",
            json={"files": [{"source": {"type": "url", "url": "https://example.com/article"}}]},
        )

    assert response.status_code == 202
    assert response.json()["tier"] == "flash"


def test_html_parse_server_no_flash_rejects_extensionless_text_html_after_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证无扩展名 URL 下载为 HTML 后仍执行禁用 Flash 的后置策略检查。"""
    url = "https://example.com/article"
    response = httpx.Response(
        200,
        content=b"<html><body><p>HTML</p></body></html>",
        headers={"Content-Type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", url),
    )
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.get.return_value = response
    parse_async_mock = AsyncMock()
    monkeypatch.setattr(api_server, "httpx", SimpleNamespace(AsyncClient=lambda **_: client))
    monkeypatch.setattr(api_server, "parse_async", parse_async_mock)
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "url", "url": url}}],
            "tier": "standard",
        }
    )
    record = api_server.JobStore().create(request, file_store)

    asyncio.run(
        api_server._run_job(
            record,
            request,
            file_store,
            image_analysis=True,
            flash_enabled=False,
        )
    )

    assert record.status == "failed"
    assert record.files[0].error is not None
    assert record.files[0].error.message == (
        "Flash parsing is disabled in this server, but this input requires the Flash backend"
    )
    parse_async_mock.assert_not_awaited()


def test_html_local_base_images_styles_and_escape_are_bounded(tmp_path: Path) -> None:
    """验证本地 base、CSS、栅格图可读取，但父目录逃逸图片只保留说明。"""
    assets = tmp_path / "assets"
    assets.mkdir()
    image_path = assets / "pixel.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(image_path)
    (assets / "styles.css").write_text(".gone { display:none }", encoding="utf-8")
    outside = tmp_path.parent / "outside-html-image.png"
    Image.new("RGB", (1, 1), "blue").save(outside)
    source = tmp_path / "sample.htm"
    source.write_text(
        """<html><head><base href="assets/"><link rel="stylesheet" href="styles.css"></head><body>
        <h1>Local</h1><p class="gone">hidden css</p><img src="pixel.png" alt="Pixel">
        <img src="../outside-html-image.png" alt="Outside"></body></html>""",
        encoding="utf-8",
    )

    result = parse(source)
    async_result = asyncio.run(parse_async(source))

    assert result.middle_json.model_dump() == async_result.middle_json.model_dump()
    assert result.middle_json.file_suffix == "html"
    assert _image_body(result.middle_json).image_base64.startswith("data:image/png;base64,")
    markdown = result.markdown()
    assert "hidden css" not in markdown
    assert "Outside" in markdown
    exported = export_middle_json(result.middle_json, tmp_path / "export")
    assert len(exported.image_paths) == 1
    assert exported.image_paths[0].read_bytes() == image_path.read_bytes()
    assert _image_body(exported.middle_json).image_base64 is None
    assert _image_body(exported.middle_json).image_path is not None


def test_html_parse_server_local_source_keeps_relative_assets(tmp_path: Path) -> None:
    """验证 parse-server 本地来源把原目录上下文传入 HTML 模型并输出严格结果。"""
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (2, 2), "green").save(image_path)
    source = tmp_path / "sample.html"
    source.write_text('<html><body><h1>API HTML</h1><img src="pixel.png" alt="Pixel"></body></html>', encoding="utf-8")
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "standard",
            "output_formats": ["markdown", "middle_json", "structured_content"],
        }
    )
    record = api_server.JobStore().create(request, file_store)

    asyncio.run(
        api_server._run_job(
            record,
            request,
            file_store,
            image_analysis=True,
            allow_local_source=True,
        )
    )

    parsed_file = record.files[0]
    assert parsed_file.status == "completed"
    assert parsed_file.output_files is not None and parsed_file.output_files.middle_json is not None
    middle_record = file_store.get_file(parsed_file.output_files.middle_json.file_id)
    assert middle_record.sha256sum is not None
    middle_payload = json.loads(file_store.read_blob(middle_record.sha256sum))
    assert middle_payload["file_suffix"] == "html"
    assert middle_payload["effort"] == "flash"
    image_body = middle_payload["pages"][0]["blocks"][1]["content"][0]
    assert image_body["image_base64"].startswith("data:image/png;base64,")


def test_html_doclib_local_bridge_uses_flash_parser(tmp_path: Path) -> None:
    """验证 Doclib 本地 Flash 桥接把 HTML 文件交给统一 MinerUParser。"""
    source = tmp_path / "doclib.html"
    source.write_text("<html><body><h1>Doclib HTML</h1><p>Body text.</p></body></html>", encoding="utf-8")
    service = object.__new__(ParseService)

    result = asyncio.run(
        service._parse_via_local(  # type: ignore[arg-type]
            {"path": str(source), "ext": "html"},
            "flash",
            "",
        )
    )

    assert result.middle_json.file_suffix == "html"
    assert result.middle_json.extensions["mineru"]["effort"] == "flash"
    assert "Doclib HTML" in result.markdown()


def test_html_rejects_page_range_and_resource_overflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 HTML 保持整本文档契约，并在输入预算超限时显式失败。"""
    source = tmp_path / "sample.html"
    source.write_text("<p>text</p>", encoding="utf-8")
    with pytest.raises(InvalidRequestError) as exc_info:
        parse(source, page_range="1")
    assert exc_info.value.code == "page_range_invalid"

    monkeypatch.setattr(html_converter_module, "MAX_HTML_BYTES", 4)
    with pytest.raises(HtmlResourceLimitError, match="max_html_bytes"):
        HtmlModel().predict(BytesIO(b"<p>x</p>"))
