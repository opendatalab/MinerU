"""从真实保存入口覆盖 API 素材往返及无需重裁的 PDF 导出。"""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import AsyncMock
import zipfile

from bs4 import BeautifulSoup
from docvortex.document.pdf import PDFDocument
from PIL import Image
import pytest
from reportlab.pdfgen.canvas import Canvas

from mineru.kit.common import save_parse_result
from mineru.kit.gradio import artifacts as gradio_artifacts
from mineru.parser import api_client, api_server
from mineru.parser.base import ParseResult
from mineru.parser.writer import DataWriter
from mineru.render import PdfLayout, render_pdf
from mineru.types import CodeBlock, MiddleJson


def _result(angle: int = 270) -> tuple[ParseResult, bytes, str]:
    """构造非连续源页及带非对称图片、HTML 表内图、代码字面量的真实结果。"""
    image = Image.new("RGB", (200, 100), "red")
    image.paste("blue", (100, 0, 200, 100))
    image.paste("green", (0, 0, 25, 25))
    output = BytesIO()
    image.save(output, "PNG")
    payload = output.getvalue()
    uri = "data:image/png;base64," + base64.b64encode(payload).decode()
    width, height = (100, 200) if angle in (90, 270) else (200, 100)
    bbox = [0.1, 0.1, 0.1 + width / 400, 0.1 + height / 600]
    markup = f"<table><tr><td><IMG alt='a &amp; b' width='20' SRC = '{uri}' /></td></tr></table>"
    middle = MiddleJson.from_dict(
        {
            "schema": "docvortex.middle",
            "schema_version": "2.0",
            "is_full_document": False,
            "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
            "extensions": {
                "mineru": {"tier": "basic", "parse_mode": "txt"},
                "docvortex_layout": {
                    "version": 1,
                    "pages": [
                        {"page_idx": 4, "width_pt": 400, "height_pt": 600, "image_rotations": {"2": angle}},
                        {"page_idx": 8, "width_pt": 400, "height_pt": 600},
                    ],
                },
            },
            "pages": [
                {
                    "page_idx": 4,
                    "blocks": [
                        {
                            "type": "table",
                            "index": 2,
                            "bbox": bbox,
                            "content": [
                                {"type": "table_body", "index": 2, "bbox": bbox, "image_base64": uri, "content": markup}
                            ],
                        },
                    ],
                },
                {"page_idx": 8, "blocks": []},
            ],
        }
    )
    return ParseResult(middle_json=middle), payload, markup


def _entries(payload: bytes) -> dict[str, bytes]:
    """读取完整 ZIP 文件集合以比较多次保存的实际内容。"""
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        return {name: archive.read(name) for name in archive.namelist()}


def test_save_externalizes_assets_without_changing_json_only_contract(tmp_path: Path) -> None:
    """API 与 CLI 使用相同保存入口，协议与所有消费格式均引用同一份素材。"""
    result, image, _ = _result()
    before = result.middle_json.to_json()
    entries = _entries(api_server._build_self_contained_zip_output(result))
    assert entries == _entries(api_server._build_self_contained_zip_output(result))
    assert set(entries) == {
        "middle_json.json",
        "markdown.md",
        "structured_content.json",
        "images/page_4_table_body_2.png",
        "images/page_4_table_body_2_1.png",
    }
    for name, payload in entries.items():
        if name.startswith("images/"):
            assert payload == image
    saved = json.loads(entries["middle_json.json"])
    assert saved["extensions"] == result.middle_json.extensions
    for name in ("middle_json.json", "markdown.md", "structured_content.json"):
        assert b"data:image/" not in entries[name]
        assert b"images/page_4_table_body_2_1.png" in entries[name]
    assert result.middle_json.to_json() == before
    assert "image_base64" not in result.to_json()
    dest = tmp_path / "result.zip"
    save_parse_result(result, dest, "zip")
    assert _entries(dest.read_bytes()) == entries


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("include_images", [False, True])
def test_server_zip_roundtrip_restores_direct_and_markup_images(
    monkeypatch: pytest.MonkeyPatch,
    asynchronous: bool,
    include_images: bool,
) -> None:
    """同步异步均消费真实服务端 ZIP，恢复 HTML 时保持属性、引号和字节。"""
    result, image, markup = _result()
    payload = api_server._build_self_contained_zip_output(result)
    monkeypatch.setattr(api_client, "_download_bytes", lambda *args: payload)
    monkeypatch.setattr(api_client, "_async_download_bytes", AsyncMock(return_value=payload))
    parser = api_client.MinerUApiParser(
        api_url="http://localhost:8000",
        tier="basic",
        include_images=include_images,
        include_model_output=True,
    )
    job = {"status": "completed", "files": [{"output_files": {"zip": {"file_id": "file-test"}}}]}
    restored = (
        asyncio.run(api_client._async_parse_result_from_job(job, "sample.pdf", parser))
        if asynchronous
        else api_client._parse_result_from_job(job, "sample.pdf", parser)
    )
    assert restored._model_output is None
    assert restored.middle_json.extensions == result.middle_json.extensions
    body = restored.pages[0].blocks[0].content[0]
    if include_images:
        assert body.image_path is None
        assert base64.b64decode(body.image_base64.split(",", 1)[1]) == image
        assert body.content == markup
        assert _entries(api_server._build_self_contained_zip_output(restored)) == _entries(payload)
    else:
        assert body.image_base64 is None and body.image_path == "images/page_4_table_body_2.png"
        assert "images/page_4_table_body_2_1.png" in body.content


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_gradio_pdf_uses_packaged_images_without_cropping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    angle: int,
) -> None:
    """API 往返及 Gradio 物化后，在源 PDF 不可用时仍保持原布局像素。"""
    result, image, _ = _result(angle)
    restored = api_client._parse_result_from_zip_bytes(
        api_server._build_self_contained_zip_output(result),
        include_images=True,
        include_model_output=False,
    )
    source = tmp_path / "source.pdf"
    buffer = BytesIO()
    canvas = Canvas(buffer, pagesize=(400, 600))
    for _ in range(9):
        canvas.showPage()
    canvas.save()
    source.write_bytes(buffer.getvalue())

    def reject_crop(*args: object, **kwargs: object) -> bytes:
        """完整素材不允许调用任何源 PDF 裁图兜底。"""
        raise AssertionError("Packaged images must not be recropped")

    monkeypatch.setattr(gradio_artifacts._ImageContext, "crop_for_block", reject_crop)
    artifacts = gradio_artifacts.persist_parse_result(
        restored,
        source,
        output_root=tmp_path / "output",
        page_range="5,9",
    )
    saved = ParseResult.from_json(artifacts.middle_json_path.read_text())
    body = saved.pages[0].blocks[0].content[0]
    assert (artifacts.root / body.image_path).read_bytes() == image
    inline_path = BeautifulSoup(body.content, "html.parser").img["src"]
    assert (artifacts.root / inline_path).read_bytes() == image
    source.unlink()
    artifacts.source_path.unlink()
    artifacts.origin_pdf_path.unlink()
    downloaded = Path(gradio_artifacts.render_download(artifacts.as_state(), "pdf"))
    with PDFDocument(render_pdf(result.middle_json, layout=PdfLayout.ORIGINAL)) as expected:
        with PDFDocument(downloaded.read_bytes()) as actual:
            assert actual.page_count == 2
            assert actual.get_page_image_infos(0)[0].bbox == pytest.approx(expected.get_page_image_infos(0)[0].bbox)
            for page_index in range(2):
                left = expected.render_page(page_index, scale=1).pil_image
                right = actual.render_page(page_index, scale=1).pil_image
                try:
                    assert left.size == right.size and left.tobytes() == right.tobytes()
                finally:
                    left.close()
                    right.close()


@pytest.mark.parametrize("location", ["direct", "html"])
def test_save_rejects_missing_assets_before_writing(location: str) -> None:
    """未提供素材字节的引用必须在任何写出之前失败。"""
    result, _, _ = _result()
    body = result.pages[0].blocks[0].content[0]
    if location == "direct":
        body.image_base64 = None
        body.image_path = "images/missing.png"
    else:
        body.content = '<table><tr><td><img src="images/missing.png"></td></tr></table>'

    class RejectWriter(DataWriter):
        """禁止缺失素材的结果写入任何文件。"""

        def write(self, path: str, data: bytes) -> None:
            """若验证未提前发生则立即暴露部分结果写出。"""
            raise AssertionError(f"Unexpected write: {path}")

    with pytest.raises(ValueError, match="Missing materialized"):
        result.save(RejectWriter())


@pytest.mark.parametrize("source", ["images/missing.png", "../escape.png"])
def test_client_rejects_missing_or_unsafe_markup_asset(source: str) -> None:
    """请求图片时，表内引用与直接引用遵守同样的完整性和路径检查。"""
    result, _, _ = _result()
    body = result.pages[0].blocks[0].content[0]
    body.content = f'<table><tr><td><img src="{source}"></td></tr></table>'
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("middle_json.json", result.to_json())
    with pytest.raises(api_client._V1APIError, match="image sidecar"):
        api_client._parse_result_from_zip_bytes(buffer.getvalue(), include_images=True, include_model_output=False)


def test_client_preserves_markup_attributes_and_code_literals() -> None:
    """带空格、实体、不同引号的 src 可恢复，代码中相同 HTML 字面量不变。"""
    data = b"image-bytes"
    path = "images/a b&c.png"
    source = "images/a%20b&amp;c.png"
    value = {
        "type": "table_body",
        "content": (
            f'<img alt="a > b" data-src="unchanged" src="{source}" width="2">'
            "<IMG src=images/second.png height=3>"
            '<img src="https://example.test/remote.png">'
        ),
    }
    code = {"type": "code_body", "content": value["content"]}
    old_code = code["content"]
    document = [value, code]
    assert api_client._collect_image_paths_from_middle_json(document) == {path, "images/second.png"}
    api_client._inline_image_sidecars(document, {path: data, "images/second.png": data})
    uri = api_client._encode_image_data_uri(data, path)
    assert value["content"] == old_code.replace(source, uri).replace("src=images/second.png", f"src={uri}")
    assert code["content"] == old_code


def test_save_and_restore_leave_code_image_literals_unchanged() -> None:
    """完整保存与恢复也不外置代码示例中的 data URI 或读取其虚构路径。"""
    result, _, markup = _result()
    literal = markup + '<img src="images/not-an-asset.png">'
    result.pages[1].blocks.append(
        CodeBlock.model_validate(
            {
                "type": "code",
                "sub_type": "code",
                "guess_lang": "html",
                "index": 3,
                "bbox": [0.1, 0.1, 0.9, 0.8],
                "content": [{"type": "code_body", "index": 3, "content": literal}],
            }
        )
    )
    packed = api_server._build_self_contained_zip_output(result)
    assert len([name for name in _entries(packed) if name.startswith("images/")]) == 2
    restored = api_client._parse_result_from_zip_bytes(packed, include_images=True, include_model_output=False)
    assert restored.pages[1].blocks[0].content[0].content == literal
    assert result.pages[1].blocks[0].content[0].content == literal
