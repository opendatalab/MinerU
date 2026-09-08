"""OFD 预览的真实转换、隔离和失败回执回归。"""

import asyncio
import io
import json
import zipfile
from pathlib import Path

import pytest
from lxml import html
from PIL import Image

from mineru.kit.gradio.ofd_preview import build_ofd_preview, prepare_ofd_preview


def ofd_bytes(text: str = "中文发票 &amp; &lt;script&gt;", *, entity: str = "") -> bytes:
    """构造包含两页、模板、路径及内嵌图片的有效 OFD 测试文档。"""
    ns = 'xmlns:ofd="http://www.ofdspec.org/2016"'
    image = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(image, format="PNG")
    files = {
        "OFD.xml": (
            f'<ofd:OFD {ns} Version="1.0" DocType="OFD">'
            "<ofd:DocBody><ofd:DocRoot>Doc/Document.xml</ofd:DocRoot></ofd:DocBody></ofd:OFD>"
        ),
        "Doc/Document.xml": f"""<ofd:Document {ns}><ofd:CommonData>
            <ofd:PageArea><ofd:PhysicalBox>0 0 210 297</ofd:PhysicalBox></ofd:PageArea>
            <ofd:DocumentRes>Res.xml</ofd:DocumentRes>
            <ofd:TemplatePage ID="9" BaseLoc="Template.xml" ZOrder="Background"/>
            </ofd:CommonData><ofd:Pages><ofd:Page ID="1" BaseLoc="Page.xml"/>
            <ofd:Page ID="2" BaseLoc="Page.xml"/></ofd:Pages></ofd:Document>""",
        "Doc/Res.xml": f"""<ofd:Res {ns}><ofd:MultiMedias><ofd:MultiMedia ID="4" Type="Image">
            <ofd:MediaFile>red.png</ofd:MediaFile></ofd:MultiMedia></ofd:MultiMedias></ofd:Res>""",
        "Doc/Template.xml": f"""<ofd:Page {ns}><ofd:Content><ofd:Layer ID="10">
            <ofd:PathObject ID="11" Boundary="10 10 190 277" Stroke="true" LineWidth="0.5">
            <ofd:AbbreviatedData>M 0 0 L 190 0 L 190 277 L 0 277 C</ofd:AbbreviatedData>
            </ofd:PathObject></ofd:Layer></ofd:Content></ofd:Page>""",
        "Doc/Page.xml": f"""{entity}<ofd:Page {ns}><ofd:Template TemplateID="9"/>
            <ofd:Content><ofd:Layer ID="1"><ofd:TextObject ID="2" Boundary="20 30 160 20" Size="5">
            <ofd:TextCode X="0" Y="8" DeltaX="g 40 5">{text}</ofd:TextCode></ofd:TextObject>
            <ofd:ImageObject ID="3" ResourceID="4" Boundary="20 60 30 30" CTM="30 0 0 30 0 0"/>
            </ofd:Layer></ofd:Content></ofd:Page>""",
        "Doc/red.png": image.getvalue(),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return output.getvalue()


def test_real_conversion_embeds_pages_and_escapes_text() -> None:
    """验证两页 SVG、模板路径、图片和文字转义，并禁止 iframe 脚本。"""
    frame = html.fromstring(build_ofd_preview(ofd_bytes()))
    assert frame.get("sandbox") == ""
    document = html.fromstring(frame.get("srcdoc"))
    assert len(document.xpath("//svg")) == 2
    assert len(document.xpath("//svg/path")) == 2
    assert "中文发票 & <script>" in document.text_content()
    assert not document.xpath("//script")
    assert "data:image/png;base64," in frame.get("srcdoc")
    assert "max-width:100%" in frame.get("srcdoc")


@pytest.mark.parametrize("payload", [b"", b"garbage", b"PK\x03\x04"])
def test_invalid_payload(payload: bytes) -> None:
    """空文件和损坏压缩包必须明确失败。"""
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        build_ofd_preview(payload)


def test_missing_ofd_xml() -> None:
    """合法 ZIP 缺少 OFD 入口时不得显示空白成功预览。"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("unrelated.xml", "<x/>")
    with pytest.raises(ValueError, match="missing OFD.xml"):
        build_ofd_preview(output.getvalue())


def test_external_entity_is_not_read(tmp_path: Path) -> None:
    """外部实体不得读取本地文件或进入最终 HTML。"""
    secret = tmp_path / "secret.txt"
    secret.write_text("OFD_EXTERNAL_ENTITY_SECRET")
    entity = f'<!DOCTYPE ofd:Page [<!ENTITY leak SYSTEM "{secret.as_uri()}">]>'
    frame = build_ofd_preview(ofd_bytes("&leak;", entity=entity))
    assert "OFD_EXTERNAL_ENTITY_SECRET" not in frame


def test_preview_receipts_and_failure(tmp_path: Path) -> None:
    """成功和失败均回传请求标识，预览错误不向解析流程抛出。"""
    source = tmp_path / "中文.OFD"
    source.write_bytes(ofd_bytes())
    ticket = json.dumps({"id": "request-1", "path": str(source)})
    receipt = json.loads(asyncio.run(prepare_ofd_preview(str(source), ticket)))
    assert receipt["id"] == "request-1" and "iframe" in receipt["html"]
    source.write_bytes(b"bad")
    receipt = json.loads(asyncio.run(prepare_ofd_preview(str(source), ticket)))
    assert "ofd_preview_failed" in receipt["html"]
    assert json.loads(asyncio.run(prepare_ofd_preview(None, ticket)))["html"] == ""
    assert json.loads(asyncio.run(prepare_ofd_preview(str(source) + "x", ticket)))["html"] == ""


@pytest.mark.parametrize("fail", [False, True])
def test_parse_preserves_source_preview(tmp_path: Path, fail: bool) -> None:
    """解析成功或失败只更新结果，OFD 源预览始终由独立事件维护。"""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from test_kit_gradio import _middle_json

    from mineru.kit.gradio.app import build_gradio_app
    from mineru.kit.gradio.client import V1ServerCapabilities
    from mineru.parser.base import ParseResult

    source = tmp_path / "source.ofd"
    source.write_bytes(ofd_bytes())
    parse = AsyncMock(return_value=ParseResult(middle_json=_middle_json(with_image=False)))
    if fail:
        parse.side_effect = ValueError("expected parse failure")
    demo = build_gradio_app(
        SimpleNamespace(parse_file=parse),
        V1ServerCapabilities("http://127.0.0.1:1", ("flash",), ("zip",), ("file_id",)),
        output_root=tmp_path / "results",
        enable_example=False,
    )
    convert = next(fn for fn in demo.fns.values() if fn.name == "convert_handler")
    ofd = next(block for block in demo.blocks.values() if "mineru-kit-ofd-preview" in (block.elem_classes or []))
    assert ofd not in convert.outputs
    upload = next(fn for fn in demo.fns.values() if fn.name == "update_file_preview")
    updates = upload.fn(str(source))
    assert all(update.get("visible") is False for update in updates[:4])

    async def collect() -> list:
        """收集真实解析事件的完整输出序列。"""
        return [value async for value in convert.fn(str(source), 0, "")]

    final = asyncio.run(collect())[-1]
    assert final[2:6] == ({"__type__": "update"},) * 4
    if fail:
        assert "expected parse failure" in final[0]
    else:
        assert final[6] is not None
        assert final[-1]


def test_browser_request_lifecycle() -> None:
    """通过真实前端脚本验证乱序完成、清除和同文件重传不会恢复旧预览。"""
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for preview lifecycle checks")
    subprocess.run([node, str(Path(__file__).with_suffix(".cjs"))], check=True, timeout=15)
