from __future__ import annotations

import asyncio
import hashlib
import json
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree
from PIL import Image
import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.core.file_io import extract_metadata
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.model.flash import OfdModel
from mineru.model.flash.ofd import OfdParseError, OfdResourceLimitError, detect_ofd
from mineru.model.flash.ofd import images as ofd_images
from mineru.model.flash.ofd import metadata as ofd_metadata
from mineru.model.flash.ofd import scene as ofd_scene
from mineru.model.flash.ofd.constants import (
    MAX_DOCUMENT_COUNT,
    MAX_DRAW_PARAM_INHERITANCE,
    MAX_EXPANDED_GLYPHS,
    MAX_GLYPH_TOKENS,
    MAX_PAGE_COUNT,
)
from mineru.model.flash.ofd.geometry import Affine, canonical_angle
from mineru.model.flash.ofd.images import build_image_item
from mineru.model.flash.ofd.models import MediaResource, OfdPageScene, ResourceRegistry
from mineru.model.flash.ofd.package import OfdPackage
from mineru.model.flash.ofd.path import OfdPathBudget, _segments, build_axis_lines
from mineru.model.flash.ofd.reading_order import OfdReadingOrderProjector
from mineru.model.flash.ofd.resources import resolve_draw_param
from mineru.model.flash.ofd.text import FontMetricResolver, OfdTextBudget, build_text_lines
from mineru.parser import MinerUParser
from mineru.parser import api_server
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.parser.file_type import guess_suffix_by_bytes, guess_suffix_by_path
from mineru.render import render_docx, render_html, render_markdown, render_structured_content
from mineru.types import BlockType

from _ofd_test_utils import build_multi_document_ofd, build_ofd_package, page_xml, path_object, text_object
from _span_test_utils import inline_text

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_SAMPLE_DIR = _PROJECT_ROOT / "tmp" / "ofd_samples" / "ofdrw_issues_20260828" / "ofd"


def _minimal_payload(*, namespace: str = "http://www.ofdspec.org/2016", version: str = "1.0") -> bytes:
    """构造包含单行文字的最小 OFD。"""
    content = text_object(3, "你好，OFD！", boundary="10 10 50 12", delta_x="g 6 5")
    return build_ofd_package(
        [("Pages/Page_0/Content.xml", page_xml(content, namespace=namespace))],
        namespace=namespace,
        version=version,
    )


def test_ofd_model_analyze_detection_and_renderers(tmp_path: Path) -> None:
    """验证 OFD 从内容识别到统一四类 renderer 的完整链路。"""
    payload = _minimal_payload()
    source = tmp_path / "disguised.csv"
    source.write_bytes(payload)

    assert detect_ofd(payload)
    assert guess_suffix_by_bytes(payload, str(source)) == "ofd"
    assert guess_suffix_by_path(source) == "ofd"
    model_pages = OfdModel().predict(BytesIO(payload))
    middle, model = doc_analyze(payload, effort="xhigh", parse_mode="ocr", file_suffix="ofd")
    async_middle, async_model = asyncio.run(aio_doc_analyze(payload, effort="medium", file_suffix="ofd"))

    assert model.pages == async_model.pages == model_pages
    assert middle.model_dump() == async_middle.model_dump()
    assert model.file_suffix == middle.file_suffix == "ofd"
    assert model.effort == middle.effort == "flash"
    assert model.parse_mode == middle.parse_mode == "txt"
    assert middle.is_full_document is True
    assert middle.pages[0].blocks[0].type == BlockType.TEXT
    assert middle.pages[0].blocks[0].bbox is not None
    assert "你好，OFD" in render_markdown(middle)
    assert "你好，OFD" in render_html(middle)
    assert render_structured_content(middle)["file_suffix"] == "ofd"
    assert render_docx(middle).startswith(b"PK")


def test_ofd_legacy_namespace_and_declared_page_order() -> None:
    """验证旧命名空间、非目录排序页树和空页保持声明顺序。"""
    namespace = "http://www.ofdspec.org"
    pages = [
        (
            "Pages/Page_0/Content.xml",
            page_xml(text_object(1, "first", boundary="10 10 30 10"), namespace=namespace),
        ),
        (
            "Pages/Page_2/Content.xml",
            page_xml(text_object(2, "second", boundary="10 10 30 10"), namespace=namespace),
        ),
        ("Pages/Page_1/Content.xml", page_xml("", namespace=namespace)),
    ]
    middle, _model = doc_analyze(
        build_ofd_package(pages, namespace=namespace, version="1.2"),
        file_suffix="ofd",
    )

    assert [page.page_idx for page in middle.pages] == [0, 1, 2]
    assert [inline_text(page.blocks[0].content) if page.blocks else "" for page in middle.pages] == ["first", "second", ""]


def test_ofd_multiple_doc_bodies_flatten_in_declared_order() -> None:
    """验证多个 DocBody 按声明顺序展开为连续物理页。"""
    middle, model = doc_analyze(build_multi_document_ofd(), file_suffix="ofd")

    assert [page.page_idx for page in middle.pages] == [0, 1]
    assert [inline_text(page[0]["content"]) for page in model.pages] == ["doc-zero", "doc-one"]


def test_ofd_declared_page_count_is_bounded_for_parser_and_metadata() -> None:
    """验证重复引用同一页面 part 也受全文页数预算限制。"""
    payload = build_ofd_package([("Pages/Page_0/Content.xml", page_xml(""))])
    source_buffer = BytesIO(payload)
    output_buffer = BytesIO()
    with ZipFile(source_buffer) as source, ZipFile(output_buffer, "w", ZIP_DEFLATED) as output:
        document_root = etree.fromstring(source.read("Doc_0/Document.xml"))
        pages = next(element for element in document_root.iter() if etree.QName(element).localname == "Pages")
        pages.clear()
        namespace = etree.QName(document_root).namespace
        for page_index in range(MAX_PAGE_COUNT + 1):
            etree.SubElement(
                pages,
                f"{{{namespace}}}Page",
                ID=str(page_index + 1),
                BaseLoc="Pages/Page_0/Content.xml",
            )
        replacement = etree.tostring(document_root, xml_declaration=True, encoding="UTF-8")
        for info in source.infolist():
            output.writestr(info, replacement if info.filename == "Doc_0/Document.xml" else source.read(info.filename))
    oversized = output_buffer.getvalue()

    with pytest.raises(OfdResourceLimitError, match="max_page_count"):
        OfdModel().predict(BytesIO(oversized))
    with pytest.raises(OfdResourceLimitError, match="max_page_count"):
        ofd_metadata.extract_ofd_metadata(BytesIO(oversized))


def test_ofd_page_count_budget_is_shared_across_doc_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证多个 DocBody 共用解析和元数据页数预算。"""
    monkeypatch.setattr(ofd_scene, "MAX_PAGE_COUNT", 1)
    monkeypatch.setattr(ofd_metadata, "MAX_PAGE_COUNT", 1)
    payload = build_multi_document_ofd()

    with pytest.raises(OfdResourceLimitError, match="max_page_count"):
        OfdModel().predict(BytesIO(payload))
    with pytest.raises(OfdResourceLimitError, match="max_page_count"):
        ofd_metadata.extract_ofd_metadata(BytesIO(payload))


def test_ofd_declared_document_count_is_bounded_before_page_parsing() -> None:
    """验证大量空 DocBody 在物化文档引用前受独立预算限制。"""
    payload = build_ofd_package([("Pages/Page_0/Content.xml", page_xml(""))])
    source_buffer = BytesIO(payload)
    output_buffer = BytesIO()
    with ZipFile(source_buffer) as source, ZipFile(output_buffer, "w", ZIP_DEFLATED) as output:
        ofd_root = etree.fromstring(source.read("OFD.xml"))
        for child in list(ofd_root):
            ofd_root.remove(child)
        namespace = etree.QName(ofd_root).namespace
        for _document_index in range(MAX_DOCUMENT_COUNT + 1):
            doc_body = etree.SubElement(ofd_root, f"{{{namespace}}}DocBody")
            doc_root = etree.SubElement(doc_body, f"{{{namespace}}}DocRoot")
            doc_root.text = "Doc_0/Document.xml"
        document_root = etree.fromstring(source.read("Doc_0/Document.xml"))
        pages = next(element for element in document_root.iter() if etree.QName(element).localname == "Pages")
        pages.clear()
        replacements = {
            "OFD.xml": etree.tostring(ofd_root, xml_declaration=True, encoding="UTF-8"),
            "Doc_0/Document.xml": etree.tostring(document_root, xml_declaration=True, encoding="UTF-8"),
        }
        for info in source.infolist():
            content = replacements[info.filename] if info.filename in replacements else source.read(info.filename)
            output.writestr(info, content)
    oversized = output_buffer.getvalue()

    with pytest.raises(OfdResourceLimitError, match="max_document_count"):
        OfdModel().predict(BytesIO(oversized))
    with pytest.raises(OfdResourceLimitError, match="max_document_count"):
        ofd_metadata.extract_ofd_metadata(BytesIO(oversized))


def test_ofd_textcode_geometry_does_not_use_oversized_boundary() -> None:
    """验证 Foxit 风格超大 Boundary 不会成为最终文字 bbox。"""
    content = text_object(
        81,
        "661016910189",
        boundary="-8.8175 -170.35411 361.51749 399.96179",
        size=9,
        x=83.00001,
        y=168.00009,
        delta_x="g 11 4.158",
        ctm="0.3527 0 0 0.3527 0 139.66919",
    )
    middle, _model = doc_analyze(
        build_ofd_package([("Pages/Page_0/Content.xml", page_xml(content, physical_box="0 0 210 140"))]),
        file_suffix="ofd",
    )

    bbox = middle.pages[0].blocks[0].bbox
    assert bbox is not None
    assert bbox[2] - bbox[0] < 0.5
    assert bbox[3] - bbox[1] < 0.2


def test_ofd_cardinal_text_directions_keep_geometry_and_angle() -> None:
    """验证 ReadDirection/CharDirection 参与字形 quad 与排序方向。"""
    content = (
        '<ofd:TextObject ID="8" Boundary="20 20 20 50" Font="1" Size="5" '
        'ReadDirection="90" CharDirection="90">'
        '<ofd:TextCode X="2" Y="-5" DeltaX="g 3 6">竖排文</ofd:TextCode>'
        "</ofd:TextObject>"
    )
    _middle, model = doc_analyze(
        build_ofd_package([("Pages/Page_0/Content.xml", page_xml(content))]),
        file_suffix="ofd",
    )

    assert model.pages[0][0]["angle"] == 90
    assert model.pages[0][0]["bbox"][2] > model.pages[0][0]["bbox"][0]
    assert model.pages[0][0]["bbox"][3] > model.pages[0][0]["bbox"][1]


@pytest.mark.parametrize(("second_x", "expected"), [(16.0, "Hello"), (17.0, "Hel lo")])
def test_ofd_same_baseline_ascii_fragments_use_measured_gap(second_x: float, expected: str) -> None:
    """验证相邻英文 run 仅在存在可见词间距时补空格。"""
    content = "".join(
        [
            text_object(1, "Hel", boundary="10 10 10 10", size=4, y=5, delta_x="2 2"),
            text_object(2, "lo", boundary=f"{second_x} 10 10 10", size=4, y=5, delta_x="2"),
        ]
    )

    _middle, model = doc_analyze(
        build_ofd_package([("Pages/Page_0/Content.xml", page_xml(content))]),
        file_suffix="ofd",
    )

    assert len(model.pages[0]) == 1
    assert inline_text(model.pages[0][0]["content"]) == expected


def test_ofd_textcode_preserves_boundary_whitespace_and_glyph_positions() -> None:
    """验证 TextCode 首尾空格参与 Delta 展开和 CGTransform 全局位置映射。"""
    text_element = etree.fromstring(
        b'<TextObject ID="9" Boundary="10 10 50 10" Font="1" Size="5">'
        b'<TextCode X="1" Y="5" DeltaX="5 7"> A </TextCode>'
        b'<CGTransform CodePosition="1" CodeCount="1"><Glyphs>42</Glyphs></CGTransform>'
        b"</TextObject>"
    )
    package_buffer = BytesIO()
    with ZipFile(package_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("OFD.xml", "<OFD/>")
    package = OfdPackage(package_buffer.getvalue())

    lines = build_text_lines(
        text_element,
        parent_transform=Affine(),
        parent_clip=(0.0, 0.0, 100.0, 100.0),
        resources=ResourceRegistry(),
        package=package,
        font_metrics=FontMetricResolver(package),
        budget=OfdTextBudget(),
        paint_order=0,
        layer_type="body",
        template_id=None,
    )

    assert len(lines) == 1
    assert lines[0].text == " A "
    assert [glyph.glyph_id for glyph in lines[0].glyphs] == [None, 42, None]
    assert [glyph.origin[0] for glyph in lines[0].glyphs] == pytest.approx([11.0, 16.0, 23.0])


def test_ofd_cgtransform_expands_only_actual_text_positions() -> None:
    """验证超大 CodeCount 只映射 TextCode 实际存在的字符位置。"""
    glyphs = "42 " + "999 " * 100_000
    text_element = etree.fromstring(
        (
            '<TextObject ID="10" Boundary="0 0 10 10" Font="1" Size="5">'
            '<TextCode X="1" Y="5">A</TextCode>'
            f'<CGTransform CodePosition="0" CodeCount="1000000000"><Glyphs>{glyphs}</Glyphs></CGTransform>'
            "</TextObject>"
        ).encode()
    )
    package_buffer = BytesIO()
    with ZipFile(package_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("OFD.xml", "<OFD/>")
    package = OfdPackage(package_buffer.getvalue())
    budget = OfdTextBudget()

    lines = build_text_lines(
        text_element,
        parent_transform=Affine(),
        parent_clip=(0.0, 0.0, 100.0, 100.0),
        resources=ResourceRegistry(),
        package=package,
        font_metrics=FontMetricResolver(package),
        budget=budget,
        paint_order=0,
        layer_type="body",
        template_id=None,
    )

    assert budget.glyph_count == 1
    assert budget.glyph_mapping_count == 1
    assert budget.glyph_token_count == 1
    assert len(lines) == 1
    assert [glyph.glyph_id for glyph in lines[0].glyphs] == [42]

    exhausted_budget = OfdTextBudget(glyph_mapping_count=MAX_EXPANDED_GLYPHS)
    with pytest.raises(OfdResourceLimitError, match="max_expanded_glyphs"):
        build_text_lines(
            text_element,
            parent_transform=Affine(),
            parent_clip=(0.0, 0.0, 100.0, 100.0),
            resources=ResourceRegistry(),
            package=package,
            font_metrics=FontMetricResolver(package),
            budget=exhausted_budget,
            paint_order=0,
            layer_type="body",
            template_id=None,
        )

    token_exhausted_budget = OfdTextBudget(glyph_token_count=MAX_GLYPH_TOKENS)
    with pytest.raises(OfdResourceLimitError, match="max_glyph_tokens"):
        build_text_lines(
            text_element,
            parent_transform=Affine(),
            parent_clip=(0.0, 0.0, 100.0, 100.0),
            resources=ResourceRegistry(),
            package=package,
            font_metrics=FontMetricResolver(package),
            budget=token_exhausted_budget,
            paint_order=0,
            layer_type="body",
            template_id=None,
        )


def test_ofd_textcode_discards_glyphs_outside_object_boundary() -> None:
    """验证 TextObject Boundary 会裁掉完整位于边界外的字符和空行。"""
    package_buffer = BytesIO()
    with ZipFile(package_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("OFD.xml", "<OFD/>")
    package = OfdPackage(package_buffer.getvalue())
    font_metrics = FontMetricResolver(package)
    common = {
        "parent_transform": Affine(),
        "parent_clip": (0.0, 0.0, 100.0, 100.0),
        "resources": ResourceRegistry(),
        "package": package,
        "font_metrics": font_metrics,
        "budget": OfdTextBudget(),
        "paint_order": 0,
        "layer_type": "body",
        "template_id": None,
    }

    partial = etree.fromstring(
        b'<TextObject ID="10" Boundary="0 0 10 10" Size="5"><TextCode X="2" Y="5" DeltaX="20">AB</TextCode></TextObject>'
    )
    outside = etree.fromstring(
        b'<TextObject ID="11" Boundary="0 0 10 10" Size="5"><TextCode X="20" Y="5">X</TextCode></TextObject>'
    )

    partial_lines = build_text_lines(partial, **common)
    outside_lines = build_text_lines(outside, **common)

    assert len(partial_lines) == 1
    assert partial_lines[0].text == "A"
    assert [glyph.text for glyph in partial_lines[0].glyphs] == ["A"]
    assert outside_lines == []


def test_ofd_draw_param_inheritance_is_ordered_and_bounded() -> None:
    """验证 DrawParam 保持父到子覆盖顺序，并在继承链超限时受控失败。"""
    ordered = ResourceRegistry(
        draw_params={
            1: {"ID": "1", "Relative": "2", "LineWidth": "2"},
            2: {"ID": "2", "LineWidth": "1", "Color": "red"},
        }
    )
    assert resolve_draw_param(ordered, 1) == {"LineWidth": "2", "Color": "red"}

    chain_length = MAX_DRAW_PARAM_INHERITANCE + 1
    oversized = ResourceRegistry(
        draw_params={
            resource_id: (
                {"ID": str(resource_id), "Relative": str(resource_id + 1)}
                if resource_id + 1 < chain_length
                else {"ID": str(resource_id), "LineWidth": "1"}
            )
            for resource_id in range(chain_length)
        }
    )
    with pytest.raises(OfdResourceLimitError, match="max_draw_param_inheritance"):
        resolve_draw_param(oversized, 0)


@pytest.mark.parametrize(
    ("raw_angle", "expected_angle"),
    [
        (89.9, 90),
        (90.1, 90),
        (179.9, 180),
        (180.1, 180),
        (359.9, 0),
        (0.1, 0),
    ],
)
def test_ofd_image_near_cardinal_rotation_accepts_both_directions(raw_angle: float, expected_angle: int) -> None:
    """验证图片旋转从直角两侧逼近时都保留载荷和阅读顺序 block。"""
    image_buffer = BytesIO()
    Image.new("RGB", (2, 2), "white").save(image_buffer, format="PNG")
    package_buffer = BytesIO()
    with ZipFile(package_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("OFD.xml", "<OFD/>")
        archive.writestr("Res/image.png", image_buffer.getvalue())
    package = OfdPackage(package_buffer.getvalue())
    transform = Affine.rotation(raw_angle)
    ctm = " ".join(str(value) for value in (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f))
    image_element = etree.fromstring(f'<ImageObject ID="1" Boundary="0 0 10 10" ResourceID="1" CTM="{ctm}"/>'.encode())
    resources = ResourceRegistry(
        media={
            1: MediaResource(
                resource_id=1,
                media_type="Image",
                media_format="PNG",
                media_part="Res/image.png",
            )
        }
    )

    item = build_image_item(
        image_element,
        parent_transform=Affine(),
        parent_clip=(0.0, 0.0, 100.0, 100.0),
        resources=resources,
        package=package,
        paint_order=0,
        layer_type="body",
        template_id=None,
    )

    assert item is not None
    assert item.diagnostic is None
    assert item.image_base64
    scene = OfdPageScene(
        page_idx=0,
        physical_box=(0.0, 0.0, 100.0, 100.0),
        content_box=None,
        images=[item],
    )
    blocks = OfdReadingOrderProjector([scene]).project_page(scene)
    assert len(blocks) == 1
    assert blocks[0]["type"] == BlockType.IMAGE
    assert blocks[0]["image_base64"] == item.image_base64
    assert canonical_angle(raw_angle) == expected_angle


def test_ofd_oversized_image_is_rejected_before_pixel_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 OFD 超限 raster 在 Pillow load 前降级为无载荷 diagnostic。"""
    image = MagicMock()
    image.__enter__.return_value = image
    image.size = (8_193, 1)
    image.load.side_effect = AssertionError("oversized image must not be decoded")
    monkeypatch.setattr(ofd_images.Image, "open", lambda _source: image)
    package_buffer = BytesIO()
    with ZipFile(package_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("OFD.xml", "<OFD/>")
        archive.writestr("Res/image.png", b"oversized")
    package = OfdPackage(package_buffer.getvalue())
    image_element = etree.fromstring(b'<ImageObject ID="1" Boundary="0 0 10 10" ResourceID="1"/>')

    item = build_image_item(
        image_element,
        parent_transform=Affine(),
        parent_clip=(0.0, 0.0, 100.0, 100.0),
        resources=ResourceRegistry(
            media={1: MediaResource(1, "Image", "PNG", "Res/image.png")},
        ),
        package=package,
        paint_order=0,
        layer_type="body",
        template_id=None,
    )

    assert item is not None
    assert item.image_base64 is None
    assert item.diagnostic == "unsupported_image_payload"
    image.load.assert_not_called()


def test_ofd_path_rejects_unexpected_numeric_tokens_without_hanging() -> None:
    """验证无活动命令或 C 后的数字 token 直接使当前非法路径降级。"""
    assert _segments("1 2", Affine(), OfdPathBudget()) == []
    assert _segments("M 0 0 C 1 2 L 3 4", Affine(), OfdPathBudget()) == []
    assert _segments("M 0 0 L 10 0", Affine(), OfdPathBudget()) == [((0.0, 0.0), (10.0, 0.0))]


def test_ofd_path_segments_are_clipped_to_object_boundary() -> None:
    """验证完全越界路径被丢弃，穿越路径裁到对象与父级裁剪交集。"""
    outside = etree.fromstring(
        b'<PathObject ID="1" Boundary="10 10 10 1" LineWidth="0.2">'
        b"<AbbreviatedData>M 20 0.5 L 30 0.5</AbbreviatedData></PathObject>"
    )
    outside_lines = build_axis_lines(
        outside,
        parent_transform=Affine(),
        parent_clip=(0.0, 0.0, 100.0, 100.0),
        paint_order=0,
        template_id=None,
        budget=OfdPathBudget(),
    )
    assert outside_lines == []

    cases = [
        ("M -5 5 L 15 5", "horizontal", (12.0, 14.9, 18.0, 15.1)),
        ("M 5 -5 L 5 15", "vertical", (14.9, 12.0, 15.1, 18.0)),
    ]
    for data, orientation, expected_bbox in cases:
        crossing = etree.fromstring(
            (
                '<PathObject ID="2" Boundary="10 10 10 10" LineWidth="0.2">'
                f"<AbbreviatedData>{data}</AbbreviatedData></PathObject>"
            ).encode()
        )
        lines = build_axis_lines(
            crossing,
            parent_transform=Affine(),
            parent_clip=(12.0, 12.0, 18.0, 18.0),
            paint_order=0,
            template_id=None,
            budget=OfdPathBudget(),
        )

        assert len(lines) == 1
        assert lines[0].orientation == orientation
        assert lines[0].bbox == pytest.approx(expected_bbox)


def test_ofd_template_grid_recovers_table() -> None:
    """验证模板路径和页面文字共同恢复高置信全线表。"""
    paths = "".join(
        [
            path_object(10, boundary="10 20 80 0.2", data="M 0 0.1 L 80 0.1"),
            path_object(11, boundary="10 50 80 0.2", data="M 0 0.1 L 80 0.1"),
            path_object(12, boundary="10 80 80 0.2", data="M 0 0.1 L 80 0.1"),
            path_object(13, boundary="10 20 0.2 60", data="M 0.1 0 L 0.1 60"),
            path_object(14, boundary="50 20 0.2 60", data="M 0.1 0 L 0.1 60"),
            path_object(15, boundary="90 20 0.2 60", data="M 0.1 0 L 0.1 60"),
        ]
    )
    template = page_xml(paths)
    body = "".join(
        [
            text_object(21, "A", boundary="20 30 10 8", size=4, y=4),
            text_object(22, "B", boundary="60 30 10 8", size=4, y=4),
            text_object(23, "C", boundary="20 60 10 8", size=4, y=4),
            text_object(24, "D", boundary="60 60 10 8", size=4, y=4),
        ]
    )
    payload = build_ofd_package(
        [("Pages/Page_0/Content.xml", page_xml(body, template_id=5))],
        templates={5: ("Tpls/Tpl_0/Content.xml", template)},
    )
    middle, model = doc_analyze(payload, file_suffix="ofd")

    assert [block["type"] for block in model.pages[0]] == [BlockType.TABLE]
    assert all(value in model.pages[0][0]["content"] for value in ("A", "B", "C", "D"))
    assert middle.pages[0].blocks[0].type == BlockType.TABLE


def test_ofd_page_resource_overrides_document_resource() -> None:
    """验证异常重复资源 ID 按 PageRes 高于 PublicRes 的规则解析。"""
    page_resource = (
        '<ofd:Res xmlns:ofd="http://www.ofdspec.org/2016"><ofd:Fonts>'
        '<ofd:Font ID="1" FontName="Page Bold" Bold="true"/></ofd:Fonts></ofd:Res>'
    )
    payload = build_ofd_package(
        [
            (
                "Pages/Page_0/Content.xml",
                page_xml(text_object(1, "page-style", boundary="10 10 40 10"), page_res="PageRes.xml"),
            )
        ],
        extra_parts={"Doc_0/Pages/Page_0/PageRes.xml": page_resource},
    )
    _middle, model = doc_analyze(payload, file_suffix="ofd")

    assert model.pages[0][0]["content"] == [{"type": "text", "content": "page-style", "styles": ["bold"]}]


def test_ofd_does_not_open_malformed_unreferenced_custom_tag() -> None:
    """验证损坏但未参与正文的扩展 XML 不会中断解析。"""
    payload = build_ofd_package(
        [("Pages/Page_0/Content.xml", page_xml(text_object(1, "visible", boundary="10 10 30 10")))],
        extra_parts={"Doc_0/Tags/CustomTag.xml": "<broken"},
    )

    middle, _model = doc_analyze(payload, file_suffix="ofd")
    assert inline_text(middle.pages[0].blocks[0].content) == "visible"


def test_ofd_rejects_foreign_namespace_and_non_v1_version() -> None:
    """验证未知命名空间和非 1.x 版本不会被宽松接受。"""
    assert not detect_ofd(_minimal_payload(namespace="https://example.com/ofd"))
    assert not detect_ofd(_minimal_payload(version="2.0"))
    with pytest.raises(OfdParseError):
        OfdModel().predict(BytesIO(_minimal_payload(namespace="https://example.com/ofd")))


def test_ofd_package_rejects_unsafe_member_and_dtd() -> None:
    """验证 ZIP 上跳成员与任意 DTD 在正文解析前被拒绝。"""
    unsafe = BytesIO()
    with ZipFile(unsafe, "w", ZIP_DEFLATED) as package:
        package.writestr("OFD.xml", '<ofd:OFD xmlns:ofd="http://www.ofdspec.org/2016" Version="1.0"/>')
        package.writestr("../escape.xml", "unsafe")
    with pytest.raises(OfdParseError, match="unsafe member"):
        OfdPackage(unsafe.getvalue())

    dtd = BytesIO()
    with ZipFile(dtd, "w", ZIP_DEFLATED) as package:
        package.writestr(
            "OFD.xml",
            '<!DOCTYPE OFD [<!ENTITY x "boom">]><ofd:OFD xmlns:ofd="http://www.ofdspec.org/2016" '
            'Version="1.0" DocType="OFD"><ofd:DocBody><ofd:DocRoot>&x;</ofd:DocRoot></ofd:DocBody></ofd:OFD>',
        )
    with OfdPackage(dtd.getvalue()) as package:
        with pytest.raises(OfdParseError, match="DTD"):
            package.root()


def test_ofd_parser_rejects_page_range_and_doclib_reads_metadata(tmp_path: Path) -> None:
    """验证 OFD 只支持整本解析且 Doclib 读取原生页数和标题。"""
    payload = _minimal_payload()
    source = tmp_path / "sample.ofd"
    source.write_bytes(payload)

    with pytest.raises(InvalidRequestError) as exc_info:
        MinerUParser(tier="flash").parse(source, page_range="1")
    assert exc_info.value.code == "page_range_invalid"

    metadata = asyncio.run(extract_metadata(str(source)))
    assert metadata["page_count"] == 1
    assert metadata["title"] == "Fixture Title"
    assert metadata["author"] == "Fixture Author"


def test_ofd_parse_server_job_emits_flash_outputs(tmp_path: Path) -> None:
    """验证本地 Parse Jobs 接受 OFD 并输出严格 Middle JSON。"""
    source = tmp_path / "sample.ofd"
    source.write_bytes(_minimal_payload())
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "standard",
            "output_formats": ["markdown", "middle_json", "structured_content", "zip"],
        }
    )
    record = api_server.JobStore().create(request, file_store)
    asyncio.run(
        api_server._run_job(
            record,
            request,
            file_store,
            ocr_mode="auto",
            image_analysis=True,
            allow_local_source=True,
        )
    )
    parsed_file = record.files[0]
    assert parsed_file.status == "completed"
    assert parsed_file.output_files is not None
    middle_record = file_store.get_file(parsed_file.output_files.middle_json.file_id)  # type: ignore[union-attr]
    assert middle_record.sha256sum is not None
    payload = json.loads(file_store.read_blob(middle_record.sha256sum))
    assert payload["file_suffix"] == "ofd"
    assert payload["effort"] == "flash"
    assert payload["parse_mode"] == "txt"
    assert payload["pages"][0]["blocks"][0]["bbox"]


def test_doclib_ingests_ofd_as_local_full_document_flash(tmp_path: Path) -> None:
    """验证 Doclib 为 OFD 建立本地整本 flash 解析任务。"""

    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, object]]:
            """关闭 parsing rules，让测试只观察 OFD 默认行为。"""
            return []

    async def run() -> None:
        """执行隔离 SQLite 入库并检查 OFD 文档与解析任务。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),  # type: ignore[arg-type]
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "sample.ofd"
        source.write_bytes(_minimal_payload())
        response = await service.request_parse(str(source), tier="flash")
        doc = await db.fetchone(
            "SELECT file_type, page_count, title FROM docs WHERE sha256=?",
            (response.sha256,),
        )
        parses = await db.fetchall(
            "SELECT tier, status, privacy FROM parses WHERE sha256=?",
            (response.sha256,),
        )
        assert response.tier == "flash"
        assert doc == {"file_type": "ofd", "page_count": 1, "title": "Fixture Title"}
        assert parses == [{"tier": "flash", "status": "pending", "privacy": "local"}]

    asyncio.run(run())


_LOCAL_SAMPLE_CASES = [
    ("issue_10_issue_10_a7b6f9eceb_ofdrw-converter_src_test_resources_helloworld_a7b6f9eceb.ofd", 1),
    ("issue_200_issue_200_a2b9080ae3_ofdrw-converter_src_test_resources_999_a2b9080ae3.ofd", 5),
    ("issue_208_issue_208_dfed483fa0_ofdrw-layout_src_test_resources_AddWatermarkAnnot_dfed483fa0.ofd", 6),
    ("issue_208_issue_208_e7cac8d149_ofdrw-layout_src_test_resources_no_page_container_e7cac8d149.ofd", 1),
    ("issue_385_issue_385_bff244b113_ofdrw-layout_src_test_resources_keyword2_bff244b113.ofd", 1),
    ("issue_183_issue_183_08a57b9736_48.3_2_08a57b9736.ofd", 1),
    ("issue_190_issue_190_6eadbaeb7b_文档2_6eadbaeb7b.ofd", 2),
    ("issue_385_issue_385_2be07ec79d_ofdrw-layout_src_test_resources_1-1_2be07ec79d.ofd", 3),
    ("issue_271_源文件_bfb11cbcab.ofd", 9),
    ("issue_293_issue_293_87024f3f31_test.zip_test_fcb3a84887.ofd", 2),
    ("issue_312_issue_312_b8973c8f48_测试_b8973c8f48.ofd", 1),
]


@pytest.mark.parametrize(("filename", "page_count"), _LOCAL_SAMPLE_CASES)
def test_ofd_local_real_samples_preserve_pages_and_normalized_bboxes(filename: str, page_count: int) -> None:
    """在用户本地真实语料存在时验证页数、稳定解析和 bbox 契约。"""
    source = _LOCAL_SAMPLE_DIR / filename
    if not source.exists():
        pytest.skip("local OFD sample corpus is unavailable")
    payload = source.read_bytes()
    assert hashlib.sha256(payload).hexdigest()[:10] in filename
    middle, model = doc_analyze(payload, file_suffix="ofd")
    assert len(model.pages) == len(middle.pages) == page_count
    for page in middle.pages:
        for block in page.blocks:
            assert block.bbox is not None
            assert all(0 <= value <= 1 for value in block.bbox)
