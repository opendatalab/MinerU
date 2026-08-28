from __future__ import annotations

import base64
from io import BytesIO
import struct
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from PIL import Image
import pytest
from lxml import etree

from mineru.model.flash._shared.hyperlink import (
    OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
    sanitize_hyperlink_target,
)
from mineru.model.flash._shared.image import image_to_b64str, image_to_bytes
from mineru.model.flash._shared.mathml import mathml_to_latex
from mineru.model.flash.office.doc import records as doc_records
from mineru.model.flash.office.legacy.binary import bounded_slice, get_f64, get_i16, get_u16, get_u32
from mineru.model.flash.office.opc import relationship_source_base_dir, write_zip_package
from mineru.model.flash.office.ppt import records as ppt_records
from mineru.model.flash.office.rich_text import OfficeRichTextSegment, build_rich_text_from_segments
from mineru.model.flash.office.xls import records as xls_records


@pytest.mark.parametrize(
    ("mode", "image_format", "expected_format"),
    [
        ("RGB", "JPEG", "JPEG"),
        ("RGBA", "PNG", "PNG"),
    ],
)
def test_shared_image_encoding_preserves_format_and_data_uri(
    mode: str,
    image_format: str,
    expected_format: str,
) -> None:
    """验证共享图片编码同时保留字节格式与 data URI 载荷。"""
    image = Image.new(mode, (2, 3), (12, 34, 56, 78) if mode == "RGBA" else (12, 34, 56))

    image_bytes = image_to_bytes(image, image_format=image_format)
    data_uri = image_to_b64str(image, image_format=image_format)
    prefix, encoded = data_uri.split(",", 1)

    assert prefix == f"data:image/{image_format.lower()};base64"
    assert base64.b64decode(encoded) == image_bytes
    with Image.open(BytesIO(image_bytes)) as decoded:
        assert decoded.format == expected_format
        assert decoded.size == (2, 3)


def test_mathml_semantics_ignores_non_tex_alternate_annotations() -> None:
    """验证 semantics 只转换主展示分支，不重复拼接非 TeX annotation。"""
    math = etree.fromstring(
        b'<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mfrac><mi>x</mi><mn>2</mn></mfrac>'
        b'<annotation-xml encoding="MathML-Content"><apply><divide/><ci>x</ci><cn>2</cn></apply></annotation-xml>'
        b"</semantics></math>"
    )

    assert mathml_to_latex(math) == r"\frac{x}{2}"


@pytest.mark.parametrize(
    ("token", "value", "expected"),
    [
        ("mi", "a_b", r"a\_b"),
        ("mi", r"\name{x}", r"\backslash{}name\{x\}"),
        ("mn", "12_3", r"12\_3"),
        ("mi", "x^~", r"x\^{}\~{}"),
        ("mi", "α", r"\alpha"),
    ],
)
def test_mathml_literal_identifier_tokens_do_not_become_latex_syntax(token: str, value: str, expected: str) -> None:
    """验证标识符和数字字面量转义 TeX 控制字符，同时保留显式希腊字母映射。"""
    math = etree.fromstring(f'<math xmlns="http://www.w3.org/1998/Math/MathML"><{token}>{value}</{token}></math>'.encode())

    assert mathml_to_latex(math) == expected


@pytest.mark.parametrize("extra_node", ["<!--producer note-->", "<?producer note?>"])
def test_mathml_annotation_scan_skips_non_element_nodes(extra_node: str) -> None:
    """验证 TeX annotation 扫描跳过 XML comment 与处理指令。"""
    math = etree.fromstring(
        (
            '<math xmlns="http://www.w3.org/1998/Math/MathML">'
            f"{extra_node}"
            '<semantics><mi>x</mi><annotation encoding="application/x-tex">x^2</annotation></semantics>'
            "</math>"
        ).encode()
    )

    assert mathml_to_latex(math) == "x^2"


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("https://example.com/path", "https://example.com/path"),
        ("mailto:user@example.com", "mailto:user@example.com"),
        ("#anchor", "#anchor"),
        ("../relative/path", "../relative/path"),
        ("javascript:alert(1)", None),
        ("file:///tmp/local", None),
        ("//example.com/path", None),
        (r"C:\local\file", None),
        ("https:///missing-host", None),
        ("https://example.com/a\x01b", None),
    ],
)
def test_shared_hyperlink_policy_rejects_active_and_local_targets(target: str, expected: str | None) -> None:
    """验证共享策略统一处理外链、fragment、相对路径和危险目标。"""
    assert (
        sanitize_hyperlink_target(
            target,
            allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
            allow_relative=True,
            allow_fragment=True,
        )
        == expected
    )


def test_shared_hyperlink_spans_preserve_url_and_tag_literals() -> None:
    """验证结构化链接保留 URL 与标签外观原文，危险目标降级为 TextSpan。"""
    unsafe = build_rich_text_from_segments(
        [OfficeRichTextSegment("<hyperlink>click</hyperlink>", hyperlink="javascript:alert(1)")]
    )
    safe = build_rich_text_from_segments(
        [OfficeRichTextSegment("<b>click</b>", style="bold", hyperlink="https://example.com/?a=1&b=2")]
    )

    assert unsafe == [{"type": "text", "content": "<hyperlink>click</hyperlink>"}]
    assert safe == [
        {
            "type": "hyperlink",
            "url": "https://example.com/?a=1&b=2",
            "content": [{"type": "text", "content": "<b>click</b>", "styles": ["bold"]}],
        }
    ]


def test_legacy_binary_readers_preserve_bounds_and_values() -> None:
    """验证旧版 Office 共用读取器只在完整边界内返回小端数值。"""
    data = struct.pack("<HhId", 0xABCD, -123, 0x1234_5678, 1.25)

    assert get_u16(data, 0) == 0xABCD
    assert get_i16(data, 2) == -123
    assert get_u32(data, 4) == 0x1234_5678
    assert get_f64(data, 8) == 1.25
    assert get_u16(data, -1) is None
    assert get_u32(data, len(data) - 3) is None
    assert get_f64(data, len(data) - 7) is None
    assert bounded_slice(data, 2, 4) == data[2:6]
    assert bounded_slice(data, -1, 1) is None
    assert bounded_slice(data, 0, -1) is None
    assert bounded_slice(data, len(data), 1) is None


def test_format_record_modules_no_longer_export_duplicate_binary_readers() -> None:
    """验证严格迁移后格式 records 模块不再暴露重复的读取函数。"""
    for module in (doc_records, ppt_records, xls_records):
        for name in ("get_u16", "get_i16", "get_u32", "get_f64", "bounded_slice"):
            assert not hasattr(module, name)


@pytest.mark.parametrize(
    ("rels_filename", "expected"),
    [
        ("_rels/.rels", ""),
        ("word/_rels/document.xml.rels", "word"),
        ("ppt/slides/_rels/slide1.xml.rels", "ppt/slides"),
        ("word/document.xml.rels", None),
        ("word/_rels/document.xml", None),
    ],
)
def test_opc_relationship_source_base_dir(rels_filename: str, expected: str | None) -> None:
    """验证 OPC relationship 路径只接受根关系或规范 part 关系。"""
    assert relationship_source_base_dir(rels_filename) == expected


def test_write_zip_package_preserves_member_data_and_metadata() -> None:
    """验证共享写包器保留成员内容与 ZipInfo 元数据。"""
    info = ZipInfo("word/document.xml", date_time=(2024, 1, 2, 3, 4, 6))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o640 << 16
    payload = b"<document/>"

    package = write_zip_package([(info, payload)])

    with ZipFile(BytesIO(package)) as archive:
        restored = archive.getinfo(info.filename)
        assert archive.read(info.filename) == payload
        assert restored.date_time == info.date_time
        assert restored.compress_type == info.compress_type
        assert restored.external_attr == info.external_attr
