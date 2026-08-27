from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

from lxml import etree

from mineru.model.flash.office.docx.package_normalizer import normalize_docx_package
from mineru.model.flash.office.pptx.package_normalizer import normalize_pptx_package
from mineru.model.flash.office.xlsx.package_normalizer import normalize_xlsx_package


_PACKAGE_RELATIONSHIPS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_SPREADSHEETML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    """把合成成员写成确定性的 OOXML 测试包。"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return output.getvalue()


def test_docx_normalizer_removes_missing_internal_relationship() -> None:
    """验证 DOCX 仍删除指向缺失成员的内部 relationship。"""
    relationships = f"""
        <Relationships xmlns="{_PACKAGE_RELATIONSHIPS_NS}">
          <Relationship Id="rId1" Type="urn:test" Target="word/missing.xml"/>
        </Relationships>
    """.encode()
    source = _zip_bytes(
        {
            "[Content_Types].xml": b"<Types/>",
            "_rels/.rels": relationships,
        }
    )

    normalized = normalize_docx_package(source)

    with ZipFile(BytesIO(normalized)) as archive:
        root = etree.fromstring(archive.read("_rels/.rels"))
    assert list(root) == []


def test_pptx_normalizer_translates_strict_ooxml_uri() -> None:
    """验证 PPTX 仍把 Strict PresentationML URI 转为 Transitional URI。"""
    strict_uri = b"http://purl.oclc.org/ooxml/presentationml/main"
    transitional_uri = b"http://schemas.openxmlformats.org/presentationml/2006/main"
    source = _zip_bytes({"ppt/presentation.xml": b'<p:presentation xmlns:p="' + strict_uri + b'"/>'})

    normalized = normalize_pptx_package(source)

    with ZipFile(BytesIO(normalized)) as archive:
        presentation = archive.read("ppt/presentation.xml")
    assert strict_uri not in presentation
    assert transitional_uri in presentation


def test_xlsx_normalizer_fills_empty_style_fill() -> None:
    """验证 XLSX 仍为空 fill 补充 patternFill。"""
    styles = f"""
        <styleSheet xmlns="{_SPREADSHEETML_NS}">
          <fills count="1"><fill/></fills>
        </styleSheet>
    """.encode()
    source = _zip_bytes({"xl/styles.xml": styles})

    normalized = normalize_xlsx_package(source)

    with ZipFile(BytesIO(normalized)) as archive:
        root = ET.fromstring(archive.read("xl/styles.xml"))
    pattern_fill = root.find(f".//{{{_SPREADSHEETML_NS}}}patternFill")
    assert pattern_fill is not None
