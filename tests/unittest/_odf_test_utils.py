from __future__ import annotations

# ruff: noqa: E501 -- 测试夹具保留紧凑 XML，便于直接核对 ODF 结构。

import base64
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


_MIME_BY_SUFFIX = {
    "odt": "application/vnd.oasis.opendocument.text",
    "ods": "application/vnd.oasis.opendocument.spreadsheet",
    "odp": "application/vnd.oasis.opendocument.presentation",
}
_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP8zwACTGCSAQANHQEDgslx/wAAAABJRU5ErkJggg=="
)


def build_odf_package(
    suffix: str,
    content_xml: str,
    *,
    styles_xml: str | None = None,
    meta_xml: str | None = None,
    extra_parts: dict[str, bytes] | None = None,
    encrypted: bool = False,
) -> bytes:
    """构造 mimetype 位于首项且不压缩的最小 ODF 测试包。"""
    mime = _MIME_BY_SUFFIX[suffix]
    manifest_entries = [
        f'<manifest:file-entry manifest:full-path="/" manifest:media-type="{mime}"/>',
        '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>',
    ]
    if encrypted:
        manifest_entries[1] = (
            '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml">'
            "<manifest:encryption-data/>"
            "</manifest:file-entry>"
        )
    for name in (extra_parts or {}):
        media_type = "image/png" if name.endswith(".png") else "text/xml"
        manifest_entries.append(
            f'<manifest:file-entry manifest:full-path="{name}" manifest:media-type="{media_type}"/>'
        )
    manifest = (
        '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
        + "".join(manifest_entries)
        + "</manifest:manifest>"
    )
    output = BytesIO()
    with ZipFile(output, "w") as package:
        package.writestr("mimetype", mime, compress_type=ZIP_STORED)
        package.writestr("META-INF/manifest.xml", manifest, compress_type=ZIP_DEFLATED)
        package.writestr("content.xml", content_xml, compress_type=ZIP_DEFLATED)
        if styles_xml is not None:
            package.writestr("styles.xml", styles_xml, compress_type=ZIP_DEFLATED)
        if meta_xml is not None:
            package.writestr("meta.xml", meta_xml, compress_type=ZIP_DEFLATED)
        for name, data in (extra_parts or {}).items():
            package.writestr(name, data, compress_type=ZIP_DEFLATED)
    return output.getvalue()


def build_odt_fixture() -> bytes:
    """构造覆盖标题、分页、列表、表格、脚注、公式和图片的 ODT。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0">
 <office:automatic-styles>
  <style:style style:name="PBreak" style:family="paragraph"><style:paragraph-properties fo:break-before="page"/></style:style>
  <style:style style:name="TBold" style:family="text"><style:text-properties fo:font-weight="bold"/></style:style>
 </office:automatic-styles>
 <office:body><office:text>
  <text:p text:style-name="Title">ODT Title</text:p>
  <text:h text:outline-level="1"><text:bookmark-start text:name="section"/>Section</text:h>
  <text:p>Plain <text:span text:style-name="TBold">bold</text:span> <text:a xlink:href="https://example.com">link</text:a></text:p>
  <text:p>literal &lt;script&gt;alert(1)&lt;/script&gt;</text:p>
  <text:list text:style-name="L1"><text:list-item text:start-value="3"><text:p>Third</text:p></text:list-item><text:list-item><text:p>Fourth</text:p></text:list-item></text:list>
  <table:table><table:table-header-rows><table:table-row><table:table-cell><text:p>A</text:p></table:table-cell><table:table-cell><text:p>B</text:p></table:table-cell></table:table-row></table:table-header-rows>
   <table:table-row><table:table-cell table:number-columns-spanned="2"><text:p>Merged</text:p></table:table-cell><table:covered-table-cell/></table:table-row></table:table>
  <text:p>Footnote<text:note text:id="n1"><text:note-citation>1</text:note-citation><text:note-body><text:p>Note body</text:p></text:note-body></text:note></text:p>
  <text:p text:style-name="PBreak">Second page</text:p>
  <text:p>Before soft<text:soft-page-break/>Third page</text:p>
  <text:p><draw:frame><draw:object xlink:href="./Object 1"/><draw:image xlink:href="Pictures/pixel.png"/></draw:frame></text:p>
  <draw:frame><draw:image xlink:href="Pictures/pixel.png"/><svg:title>pixel</svg:title></draw:frame>
 </office:text></office:body>
</office:document-content>"""
    styles = """<office:document-styles
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
 <office:styles>
  <style:style style:name="Title" style:display-name="Title" style:family="paragraph"/>
  <text:list-style style:name="L1"><text:list-level-style-number text:level="1" style:num-format="1" text:start-value="3"/></text:list-style>
 </office:styles>
 <office:master-styles><style:master-page style:name="Standard"><style:header><text:p>Header</text:p></style:header><style:footer><text:p>Footer</text:p></style:footer></style:master-page></office:master-styles>
</office:document-styles>"""
    formula = """<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:math="http://www.w3.org/1998/Math/MathML"><office:body><office:formula><math:math><math:mfrac><math:mi>x</math:mi><math:mn>2</math:mn></math:mfrac></math:math></office:formula></office:body></office:document-content>"""
    meta = """<office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0"><office:meta><dc:title>ODT Meta</dc:title><dc:creator>Alice</dc:creator><meta:keyword>one</meta:keyword><meta:document-statistic meta:page-count="3"/></office:meta></office:document-meta>"""
    return build_odf_package(
        "odt",
        content,
        styles_xml=styles,
        meta_xml=meta,
        extra_parts={"Object 1/content.xml": formula.encode(), "Pictures/pixel.png": _PIXEL_PNG},
    )


def _chart_object_xml() -> bytes:
    """返回带精确 series 引用和内嵌数据表的 ODF chart 子文档。"""
    return b"""<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
 <office:body><office:chart><chart:chart><chart:plot-area><chart:categories table:cell-range-address="local-table.A2:A3"/><chart:series chart:label-cell-address="local-table.B1" chart:values-cell-range-address="local-table.B2:B3"/></chart:plot-area></chart:chart>
 <table:table><table:table-row><table:table-cell><text:p>Category</text:p></table:table-cell><table:table-cell><text:p>Value</text:p></table:table-cell></table:table-row><table:table-row><table:table-cell><text:p>A</text:p></table:table-cell><table:table-cell office:value-type="float" office:value="1"/></table:table-row><table:table-row><table:table-cell><text:p>B</text:p></table:table-cell><table:table-cell office:value-type="float" office:value="2"/></table:table-row></table:table>
 </office:chart></office:body></office:document-content>"""


def build_odp_fixture() -> bytes:
    """构造覆盖空 slide、图表预览和 speaker notes 的 ODP。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0">
 <office:body><office:presentation>
  <draw:page draw:name="Slide 1"><draw:frame presentation:class="title" svg:x="1cm" svg:y="1cm"><draw:text-box><text:p>Deck title</text:p></draw:text-box></draw:frame><draw:frame svg:x="1cm" svg:y="3cm"><draw:text-box><text:p>Body</text:p></draw:text-box></draw:frame></draw:page>
  <draw:page draw:name="Slide 2"/>
  <draw:page draw:name="Slide 3"><draw:frame svg:x="2cm" svg:y="2cm"><draw:object xlink:href="./Object 1"/><draw:image xlink:href="Pictures/pixel.png"/></draw:frame><presentation:notes><draw:frame><draw:text-box><text:p>Speaker note</text:p></draw:text-box></draw:frame></presentation:notes></draw:page>
 </office:presentation></office:body>
</office:document-content>"""
    return build_odf_package(
        "odp",
        content,
        extra_parts={"Object 1/content.xml": _chart_object_xml(), "Pictures/pixel.png": _PIXEL_PNG},
    )


def build_ods_fixture() -> bytes:
    """构造覆盖可见/隐藏 sheet、离散数据区、合并和图表的 ODS。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:body><office:spreadsheet>
  <table:table table:name="Visible A"><table:table-row><table:table-cell><text:p>Name</text:p></table:table-cell><table:table-cell><text:p>Value</text:p></table:table-cell></table:table-row><table:table-row><table:table-cell><text:p>A</text:p></table:table-cell><table:table-cell office:value-type="percentage" office:value="0.5"/></table:table-row><table:shapes><draw:frame><draw:object xlink:href="./Object 1"/><draw:image xlink:href="Pictures/pixel.png"/></draw:frame></table:shapes></table:table>
  <table:table table:name="Hidden" table:display="false"><table:table-row><table:table-cell><text:p>secret</text:p></table:table-cell></table:table-row></table:table>
  <table:table table:name="Visible B"><table:table-row><table:table-cell table:number-columns-spanned="2"><text:p>Merged</text:p></table:table-cell><table:covered-table-cell/></table:table-row><table:table-row table:number-rows-repeated="2"><table:table-cell table:number-columns-repeated="2"/></table:table-row><table:table-row><table:table-cell table:number-columns-repeated="3"/><table:table-cell><text:p>Far</text:p></table:table-cell></table:table-row></table:table>
 </office:spreadsheet></office:body>
</office:document-content>"""
    return build_odf_package(
        "ods",
        content,
        extra_parts={"Object 1/content.xml": _chart_object_xml(), "Pictures/pixel.png": _PIXEL_PNG},
    )


__all__ = ["build_odf_package", "build_odp_fixture", "build_ods_fixture", "build_odt_fixture"]
