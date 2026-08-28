# Copyright (c) Opendatalab. All rights reserved.
"""EPUB 3.3 的元数据 XML、导航文档与确定性 OCF 打包。"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from lxml import etree

from .assets import EpubAsset

_CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_EPUB_NS = "http://www.idpf.org/2007/ops"
_OPF_NS = "http://www.idpf.org/2007/opf"
_XHTML_NS = "http://www.w3.org/1999/xhtml"
_XML_NS = "http://www.w3.org/XML/1998/namespace"
_EPUB_MIME = b"application/epub+zip"
_CONTENT_PATH = "EPUB/text/content.xhtml"
_STYLE_PATH = "EPUB/styles/mineru.css"


@dataclass(frozen=True, slots=True)
class EpubMetadata:
    """保存 package document 所需的规范化书籍元数据。"""

    title: str
    authors: tuple[str, ...]
    language: str
    identifier: str
    modified_at: datetime


@dataclass(slots=True)
class NavigationItem:
    """保存 EPUB toc nav 中一个可链接的层级条目。"""

    title: str
    href: str
    children: list[NavigationItem] = field(default_factory=list)


def build_epub_package(
    *,
    metadata: EpubMetadata,
    content_xhtml: bytes,
    navigation: list[NavigationItem],
    stylesheet: bytes,
    assets: tuple[EpubAsset, ...],
    has_mathml: bool,
) -> bytes:
    """在内存中按固定成员顺序构造完整 EPUB 3.3 容器。"""
    container_xml = _build_container_xml()
    navigation_xhtml = _build_navigation_xhtml(metadata, navigation)
    package_document = _build_package_document(metadata, assets, has_mathml=has_mathml)
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        _write_member(archive, "mimetype", _EPUB_MIME, metadata.modified_at, compression=ZIP_STORED)
        _write_member(archive, "META-INF/container.xml", container_xml, metadata.modified_at)
        _write_member(archive, "EPUB/package.opf", package_document, metadata.modified_at)
        _write_member(archive, "EPUB/nav.xhtml", navigation_xhtml, metadata.modified_at)
        _write_member(archive, _CONTENT_PATH, content_xhtml, metadata.modified_at)
        _write_member(archive, _STYLE_PATH, stylesheet, metadata.modified_at)
        for asset in assets:
            _write_member(archive, f"EPUB/assets/{asset.file_name}", asset.data, metadata.modified_at)
    return output.getvalue()


def _build_container_xml() -> bytes:
    """生成指向唯一 OPF package document 的 container.xml。"""
    root = etree.Element(f"{{{_CONTAINER_NS}}}container", nsmap={None: _CONTAINER_NS}, version="1.0")
    rootfiles = etree.SubElement(root, f"{{{_CONTAINER_NS}}}rootfiles")
    etree.SubElement(
        rootfiles,
        f"{{{_CONTAINER_NS}}}rootfile",
        attrib={
            "full-path": "EPUB/package.opf",
            "media-type": "application/oebps-package+xml",
        },
    )
    return _serialize_xml(root)


def _build_package_document(metadata: EpubMetadata, assets: tuple[EpubAsset, ...], *, has_mathml: bool) -> bytes:
    """生成包含必需元数据、完整 manifest 与单正文 spine 的 OPF。"""
    root = etree.Element(
        f"{{{_OPF_NS}}}package",
        nsmap={None: _OPF_NS, "dc": _DC_NS},
        attrib={
            "version": "3.0",
            "unique-identifier": "pub-id",
            f"{{{_XML_NS}}}lang": metadata.language,
        },
    )
    metadata_element = etree.SubElement(root, f"{{{_OPF_NS}}}metadata")
    identifier = etree.SubElement(metadata_element, f"{{{_DC_NS}}}identifier", id="pub-id")
    identifier.text = metadata.identifier
    title = etree.SubElement(metadata_element, f"{{{_DC_NS}}}title")
    title.text = metadata.title
    language = etree.SubElement(metadata_element, f"{{{_DC_NS}}}language")
    language.text = metadata.language
    for author in metadata.authors:
        creator = etree.SubElement(metadata_element, f"{{{_DC_NS}}}creator")
        creator.text = author
    modified = etree.SubElement(metadata_element, f"{{{_OPF_NS}}}meta", property="dcterms:modified")
    modified.text = metadata.modified_at.strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest = etree.SubElement(root, f"{{{_OPF_NS}}}manifest")
    etree.SubElement(
        manifest,
        f"{{{_OPF_NS}}}item",
        id="nav",
        href="nav.xhtml",
        attrib={"media-type": "application/xhtml+xml", "properties": "nav"},
    )
    content_attributes = {
        "id": "content",
        "href": "text/content.xhtml",
        "media-type": "application/xhtml+xml",
    }
    if has_mathml:
        content_attributes["properties"] = "mathml"
    etree.SubElement(manifest, f"{{{_OPF_NS}}}item", attrib=content_attributes)
    etree.SubElement(
        manifest,
        f"{{{_OPF_NS}}}item",
        id="style",
        href="styles/mineru.css",
        attrib={"media-type": "text/css"},
    )
    for position, asset in enumerate(assets, start=1):
        etree.SubElement(
            manifest,
            f"{{{_OPF_NS}}}item",
            id=f"asset-{position}",
            href=f"assets/{asset.file_name}",
            attrib={"media-type": asset.media_type},
        )
    spine = etree.SubElement(root, f"{{{_OPF_NS}}}spine")
    etree.SubElement(spine, f"{{{_OPF_NS}}}itemref", idref="content")
    return _serialize_xml(root)


def _build_navigation_xhtml(metadata: EpubMetadata, navigation: list[NavigationItem]) -> bytes:
    """生成恰含一个 toc nav 和一个 landmarks nav 的 EPUB 导航文档。"""
    root = etree.Element(
        f"{{{_XHTML_NS}}}html",
        nsmap={None: _XHTML_NS, "epub": _EPUB_NS},
        attrib={f"{{{_XML_NS}}}lang": metadata.language, "lang": metadata.language},
    )
    head = etree.SubElement(root, f"{{{_XHTML_NS}}}head")
    etree.SubElement(head, f"{{{_XHTML_NS}}}meta", charset="utf-8")
    title = etree.SubElement(head, f"{{{_XHTML_NS}}}title")
    title.text = f"{metadata.title} — Table of Contents"
    etree.SubElement(
        head,
        f"{{{_XHTML_NS}}}link",
        rel="stylesheet",
        href="styles/mineru.css",
        type="text/css",
    )
    body = etree.SubElement(root, f"{{{_XHTML_NS}}}body")
    toc = etree.SubElement(
        body,
        f"{{{_XHTML_NS}}}nav",
        id="toc",
        role="doc-toc",
        attrib={f"{{{_EPUB_NS}}}type": "toc"},
    )
    heading = etree.SubElement(toc, f"{{{_XHTML_NS}}}h1")
    heading.text = "Table of Contents"
    ordered = etree.SubElement(toc, f"{{{_XHTML_NS}}}ol")
    for item in navigation:
        _append_navigation_item(ordered, item)

    landmarks = etree.SubElement(
        body,
        f"{{{_XHTML_NS}}}nav",
        id="landmarks",
        hidden="hidden",
        attrib={f"{{{_EPUB_NS}}}type": "landmarks"},
    )
    landmarks_heading = etree.SubElement(landmarks, f"{{{_XHTML_NS}}}h2")
    landmarks_heading.text = "Landmarks"
    landmarks_list = etree.SubElement(landmarks, f"{{{_XHTML_NS}}}ol")
    item = etree.SubElement(landmarks_list, f"{{{_XHTML_NS}}}li")
    link = etree.SubElement(
        item,
        f"{{{_XHTML_NS}}}a",
        href="text/content.xhtml#content-start",
        attrib={f"{{{_EPUB_NS}}}type": "bodymatter"},
    )
    link.text = "Start of Content"
    return _serialize_xml(root, doctype="<!DOCTYPE html>")


def _append_navigation_item(parent: etree._Element, item: NavigationItem) -> None:
    """递归把一个导航条目及其子项写入有序列表。"""
    entry = etree.SubElement(parent, f"{{{_XHTML_NS}}}li")
    link = etree.SubElement(entry, f"{{{_XHTML_NS}}}a", href=item.href)
    link.text = item.title
    if item.children:
        nested = etree.SubElement(entry, f"{{{_XHTML_NS}}}ol")
        for child in item.children:
            _append_navigation_item(nested, child)


def _serialize_xml(root: etree._Element, *, doctype: str | None = None) -> bytes:
    """以 UTF-8 XML 声明和稳定缩进序列化一个 EPUB XML 文档。"""
    etree.indent(root, space="  ")
    return etree.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True,
        doctype=doctype,
        pretty_print=True,
    )


def _write_member(
    archive: ZipFile,
    name: str,
    data: bytes,
    modified_at: datetime,
    *,
    compression: int = ZIP_DEFLATED,
) -> None:
    """用固定权限、无 extra 字段和规范时间写入一个 OCF ZIP 成员。"""
    info = ZipInfo(name, date_time=_zip_datetime(modified_at))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.extra = b""
    info.comment = b""
    archive.writestr(info, data)


def _zip_datetime(modified_at: datetime) -> tuple[int, int, int, int, int, int]:
    """把 UTC 修改时间约束到 ZIP DOS 时间范围并截断到双秒。"""
    year = min(max(modified_at.year, 1980), 2107)
    day = min(modified_at.day, calendar.monthrange(year, modified_at.month)[1])
    second = modified_at.second - modified_at.second % 2
    return year, modified_at.month, day, modified_at.hour, modified_at.minute, second


__all__ = ["EpubMetadata", "NavigationItem", "build_epub_package"]
