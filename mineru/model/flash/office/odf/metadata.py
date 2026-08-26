# Copyright (c) Opendatalab. All rights reserved.
"""读取 OpenDocument meta.xml 与结构页数。"""

from __future__ import annotations

from typing import BinaryIO, Final

from lxml import etree  # type: ignore[reportMissingImports]

from .constants import OdfSuffix, qname
from .package import OdfPackage
from .styles import OdfStyles


MAX_ODT_METADATA_PAGE_COUNT: Final = 10_000


def _first_text(root: etree._Element | None, *tags: str) -> str | None:
    """返回多个候选标签中首个非空文本。"""
    if root is None:
        return None
    for tag in tags:
        element = root.find(f".//{tag}")
        if element is not None:
            value = "".join(element.itertext()).strip()
            if value:
                return value
    return None


def _odt_page_count(meta_root: etree._Element | None) -> int | None:
    """读取 ODT 生产者记录的布局页数，缺失或非法时返回空。"""
    if meta_root is None:
        return None
    statistic = meta_root.find(f".//{qname('meta', 'document-statistic')}")
    if statistic is None:
        return None
    try:
        value = int(statistic.get(qname("meta", "page-count"), ""))
    except ValueError:
        return None
    return min(value, MAX_ODT_METADATA_PAGE_COUNT) if value >= 1 else None


def _visible_sheet_count(body: etree._Element, styles: OdfStyles) -> int:
    """统计未被 table:display 或表格样式隐藏的 ODS 工作表。"""
    count = 0
    for sheet in body:
        if sheet.tag != qname("table", "table"):
            continue
        if sheet.get(qname("table", "display"), "true").casefold() == "false":
            continue
        if styles.table_is_visible(sheet.get(qname("table", "style-name"))):
            count += 1
    return count


def extract_odf_metadata(file_binary: BinaryIO, suffix: OdfSuffix) -> dict[str, object | None]:
    """提取 ODF 标题作者等元数据及稳定文档页数。"""
    package = OdfPackage(file_binary.read())
    try:
        content_root = package.validate_document(suffix)
        styles_root = package.xml_part("styles.xml")
        styles = OdfStyles(styles_root, content_root)
        body = package.body_element(content_root, suffix)
        meta_root = package.xml_part("meta.xml")
        keywords = []
        if meta_root is not None:
            for keyword in meta_root.iter(qname("meta", "keyword")):
                value = "".join(keyword.itertext()).strip()
                if value:
                    keywords.append(value)
        if suffix == "odt":
            page_count = _odt_page_count(meta_root)
        elif suffix == "odp":
            page_count = sum(
                1 for child in body if child.tag == qname("draw", "page") and styles.drawing_page_is_visible(child)
            )
        else:
            page_count = _visible_sheet_count(body, styles)
        return {
            "page_count": page_count or 1,
            "title": _first_text(meta_root, qname("dc", "title")),
            "author": _first_text(meta_root, qname("dc", "creator"), qname("meta", "initial-creator")),
            "subject": _first_text(meta_root, qname("dc", "subject")),
            "keywords": ", ".join(keywords) or None,
        }
    finally:
        package.close()


__all__ = ["extract_odf_metadata"]
