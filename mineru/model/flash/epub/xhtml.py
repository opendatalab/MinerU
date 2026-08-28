# Copyright (c) Opendatalab. All rights reserved.
"""把 EPUB XHTML/SVG 内容文档转换为 MinerU raw blocks。"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from lxml import etree  # type: ignore[reportMissingImports]

from ....utils.image_payload import parse_image_data_uri_strict
from .._shared.hyperlink import sanitize_hyperlink_target
from .._shared.markup import (
    MarkupAnchorDocument,
    MarkupAnchorRegistry,
    MarkupProjector,
    MarkupStylesheet,
    ResolvedMarkupImage,
    element_id,
    visible_element_text,
)
from .._shared.markup.projector import (
    BLOCK_TAGS as _BLOCK_TAGS,
    SKIPPED_TAGS as _SKIPPED_TAGS,
    clean_text_node as _clean_text_node,
    entity_text as _entity_text,
    local_name as _local_name,
)
from .constants import IMAGE_MEDIA_BY_EXTENSION, SVG_MEDIA_TYPE
from .package import EpubPackage

_INDIVIDUAL_NOTE_TYPE_ORDER = ("footnote", "endnote", "rearnote")
_INDIVIDUAL_NOTE_ROLE_ORDER = ("doc-footnote", "doc-endnote")
_INDIVIDUAL_NOTE_TYPES = frozenset(_INDIVIDUAL_NOTE_TYPE_ORDER)
_INDIVIDUAL_NOTE_ROLES = frozenset(_INDIVIDUAL_NOTE_ROLE_ORDER)
_NOTE_BLOCK_TAGS = _BLOCK_TAGS | {"li"}
_NOTE_NON_TEXT_SUBTREES = frozenset(
    {
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "math",
        "ol",
        "pre",
        "svg",
        "table",
        "ul",
    }
)


def _epub_types(element: etree._Element) -> frozenset[str]:
    """读取 EPUB 命名空间或未命名 type 属性中的结构语义 token。"""
    values: list[str] = []
    for name, value in element.attrib.items():
        local_name = etree.QName(name).localname if name.startswith("{") else name.split(":", 1)[-1]
        if local_name == "type":
            values.extend(value.casefold().split())
    return frozenset(values)


def _roles(element: etree._Element) -> frozenset[str]:
    """读取 ARIA role 属性中的小写语义 token。"""
    return frozenset((element.get("role") or "").casefold().split())


def _is_individual_note(element: etree._Element) -> bool:
    """判断块级元素是否表示单条 EPUB Footnote/Endnote。"""
    if _local_name(element) not in _NOTE_BLOCK_TAGS:
        return False
    return bool(_epub_types(element) & _INDIVIDUAL_NOTE_TYPES or _roles(element) & _INDIVIDUAL_NOTE_ROLES)


def _note_semantic(element: etree._Element) -> str:
    """按固定优先级返回 note 的 EPUB type 或 ARIA role。"""
    epub_types = _epub_types(element)
    for note_type in _INDIVIDUAL_NOTE_TYPE_ORDER:
        if note_type in epub_types:
            return note_type
    roles = _roles(element)
    for role in _INDIVIDUAL_NOTE_ROLE_ORDER:
        if role in roles:
            return role
    return "note"


def _note_has_text_block(element: etree._Element) -> bool:
    """判断 note 是否能产生非空 text，从而避免注册没有正文目标的 anchor。"""
    if _clean_text_node(element.text).strip():
        return True
    for child in element:
        if child.tail and _clean_text_node(child.tail).strip():
            return True
        if not isinstance(child.tag, str):
            if _entity_text(child):
                return True
            continue
        name = _local_name(child)
        if name in _SKIPPED_TAGS or name in _NOTE_NON_TEXT_SUBTREES or name in {"img", "image"}:
            continue
        if _note_has_text_block(child):
            return True
    return False


def _load_chapter_stylesheet(package: EpubPackage, chapter_path: str, root: etree._Element) -> MarkupStylesheet:
    """按章节 head 顺序加载包内 CSS 与内联 style。"""
    stylesheet = MarkupStylesheet()
    for element in root.iter():
        if not isinstance(element.tag, str):
            continue
        name = _local_name(element)
        if name == "link" and "stylesheet" in (element.get("rel") or "").casefold().split():
            target = package.resolve_reference(element.get("href") or "", base_part=chapter_path)
            if target is None:
                continue
            data = package.read_part(target.path)
            if data is not None:
                stylesheet.add(data.decode("utf-8-sig", errors="replace"))
        elif name == "style":
            stylesheet.add("".join(element.itertext()))
    return stylesheet


class _EpubAnchorPolicy:
    """保持 EPUB 标题、脚注 identity 与可落地性判定的既有契约。"""

    anchor_prefix = "epub"
    register_document_start = True

    @staticmethod
    def heading_identity(element: etree._Element, ordinal: int) -> str:
        """按源 ID 或匿名标题序号生成 EPUB 标题 identity。"""
        return f"{element_id(element) or 'heading'}-{ordinal}"

    @staticmethod
    def is_materializable_note(element: etree._Element, document: MarkupAnchorDocument) -> bool:
        """沿用 EPUB note 语义、文本块能力和最终可见性判断。"""
        return _is_individual_note(element) and _note_has_text_block(element) and bool(visible_element_text(element, document))

    @staticmethod
    def note_identity(element: etree._Element, ordinal: int) -> str:
        """按 note 类型、源 ID 与章节内序号生成 EPUB 脚注 identity。"""
        source_id = element_id(element)
        return f"note-{_note_semantic(element)}-{source_id or 'anonymous'}-{ordinal}"


class EpubAnchorRegistry:
    """建立章节路径、标题与 note fragment 到实际 canonical anchor 的别名表。"""

    def __init__(self, chapters: list[tuple[str, etree._Element]], package: EpubPackage) -> None:
        """预扫描全部选中 XHTML 章节，建立标题、note 与章节起点映射。"""
        self._package = package
        documents = [
            MarkupAnchorDocument(
                key=chapter_path,
                root=root,
                stylesheet=_load_chapter_stylesheet(package, chapter_path, root),
                visibility_scope="nearest_body",
                text_normalization="xhtml_whitespace",
            )
            for chapter_path, root in chapters
        ]
        self._registry = MarkupAnchorRegistry(documents, _EpubAnchorPolicy())

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回一个已预扫描 EPUB 标题的规范 anchor。"""
        return self._registry.heading_anchor(heading)

    def heading_label(self, anchor: str) -> str | None:
        """返回规范 EPUB 标题 anchor 对应的可见标签。"""
        return self._registry.heading_label(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回一个已预扫描 EPUB Footnote/Endnote anchor。"""
        return self._registry.note_anchor(note)

    def resolve_anchor(self, href: str, *, base_part: str) -> str | None:
        """解析指向正文标题或 note 的 EPUB 包内链接，并返回不带井号的 anchor。"""
        normalized = sanitize_hyperlink_target(
            href,
            allowed_schemes=(),
            allow_relative=True,
            allow_fragment=True,
        )
        if normalized is None:
            return None
        target = self._package.resolve_reference(normalized, base_part=base_part)
        if target is None:
            return None
        return self._registry.resolve_target(target.path, target.fragment)

    def resolve_link(self, href: str, *, base_part: str) -> str | None:
        """解析安全外部链接或指向已输出标题/note 的 EPUB 内部链接。"""
        external = sanitize_hyperlink_target(href)
        if external is not None:
            return external
        anchor = self.resolve_anchor(href, base_part=base_part)
        return f"#{anchor}" if anchor else None


def build_anchor_registry(chapters: list[tuple[str, etree._Element]], package: EpubPackage) -> EpubAnchorRegistry:
    """从已解析章节元组建立跨章节锚点注册表。"""
    return EpubAnchorRegistry(chapters, package)


@dataclass(frozen=True, slots=True)
class _EpubMarkupContext:
    """把 EPUB 包资源与文档级 anchor 适配到共享 markup projector。"""

    package: EpubPackage
    chapter_path: str
    anchors: EpubAnchorRegistry

    def resolve_link(self, href: str) -> str | None:
        """解析安全外部链接或实际存在的 EPUB 包内 anchor。"""
        return self.anchors.resolve_link(href, base_part=self.chapter_path)

    def resolve_image(self, source: str, *, alt: str = "") -> ResolvedMarkupImage | None:
        """读取并严格校验一个 EPUB 包内栅格图片引用。"""
        target = self.package.resolve_reference(source, base_part=self.chapter_path)
        if target is None:
            return ResolvedMarkupImage(alt=alt) if alt else None
        media_type = (self.package.content_type_for(target.path) or "").casefold()
        extension = target.path.rsplit(".", 1)[-1].casefold() if "." in target.path else ""
        media_type = media_type or IMAGE_MEDIA_BY_EXTENSION.get(extension, "")
        if not media_type.startswith("image/") or media_type == SVG_MEDIA_TYPE:
            return ResolvedMarkupImage(alt=alt) if alt else None
        payload = self.package.read_part(target.path, asset=True)
        if payload is None:
            return ResolvedMarkupImage(alt=alt) if alt else None
        data_uri = f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"
        try:
            parse_image_data_uri_strict(data_uri)
        except ValueError:
            return ResolvedMarkupImage(alt=alt) if alt else None
        return ResolvedMarkupImage(image_base64=data_uri, alt=alt)

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回一个已预扫描 EPUB 标题的规范 anchor。"""
        return self.anchors.heading_anchor(heading)

    def heading_label(self, anchor: str) -> str | None:
        """返回规范 EPUB 标题 anchor 对应的可见标签。"""
        return self.anchors.heading_label(anchor)

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回一个已预扫描 EPUB Footnote/Endnote anchor。"""
        return self.anchors.note_anchor(note)


class EpubChapterConverter:
    """把一个 XHTML spine item 通过共享 projector 投影为 raw blocks。"""

    def __init__(
        self,
        package: EpubPackage,
        chapter_path: str,
        root: etree._Element,
        anchors: EpubAnchorRegistry,
    ) -> None:
        """绑定单个章节的包、路径、DOM 与文档级 anchor 注册表。"""
        self.package = package
        self.chapter_path = chapter_path
        self.root = root
        self.anchors = anchors
        self.stylesheet = _load_chapter_stylesheet(package, chapter_path, root)

    def convert(self) -> list[dict[str, object]]:
        """解析 XHTML body，并保持既有 EPUB 标题与脚注语义。"""
        body = next(
            (element for element in self.root.iter() if isinstance(element.tag, str) and _local_name(element) == "body"),
            None,
        )
        if body is None:
            return []
        context = _EpubMarkupContext(self.package, self.chapter_path, self.anchors)
        return MarkupProjector(
            body,
            context,
            self.stylesheet,
            single_document_title=False,
        ).convert()


def convert_svg_spine(
    package: EpubPackage,
    chapter_path: str,
    root: etree._Element,
) -> list[dict[str, object]]:
    """把 standalone SVG spine item 尽力转换为文本和包内栅格图片。"""
    empty_registry = EpubAnchorRegistry([], package)
    context = _EpubMarkupContext(package, chapter_path, empty_registry)
    stylesheet = _load_chapter_stylesheet(package, chapter_path, root)
    return MarkupProjector(root, context, stylesheet).convert_svg()


__all__ = [
    "EpubAnchorRegistry",
    "EpubChapterConverter",
    "build_anchor_registry",
    "convert_svg_spine",
]
