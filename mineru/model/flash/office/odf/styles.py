# Copyright (c) Opendatalab. All rights reserved.
"""解析 OpenDocument 样式继承、列表和逻辑分页属性。"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from lxml import etree  # type: ignore[reportMissingImports]

from .constants import qname
from .models import ListLevel, ParagraphProperties, ParagraphPropertiesDelta, TextStyle, TextStyleDelta


@dataclass(frozen=True, slots=True)
class _StyleDefinition:
    """保存一个命名样式的父级、文本增量和分页属性。"""

    name: str
    family: str
    parent: str | None
    display_name: str | None
    text_delta: TextStyleDelta
    paragraph: ParagraphPropertiesDelta
    table_display: bool | None


class OdfStyles:
    """合并 styles.xml 与 content.xml 中的 ODF 样式定义。"""

    def __init__(self, *roots: etree._Element | None) -> None:
        """按传入顺序收集文档样式，使 content.xml 自动样式覆盖基础定义。"""
        self._styles: dict[tuple[str, str], _StyleDefinition] = {}
        self._defaults: dict[str, TextStyleDelta] = {}
        self._list_styles: dict[str, dict[int, ListLevel]] = {}
        self._resolved_text: dict[tuple[str, str], TextStyleDelta] = {}
        self._master_pages: dict[str, etree._Element] = {}
        for root in roots:
            if root is not None:
                self._collect(root)

    def _collect(self, root: etree._Element) -> None:
        """从一棵 ODF XML 树收集默认、命名、列表和 master-page 样式。"""
        for default in root.iter(qname("style", "default-style")):
            family = default.get(qname("style", "family"))
            if family:
                self._defaults[family] = self._text_delta(default)
        for style in root.iter(qname("style", "style")):
            name = style.get(qname("style", "name"))
            family = style.get(qname("style", "family"))
            if not name or not family:
                continue
            self._styles[(family, name)] = _StyleDefinition(
                name=name,
                family=family,
                parent=style.get(qname("style", "parent-style-name")),
                display_name=style.get(qname("style", "display-name")),
                text_delta=self._text_delta(style),
                paragraph=self._paragraph_properties(style),
                table_display=self._table_display(style),
            )
        for list_style in root.iter(qname("text", "list-style")):
            name = list_style.get(qname("style", "name"))
            if name:
                self._list_styles[name] = self._parse_list_style(list_style)
        for master_page in root.iter(qname("style", "master-page")):
            name = master_page.get(qname("style", "name"))
            if name:
                self._master_pages[name] = master_page

    @staticmethod
    def _text_delta(style: etree._Element) -> TextStyleDelta:
        """把 style:text-properties 转换为可继承的三态样式增量。"""
        properties = style.find(qname("style", "text-properties"))
        if properties is None:
            return TextStyleDelta()
        weight = properties.get(qname("fo", "font-weight"))
        bold: bool | None = None
        if weight is not None:
            try:
                bold = int(weight) >= 600
            except ValueError:
                bold = weight.casefold() == "bold"
        font_style = properties.get(qname("fo", "font-style"))
        italic = None if font_style is None else font_style.casefold() in {"italic", "oblique"}
        underline_style = properties.get(qname("style", "text-underline-style"))
        underline = None if underline_style is None else underline_style.casefold() != "none"
        strike_style = properties.get(qname("style", "text-line-through-style"))
        strikethrough = None if strike_style is None else strike_style.casefold() != "none"
        position = (properties.get(qname("style", "text-position")) or "").strip().casefold()
        superscript: bool | None = None
        subscript: bool | None = None
        if position:
            superscript = position.startswith("super") or OdfStyles._position_is_positive(position)
            subscript = position.startswith("sub") or OdfStyles._position_is_negative(position)
            if position == "0%":
                superscript = False
                subscript = False
        return TextStyleDelta(
            bold=bold,
            italic=italic,
            underline=underline,
            strikethrough=strikethrough,
            superscript=superscript,
            subscript=subscript,
        )

    @staticmethod
    def _position_is_positive(position: str) -> bool:
        """判断百分比形式的 text-position 是否表示上标。"""
        try:
            return float(position.split("%", 1)[0]) > 0
        except ValueError:
            return False

    @staticmethod
    def _position_is_negative(position: str) -> bool:
        """判断百分比形式的 text-position 是否表示下标。"""
        try:
            return float(position.split("%", 1)[0]) < 0
        except ValueError:
            return False

    @staticmethod
    def _paragraph_properties(style: etree._Element) -> ParagraphPropertiesDelta:
        """读取可区分缺省与显式非分页值的段落属性增量。"""
        properties = style.find(qname("style", "paragraph-properties"))
        before = properties.get(qname("fo", "break-before")) if properties is not None else None
        after = properties.get(qname("fo", "break-after")) if properties is not None else None
        return ParagraphPropertiesDelta(
            break_before=None if before is None else before.casefold() == "page",
            break_after=None if after is None else after.casefold() == "page",
            master_page_name=style.get(qname("style", "master-page-name")),
        )

    @staticmethod
    def _table_display(style: etree._Element) -> bool | None:
        """读取表格样式的 display 开关，用于过滤隐藏工作表。"""
        properties = style.find(qname("style", "table-properties"))
        if properties is None:
            return None
        display = properties.get(qname("table", "display"))
        if display is None:
            return None
        return display.casefold() != "false"

    @staticmethod
    def _parse_list_style(element: etree._Element) -> dict[int, ListLevel]:
        """解析列表样式的层级、起始值和前后缀。"""
        levels: dict[int, ListLevel] = {}
        for child in element:
            if not isinstance(child.tag, str):
                continue
            local_name = etree.QName(child).localname
            if local_name not in {"list-level-style-number", "list-level-style-bullet", "list-level-style-image"}:
                continue
            try:
                level = max(1, int(child.get(qname("text", "level"), "1")))
            except ValueError:
                level = 1
            try:
                start = max(1, int(child.get(qname("text", "start-value"), "1")))
            except ValueError:
                start = 1
            ordered = local_name == "list-level-style-number" and bool(child.get(qname("style", "num-format"), "1"))
            levels[level - 1] = ListLevel(
                ordered=ordered,
                start=start,
                prefix=child.get(qname("style", "num-prefix"), ""),
                suffix=child.get(qname("style", "num-suffix"), "." if ordered else ""),
            )
        return levels

    def _resolved_delta(self, family: str, name: str | None) -> TextStyleDelta:
        """沿 parent-style-name 合并样式，并在循环处安全截断。"""
        if not name:
            return self._defaults.get(family, TextStyleDelta())
        key = (family, name)
        if key in self._resolved_text:
            return self._resolved_text[key]
        chain: list[_StyleDefinition] = []
        seen: set[str] = set()
        current = name
        while current:
            if current in seen:
                logger.warning("ODF style inheritance cycle detected: family={}, style={}", family, current)
                break
            seen.add(current)
            definition = self._styles.get((family, current))
            if definition is None:
                break
            chain.append(definition)
            current = definition.parent or ""
        result = self._defaults.get(family, TextStyleDelta())
        for definition in reversed(chain):
            result = result.merge(definition.text_delta)
        self._resolved_text[key] = result
        return result

    def text_style(self, style_name: str | None, *, family: str, inherited: TextStyle | None = None) -> TextStyle:
        """解析指定样式，并让 span 的未声明属性继承当前段落样式。"""
        delta = self._resolved_delta(family, style_name)
        if inherited is not None:
            base = TextStyleDelta(
                bold=inherited.bold,
                italic=inherited.italic,
                underline=inherited.underline,
                strikethrough=inherited.strikethrough,
                superscript=inherited.superscript,
                subscript=inherited.subscript,
            )
            delta = base.merge(delta)
        return delta.resolve()

    def paragraph_properties(self, style_name: str | None) -> ParagraphProperties:
        """沿段落父样式继承分页标记和 master-page 名称。"""
        if not style_name:
            return ParagraphProperties()
        chain: list[_StyleDefinition] = []
        seen: set[str] = set()
        current = style_name
        while current and current not in seen:
            seen.add(current)
            definition = self._styles.get(("paragraph", current))
            if definition is None:
                break
            chain.append(definition)
            current = definition.parent or ""
        before = False
        after = False
        master: str | None = None
        for definition in reversed(chain):
            if definition.paragraph.break_before is not None:
                before = definition.paragraph.break_before
            if definition.paragraph.break_after is not None:
                after = definition.paragraph.break_after
            master = definition.paragraph.master_page_name or master
        return ParagraphProperties(before, after, master)

    def is_document_title(self, style_name: str | None) -> bool:
        """根据样式名称和 display-name 判断段落是否为文档标题。"""
        if not style_name:
            return False
        definition = self._styles.get(("paragraph", style_name))
        names = {style_name, definition.display_name if definition is not None else ""}
        normalized = {name.replace("_20_", " ").replace("_", " ").strip().casefold() for name in names if name}
        return bool(normalized & {"title", "document title", "标题"})

    def list_level(self, style_name: str | None, depth: int) -> ListLevel:
        """返回指定列表深度的定义，不存在时使用无序默认值。"""
        if not style_name:
            return ListLevel()
        levels = self._list_styles.get(style_name, {})
        return levels.get(depth, ListLevel())

    def table_is_visible(self, style_name: str | None) -> bool:
        """判断工作表样式是否显式隐藏，未知样式默认可见。"""
        if not style_name:
            return True
        definition = self._styles.get(("table", style_name))
        return definition is None or definition.table_display is not False

    def master_page(self, name: str | None) -> etree._Element | None:
        """返回指定 master-page；空名称时优先使用第一个定义。"""
        if name and name in self._master_pages:
            return self._master_pages[name]
        return next(iter(self._master_pages.values()), None)


__all__ = ["OdfStyles"]
