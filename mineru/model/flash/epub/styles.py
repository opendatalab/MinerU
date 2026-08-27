# Copyright (c) Opendatalab. All rights reserved.
"""解析 EPUB XHTML 使用的有限语义 CSS 子集。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree  # type: ignore[reportMissingImports]


_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_IMPORTANT_RE = re.compile(r"!\s*important\s*$", re.IGNORECASE)
_TEXT_STYLE_FIELDS = ("bold", "italic", "underline", "strikethrough", "superscript", "subscript")
_VISIBILITY_FIELDS = ("display", "visibility")


@dataclass(frozen=True, slots=True)
class TextStyle:
    """保存可投影到 Middle JSON 行内协议的文字样式。"""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    superscript: bool = False
    subscript: bool = False

    def merge(self, other: TextStyle) -> TextStyle:
        """合并继承样式和当前元素显式开启的样式。"""
        return TextStyle(
            bold=self.bold or other.bold,
            italic=self.italic or other.italic,
            underline=self.underline or other.underline,
            strikethrough=self.strikethrough or other.strikethrough,
            superscript=self.superscript or other.superscript,
            subscript=self.subscript or other.subscript,
        )

    def names(self) -> tuple[str, ...]:
        """按稳定顺序返回现有行内协议识别的样式名称。"""
        return tuple(
            name
            for enabled, name in (
                (self.bold, "bold"),
                (self.italic, "italic"),
                (self.underline, "underline"),
                (self.strikethrough, "strikethrough"),
                (self.superscript, "superscript"),
                (self.subscript, "subscript"),
            )
            if enabled
        )


@dataclass(frozen=True, slots=True)
class TextStyleDelta:
    """保存 CSS 对各文字样式的显式开启、关闭或未声明状态。"""

    bold: bool | None = None
    italic: bool | None = None
    underline: bool | None = None
    strikethrough: bool | None = None
    superscript: bool | None = None
    subscript: bool | None = None

    def apply(self, base: TextStyle) -> TextStyle:
        """把当前声明覆盖到已解析的继承/标签样式。"""
        return TextStyle(
            bold=base.bold if self.bold is None else self.bold,
            italic=base.italic if self.italic is None else self.italic,
            underline=base.underline if self.underline is None else self.underline,
            strikethrough=base.strikethrough if self.strikethrough is None else self.strikethrough,
            superscript=base.superscript if self.superscript is None else self.superscript,
            subscript=base.subscript if self.subscript is None else self.subscript,
        )

    def is_empty(self) -> bool:
        """返回当前声明是否没有触及任何受支持样式。"""
        return all(
            value is None
            for value in (
                self.bold,
                self.italic,
                self.underline,
                self.strikethrough,
                self.superscript,
                self.subscript,
            )
        )


@dataclass(frozen=True, slots=True)
class ElementStyle:
    """保存元素最终文字样式、整树隐藏状态和继承可见性。"""

    text: TextStyle
    subtree_hidden: bool = False
    visibility_hidden: bool = False

    @property
    def hidden(self) -> bool:
        """返回当前元素是否因任一种受支持的隐藏语义而不可见。"""
        return self.subtree_hidden or self.visibility_hidden


@dataclass(slots=True)
class _SelectorCascade:
    """按 selector 聚合各属性最后一次声明及其源码顺序。"""

    priority: int
    declarations: dict[str, tuple[bool, int, bool]] = field(default_factory=dict)
    visibility: dict[str, tuple[bool, int, bool]] = field(default_factory=dict)

    def update(self, parsed: _ParsedDeclarations, order: int) -> None:
        """按 importance 和源码顺序更新同 selector 的逐属性级联结果。"""
        for name, (important, value) in parsed.text.items():
            current = self.declarations.get(name)
            if current is None or (important, order) >= current[:2]:
                self.declarations[name] = (important, order, value)
        for name, (important, value) in parsed.visibility.items():
            current = self.visibility.get(name)
            if current is None or (important, order) >= current[:2]:
                self.visibility[name] = (important, order, value)


@dataclass(frozen=True, slots=True)
class _ParsedDeclarations:
    """保存已投影 CSS 属性的 importance 与布尔值。"""

    text: dict[str, tuple[bool, bool]]
    visibility: dict[str, tuple[bool, bool]]


def _local_name(element: etree._Element) -> str:
    """返回 XHTML 元素不含命名空间的小写本地名。"""
    return etree.QName(element).localname.casefold()


def _numeric_font_weight(value: str) -> int | None:
    """在整数转换前解析 CSS Fonts 允许的一到一千字重。"""
    if not value.isascii() or not value.isdigit() or len(value) > 4:
        return None
    weight = int(value)
    return weight if 1 <= weight <= 1_000 else None


def _parse_declarations(value: str) -> _ParsedDeclarations:
    """从声明串逐属性提取字体语义、隐藏状态和 important 优先级。"""
    text: dict[str, tuple[bool, bool]] = {}
    visibility: dict[str, tuple[bool, bool]] = {}
    for raw_declaration in value.split(";"):
        if ":" not in raw_declaration:
            continue
        name, raw_value = raw_declaration.split(":", 1)
        name = name.strip().casefold()
        important_match = _CSS_IMPORTANT_RE.search(raw_value)
        important = important_match is not None
        normalized = raw_value[: important_match.start() if important_match is not None else None].strip().casefold()
        text_updates: dict[str, bool] = {}
        visibility_update: tuple[str, bool] | None = None
        if name == "font-weight":
            if normalized in {"bold", "bolder"}:
                text_updates["bold"] = True
            elif normalized in {"normal", "lighter"}:
                text_updates["bold"] = False
            elif (weight := _numeric_font_weight(normalized)) is not None:
                text_updates["bold"] = weight >= 600
        elif name == "font-style":
            text_updates["italic"] = normalized in {"italic", "oblique"}
        elif name in {"text-decoration", "text-decoration-line"}:
            if normalized == "none":
                text_updates["underline"] = False
                text_updates["strikethrough"] = False
            else:
                text_updates["underline"] = "underline" in normalized
                text_updates["strikethrough"] = "line-through" in normalized
        elif name == "vertical-align":
            text_updates["superscript"] = normalized in {"super", "text-top"}
            text_updates["subscript"] = normalized in {"sub", "text-bottom"}
        elif name == "display":
            visibility_update = ("display", normalized == "none")
        elif name == "visibility":
            if normalized in {"hidden", "collapse"}:
                visibility_update = ("visibility", True)
            elif normalized in {"visible", "initial"}:
                visibility_update = ("visibility", False)
        for field_name, field_value in text_updates.items():
            current = text.get(field_name)
            if current is None or important or not current[0]:
                text[field_name] = (important, field_value)
        if visibility_update is not None:
            field_name, field_value = visibility_update
            current = visibility.get(field_name)
            if current is None or important or not current[0]:
                visibility[field_name] = (important, field_value)
    return _ParsedDeclarations(text=text, visibility=visibility)


class EpubStylesheet:
    """保存按文档顺序解析的简单 tag/class CSS 规则。"""

    def __init__(self) -> None:
        """初始化按 tag、class 与 tag.class 分桶的 selector 索引。"""
        self._tag_cascades: dict[str, _SelectorCascade] = {}
        self._class_cascades: dict[str, _SelectorCascade] = {}
        self._tag_class_cascades: dict[tuple[str, str], _SelectorCascade] = {}
        self._source_order = 0

    def _selector_cascade(self, tag: str | None, class_name: str | None, priority: int) -> _SelectorCascade:
        """返回指定简单 selector 的聚合级联槽。"""
        if class_name is None:
            assert tag is not None
            return self._tag_cascades.setdefault(tag, _SelectorCascade(priority))
        if tag is None:
            return self._class_cascades.setdefault(class_name, _SelectorCascade(priority))
        return self._tag_class_cascades.setdefault((tag, class_name), _SelectorCascade(priority))

    def add(self, css: str) -> None:
        """追加一个 stylesheet 中受支持的简单 selector 规则。"""
        normalized_css = _CSS_COMMENT_RE.sub("", css)
        for chunk in normalized_css.split("}"):
            if "{" not in chunk:
                continue
            selectors, declarations = chunk.split("{", 1)
            parsed_declarations = _parse_declarations(declarations)
            if not parsed_declarations.text and not parsed_declarations.visibility:
                continue
            for selector in selectors.split(","):
                parsed = self._parse_selector(selector)
                if parsed is None:
                    continue
                tag, class_name, priority = parsed
                cascade = self._selector_cascade(tag, class_name, priority)
                cascade.update(parsed_declarations, self._source_order)
                self._source_order += 1

    @staticmethod
    def _parse_selector(selector: str) -> tuple[str | None, str | None, int] | None:
        """只接受 tag、.class 和 tag.class，拒绝组合器及伪类。"""
        normalized = selector.strip()
        if not normalized or any(token in normalized for token in (" ", ">", "+", "~", ":", "[", "#")):
            return None
        if "." in normalized:
            tag_text, class_name = normalized.split(".", 1)
            if not class_name or "." in class_name:
                return None
            tag = tag_text.casefold() or None
            return tag, class_name, 10 + (1 if tag else 0)
        return normalized.casefold(), None, 1

    def resolve(
        self,
        element: etree._Element,
        inherited: TextStyle,
        inherited_visibility_hidden: bool = False,
    ) -> ElementStyle:
        """计算元素的继承样式、标签默认样式、CSS 规则和 inline style。"""
        tag = _local_name(element)
        classes = frozenset((element.get("class") or "").split())
        tag_style = TextStyle(
            bold=tag in {"b", "strong"},
            italic=tag in {"cite", "dfn", "em", "i", "var"},
            strikethrough=tag in {"del", "s", "strike"},
            superscript=tag == "sup",
            subscript=tag == "sub",
        )
        style = inherited.merge(tag_style)
        subtree_hidden = element.get("hidden") is not None or (element.get("aria-hidden") or "").casefold() == "true"
        matching: list[_SelectorCascade] = []
        if cascade := self._tag_cascades.get(tag):
            matching.append(cascade)
        for class_name in classes:
            if cascade := self._class_cascades.get(class_name):
                matching.append(cascade)
            if cascade := self._tag_class_cascades.get((tag, class_name)):
                matching.append(cascade)

        inline = _parse_declarations(element.get("style") or "")
        resolved_values: dict[str, bool] = {}
        for name in _TEXT_STYLE_FIELDS:
            candidates = [
                (important, cascade.priority, order, value)
                for cascade in matching
                if (declaration := cascade.declarations.get(name)) is not None
                for important, order, value in (declaration,)
            ]
            if (inline_declaration := inline.text.get(name)) is not None:
                important, value = inline_declaration
                candidates.append((important, 1_000, self._source_order, value))
            if candidates:
                resolved_values[name] = max(candidates, key=lambda item: item[:3])[3]
        style = TextStyleDelta(**resolved_values).apply(style)

        resolved_visibility: dict[str, bool] = {}
        for name in _VISIBILITY_FIELDS:
            candidates = [
                (important, cascade.priority, order, value)
                for cascade in matching
                if (declaration := cascade.visibility.get(name)) is not None
                for important, order, value in (declaration,)
            ]
            if (inline_declaration := inline.visibility.get(name)) is not None:
                important, value = inline_declaration
                candidates.append((important, 1_000, self._source_order, value))
            if candidates:
                resolved_visibility[name] = max(candidates, key=lambda item: item[:3])[3]
        subtree_hidden = subtree_hidden or resolved_visibility.get("display", False)
        visibility_hidden = resolved_visibility.get("visibility", inherited_visibility_hidden)
        return ElementStyle(style, subtree_hidden, visibility_hidden)


__all__ = ["ElementStyle", "EpubStylesheet", "TextStyle", "TextStyleDelta"]
