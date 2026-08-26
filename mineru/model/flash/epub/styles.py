# Copyright (c) Opendatalab. All rights reserved.
"""解析 EPUB XHTML 使用的有限语义 CSS 子集。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import etree  # type: ignore[reportMissingImports]


_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_TEXT_STYLE_FIELDS = ("bold", "italic", "underline", "strikethrough", "superscript", "subscript")


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
    """保存元素最终文字样式和可见性。"""

    text: TextStyle
    hidden: bool = False


@dataclass(slots=True)
class _SelectorCascade:
    """按 selector 聚合各属性最后一次声明及其源码顺序。"""

    priority: int
    declarations: dict[str, tuple[int, bool]] = field(default_factory=dict)
    hidden: tuple[int, bool] | None = None

    def update(self, style: TextStyleDelta, hidden: bool | None, order: int) -> None:
        """用同 selector 的较新声明更新逐属性级联结果。"""
        for name in _TEXT_STYLE_FIELDS:
            value = getattr(style, name)
            if value is not None:
                self.declarations[name] = (order, value)
        if hidden is not None:
            self.hidden = (order, hidden)


def _local_name(element: etree._Element) -> str:
    """返回 XHTML 元素不含命名空间的小写本地名。"""
    return etree.QName(element).localname.casefold()


def _parse_declarations(value: str) -> tuple[TextStyleDelta, bool | None]:
    """从声明串提取字体语义和 display/visibility 隐藏状态。"""
    bold = italic = underline = strikethrough = superscript = subscript = None
    hidden: bool | None = None
    for raw_declaration in value.split(";"):
        if ":" not in raw_declaration:
            continue
        name, raw_value = raw_declaration.split(":", 1)
        name = name.strip().casefold()
        normalized = raw_value.split("!important", 1)[0].strip().casefold()
        if name == "font-weight":
            bold = normalized in {"bold", "bolder"} or normalized.isdigit() and int(normalized) >= 600
        elif name == "font-style":
            italic = normalized in {"italic", "oblique"}
        elif name in {"text-decoration", "text-decoration-line"}:
            if normalized == "none":
                underline = False
                strikethrough = False
            else:
                underline = "underline" in normalized
                strikethrough = "line-through" in normalized
        elif name == "vertical-align":
            superscript = normalized in {"super", "text-top"}
            subscript = normalized in {"sub", "text-bottom"}
        elif name == "display":
            hidden = normalized == "none"
        elif name == "visibility":
            hidden = normalized in {"hidden", "collapse"}
    return TextStyleDelta(bold, italic, underline, strikethrough, superscript, subscript), hidden


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
            style, hidden = _parse_declarations(declarations)
            if style.is_empty() and hidden is None:
                continue
            for selector in selectors.split(","):
                parsed = self._parse_selector(selector)
                if parsed is None:
                    continue
                tag, class_name, priority = parsed
                cascade = self._selector_cascade(tag, class_name, priority)
                cascade.update(style, hidden, self._source_order)
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

    def resolve(self, element: etree._Element, inherited: TextStyle) -> ElementStyle:
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
        hidden = element.get("hidden") is not None or (element.get("aria-hidden") or "").casefold() == "true"
        matching: list[_SelectorCascade] = []
        if cascade := self._tag_cascades.get(tag):
            matching.append(cascade)
        for class_name in classes:
            if cascade := self._class_cascades.get(class_name):
                matching.append(cascade)
            if cascade := self._tag_class_cascades.get((tag, class_name)):
                matching.append(cascade)

        resolved_values: dict[str, bool] = {}
        for name in _TEXT_STYLE_FIELDS:
            candidates = [
                (cascade.priority, order, value)
                for cascade in matching
                if (declaration := cascade.declarations.get(name)) is not None
                for order, value in (declaration,)
            ]
            if candidates:
                resolved_values[name] = max(candidates, key=lambda item: (item[0], item[1]))[2]
        style = TextStyleDelta(**resolved_values).apply(style)

        hidden_candidates = [
            (cascade.priority, order, value)
            for cascade in matching
            if cascade.hidden is not None
            for order, value in (cascade.hidden,)
        ]
        if hidden_candidates:
            hidden = max(hidden_candidates, key=lambda item: (item[0], item[1]))[2]
        inline_style, inline_hidden = _parse_declarations(element.get("style") or "")
        style = inline_style.apply(style)
        if inline_hidden is not None:
            hidden = inline_hidden
        return ElementStyle(style, hidden)


__all__ = ["ElementStyle", "EpubStylesheet", "TextStyle", "TextStyleDelta"]
