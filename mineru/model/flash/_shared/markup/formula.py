# Copyright (c) Opendatalab. All rights reserved.
"""集中识别静态 HTML/XHTML 公式并归一化为裸 LaTeX。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, TypeAlias

from lxml import etree  # type: ignore[reportMissingImports]

from ..mathml import mathml_to_latex


FormulaDisplay: TypeAlias = Literal["inline", "block"]
FormulaSourceKind: TypeAlias = Literal[
    "mineru_latex",
    "mathml_tex_annotation",
    "tex_script",
    "data_tex",
    "legacy_mineru_text",
    "mathml_alttext",
    "embedded_mathml",
    "presentation_mathml",
]

_MATH_SCRIPT_TYPE_RE = re.compile(r"^math/(?:tex|latex)(?:\s*;\s*mode\s*=\s*display)?$", re.IGNORECASE)
_DISPLAY_FORMULA_TOKENS = frozenset({"display", "katex-display", "math-display", "mathjax-display", "mineru-math--block"})
_TEX_ANNOTATION_ENCODINGS = frozenset(
    {
        "application/tex",
        "application/x-latex",
        "application/x-tex",
        "text/latex",
        "text/tex",
    }
)
_PRESENTATION_MATHML_TAGS = frozenset(
    {
        "annotation",
        "annotation-xml",
        "math",
        "mfenced",
        "mfrac",
        "mi",
        "mlabeledtr",
        "mn",
        "mo",
        "mover",
        "mpadded",
        "mphantom",
        "mroot",
        "mrow",
        "mspace",
        "msqrt",
        "mstyle",
        "msub",
        "msubsup",
        "msup",
        "mtable",
        "mtd",
        "mtext",
        "mtr",
        "munder",
        "munderover",
        "semantics",
    }
)
_PRESENTATION_ARITIES: dict[str, int] = {
    "mfrac": 2,
    "mover": 2,
    "mroot": 2,
    "msub": 2,
    "msubsup": 3,
    "msup": 2,
    "munder": 2,
    "munderover": 3,
}


@dataclass(frozen=True, slots=True)
class FormulaExtraction:
    """保存一次成功公式识别的裸 LaTeX、显示方式和来源。"""

    latex: str
    display: FormulaDisplay
    source_kind: FormulaSourceKind


def extract_formula(element: etree._Element) -> FormulaExtraction | None:
    """按固定优先级从公式节点或常见公式包装器中提取裸 LaTeX。"""
    display = _formula_display(element)

    if latex := _subtree_attribute(element, "data-mineru-latex"):
        return FormulaExtraction(latex, display, "mineru_latex")

    if latex := _tex_annotation(element):
        return FormulaExtraction(latex, display, "mathml_tex_annotation")

    if latex := _tex_script(element):
        return FormulaExtraction(latex, _script_display(element, display), "tex_script")

    if latex := _data_formula(element):
        return FormulaExtraction(latex, display, "data_tex")

    if "mineru-math" in _class_tokens(element):
        latex = strip_formula_delimiters("".join(element.itertext()))
        if latex:
            return FormulaExtraction(latex, display, "legacy_mineru_text")

    if latex := _mathml_alttext(element):
        return FormulaExtraction(latex, display, "mathml_alttext")

    name = _local_name(element)
    if name != "math":
        math_element = next(
            (child for child in element.iterdescendants() if isinstance(child.tag, str) and _local_name(child) == "math"),
            None,
        )
        if math_element is not None and (latex := _presentation_mathml(math_element)):
            return FormulaExtraction(latex, _formula_display(math_element, display), "embedded_mathml")
        return None

    if latex := _presentation_mathml(element):
        return FormulaExtraction(latex, display, "presentation_mathml")
    return None


def strip_formula_delimiters(value: str) -> str:
    """重复移除常见外层 TeX 定界符，不改动公式内部结构。"""
    normalized = value.strip()
    pairs = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
    for _ in range(4):
        stripped = normalized
        for opening, closing in pairs:
            has_outer_pair = normalized.startswith(opening) and normalized.endswith(closing)
            if has_outer_pair and len(normalized) > len(opening) + len(closing):
                stripped = normalized[len(opening) : -len(closing)].strip()
                break
        if stripped == normalized:
            break
        normalized = stripped
    return normalized


def is_tex_script(element: etree._Element) -> bool:
    """判断元素是否是受支持的静态 TeX/LaTeX script carrier。"""
    return _local_name(element) == "script" and _MATH_SCRIPT_TYPE_RE.fullmatch((element.get("type") or "").strip()) is not None


def _normalized_attribute(element: etree._Element, name: str) -> str | None:
    """读取并去除外层定界符，空值统一返回 None。"""
    value = strip_formula_delimiters(element.get(name) or "")
    return value or None


def _subtree_attribute(element: etree._Element, name: str) -> str | None:
    """按文档顺序返回自身或后代首个非空规范公式属性。"""
    for candidate in [element, *element.iterdescendants()]:
        if isinstance(candidate.tag, str) and (value := _normalized_attribute(candidate, name)):
            return value
    return None


def _tex_annotation(element: etree._Element) -> str | None:
    """返回节点子树中首个受支持 TeX annotation。"""
    candidates = [element, *element.iterdescendants()]
    for candidate in candidates:
        if not isinstance(candidate.tag, str) or _local_name(candidate) != "annotation":
            continue
        encoding = (candidate.get("encoding") or "").strip().casefold()
        if encoding not in _TEX_ANNOTATION_ENCODINGS:
            continue
        latex = strip_formula_delimiters("".join(candidate.itertext()))
        if latex:
            return latex
    return None


def _tex_script(element: etree._Element) -> str | None:
    """返回当前节点或后代中首个受支持 TeX script 内容。"""
    candidates = [element, *element.iterdescendants()]
    for candidate in candidates:
        if not isinstance(candidate.tag, str) or not is_tex_script(candidate):
            continue
        latex = strip_formula_delimiters("".join(candidate.itertext()))
        if latex:
            return latex
    return None


def _data_formula(element: etree._Element) -> str | None:
    """按 data-tex、data-expr 顺序读取节点及其后代。"""
    for candidate in [element, *element.iterdescendants()]:
        if not isinstance(candidate.tag, str):
            continue
        for attribute in ("data-tex", "data-expr"):
            if latex := _normalized_attribute(candidate, attribute):
                return latex
    return None


def _mathml_alttext(element: etree._Element) -> str | None:
    """返回首个 MathML alttext 中声明的 LaTeX。"""
    for candidate in [element, *element.iterdescendants()]:
        if not isinstance(candidate.tag, str) or _local_name(candidate) != "math":
            continue
        if latex := _normalized_attribute(candidate, "alttext"):
            return latex
    return None


def _presentation_mathml(element: etree._Element) -> str | None:
    """尽力把受支持 Presentation MathML 转换为裸 LaTeX。"""
    if not _is_supported_presentation_mathml(element):
        return None
    latex = mathml_to_latex(element)
    normalized = strip_formula_delimiters(latex or "")
    return normalized or None


def _is_supported_presentation_mathml(element: etree._Element) -> bool:
    """拒绝未知节点和畸形固定元数结构，避免把普通 XML 文本冒充为 LaTeX。"""
    if _local_name(element) != "math":
        return False
    for candidate in [element, *element.iterdescendants()]:
        if not isinstance(candidate.tag, str):
            continue
        name = _local_name(candidate)
        if name not in _PRESENTATION_MATHML_TAGS:
            return False
        children = [child for child in candidate if isinstance(child.tag, str)]
        if name in _PRESENTATION_ARITIES and len(children) != _PRESENTATION_ARITIES[name]:
            return False
        if name in {"mi", "mn", "mo", "mtext", "mspace"} and children:
            return False
        if name == "semantics" and (not children or _local_name(children[0]) in {"annotation", "annotation-xml"}):
            return False
    return True


def _formula_display(element: etree._Element, default: FormulaDisplay = "inline") -> FormulaDisplay:
    """从显式属性、MathML display 和完整 class token 推断显示方式。"""
    declared = (element.get("data-formula-display") or "").strip().casefold()
    if declared in {"inline", "block"}:
        return declared  # type: ignore[return-value]
    if (element.get("display") or "").strip().casefold() == "block":
        return "block"
    classes = frozenset((element.get("class") or "").casefold().split())
    return "block" if classes & _DISPLAY_FORMULA_TOKENS else default


def _script_display(element: etree._Element, default: FormulaDisplay) -> FormulaDisplay:
    """在脚本来源中识别 mode=display，并保留外层显式显示方式。"""
    for candidate in [element, *element.iterdescendants()]:
        if not isinstance(candidate.tag, str) or not is_tex_script(candidate):
            continue
        if "mode=display" in re.sub(r"\s+", "", (candidate.get("type") or "").casefold()):
            return "block"
        break
    return default


def _local_name(element: etree._Element) -> str:
    """返回不含命名空间的小写本地标签名。"""
    return etree.QName(element).localname.casefold()


def _class_tokens(element: etree._Element) -> frozenset[str]:
    """返回不做任意 substring 匹配的完整 class token。"""
    return frozenset((element.get("class") or "").casefold().split())


__all__ = [
    "FormulaDisplay",
    "FormulaExtraction",
    "FormulaSourceKind",
    "extract_formula",
    "is_tex_script",
    "strip_formula_delimiters",
]
