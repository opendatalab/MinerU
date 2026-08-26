# Copyright (c) Opendatalab. All rights reserved.
"""把常用 Presentation MathML 结构转换为 LaTeX。"""

from __future__ import annotations

import re

from lxml import etree  # type: ignore[reportMissingImports]


_OPERATOR_MAP = {
    "−": "-",
    "×": r"\times ",
    "÷": r"\div ",
    "·": r"\cdot ",
    "±": r"\pm ",
    "∓": r"\mp ",
    "∞": r"\infty ",
    "≠": r"\ne ",
    "≤": r"\le ",
    "≥": r"\ge ",
    "≈": r"\approx ",
    "≡": r"\equiv ",
    "∈": r"\in ",
    "∉": r"\notin ",
    "⊂": r"\subset ",
    "⊆": r"\subseteq ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "∑": r"\sum ",
    "∏": r"\prod ",
    "∫": r"\int ",
    "∂": r"\partial ",
    "√": r"\sqrt{}",
    "→": r"\to ",
    "←": r"\leftarrow ",
    "↔": r"\leftrightarrow ",
}
_GREEK_MAP = {
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "δ": r"\delta ",
    "ε": r"\epsilon ",
    "θ": r"\theta ",
    "λ": r"\lambda ",
    "μ": r"\mu ",
    "π": r"\pi ",
    "σ": r"\sigma ",
    "φ": r"\phi ",
    "ω": r"\omega ",
    "Γ": r"\Gamma ",
    "Δ": r"\Delta ",
    "Θ": r"\Theta ",
    "Λ": r"\Lambda ",
    "Π": r"\Pi ",
    "Σ": r"\Sigma ",
    "Φ": r"\Phi ",
    "Ω": r"\Omega ",
}
_LATEX_ESCAPE_RE = re.compile(r"([#$%&_{}])")


def _local_name(element: etree._Element) -> str:
    """返回 XML 元素不含命名空间的本地名。"""
    return etree.QName(element).localname


def _escape_text(value: str) -> str:
    """转义进入 LaTeX 文本命令的保留字符。"""
    return _LATEX_ESCAPE_RE.sub(r"\\\1", value)


def _children(element: etree._Element) -> list[etree._Element]:
    """返回当前元素的全部普通 XML 子元素。"""
    return [child for child in element if isinstance(child.tag, str)]


def _join_children(element: etree._Element) -> str:
    """按文档顺序拼接所有子 MathML 节点。"""
    return "".join(_convert(child) for child in _children(element))


def _convert(element: etree._Element) -> str:
    """递归转换一个常用 MathML 节点，未知容器保留其可解析子项。"""
    name = _local_name(element)
    children = _children(element)
    text = (element.text or "").strip()
    if name in {"math", "mrow", "mstyle", "mpadded", "mphantom", "semantics"}:
        return _join_children(element)
    if name in {"mi", "mn"}:
        return _GREEK_MAP.get(text, text)
    if name == "mo":
        return _OPERATOR_MAP.get(text, text)
    if name == "mtext":
        return rf"\text{{{_escape_text(text)}}}"
    if name == "mspace":
        return r"\,"
    if name == "mfrac" and len(children) >= 2:
        return rf"\frac{{{_convert(children[0])}}}{{{_convert(children[1])}}}"
    if name == "msqrt":
        return rf"\sqrt{{{_join_children(element)}}}"
    if name == "mroot" and len(children) >= 2:
        return rf"\sqrt[{_convert(children[1])}]{{{_convert(children[0])}}}"
    if name == "msup" and len(children) >= 2:
        return rf"{{{_convert(children[0])}}}^{{{_convert(children[1])}}}"
    if name == "msub" and len(children) >= 2:
        return rf"{{{_convert(children[0])}}}_{{{_convert(children[1])}}}"
    if name == "msubsup" and len(children) >= 3:
        return rf"{{{_convert(children[0])}}}_{{{_convert(children[1])}}}^{{{_convert(children[2])}}}"
    if name == "mover" and len(children) >= 2:
        return rf"\overset{{{_convert(children[1])}}}{{{_convert(children[0])}}}"
    if name == "munder" and len(children) >= 2:
        return rf"\underset{{{_convert(children[1])}}}{{{_convert(children[0])}}}"
    if name == "munderover" and len(children) >= 3:
        base = _convert(children[0])
        return rf"\underset{{{_convert(children[1])}}}{{\overset{{{_convert(children[2])}}}{{{base}}}}}"
    if name == "mfenced":
        opening = element.get("open", "(")
        closing = element.get("close", ")")
        separators = element.get("separators", ",") or ","
        values = [_convert(child) for child in children]
        return rf"\left{opening}{separators[0].join(values)}\right{closing}"
    if name == "mtable":
        rows = [_convert(child) for child in children if _local_name(child) in {"mtr", "mlabeledtr"}]
        return r"\begin{matrix}" + r" \\ ".join(rows) + r"\end{matrix}"
    if name in {"mtr", "mlabeledtr"}:
        return " & ".join(_convert(child) for child in children)
    if name == "mtd":
        return _join_children(element)
    if name == "annotation" and "tex" in (element.get("encoding", "").casefold()):
        return text
    return _join_children(element) or text


def mathml_to_latex(math_element: etree._Element) -> str | None:
    """转换 MathML 根节点，并优先采用生产者保留的 TeX annotation。"""
    for annotation in math_element.iter():
        if _local_name(annotation) != "annotation":
            continue
        encoding = (annotation.get("encoding") or "").casefold()
        if "tex" in encoding and (annotation.text or "").strip():
            return (annotation.text or "").strip()
    latex = _convert(math_element).strip()
    return latex or None


__all__ = ["mathml_to_latex"]
