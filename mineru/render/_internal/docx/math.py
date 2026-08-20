# Copyright (c) Opendatalab. All rights reserved.
"""DOCX renderer 使用的 LaTeX 与 OMML 转换。"""

from __future__ import annotations

from latex2mathml.converter import convert as latex_to_mathml
from lxml import etree
from mathml2omml import convert as mathml_to_omml
import re

_OFFICE_MATH_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_EMPTY_SCRIPT_RE = re.compile(r"(?:_\s*\{\s*\}|\^\s*\{\s*\})")
_GENFRAC_COMMAND = r"\genfrac"
_GENFRAC_ARGUMENT_COUNT = 6


class DocxFormulaError(ValueError):
    """表示 LaTeX 无法转换为可插入 DOCX 的 OMML。"""


def split_formula_tag(content: str) -> tuple[str, str | None]:
    """剥离公式末尾括号平衡的 ``\\tag{...}``，并返回正文与编号。"""
    stripped_end = len(content.rstrip())
    if stripped_end == 0 or content[stripped_end - 1] != "}":
        return content, None

    search_end = stripped_end
    while (tag_start := content.rfind(r"\tag", 0, search_end)) >= 0:
        if not _is_escaped_command(content, tag_start):
            opening_brace = _find_tag_opening_brace(content, tag_start, stripped_end)
            if opening_brace is not None:
                closing_brace = _find_balanced_closing_brace(content, opening_brace, stripped_end)
                if closing_brace == stripped_end - 1:
                    formula = content[:tag_start].rstrip()
                    tag = content[opening_brace + 1 : closing_brace].strip()
                    return formula, tag
        search_end = tag_start

    return content, None


def latex_to_omml(latex: str, *, display: bool) -> etree._Element:
    """把 LaTeX 转换为带命名空间、可直接插入 DOCX 的 OMML 节点。"""
    try:
        normalized_latex = _normalize_supported_genfrac(latex)
        normalized_latex = _EMPTY_SCRIPT_RE.sub("", normalized_latex)
        mathml = latex_to_mathml(normalized_latex, display="block" if display else "inline")
        omml_xml = _repair_mathml2omml_xml(mathml_to_omml(mathml))
        equation = _parse_omml(omml_xml)
        _normalize_omml(equation)
        if not display:
            return equation

        paragraph = etree.Element(
            etree.QName(_OFFICE_MATH_NAMESPACE, "oMathPara"),
            nsmap={"m": _OFFICE_MATH_NAMESPACE},
        )
        paragraph.append(equation)
        return paragraph
    except Exception as exc:
        raise DocxFormulaError(f"LaTeX 公式无法转换为 OMML: {latex!r}") from exc


def _normalize_supported_genfrac(content: str) -> str:
    """把规范无横线 genfrac 转为 latex2mathml 支持的双行 matrix。"""
    chunks: list[str] = []
    preserved_start = 0
    search_start = 0
    while (command_start := content.find(_GENFRAC_COMMAND, search_start)) >= 0:
        command_end = command_start + len(_GENFRAC_COMMAND)
        if _is_escaped_command(content, command_start):
            search_start = command_end
            continue

        parsed = _parse_braced_arguments(content, command_end, _GENFRAC_ARGUMENT_COUNT)
        if parsed is None:
            search_start = command_end
            continue
        arguments, expression_end = parsed
        left_delimiter, right_delimiter, thickness, math_style, numerator, denominator = arguments
        if left_delimiter.strip() or right_delimiter.strip() or thickness.strip() != "0pt" or math_style.strip():
            search_start = expression_end
            continue

        normalized_numerator = _normalize_supported_genfrac(numerator)
        normalized_denominator = _normalize_supported_genfrac(denominator)
        replacement = (
            rf"\begin{{matrix}}{{{normalized_numerator}}}"
            r"\\"
            rf"{{{normalized_denominator}}}\end{{matrix}}"
        )
        chunks.append(content[preserved_start:command_start])
        chunks.append(replacement)
        preserved_start = expression_end
        search_start = expression_end

    if not chunks:
        return content
    chunks.append(content[preserved_start:])
    return "".join(chunks)


def _parse_braced_arguments(
    content: str,
    cursor: int,
    count: int,
) -> tuple[tuple[str, ...], int] | None:
    """从指定位置读取固定数量的平衡花括号参数，并返回参数与结束位置。"""
    arguments: list[str] = []
    content_end = len(content)
    for _ in range(count):
        while cursor < content_end and content[cursor].isspace():
            cursor += 1
        if cursor >= content_end or content[cursor] != "{":
            return None
        closing_brace = _find_balanced_closing_brace(content, cursor, content_end)
        if closing_brace is None:
            return None
        arguments.append(content[cursor + 1 : closing_brace])
        cursor = closing_brace + 1
    return tuple(arguments), cursor


def _repair_mathml2omml_xml(omml_xml: str) -> str:
    """修复 mathml2omml 0.0.2 对 groupChrPr 写出的错误闭合标签。"""
    return omml_xml.replace(
        "</m:groupChr><m:e>",
        "</m:groupChrPr><m:e>",
    )


def _normalize_omml(equation: etree._Element) -> None:
    """补齐 Word 要求的根号属性，并隐藏无显式底数的脚本占位框。"""
    namespace = f"{{{_OFFICE_MATH_NAMESPACE}}}"
    for radical in equation.findall(f".//{namespace}rad"):
        if radical.find(f"{namespace}deg") is not None:
            continue
        properties = etree.Element(etree.QName(_OFFICE_MATH_NAMESPACE, "radPr"))
        degree_hidden = etree.SubElement(
            properties,
            etree.QName(_OFFICE_MATH_NAMESPACE, "degHide"),
        )
        degree_hidden.set(etree.QName(_OFFICE_MATH_NAMESPACE, "val"), "1")
        degree = etree.Element(etree.QName(_OFFICE_MATH_NAMESPACE, "deg"))
        radical.insert(0, properties)
        radical.insert(1, degree)

    for script_name in ("sSup", "sSub", "sSubSup"):
        for script in equation.findall(f".//{namespace}{script_name}"):
            base = script.find(f"{namespace}e")
            if base is None or "".join(base.itertext()).strip():
                continue
            text = base.find(f".//{namespace}t")
            if text is None:
                run = etree.SubElement(base, etree.QName(_OFFICE_MATH_NAMESPACE, "r"))
                text = etree.SubElement(run, etree.QName(_OFFICE_MATH_NAMESPACE, "t"))
            text.text = "\u200b"


def _is_escaped_command(content: str, command_start: int) -> bool:
    """判断命令起始反斜杠是否被前一个反斜杠转义。"""
    preceding_backslashes = 0
    cursor = command_start - 1
    while cursor >= 0 and content[cursor] == "\\":
        preceding_backslashes += 1
        cursor -= 1
    return preceding_backslashes % 2 == 1


def _find_tag_opening_brace(content: str, tag_start: int, content_end: int) -> int | None:
    """查找 tag 命令允许空白后的左花括号。"""
    cursor = tag_start + len(r"\tag")
    while cursor < content_end and content[cursor].isspace():
        cursor += 1
    if cursor >= content_end or content[cursor] != "{":
        return None
    return cursor


def _find_balanced_closing_brace(content: str, opening_brace: int, content_end: int) -> int | None:
    """查找与 tag 左花括号配对的右花括号，并忽略转义花括号。"""
    depth = 0
    for cursor in range(opening_brace, content_end):
        character = content[cursor]
        if character not in "{}" or _is_escaped_character(content, cursor):
            continue
        depth += 1 if character == "{" else -1
        if depth == 0:
            return cursor
        if depth < 0:
            return None
    return None


def _is_escaped_character(content: str, position: int) -> bool:
    """判断指定字符前是否存在奇数个连续反斜杠。"""
    preceding_backslashes = 0
    cursor = position - 1
    while cursor >= 0 and content[cursor] == "\\":
        preceding_backslashes += 1
        cursor -= 1
    return preceding_backslashes % 2 == 1


def _parse_omml(omml_xml: str) -> etree._Element:
    """为第三方库返回的未绑定 ``m`` 前缀补充命名空间并解析节点。"""
    wrapper_xml = f'<docx-math-root xmlns:m="{_OFFICE_MATH_NAMESPACE}">{omml_xml}</docx-math-root>'
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    wrapper = etree.fromstring(wrapper_xml.encode("utf-8"), parser=parser)
    if len(wrapper) != 1:
        raise ValueError("OMML 转换结果必须只包含一个根节点")

    equation = wrapper[0]
    expected_tag = etree.QName(_OFFICE_MATH_NAMESPACE, "oMath")
    if equation.tag != expected_tag:
        raise ValueError("OMML 转换结果的根节点必须是 m:oMath")
    wrapper.remove(equation)
    return equation


__all__ = ["DocxFormulaError", "latex_to_omml", "split_formula_tag"]
