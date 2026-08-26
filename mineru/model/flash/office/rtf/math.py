# Copyright (c) Opendatalab. All rights reserved.
"""把 RTF Office Math destination 规范化为 OMML 并复用现有 LaTeX 转换器。"""

from __future__ import annotations

from dataclasses import dataclass, field

from lxml import etree  # type: ignore[reportAttributeAccessIssue]
from loguru import logger

from ..docx.tools.math.omml import oMath2Latex
from .lexer import RtfClose, RtfControlWord, RtfHexByte, RtfLexer, RtfOpen, RtfTextBytes

_MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_MATH_PREFIX = f"{{{_MATH_NS}}}"

# RTF Office Math control words为 OMML local name 增加一个 m 前缀；映射保留 XML 所需大小写。
_OMML_LOCAL_NAMES = (
    "acc",
    "accPr",
    "aln",
    "alnScr",
    "argPr",
    "argSz",
    "bar",
    "barPr",
    "baseJc",
    "begChr",
    "borderBox",
    "borderBoxPr",
    "box",
    "boxPr",
    "brk",
    "brkBin",
    "brkBinSub",
    "cGp",
    "cGpRule",
    "chr",
    "count",
    "cSp",
    "ctrlPr",
    "d",
    "defJc",
    "deg",
    "degHide",
    "den",
    "diff",
    "dispDef",
    "dPr",
    "e",
    "endChr",
    "eqArr",
    "eqArrPr",
    "f",
    "fName",
    "fPr",
    "func",
    "funcPr",
    "groupChr",
    "groupChrPr",
    "grow",
    "hideBot",
    "hideLeft",
    "hideRight",
    "hideTop",
    "interSp",
    "intLim",
    "intraSp",
    "jc",
    "lim",
    "limLoc",
    "limLow",
    "limLowPr",
    "limUpp",
    "limUppPr",
    "lit",
    "lMargin",
    "m",
    "mathFont",
    "mathPr",
    "maxDist",
    "mc",
    "mcJc",
    "mcPr",
    "mcs",
    "mPr",
    "mr",
    "nary",
    "naryLim",
    "naryPr",
    "noBreak",
    "nor",
    "num",
    "objDist",
    "oMath",
    "oMathPara",
    "oMathParaPr",
    "opEmu",
    "phant",
    "phantPr",
    "plcHide",
    "pos",
    "postSp",
    "preSp",
    "r",
    "rad",
    "radPr",
    "rMargin",
    "rPr",
    "rSp",
    "rSpRule",
    "scr",
    "sepChr",
    "show",
    "shp",
    "smallFrac",
    "sPre",
    "sPrePr",
    "sSub",
    "sSubPr",
    "sSubSup",
    "sSubSupPr",
    "sSup",
    "sSupPr",
    "strikeBLTR",
    "strikeH",
    "strikeTLBR",
    "strikeV",
    "sty",
    "sub",
    "subHide",
    "sup",
    "supHide",
    "t",
    "transp",
    "type",
    "vertJc",
    "wrapIndent",
    "wrapRight",
    "zeroAsc",
    "zeroDesc",
    "zeroWid",
)
_CONTROL_TO_LOCAL = {f"m{name}".lower(): name for name in _OMML_LOCAL_NAMES}
_PROPERTY_LOCAL_NAMES = {
    "baseJc",
    "begChr",
    "chr",
    "degHide",
    "endChr",
    "grow",
    "jc",
    "limLoc",
    "nor",
    "pos",
    "scr",
    "sepChr",
    "show",
    "sty",
    "subHide",
    "supHide",
    "type",
}
_SCR_VALUES = {
    0: "roman",
    1: "script",
    2: "fraktur",
    3: "double-struck",
    4: "sans-serif",
    5: "monospace",
}


@dataclass(slots=True)
class _MathGroup:
    """保存一个 RTF group 对应的 Office Math 局部树。"""

    local: str | None = None
    text: list[str] = field(default_factory=list)
    controls: list[tuple[str, int | None]] = field(default_factory=list)
    children: list[_MathGroup] = field(default_factory=list)


def _append_unicode(group: _MathGroup, value: int | None, surrogate: list[int | None]) -> int:
    """解码有符号 UTF-16 code unit，并返回需要跳过的 fallback 字符数占位。"""
    if value is None:
        return 0
    unit = value + 65536 if value < 0 else value
    unit &= 0xFFFF
    pending = surrogate[0]
    if 0xD800 <= unit <= 0xDBFF:
        surrogate[0] = unit
        return 1
    if 0xDC00 <= unit <= 0xDFFF and pending is not None:
        codepoint = 0x10000 + ((pending - 0xD800) << 10) + (unit - 0xDC00)
        group.text.append(chr(codepoint))
        surrogate[0] = None
        return 1
    if pending is not None:
        group.text.append("\ufffd")
        surrogate[0] = None
    group.text.append(chr(unit) if not 0xD800 <= unit <= 0xDFFF else "\ufffd")
    return 1


def _parse_math_groups(data: bytes, encoding: str) -> _MathGroup:
    """把 math destination token 流构造成与 RTF group 同构的轻量树。"""
    root = _MathGroup(local="math")
    stack = [root]
    uc_skip = 1
    fallback_skip = 0
    surrogate: list[int | None] = [None]
    for token in RtfLexer(data):
        if isinstance(token, RtfOpen):
            group = _MathGroup()
            stack[-1].children.append(group)
            stack.append(group)
            continue
        if isinstance(token, RtfClose):
            if len(stack) > 1:
                stack.pop()
            continue
        group = stack[-1]
        if isinstance(token, RtfControlWord):
            if token.name == "uc":
                uc_skip = max(token.param or 0, 0)
                continue
            if token.name == "u":
                _append_unicode(group, token.param, surrogate)
                fallback_skip = uc_skip
                continue
            local = _CONTROL_TO_LOCAL.get(token.name)
            if local is not None:
                if group.local is None:
                    group.local = local
                else:
                    group.controls.append((local, token.param))
            continue
        if isinstance(token, RtfHexByte):
            if fallback_skip:
                fallback_skip -= 1
                continue
            group.text.append(bytes([token.value]).decode(encoding, errors="replace"))
            continue
        if isinstance(token, RtfTextBytes):
            raw = token.data.replace(b"\r", b"").replace(b"\n", b"")
            if fallback_skip:
                skipped = min(fallback_skip, len(raw))
                raw = raw[skipped:]
                fallback_skip -= skipped
            if raw:
                group.text.append(raw.decode(encoding, errors="replace"))
    if surrogate[0] is not None:
        stack[-1].text.append("\ufffd")
    return root


def _value_text(group: _MathGroup) -> str:
    """返回属性 group 的规范化直接文本。"""
    return "".join(group.text).strip()


def _build_omml_element(group: _MathGroup) -> etree._Element | None:
    """把一个已命名 math group 转换为规范 OMML element。"""
    if group.local is None or group.local == "math":
        return None
    element = etree.Element(f"{_MATH_PREFIX}{group.local}", nsmap={"m": _MATH_NS})
    if group.local == "r":
        properties: list[tuple[str, str]] = []
        for local, param in group.controls:
            if local == "scr" and param is not None:
                properties.append((local, _SCR_VALUES.get(param, str(param))))
            elif local in {"sty", "nor"}:
                properties.append((local, str(param if param is not None else 1)))
        if properties:
            r_pr = etree.SubElement(element, f"{_MATH_PREFIX}rPr")
            for local, value in properties:
                prop = etree.SubElement(r_pr, f"{_MATH_PREFIX}{local}")
                prop.set(f"{_MATH_PREFIX}val", value)
        text = "".join(group.text)
        if text:
            etree.SubElement(element, f"{_MATH_PREFIX}t").text = text
    elif group.local in _PROPERTY_LOCAL_NAMES:
        value = _value_text(group)
        if value:
            element.set(f"{_MATH_PREFIX}val", value)
        elif group.controls:
            element.set(f"{_MATH_PREFIX}val", str(group.controls[-1][1] or 0))
        else:
            element.set(f"{_MATH_PREFIX}val", "1")
    elif group.text:
        text = "".join(group.text)
        run = etree.SubElement(element, f"{_MATH_PREFIX}r")
        etree.SubElement(run, f"{_MATH_PREFIX}t").text = text

    for child in group.children:
        child_element = _build_omml_element(child)
        if child_element is not None:
            element.append(child_element)
    return element


def _find_groups(group: _MathGroup, local: str) -> list[_MathGroup]:
    """按深度优先顺序收集指定 OMML local name 的 group。"""
    result: list[_MathGroup] = []
    pending = [group]
    while pending:
        current = pending.pop()
        if current.local == local:
            result.append(current)
        pending.extend(reversed(current.children))
    return result


def parse_rtf_math(data: bytes, *, encoding: str = "cp1252") -> tuple[list[str], bool]:
    """解析一个 ``mmath`` group，返回 LaTeX 公式列表和行间公式标记。"""
    try:
        root = _parse_math_groups(data, encoding)
        paragraphs = _find_groups(root, "oMathPara")
        equation_groups: list[_MathGroup] = []
        if paragraphs:
            for paragraph in paragraphs:
                equation_groups.extend(_find_groups(paragraph, "oMath"))
        else:
            equation_groups = _find_groups(root, "oMath")
        formulas: list[str] = []
        for equation_group in equation_groups:
            element = _build_omml_element(equation_group)
            if element is None:
                continue
            latex = str(oMath2Latex(element)).strip()
            if latex:
                formulas.append(latex)
        return formulas, bool(paragraphs)
    except Exception as exc:
        logger.warning("RTF Office Math fallback: {}", exc)
        return [], False


__all__ = ["parse_rtf_math"]
