# Copyright (c) Opendatalab. All rights reserved.

"""安全解析 Equation Editor 3.x 的 Equation Native/MTEF v3 公式。"""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct

from loguru import logger

from .limits import MAX_RECORD_DEPTH, MAX_RECORDS
from .ole import BoundedOleReader

_OBJECT_POOL_EQUATION_RE = re.compile(
    r"^ObjectPool/_([0-9]+)/Equation Native$",
    re.IGNORECASE,
)

_XF_NULL = 0x10
_XF_RULER = 0x20
_XF_EMBELL = 0x20
_XF_LSPACE = 0x40
_XF_LMOVE = 0x80


class _MtefError(ValueError):
    """MTEF 输入不完整、不受支持或超过安全限制。"""


@dataclass(frozen=True, slots=True)
class _Node:
    """MTEF 解析后的最小公式语法节点。"""

    kind: str
    value: object | None = None
    children: tuple[_Node, ...] = ()


class _MtefReader:
    """有界读取 MTEF v3 record tree。"""

    def __init__(self, data: bytes) -> None:
        """校验五字节 MTEF v3 头并初始化游标。"""

        if len(data) < 5 or data[0] != 3:
            raise _MtefError("Equation Native does not contain MTEF v3")
        self.data = data
        self.pos = 5
        self.depth = 0
        self.records = 0

    def _charge(self) -> None:
        """计入一条 record，超过共享上限时拒绝对象。"""

        self.records += 1
        if self.records > MAX_RECORDS:
            raise _MtefError("MTEF record count exceeds the safety limit")

    def _u8(self) -> int:
        """有界读取一个无符号字节。"""

        if self.pos >= len(self.data):
            raise _MtefError("MTEF record is truncated")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def _u16(self) -> int:
        """有界读取小端 u16。"""

        if self.pos + 2 > len(self.data):
            raise _MtefError("MTEF u16 is truncated")
        value = int(struct.unpack_from("<H", self.data, self.pos)[0])
        self.pos += 2
        return value

    def _skip_nudge(self) -> None:
        """跳过短或长格式的 nudge 偏移。"""

        dx = self._u8()
        dy = self._u8()
        if dx == 128 and dy == 128:
            self._u16()
            self._u16()

    def _enter(self) -> None:
        """进入一个嵌套 object list 并限制深度。"""

        self.depth += 1
        if self.depth > MAX_RECORD_DEPTH:
            raise _MtefError("MTEF nesting exceeds the safety limit")

    def _leave(self) -> None:
        """离开一个嵌套 object list。"""

        self.depth -= 1

    def _skip_ruler(self) -> None:
        """跳过不影响公式语义的 RULER record。"""

        count = self._u8()
        for _ in range(count):
            self._u8()
            self._u16()

    def _skip_font(self) -> None:
        """跳过 FONT 的 typeface、style 和零结尾名称。"""

        self._u8()
        self._u8()
        while self._u8() != 0:
            pass

    def _skip_size(self) -> None:
        """跳过三种长度形式的 SIZE record。"""

        first = self._u8()
        if first == 100:
            self._u8()
            self._u16()
        elif first == 101:
            self._u16()
        else:
            self._u8()

    def _parse_embellishments(self) -> tuple[int, ...]:
        """读取 CHAR 后以 END 终止的 embellishment 列表。"""

        values: list[int] = []
        while True:
            tag = self._u8()
            self._charge()
            record_type = tag & 0x0F
            flags = tag & 0xF0
            if record_type == 0:
                return tuple(values)
            if record_type != 6:
                raise _MtefError("MTEF embellishment list contains a non-EMBELL record")
            if flags & _XF_LMOVE:
                self._skip_nudge()
            values.append(self._u8())

    def _parse_line(self, flags: int) -> _Node:
        """解析一个 LINE slot。"""

        if flags & _XF_NULL:
            return _Node("sequence")
        if flags & _XF_LMOVE:
            self._skip_nudge()
        if flags & _XF_LSPACE:
            self._u8()
        if flags & _XF_RULER:
            ruler_tag = self._u8()
            if ruler_tag & 0x0F != 7:
                raise _MtefError("MTEF LINE ruler tag is invalid")
            self._skip_ruler()
        self._enter()
        children = self._parse_list()
        self._leave()
        return _Node("sequence", children=children)

    def _parse_template(self, flags: int) -> _Node:
        """解析 TMPL selector、variation、options 和 slots。"""

        if flags & _XF_LMOVE:
            self._skip_nudge()
        selector = self._u8()
        variation = self._u8()
        options = self._u8()
        self._enter()
        slots = self._parse_slots()
        self._leave()
        return _Node("template", (selector, variation, options), slots)

    def _parse_pile(self, flags: int) -> _Node:
        """解析纵向 PILE 中的 LINE slots。"""

        if flags & _XF_LMOVE:
            self._skip_nudge()
        self._u8()
        self._u8()
        if flags & _XF_RULER:
            ruler_tag = self._u8()
            if ruler_tag & 0x0F != 7:
                raise _MtefError("MTEF PILE ruler tag is invalid")
            self._skip_ruler()
        self._enter()
        slots = self._parse_slots()
        self._leave()
        return _Node("pile", children=slots)

    def _parse_matrix(self, flags: int) -> _Node:
        """解析 MATRIX 维度、分隔线位图和逐格 LINE。"""

        if flags & _XF_LMOVE:
            self._skip_nudge()
        self._u8()
        self._u8()
        self._u8()
        rows = self._u8()
        cols = self._u8()
        if rows == 0 or cols == 0 or rows * cols > 4096:
            raise _MtefError("MTEF matrix dimensions are invalid")
        for entry_count in (rows + 1, cols + 1):
            for _ in range((2 * entry_count + 7) // 8):
                self._u8()
        cells: list[_Node] = []
        self._enter()
        while len(cells) < rows * cols:
            tag = self._u8()
            self._charge()
            record_type = tag & 0x0F
            flags = tag & 0xF0
            if record_type == 0:
                continue
            if record_type == 1:
                cells.append(self._parse_line(flags))
            elif record_type == 7:
                self._skip_ruler()
            elif record_type == 8:
                self._skip_font()
            elif record_type == 9:
                self._skip_size()
            elif 10 <= record_type <= 14:
                continue
            else:
                cells.append(self._parse_record(record_type, flags))
        self._leave()
        if self.pos < len(self.data) and self.data[self.pos] & 0x0F == 0:
            self.pos += 1
            self._charge()
        return _Node("matrix", (rows, cols), tuple(cells))

    def _parse_record(self, record_type: int, flags: int) -> _Node:
        """解析一条已读取 tag 的语义 record。"""

        if record_type == 1:
            return self._parse_line(flags)
        if record_type == 2:
            if flags & _XF_LMOVE:
                self._skip_nudge()
            typeface = self._u8()
            character = self._u16()
            embellishments = self._parse_embellishments() if flags & _XF_EMBELL else ()
            return _Node("character", (typeface, character, embellishments))
        if record_type == 3:
            return self._parse_template(flags)
        if record_type == 4:
            return self._parse_pile(flags)
        if record_type == 5:
            return self._parse_matrix(flags)
        if record_type == 6:
            if flags & _XF_LMOVE:
                self._skip_nudge()
            return _Node("embellishment", self._u8())
        if record_type == 7:
            self._skip_ruler()
            return _Node("metadata")
        if record_type == 8:
            self._skip_font()
            return _Node("metadata")
        if record_type == 9:
            self._skip_size()
            return _Node("metadata")
        if 10 <= record_type <= 14:
            return _Node("metadata")
        raise _MtefError(f"unsupported MTEF v3 record type: {record_type}")

    def _parse_list(self) -> tuple[_Node, ...]:
        """解析以 END 终止的普通 object list。"""

        nodes: list[_Node] = []
        while True:
            tag = self._u8()
            self._charge()
            record_type = tag & 0x0F
            if record_type == 0:
                return tuple(nodes)
            node = self._parse_record(record_type, tag & 0xF0)
            if node.kind != "metadata":
                nodes.append(node)

    def _parse_slots(self) -> tuple[_Node, ...]:
        """解析 template/pile 中每个 LINE 对应的 slot。"""

        slots: list[_Node] = []
        while True:
            tag = self._u8()
            self._charge()
            record_type = tag & 0x0F
            flags = tag & 0xF0
            if record_type == 0:
                return tuple(slots)
            if record_type == 1:
                slots.append(self._parse_line(flags))
            elif record_type == 7:
                self._skip_ruler()
            elif record_type == 8:
                self._skip_font()
            elif record_type == 9:
                self._skip_size()
            elif 10 <= record_type <= 14:
                continue
            else:
                slots.append(self._parse_record(record_type, flags))

    def parse(self) -> _Node:
        """解析根 object list 并拒绝没有可见内容的对象。"""

        children = self._parse_list()
        root = _Node("sequence", children=children)
        if not _render_node(root).strip():
            raise _MtefError("MTEF formula contains no visible content")
        return root


_CHAR_LATEX = {
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "δ": r"\delta ",
    "ε": r"\epsilon ",
    "θ": r"\theta ",
    "λ": r"\lambda ",
    "μ": r"\mu ",
    "π": r"\pi ",
    "ρ": r"\rho ",
    "σ": r"\sigma ",
    "φ": r"\phi ",
    "ω": r"\omega ",
    "Γ": r"\Gamma ",
    "Δ": r"\Delta ",
    "Θ": r"\Theta ",
    "Λ": r"\Lambda ",
    "Ξ": r"\Xi ",
    "Π": r"\Pi ",
    "Σ": r"\Sigma ",
    "Φ": r"\Phi ",
    "Ψ": r"\Psi ",
    "Ω": r"\Omega ",
    "≤": r"\leq ",
    "≥": r"\geq ",
    "≠": r"\ne ",
    "±": r"\pm ",
    "∓": r"\mp ",
    "−": "-",
    "×": r"\times ",
    "÷": r"\div ",
    "∞": r"\infty ",
    "∂": r"\partial ",
    "∈": r"\in ",
    "∉": r"\notin ",
    "∪": r"\cup ",
    "∩": r"\cap ",
    "→": r"\rightarrow ",
    "←": r"\leftarrow ",
    "↔": r"\leftrightarrow ",
    "·": r"\cdot ",
    "∑": r"\sum ",
    "∫": r"\int ",
}


def _render_character(character: int) -> str:
    """把 MTEF 字符转为安全 LaTeX。"""

    try:
        value = chr(character)
    except ValueError as exc:
        raise _MtefError("MTEF character code is invalid") from exc
    mapped = _CHAR_LATEX.get(value)
    if mapped is not None:
        return mapped
    if value == "\\":
        return r"\backslash "
    if value in "#$%&_{}":
        return f"\\{value}"
    if value == "~":
        return r"\sim "
    if value == "^":
        return r"\hat{}"
    if ord(value) < 0x20:
        raise _MtefError("MTEF formula contains a control character")
    return value


def _apply_embellishments(value: str, embellishments: tuple[int, ...]) -> str:
    """按记录顺序把常见 embellishment 转为 LaTeX。"""

    for embellishment in embellishments:
        if embellishment == 2:
            value = rf"\dot{{{value}}}"
        elif embellishment == 3:
            value = rf"\ddot{{{value}}}"
        elif embellishment == 4:
            value = rf"\overset{{\ldots}}{{{value}}}"
        elif embellishment == 5:
            value = f"{value}'"
        elif embellishment == 6:
            value = f"{value}''"
        elif embellishment == 8:
            value = rf"\widetilde{{{value}}}"
        elif embellishment == 9:
            value = rf"\widehat{{{value}}}"
        elif embellishment == 10:
            value = rf"\not{{{value}}}"
        elif embellishment == 11:
            value = rf"\overrightarrow{{{value}}}"
        elif embellishment == 12:
            value = rf"\overleftarrow{{{value}}}"
        elif embellishment == 13:
            value = rf"\overleftrightarrow{{{value}}}"
        elif embellishment == 16:
            value = rf"\bar{{{value}}}"
        elif embellishment == 17:
            value = rf"\overline{{{value}}}"
        elif embellishment == 18:
            value = f"{value}'''"
        elif embellishment == 19:
            value = rf"\overparen{{{value}}}"
        elif embellishment == 20:
            value = rf"\underparen{{{value}}}"
        else:
            raise _MtefError(f"unsupported MTEF embellishment: {embellishment}")
    return value


def _slot(slots: tuple[_Node, ...], index: int) -> str:
    """渲染一个可选 template slot。"""

    return _render_node(slots[index]).strip() if index < len(slots) else ""


def _fence(selector: int, variation: int, body: str) -> str:
    """渲染左右 fence 及单边 variation。"""

    pairs = {
        0: (r"\langle", r"\rangle"),
        1: ("(", ")"),
        2: (r"\{", r"\}"),
        3: ("[", "]"),
        4: ("|", "|"),
        5: (r"\|", r"\|"),
        6: (r"\lfloor", r"\rfloor"),
        7: (r"\lceil", r"\rceil"),
    }
    left, right = pairs[selector]
    if variation == 1:
        right = "."
    elif variation == 2:
        left = "."
    elif variation != 0:
        raise _MtefError("unsupported MTEF fence variation")
    return rf"\left{left}{body}\right{right}"


def _big_operator(selector: int, slots: tuple[_Node, ...]) -> str:
    """渲染积分、求和、乘积及集合大运算。"""

    operators = {
        21: r"\int",
        22: r"\iint",
        23: r"\iiint",
        24: r"\int",
        25: r"\iint",
        26: r"\iiint",
        29: r"\sum",
        30: r"\sum",
        31: r"\prod",
        32: r"\prod",
        33: r"\coprod",
        34: r"\coprod",
        35: r"\bigcup",
        36: r"\bigcup",
        37: r"\bigcap",
        38: r"\bigcap",
    }
    operator = operators[selector]
    main = _slot(slots, 0)
    upper = _slot(slots, 1)
    lower = _slot(slots, 2)
    limits = (rf"_{{{lower}}}" if lower else "") + (rf"^{{{upper}}}" if upper else "")
    return rf"{operator}{limits}{{{main}}}"


def _render_template(selector: int, variation: int, options: int, slots: tuple[_Node, ...]) -> str:
    """把一个受支持的 MTEF v3 template 转为 LaTeX。"""

    del options
    if 0 <= selector <= 7:
        return _fence(selector, variation, _slot(slots, 0))
    if selector == 8:
        return rf"\left\{{\left\{{{_slot(slots, 0)}"
    if selector == 9:
        return rf"{_slot(slots, 0)}\right\}}\right\}}"
    if selector == 10:
        return rf"{_slot(slots, 0)}\right\}}\left\{{"
    if selector == 11:
        return rf"\left\{{{_slot(slots, 0)}\right)"
    if selector == 12:
        return rf"\left({_slot(slots, 0)}\right\}}"
    if selector == 13:
        radicand = _slot(slots, 0)
        index = _slot(slots, 1)
        return rf"\sqrt[{index}]{{{radicand}}}" if variation == 1 and index else rf"\sqrt{{{radicand}}}"
    if selector == 14:
        return rf"\frac{{{_slot(slots, 0)}}}{{{_slot(slots, 1)}}}"
    if selector in {15, 44}:
        subscript = _slot(slots, 0)
        superscript = _slot(slots, 1)
        return (rf"_{{{subscript}}}" if subscript else "") + (rf"^{{{superscript}}}" if superscript else "")
    if selector == 16:
        return rf"\underline{{{_slot(slots, 0)}}}"
    if selector == 17:
        return rf"\overline{{{_slot(slots, 0)}}}"
    if selector in {18, 19, 20}:
        arrow = {18: r"\leftarrow", 19: r"\rightarrow", 20: r"\leftrightarrow"}[selector]
        return rf"\overset{{{_slot(slots, 0)}}}{{{arrow}}}" if variation == 0 else rf"\underset{{{_slot(slots, 0)}}}{{{arrow}}}"
    if selector == 27:
        return rf"\overbrace{{{_slot(slots, 0)}}}^{{{_slot(slots, 1)}}}"
    if selector == 28:
        return rf"\underbrace{{{_slot(slots, 0)}}}_{{{_slot(slots, 1)}}}"
    if 21 <= selector <= 38:
        return _big_operator(selector, slots)
    if selector == 39:
        main = _slot(slots, 0)
        lower = _slot(slots, 1)
        upper = _slot(slots, 2)
        return rf"\lim_{{{lower}}}^{{{upper}}}{{{main}}}"
    if selector == 40:
        return rf"\overline{{{_slot(slots, 1)}}}\smash{{\big) {_slot(slots, 0)}}}"
    if selector == 41:
        return rf"{{{_slot(slots, 0)}}}/{{{_slot(slots, 1)}}}"
    if selector in {42, 43}:
        operator = _slot(slots, 3) or (r"\int" if selector == 42 else r"\sum")
        main = _slot(slots, 0)
        upper = _slot(slots, 1)
        lower = _slot(slots, 2)
        return operator + (rf"_{{{lower}}}" if lower else "") + (rf"^{{{upper}}}" if upper else "") + rf"{{{main}}}"
    if selector == 45:
        return rf"\left\langle {_slot(slots, 0)}\middle|{_slot(slots, 1)}\right\rangle"
    if selector in {46, 47}:
        command = "under" if selector == 46 else "over"
        direction = {0: "leftarrow", 1: "rightarrow", 2: "leftrightarrow"}.get(variation)
        if direction is None:
            raise _MtefError("unsupported MTEF vector variation")
        return rf"\{command}{direction}{{{_slot(slots, 0)}}}"
    if selector == 48:
        return rf"\overparen{{{_slot(slots, 0)}}}"
    raise _MtefError(f"unsupported MTEF v3 template selector: {selector}")


def _render_node(node: _Node) -> str:
    """递归序列化 MTEF AST。"""

    if node.kind == "sequence":
        return "".join(_render_node(child) for child in node.children)
    if node.kind == "character":
        _typeface, character, embellishments = node.value  # type: ignore[misc]
        return _apply_embellishments(_render_character(int(character)), tuple(embellishments))
    if node.kind == "template":
        selector, variation, options = node.value  # type: ignore[misc]
        return _render_template(int(selector), int(variation), int(options), node.children)
    if node.kind == "pile":
        rows = [_render_node(child).strip() for child in node.children]
        return r"\begin{gathered}" + r"\\".join(rows) + r"\end{gathered}"
    if node.kind == "matrix":
        rows, cols = node.value  # type: ignore[misc]
        rendered = [_render_node(child).strip() for child in node.children]
        matrix_rows = ["&".join(rendered[row * int(cols) : (row + 1) * int(cols)]) for row in range(int(rows))]
        return r"\begin{matrix}" + r"\\".join(matrix_rows) + r"\end{matrix}"
    if node.kind in {"metadata", "embellishment"}:
        return ""
    raise _MtefError(f"unknown MTEF AST node: {node.kind}")


def decode_mtef_v3(data: bytes) -> str | None:
    """把 MTEF v3 字节流转换为 LaTeX；不可信对象失败时返回 None。"""

    try:
        latex = _render_node(_MtefReader(data).parse()).strip()
    except (ArithmeticError, IndexError, struct.error, UnicodeError, _MtefError, ValueError):
        return None
    return latex or None


def decode_equation_native(data: bytes) -> str | None:
    """校验 28 字节 EQNOLEFILEHDR 并解码其中的 MTEF v3。"""

    if len(data) < 28:
        return None
    header_size, version, _clipboard_format, object_size = struct.unpack_from("<HIHI", data, 0)
    if header_size < 28 or header_size > len(data) or version != 0x0002_0000:
        return None
    object_end = header_size + object_size
    if object_end < header_size or object_end > len(data):
        return None
    return decode_mtef_v3(data[header_size:object_end])


def decode_equation_object(data: bytes) -> str | None:
    """从独立 Equation Editor OLE 对象读取 Equation Native 并转为 LaTeX。"""

    try:
        with BoundedOleReader(data) as ole:
            native = ole.read_stream("Equation Native")
    except ValueError:
        return None
    return decode_equation_native(native)


def read_object_pool_equations(ole: BoundedOleReader) -> dict[int, str]:
    """只读提取 DOC ObjectPool 中可安全解码的 Equation Native streams。"""

    equations: dict[int, str] = {}
    for stream_name in ole.stream_names(prefix="ObjectPool/"):
        match = _OBJECT_POOL_EQUATION_RE.match(stream_name)
        if match is None:
            continue
        storage_id = int(match.group(1))
        latex = decode_equation_native(ole.read_stream(stream_name))
        if latex is None:
            logger.warning(
                "DOC_MTEF_FALLBACK: storage_id={} has an invalid or unsupported Equation Native stream",
                storage_id,
            )
            continue
        equations.setdefault(storage_id, latex)
    return equations
