# Copyright (c) Opendatalab. All rights reserved.

"""按照 WIRIS 规范安全读取 MathType MTEF v5 record tree。"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from ..errors import LegacyOfficeResourceLimitError
from ..limits import MAX_RECORD_DEPTH, MAX_RECORDS
from .mtef import _MtefError, _Node, _render_node

_OPT_NUDGE = 0x08
_CHAR_EMBELL = 0x01
_CHAR_FUNC_START = 0x02
_CHAR_ENC_8 = 0x04
_CHAR_ENC_16 = 0x10
_CHAR_NO_MTCODE = 0x20
_LINE_NULL = 0x01
_LINE_RULER = 0x02
_LINE_LSPACE = 0x04
_PILE_RULER = 0x02
_BUILTIN_TYPEFACES = frozenset({*range(1, 13), 22, 23, 24})

_PREDEFINED_ENCODINGS = {
    1: "MTCode",
    2: "Unknown",
    3: "Symbol",
    4: "MTExtra",
}

# Adobe Symbol/MathType Symbol 编码中常见且语义稳定的字符。
_SYMBOL_FONT_POSITION_TO_UNICODE = {
    0x41: 0x0391,
    0x42: 0x0392,
    0x43: 0x03A7,
    0x44: 0x0394,
    0x45: 0x0395,
    0x46: 0x03A6,
    0x47: 0x0393,
    0x48: 0x0397,
    0x49: 0x0399,
    0x4A: 0x03D1,
    0x4B: 0x039A,
    0x4C: 0x039B,
    0x4D: 0x039C,
    0x4E: 0x039D,
    0x4F: 0x039F,
    0x50: 0x03A0,
    0x51: 0x0398,
    0x52: 0x03A1,
    0x53: 0x03A3,
    0x54: 0x03A4,
    0x55: 0x03A5,
    0x56: 0x03C2,
    0x57: 0x03A9,
    0x58: 0x039E,
    0x59: 0x03A8,
    0x5A: 0x0396,
    0x61: 0x03B1,
    0x62: 0x03B2,
    0x63: 0x03C7,
    0x64: 0x03B4,
    0x65: 0x03B5,
    0x66: 0x03C6,
    0x67: 0x03B3,
    0x68: 0x03B7,
    0x69: 0x03B9,
    0x6A: 0x03D5,
    0x6B: 0x03BA,
    0x6C: 0x03BB,
    0x6D: 0x03BC,
    0x6E: 0x03BD,
    0x6F: 0x03BF,
    0x70: 0x03C0,
    0x71: 0x03B8,
    0x72: 0x03C1,
    0x73: 0x03C3,
    0x74: 0x03C4,
    0x75: 0x03C5,
    0x76: 0x03D6,
    0x77: 0x03C9,
    0x78: 0x03BE,
    0x79: 0x03C8,
    0x7A: 0x03B6,
    0xA3: 0x2264,
    0xA5: 0x221E,
    0xAB: 0x2194,
    0xAC: 0x2190,
    0xAE: 0x2192,
    0xB1: 0x00B1,
    0xB3: 0x2265,
    0xB9: 0x2260,
    0xC7: 0x2229,
    0xC8: 0x222A,
    0xD5: 0x220F,
    0xE5: 0x2211,
    0xF2: 0x222B,
}

# MathType v5 常见私有 MTCode；未列出的 PUA 继续整体回退，避免猜测字符。
_MTCODE_PUA_TO_LATEX = {
    0xE90B: r"\supseteqq ",
    0xE90C: r"\subseteqq ",
    0xE922: r"\lesseqqgtr ",
    0xE92D: r"\gtreqqless ",
    0xE92E: r"\shortmid ",
    0xE92F: r"\shortparallel ",
    0xE930: r"\leqslant ",
    0xE931: r"\geqslant ",
    0xE932: r"\lessapprox ",
    0xE933: r"\gtrapprox ",
    0xE938: r"\preceq ",
    0xE939: r"\succeq ",
    0xE981: r"\circleddash ",
    0xE98F: r"\centerdot ",
    0xEA06: r"\nleq ",
    0xEA07: r"\ngeq ",
    0xEA11: r"\nsim ",
    0xEF02: r"\,",
    0xEF03: r"\;",
    0xEF04: r"\ ",
    0xEF05: r"\quad ",
    0xEF06: r"\qquad ",
    0xEF22: r"\!",
}

_MTEXTRA_FONT_POSITION_TO_UNICODE = {
    0x20: 0x0020,
    0x3C: 0x25C3,
    0x3E: 0x25B9,
    0x43: 0x2210,
    0x44: 0x019B,
    0x49: 0x2229,
    0x4B: 0x2026,
    0x4C: 0x22EF,
    0x4D: 0x22EE,
    0x4E: 0x22F0,
    0x4F: 0x22F1,
    0x51: 0x2235,
    0x55: 0x222A,
    0x60: 0x2035,
    0x61: 0x21A6,
    0x62: 0x2195,
    0x63: 0x21D5,
    0x66: 0x227B,
    0x68: 0x210F,
    0x6C: 0x2113,
    0x6D: 0x2213,
    0x6F: 0x2218,
    0x70: 0x227A,
}


@dataclass(frozen=True, slots=True)
class _FontDefinition:
    """记录一个 FONT_DEF 的 encoding 索引和字体名称。"""

    encoding_index: int
    name: str


@dataclass(frozen=True, slots=True)
class _FontStyleDefinition:
    """记录一个 FONT_STYLE_DEF 的字体引用和粗斜体位。"""

    font_definition_index: int
    style_bits: int


class _MtefV5Reader:
    """有界读取 MTEF v5 header、定义记录和公式对象树。"""

    def __init__(self, data: bytes) -> None:
        """校验 v5 header 并初始化 definition tables 与安全计数器。"""

        if len(data) < 7 or data[0] != 5:
            raise _MtefError("Equation Native does not contain MTEF v5")
        if data[1] not in {0, 1} or data[2] not in {0, 1} or data[3] < 4:
            raise _MtefError("MTEF v5 generator header is invalid")
        self.data = data
        self.pos = 5
        self.depth = 0
        self.records = 0
        self.encoding_definitions: dict[int, str] = dict(_PREDEFINED_ENCODINGS)
        self.font_definitions: list[_FontDefinition | None] = [None]
        self.font_style_definitions: list[_FontStyleDefinition | None] = [None]
        self.equation_styles: list[_FontStyleDefinition | None] = []
        self.color_definition_count = 0
        application_key = self._cstring(max_bytes=4096)
        if not application_key:
            raise _MtefError("MTEF v5 application key is empty")
        equation_options = self._u8()
        if equation_options & ~0x01:
            raise _MtefError("MTEF v5 equation options are invalid")

    def _charge(self) -> None:
        """计入一条 record，超过共享上限时抛稳定资源错误。"""

        self.records += 1
        if self.records > MAX_RECORDS:
            raise LegacyOfficeResourceLimitError(
                f"MTEF record count exceeds max_records={MAX_RECORDS}"
            )

    def _enter(self) -> None:
        """进入嵌套 object list 并限制共享深度。"""

        self.depth += 1
        if self.depth > MAX_RECORD_DEPTH:
            raise LegacyOfficeResourceLimitError(
                f"MTEF nesting exceeds max_record_depth={MAX_RECORD_DEPTH}"
            )

    def _leave(self) -> None:
        """离开当前 object list。"""

        self.depth -= 1

    def _u8(self) -> int:
        """有界读取一个无符号字节。"""

        if self.pos >= len(self.data):
            raise _MtefError("MTEF v5 record is truncated")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def _u16(self) -> int:
        """有界读取一个小端 u16。"""

        if self.pos + 2 > len(self.data):
            raise _MtefError("MTEF v5 u16 is truncated")
        value = int(struct.unpack_from("<H", self.data, self.pos)[0])
        self.pos += 2
        return value

    def _bytes(self, size: int) -> bytes:
        """按显式长度有界读取 payload。"""

        if size < 0 or self.pos + size < self.pos or self.pos + size > len(self.data):
            raise _MtefError("MTEF v5 payload length is invalid")
        value = self.data[self.pos : self.pos + size]
        self.pos += size
        return value

    def _cstring(self, *, max_bytes: int) -> str:
        """读取有长度上限的零结尾单字节字符串。"""

        end_limit = min(len(self.data), self.pos + max_bytes + 1)
        end = self.data.find(b"\x00", self.pos, end_limit)
        if end < 0:
            raise _MtefError("MTEF v5 string is not null-terminated")
        raw = self.data[self.pos:end]
        self.pos = end + 1
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    def _signed(self) -> int:
        """读取 MTEF v5 紧凑有符号整数。"""

        first = self._u8()
        if first == 0xFF:
            return self._u16() - 0x8000
        return first - 0x80

    def _unsigned(self) -> int:
        """读取 MTEF v5 紧凑无符号整数。"""

        first = self._u8()
        return self._u16() if first == 0xFF else first

    def _variation(self) -> int:
        """读取一或两字节 template variation。"""

        first = self._u8()
        if first & 0x80:
            return (first & 0x7F) | (self._u8() << 8)
        return first

    def _skip_nudge(self) -> None:
        """跳过短或长格式 nudge。"""

        dx = self._u8()
        dy = self._u8()
        if dx == 0x80 and dy == 0x80:
            self._u16()
            self._u16()

    def _skip_ruler(self) -> None:
        """解析并跳过 RULER record 的 tab stops。"""

        count = self._u8()
        for _ in range(count):
            tab_type = self._u8()
            if tab_type > 4:
                raise _MtefError("MTEF v5 ruler tab type is invalid")
            self._u16()

    def _skip_size(self) -> None:
        """解析并跳过 SIZE record 的三种编码。"""

        first = self._u8()
        if first == 100:
            logical_size = self._u8()
            if logical_size > 7:
                raise _MtefError("MTEF v5 logical size is invalid")
            self._u16()
        elif first == 101:
            self._u16()
        else:
            if first > 7:
                raise _MtefError("MTEF v5 logical size is invalid")
            self._u8()

    def _dimension_array(self) -> None:
        """有界消费 EQN_PREFS 中按 nibble 编码的 dimension array。"""

        count = self._u8()
        current = 0
        use_low = False

        def next_nibble() -> int:
            """按高四位优先顺序读取一个 nibble。"""

            nonlocal current, use_low
            if not use_low:
                current = self._u8()
                use_low = True
                return current >> 4
            use_low = False
            return current & 0x0F

        for _ in range(count):
            units = next_nibble()
            if units > 4:
                raise _MtefError("MTEF v5 dimension units are invalid")
            while True:
                value = next_nibble()
                if value == 0x0F:
                    break
                if value > 0x0B:
                    raise _MtefError("MTEF v5 dimension digit is invalid")
        if use_low and current & 0x0F:
            raise _MtefError("MTEF v5 dimension padding nibble is invalid")

    def _parse_font_style_definition(self) -> _Node:
        """读取 FONT_STYLE_DEF 并验证先前 FONT_DEF 引用。"""

        font_index = self._unsigned()
        style_bits = self._u8()
        if style_bits & ~0x03:
            raise _MtefError("MTEF v5 font style bits are invalid")
        if font_index <= 0 or font_index >= len(self.font_definitions):
            raise _MtefError("MTEF v5 FONT_STYLE_DEF reference is invalid")
        self.font_style_definitions.append(
            _FontStyleDefinition(font_index, style_bits)
        )
        return _Node("metadata")

    def _parse_color_definition(self) -> _Node:
        """读取 COLOR_DEF，颜色只校验结构而不进入现有 schema。"""

        options = self._u8()
        if options & ~0x07:
            raise _MtefError("MTEF v5 color options are invalid")
        component_count = 4 if options & 0x01 else 3
        for _ in range(component_count):
            if self._u16() > 1000:
                raise _MtefError("MTEF v5 color component is invalid")
        if options & 0x04:
            self._cstring(max_bytes=4096)
        self.color_definition_count += 1
        return _Node("metadata")

    def _parse_font_definition(self) -> _Node:
        """读取 FONT_DEF 并绑定已声明 encoding。"""

        encoding_index = self._unsigned()
        if encoding_index not in self.encoding_definitions:
            raise _MtefError("MTEF v5 FONT_DEF encoding reference is invalid")
        name = self._cstring(max_bytes=4096)
        if not name:
            raise _MtefError("MTEF v5 font name is empty")
        self.font_definitions.append(_FontDefinition(encoding_index, name))
        return _Node("metadata")

    def _parse_equation_preferences(self) -> _Node:
        """读取 EQN_PREFS 的尺寸、间距和 style definition 数组。"""

        if self._u8() != 0:
            raise _MtefError("MTEF v5 equation preference options are invalid")
        self._dimension_array()
        self._dimension_array()
        style_count = self._u8()
        styles: list[_FontStyleDefinition | None] = []
        for _ in range(style_count):
            font_index = self._unsigned()
            if font_index == 0:
                styles.append(None)
                continue
            if font_index >= len(self.font_definitions):
                raise _MtefError("MTEF v5 equation style reference is invalid")
            style_bits = self._u8()
            if style_bits & ~0x03:
                raise _MtefError("MTEF v5 equation style bits are invalid")
            styles.append(_FontStyleDefinition(font_index, style_bits))
        self.equation_styles = styles
        return _Node("metadata")

    def _parse_encoding_definition(self) -> _Node:
        """读取自定义 ENCODING_DEF 并按出现顺序从索引 5 编号。"""

        name = self._cstring(max_bytes=4096)
        if not name:
            raise _MtefError("MTEF v5 encoding name is empty")
        index = max(self.encoding_definitions) + 1
        self.encoding_definitions[index] = name
        return _Node("metadata")

    def _character_style(self, typeface: int) -> int:
        """从显式字体或 EQN_PREFS style 解析粗斜体位。"""

        definition: _FontStyleDefinition | None = None
        if typeface < 0:
            index = -typeface
            if 0 < index < len(self.font_style_definitions):
                definition = self.font_style_definitions[index]
        elif 0 < typeface <= len(self.equation_styles):
            definition = self.equation_styles[typeface - 1]
        return definition.style_bits if definition is not None else 0

    def _character_encoding(self, typeface: int) -> str:
        """解析无 MTCode 字符所引用的字体 encoding。"""

        definition: _FontStyleDefinition | None = None
        if typeface < 0:
            index = -typeface
            if 0 < index < len(self.font_style_definitions):
                definition = self.font_style_definitions[index]
        elif 0 < typeface <= len(self.equation_styles):
            definition = self.equation_styles[typeface - 1]
        if definition is not None:
            font_definition = self.font_definitions[definition.font_definition_index]
            if font_definition is not None:
                return self.encoding_definitions[font_definition.encoding_index]
        if typeface in {4, 5, 6}:
            return "Symbol"
        if typeface == 11:
            return "MTExtra"
        return "WinAllBasicCodePages"

    @staticmethod
    def _mtcode_character(code: int) -> int | str:
        """把可验证的 MTCode 转为 Unicode/LaTeX，未知 PUA 整体回退。"""

        if 0xD800 <= code <= 0xDFFF or code < 0x20:
            raise _MtefError("MTEF v5 MTCode character is invalid")
        if 0xE000 <= code <= 0xF8FF:
            mapped = _MTCODE_PUA_TO_LATEX.get(code)
            if mapped is not None:
                return mapped
            if 0xF000 <= code <= 0xF019:
                return rf"\mathfrak{{{chr(ord('A') + code - 0xF000)}}}"
            if 0xF01A <= code <= 0xF033:
                return rf"\mathfrak{{{chr(ord('a') + code - 0xF01A)}}}"
            if 0xF080 <= code <= 0xF099:
                return rf"\mathbb{{{chr(ord('A') + code - 0xF080)}}}"
            if 0xF100 <= code <= 0xF119:
                return rf"\mathcal{{{chr(ord('A') + code - 0xF100)}}}"
            raise _MtefError("MTEF v5 private MTCode is unsupported")
        return code

    @staticmethod
    def _font_position_character(encoding: str, position: int) -> int | str:
        """从已知 encoding 的 font position 恢复 Unicode/LaTeX。"""

        normalized = encoding.strip().casefold()
        if normalized == "mtcode":
            return _MtefV5Reader._mtcode_character(position)
        if normalized == "symbol":
            character = _SYMBOL_FONT_POSITION_TO_UNICODE.get(position)
            if character is None:
                raise _MtefError("MTEF v5 Symbol font position is unsupported")
            return character
        if normalized == "mtextra":
            character = _MTEXTRA_FONT_POSITION_TO_UNICODE.get(position)
            if character is None:
                raise _MtefError("MTEF v5 MTExtra font position is unsupported")
            return character
        if normalized in {
            "winallbasiccodepages",
            "windowsansi",
            "windows-1252",
            "cp1252",
            "unicode",
        }:
            if position <= 0xFF:
                value = bytes([position]).decode("cp1252")
                return ord(value)
            return _MtefV5Reader._mtcode_character(position)
        raise _MtefError(f"unsupported MTEF v5 font encoding: {encoding}")

    def _parse_embellishments(self) -> tuple[int, ...]:
        """读取 CHAR 后以 END 终止的 EMBELL 列表。"""

        values: list[int] = []
        while True:
            record_type = self._u8()
            self._charge()
            if record_type == 0:
                return tuple(values)
            if record_type != 6:
                raise _MtefError("MTEF v5 embellishment list is invalid")
            options = self._u8()
            if options & ~_OPT_NUDGE:
                raise _MtefError("MTEF v5 embellishment options are invalid")
            if options & _OPT_NUDGE:
                self._skip_nudge()
            embellishment = self._u8()
            if not 2 <= embellishment <= 37:
                raise _MtefError("MTEF v5 embellishment type is invalid")
            values.append(embellishment)

    def _parse_character(self) -> _Node:
        """读取 CHAR 的 typeface、MTCode/font position 和 embellishments。"""

        options = self._u8()
        allowed = (
            _OPT_NUDGE
            | _CHAR_EMBELL
            | _CHAR_FUNC_START
            | _CHAR_ENC_8
            | _CHAR_ENC_16
            | _CHAR_NO_MTCODE
        )
        if options & ~allowed or options & _CHAR_ENC_8 and options & _CHAR_ENC_16:
            raise _MtefError("MTEF v5 CHAR options are invalid")
        if options & _OPT_NUDGE:
            self._skip_nudge()
        typeface = self._signed()
        if typeface < 0:
            definition_index = -typeface
            if not (
                0 < definition_index < len(self.font_style_definitions)
                and self.font_style_definitions[definition_index] is not None
            ):
                raise _MtefError("MTEF v5 explicit typeface reference is invalid")
        elif typeface not in _BUILTIN_TYPEFACES:
            raise _MtefError("MTEF v5 typeface value is invalid")
        mtcode = None if options & _CHAR_NO_MTCODE else self._u16()
        font_position: int | None = None
        if options & _CHAR_ENC_8:
            font_position = self._u8()
        elif options & _CHAR_ENC_16:
            font_position = self._u16()
        if mtcode is None and font_position is None:
            raise _MtefError("MTEF v5 CHAR has no character value")
        if mtcode is not None:
            character = self._mtcode_character(mtcode)
        else:
            if font_position is None:
                raise _MtefError("MTEF v5 CHAR font position is missing")
            character = self._font_position_character(
                self._character_encoding(typeface),
                font_position,
            )
        embellishments = (
            self._parse_embellishments() if options & _CHAR_EMBELL else ()
        )
        return _Node(
            "character_v5",
            (
                character,
                embellishments,
                self._character_style(typeface),
                typeface,
                bool(options & _CHAR_FUNC_START),
            ),
        )

    def _parse_line(self) -> _Node:
        """读取 LINE options、可选 ruler 和内部 object list。"""

        options = self._u8()
        if options & ~(_OPT_NUDGE | _LINE_NULL | _LINE_RULER | _LINE_LSPACE):
            raise _MtefError("MTEF v5 LINE options are invalid")
        if options & _OPT_NUDGE:
            self._skip_nudge()
        if options & _LINE_LSPACE:
            self._u16()
        if options & _LINE_RULER:
            if self._u8() != 7:
                raise _MtefError("MTEF v5 LINE ruler tag is invalid")
            self._charge()
            self._skip_ruler()
        if options & _LINE_NULL:
            return _Node("sequence")
        self._enter()
        children = self._parse_list()
        self._leave()
        return _Node("sequence", children=children)

    def _parse_pile(self) -> _Node:
        """读取 PILE alignment、ruler 和逐行 object list。"""

        options = self._u8()
        if options & ~(_OPT_NUDGE | _PILE_RULER):
            raise _MtefError("MTEF v5 PILE options are invalid")
        if options & _OPT_NUDGE:
            self._skip_nudge()
        horizontal_alignment = self._u8()
        vertical_alignment = self._u8()
        if horizontal_alignment not in {1, 2, 3, 4, 5} or vertical_alignment > 4:
            raise _MtefError("MTEF v5 PILE alignment is invalid")
        if options & _PILE_RULER:
            if self._u8() != 7:
                raise _MtefError("MTEF v5 PILE ruler tag is invalid")
            self._charge()
            self._skip_ruler()
        self._enter()
        children = self._parse_list()
        self._leave()
        if any(child.kind != "sequence" for child in children):
            raise _MtefError("MTEF v5 PILE contains a non-LINE object")
        return _Node("pile", children=children)

    def _parse_matrix(self) -> _Node:
        """读取 MATRIX 维度、partition bits 和逐格 LINE。"""

        options = self._u8()
        if options & ~_OPT_NUDGE:
            raise _MtefError("MTEF v5 MATRIX options are invalid")
        if options & _OPT_NUDGE:
            self._skip_nudge()
        vertical_alignment = self._u8()
        horizontal_justification = self._u8()
        vertical_justification = self._u8()
        if (
            vertical_alignment > 4
            or horizontal_justification not in {1, 2, 3, 4, 5}
            or vertical_justification > 4
        ):
            raise _MtefError("MTEF v5 MATRIX alignment is invalid")
        rows = self._u8()
        cols = self._u8()
        if rows == 0 or cols == 0 or rows * cols > 4096:
            raise _MtefError("MTEF v5 matrix dimensions are invalid")
        self._bytes((2 * (rows + 1) + 7) // 8)
        self._bytes((2 * (cols + 1) + 7) // 8)
        self._enter()
        children = self._parse_list()
        self._leave()
        cells = tuple(child for child in children if child.kind == "sequence")
        if len(cells) != rows * cols or len(cells) != len(children):
            raise _MtefError("MTEF v5 matrix cell list is invalid")
        return _Node("matrix", (rows, cols), cells)

    def _parse_template(self) -> _Node:
        """读取 TMPL selector、variation、options 并转换为语义 AST。"""

        record_options = self._u8()
        if record_options & ~_OPT_NUDGE:
            raise _MtefError("MTEF v5 TMPL options are invalid")
        if record_options & _OPT_NUDGE:
            self._skip_nudge()
        selector = self._u8()
        variation = self._variation()
        template_options = self._u8()
        self._enter()
        slots = self._parse_list()
        self._leave()
        return _semantic_template(selector, variation, template_options, slots)

    def _parse_record(self, record_type: int) -> _Node:
        """解析一条已读取 type 的 v5 record。"""

        if record_type == 1:
            return self._parse_line()
        if record_type == 2:
            return self._parse_character()
        if record_type == 3:
            return self._parse_template()
        if record_type == 4:
            return self._parse_pile()
        if record_type == 5:
            return self._parse_matrix()
        if record_type == 6:
            raise _MtefError("MTEF v5 EMBELL appears outside a CHAR list")
        if record_type == 7:
            self._skip_ruler()
            return _Node("metadata")
        if record_type == 8:
            return self._parse_font_style_definition()
        if record_type == 9:
            self._skip_size()
            return _Node("metadata")
        if 10 <= record_type <= 14:
            return _Node("metadata")
        if record_type == 15:
            color_index = self._unsigned()
            if not 0 < color_index <= self.color_definition_count:
                raise _MtefError("MTEF v5 COLOR reference is invalid")
            return _Node("metadata")
        if record_type == 16:
            return self._parse_color_definition()
        if record_type == 17:
            return self._parse_font_definition()
        if record_type == 18:
            return self._parse_equation_preferences()
        if record_type == 19:
            return self._parse_encoding_definition()
        if record_type >= 100:
            self._bytes(self._unsigned())
            return _Node("metadata")
        raise _MtefError(f"unsupported MTEF v5 record type: {record_type}")

    @staticmethod
    def _group_character_runs(nodes: list[_Node]) -> tuple[_Node, ...]:
        """把连续 TEXT 字符和 FUNC_START 函数字符折叠为语义节点。"""

        grouped: list[_Node] = []
        index = 0
        while index < len(nodes):
            node = nodes[index]
            if node.kind != "character_v5":
                grouped.append(node)
                index += 1
                continue
            character, embellishments, style_bits, typeface, function_start = node.value  # type: ignore[misc]
            if (
                isinstance(character, int)
                and int(typeface) in {1, 12}
                and not embellishments
                and not style_bits
            ):
                characters = [chr(character)]
                cursor = index + 1
                while cursor < len(nodes):
                    candidate = nodes[cursor]
                    if candidate.kind != "character_v5":
                        break
                    c_char, c_embell, c_style, c_typeface, _c_start = candidate.value  # type: ignore[misc]
                    if (
                        not isinstance(c_char, int)
                        or c_embell
                        or c_style
                        or int(c_typeface) != int(typeface)
                    ):
                        break
                    characters.append(chr(c_char))
                    cursor += 1
                grouped.append(_Node("text", "".join(characters)))
                index = cursor
                continue
            if (
                not isinstance(character, int)
                or not function_start
                or int(typeface) != 2
                or embellishments
                or style_bits
            ):
                grouped.append(node)
                index += 1
                continue
            characters = [chr(int(character))]
            cursor = index + 1
            while cursor < len(nodes):
                candidate = nodes[cursor]
                if candidate.kind != "character_v5":
                    break
                c_char, c_embell, c_style, c_typeface, _c_start = candidate.value  # type: ignore[misc]
                if (
                    not isinstance(c_char, int)
                    or c_embell
                    or c_style
                    or int(c_typeface) != 2
                ):
                    break
                characters.append(chr(int(c_char)))
                cursor += 1
            grouped.append(_Node("function", "".join(characters)))
            index = cursor
        return tuple(grouped)

    def _parse_list(self) -> tuple[_Node, ...]:
        """解析以 END 终止的 v5 object list。"""

        nodes: list[_Node] = []
        while True:
            record_type = self._u8()
            self._charge()
            if record_type == 0:
                return self._group_character_runs(nodes)
            node = self._parse_record(record_type)
            if node.kind != "metadata":
                nodes.append(node)

    def parse(self) -> _Node:
        """解析根 object list，要求完整消费且产生可见公式。"""

        children = self._parse_list()
        if self.pos != len(self.data):
            raise _MtefError("MTEF v5 contains trailing bytes")
        root = _Node("sequence", children=children)
        if not _render_node(root).strip():
            raise _MtefError("MTEF v5 formula contains no visible content")
        return root


def _slot_node(slots: tuple[_Node, ...], index: int) -> _Node:
    """返回一个 template slot，不存在时补空 sequence。"""

    return slots[index] if index < len(slots) else _Node("sequence")


def _semantic_template(
    selector: int,
    variation: int,
    options: int,
    slots: tuple[_Node, ...],
) -> _Node:
    """把 MTEF v5 selector/variation 转换为版本无关语义节点。"""

    if 0 <= selector <= 8:
        if variation & ~0x03 or options not in {0, 1, 2}:
            raise _MtefError("unsupported MTEF v5 fence variation")
        pairs = {
            0: (r"\langle", r"\rangle"),
            1: ("(", ")"),
            2: (r"\{", r"\}"),
            3: ("[", "]"),
            4: ("|", "|"),
            5: (r"\|", r"\|"),
            6: (r"\lfloor", r"\rfloor"),
            7: (r"\lceil", r"\rceil"),
            8: (r"\llbracket", r"\rrbracket"),
        }
        default_left, default_right = pairs[selector]
        left = default_left if variation & 0x01 else "."
        right = default_right if variation & 0x02 else "."
        return _Node(
            "fence_semantic",
            (left, right),
            (_slot_node(slots, 0), _slot_node(slots, 1), _slot_node(slots, 2)),
        )
    if selector == 9:
        if variation & ~0x33 or options not in {0, 1, 2}:
            raise _MtefError("unsupported MTEF v5 interval variation")
        left = {0: "(", 1: ")", 2: "[", 3: "]"}[variation & 0x03]
        right = {0: "(", 1: ")", 2: "[", 3: "]"}[(variation >> 4) & 0x03]
        return _Node("fence_semantic", (left, right), (_slot_node(slots, 0),))
    if selector == 10:
        if variation not in {0, 1} or options != 0:
            raise _MtefError("unsupported MTEF v5 root variation")
        return _Node(
            "root_semantic",
            variation == 1,
            (_slot_node(slots, 0), _slot_node(slots, 1)),
        )
    if selector == 11:
        if variation & ~0x07 or options != 0:
            raise _MtefError("unsupported MTEF v5 fraction variation")
        return _Node(
            "fraction_semantic",
            bool(variation & 0x02),
            (_slot_node(slots, 0), _slot_node(slots, 1)),
        )
    if selector in {12, 13}:
        if variation & ~0x01 or options != 0:
            raise _MtefError("unsupported MTEF v5 bar variation")
        return _Node(
            "bar_semantic",
            (selector == 12, bool(variation & 0x01)),
            (_slot_node(slots, 0),),
        )
    if selector == 14:
        if variation & ~0x3F or options != 0:
            raise _MtefError("unsupported MTEF v5 arrow variation")
        return _Node("arrow_semantic", variation, slots)
    if 15 <= selector <= 22:
        if options not in {0, 1} or variation & ~0x01FF:
            raise _MtefError("unsupported MTEF v5 big-operator variation")
        defaults = {
            15: r"\int",
            16: r"\sum",
            17: r"\prod",
            18: r"\coprod",
            19: r"\bigcup",
            20: r"\bigcap",
            21: "",
            22: "",
        }
        return _Node(
            "big_operator_semantic",
            defaults[selector],
            tuple(_slot_node(slots, index) for index in range(4)),
        )
    if selector == 23:
        if variation not in {0, 1} or options != 0:
            raise _MtefError("unsupported MTEF v5 limit variation")
        return _Node(
            "limit_semantic",
            None,
            tuple(_slot_node(slots, index) for index in range(3)),
        )
    if selector in {24, 25}:
        if variation & ~0x01 or options != 0:
            raise _MtefError("unsupported MTEF v5 horizontal-fence variation")
        return _Node(
            "horizontal_fence_semantic",
            (selector == 24, bool(variation & 0x01)),
            tuple(_slot_node(slots, index) for index in range(3)),
        )
    if selector == 26:
        if variation & ~0x01 or options != 0:
            raise _MtefError("unsupported MTEF v5 long-division variation")
        return _Node(
            "long_division_semantic",
            bool(variation & 0x01),
            (_slot_node(slots, 0), _slot_node(slots, 1)),
        )
    if selector in {27, 28, 29}:
        if variation & ~0x01 or options != 0:
            raise _MtefError("unsupported MTEF v5 script variation")
        subscript = (
            _slot_node(slots, 0)
            if selector in {27, 29}
            else _Node("sequence")
        )
        superscript = (
            _slot_node(slots, 1)
            if selector in {28, 29}
            else _Node("sequence")
        )
        return _Node(
            "scripts_semantic",
            bool(variation & 0x01),
            (subscript, superscript),
        )
    if selector == 30:
        if variation & ~0x03 or options != 0:
            raise _MtefError("unsupported MTEF v5 Dirac variation")
        return _Node(
            "dirac_semantic",
            variation,
            tuple(_slot_node(slots, index) for index in range(5)),
        )
    if selector == 31:
        if variation & ~0x0F or options != 0 or variation & 0x08 and variation & 0x03 == 0x03:
            raise _MtefError("unsupported MTEF v5 vector variation")
        return _Node("vector_semantic", variation, (_slot_node(slots, 0),))
    if selector in {32, 33, 34}:
        if variation != 0 or options != 0:
            raise _MtefError("unsupported MTEF v5 accent variation")
        command = {32: "widetilde", 33: "widehat", 34: "overparen"}[selector]
        return _Node("accent_semantic", command, (_slot_node(slots, 0),))
    if selector == 35:
        raise _MtefError("unsupported MTEF v5 joint-status template")
    if selector == 36:
        if variation & ~0x07 or variation == 0 or variation & 0x01 or options != 0:
            raise _MtefError("unsupported MTEF v5 strike variation")
        command = "xcancel" if variation & 0x06 == 0x06 else "cancel" if variation & 0x02 else "bcancel"
        return _Node("strike_semantic", command, (_slot_node(slots, 0),))
    if selector == 37:
        if options != 0 or variation not in {0, 0x1E}:
            raise _MtefError("unsupported MTEF v5 box variation")
        return _Node("box_semantic", None, (_slot_node(slots, 0),))
    raise _MtefError(f"unsupported MTEF v5 template selector: {selector}")


def decode_mtef_v5(data: bytes) -> str | None:
    """把 MTEF v5 字节流转换为 LaTeX，损坏或不支持时整体返回空。"""

    try:
        latex = _render_node(_MtefV5Reader(data).parse()).strip()
    except LegacyOfficeResourceLimitError:
        raise
    except (
        ArithmeticError,
        IndexError,
        struct.error,
        UnicodeError,
        _MtefError,
        ValueError,
    ):
        return None
    return latex or None
