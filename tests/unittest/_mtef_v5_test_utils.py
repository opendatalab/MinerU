"""构造结果已知的 MathType MTEF v5 测试字节流。"""

from __future__ import annotations

import struct


def v5_signed(value: int) -> bytes:
    """按 MTEF v5 紧凑格式编码一个有符号整数。"""

    if not -0x8000 <= value <= 0x7FFF:
        raise ValueError("MTEF v5 signed fixture is out of range")
    if -128 <= value < 127:
        return bytes([value + 128])
    return b"\xff" + struct.pack("<H", value + 0x8000)


def v5_unsigned(value: int) -> bytes:
    """按 MTEF v5 紧凑格式编码一个无符号整数。"""

    if not 0 <= value <= 0xFFFF:
        raise ValueError("MTEF v5 unsigned fixture is out of range")
    return bytes([value]) if value < 0xFF else b"\xff" + struct.pack("<H", value)


def v5_variation(value: int) -> bytes:
    """按 MTEF v5 一或两字节格式编码 template variation。"""

    if not 0 <= value <= 0x7FFF:
        raise ValueError("MTEF v5 variation fixture is out of range")
    if value < 0x80:
        return bytes([value])
    return bytes([0x80 | (value & 0x7F), value >> 8])


def v5_char(
    value: str,
    *,
    typeface: int = 3,
    embellishments: tuple[int, ...] = (),
    function_start: bool = False,
    font_position: int | None = None,
    omit_mtcode: bool = False,
) -> bytes:
    """构造一个带可选字体位置、函数起点和 embellishments 的 CHAR。"""

    if len(value) != 1 or ord(value) > 0xFFFF:
        raise ValueError("MTEF v5 CHAR fixture requires one BMP character")
    options = 0
    if embellishments:
        options |= 0x01
    if function_start:
        options |= 0x02
    if font_position is not None:
        options |= 0x04 if font_position <= 0xFF else 0x10
    if omit_mtcode:
        options |= 0x20
    payload = bytes([2, options]) + v5_signed(typeface)
    if not omit_mtcode:
        payload += struct.pack("<H", ord(value))
    if font_position is not None:
        payload += (
            bytes([font_position])
            if font_position <= 0xFF
            else struct.pack("<H", font_position)
        )
    if embellishments:
        payload += b"".join(bytes([6, 0, item]) for item in embellishments)
        payload += b"\x00"
    return payload


def v5_text(value: str, *, typeface: int = 3) -> bytes:
    """把 BMP 字符串编码为连续 MTEF v5 CHAR records。"""

    return b"".join(v5_char(character, typeface=typeface) for character in value)


def v5_line(*records: bytes, null: bool = False) -> bytes:
    """构造一个普通或 NULL LINE。"""

    if null:
        if records:
            raise ValueError("NULL LINE fixture cannot contain records")
        return b"\x01\x01"
    return b"\x01\x00" + b"".join(records) + b"\x00"


def v5_template(
    selector: int,
    *slots: bytes,
    variation: int = 0,
    options: int = 0,
) -> bytes:
    """构造一个含完整 subobject list 的 MTEF v5 TMPL。"""

    return (
        bytes([3, 0, selector])
        + v5_variation(variation)
        + bytes([options])
        + b"".join(slots)
        + b"\x00"
    )


def v5_pile(*lines: bytes) -> bytes:
    """构造居中的 MTEF v5 PILE。"""

    return b"\x04\x00\x02\x00" + b"".join(lines) + b"\x00"


def v5_matrix(rows: list[list[bytes]]) -> bytes:
    """构造无 partition lines 的 MTEF v5 MATRIX。"""

    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("MTEF v5 matrix fixture must be rectangular")
    row_count = len(rows)
    col_count = len(rows[0])
    row_parts = b"\x00" * ((2 * (row_count + 1) + 7) // 8)
    col_parts = b"\x00" * ((2 * (col_count + 1) + 7) // 8)
    cells = b"".join(v5_line(cell) for row in rows for cell in row)
    return (
        bytes([5, 0, 0, 2, 0, row_count, col_count])
        + row_parts
        + col_parts
        + cells
        + b"\x00"
    )


def v5_encoding_definition(name: str) -> bytes:
    """构造一个 ENCODING_DEF。"""

    return b"\x13" + name.encode("ascii") + b"\x00"


def v5_font_definition(encoding_index: int, name: str) -> bytes:
    """构造一个 FONT_DEF。"""

    return b"\x11" + v5_unsigned(encoding_index) + name.encode("ascii") + b"\x00"


def v5_font_style_definition(font_index: int, style_bits: int) -> bytes:
    """构造一个 FONT_STYLE_DEF。"""

    return b"\x08" + v5_unsigned(font_index) + bytes([style_bits])


def v5_equation_preferences(
    styles: list[tuple[int, int] | None],
) -> bytes:
    """构造尺寸和间距为空、仅含 style definitions 的 EQN_PREFS。"""

    payload = bytearray([18, 0, 0, 0, len(styles)])
    for style in styles:
        if style is None:
            payload.extend(v5_unsigned(0))
        else:
            font_index, style_bits = style
            payload.extend(v5_unsigned(font_index))
            payload.append(style_bits)
    return bytes(payload)


def v5_future_record(payload: bytes) -> bytes:
    """构造可由旧 reader 按显式长度跳过的 future record。"""

    return b"\x64" + v5_unsigned(len(payload)) + payload


def v5_equation(
    *records: bytes,
    definitions: tuple[bytes, ...] = (),
    application_key: str = "DSMT7",
    inline: bool = False,
) -> bytes:
    """构造完整 v5 header、定义记录、初始 SIZE、根 LINE 和 END。"""

    header = bytes([5, 1, 0, 7, 0])
    header += application_key.encode("ascii") + b"\x00" + bytes([1 if inline else 0])
    return header + b"".join(definitions) + b"\x0a" + v5_line(*records) + b"\x00"


def v5_formula_corpus() -> list[tuple[str, bytes, str]]:
    """返回覆盖常见 v5 CHAR、TMPL、PILE 与 MATRIX 的公式语料。"""

    fraction = v5_template(
        11,
        v5_line(v5_text("a+b")),
        v5_line(v5_char("c")),
    )
    square_root = v5_template(
        10,
        v5_line(
            v5_char("x"),
            v5_template(
                28,
                v5_line(null=True),
                v5_line(v5_char("2")),
            ),
            v5_text("+1"),
        ),
        v5_line(null=True),
    )
    nth_root = v5_template(
        10,
        v5_line(v5_char("x")),
        v5_line(v5_char("3")),
        variation=1,
    )
    scripts = v5_char("x") + v5_template(
        29,
        v5_line(v5_char("i")),
        v5_line(v5_char("2")),
    )
    fenced_fraction = v5_template(
        1,
        v5_line(fraction),
        variation=3,
    )
    summation = v5_template(
        16,
        v5_line(v5_char("i")),
        v5_line(v5_char("n")),
        v5_line(v5_text("i=1")),
        v5_char("∑", typeface=6),
        variation=0x43,
    )
    integral = v5_template(
        15,
        v5_line(v5_char("x")),
        v5_line(v5_char("1")),
        v5_line(v5_char("0")),
        v5_char("∫", typeface=6),
        variation=3,
    )
    matrix = v5_matrix(
        [
            [v5_char("a"), v5_char("b")],
            [v5_char("c"), v5_char("d")],
        ]
    )
    return [
        ("linear", v5_equation(v5_text("x+y")), "x+y"),
        ("fraction", v5_equation(fraction), r"\frac{a+b}{c}"),
        ("square_root", v5_equation(square_root), r"\sqrt{x^{2}+1}"),
        ("nth_root", v5_equation(nth_root), r"\sqrt[3]{x}"),
        ("sub_sup", v5_equation(scripts), r"x_{i}^{2}"),
        ("fence", v5_equation(fenced_fraction), r"\left(\frac{a+b}{c}\right)"),
        ("summation", v5_equation(summation), r"\sum_{i=1}^{n}{i}"),
        ("integral", v5_equation(integral), r"\int_{0}^{1}{x}"),
        ("matrix", v5_equation(matrix), r"\begin{matrix}a&b\\c&d\end{matrix}"),
        ("pile", v5_equation(v5_pile(v5_line(v5_char("x")), v5_line(v5_char("y")))), r"\begin{gathered}x\\y\end{gathered}"),
        ("embellishment", v5_equation(v5_char("x", embellishments=(9,))), r"\widehat{x}"),
        ("greek_relation", v5_equation(v5_char("α"), v5_char("≤"), v5_char("β")), r"\alpha \leq \beta"),
        ("interval", v5_equation(v5_template(9, v5_line(v5_char("x")), variation=0x10)), r"\left(x\right)"),
        ("vector", v5_equation(v5_template(31, v5_line(v5_char("x")), variation=2)), r"\overrightarrow{x}"),
        (
            "dirac",
            v5_equation(
                v5_template(
                    30,
                    v5_line(v5_char("a")),
                    v5_line(v5_char("b")),
                    variation=3,
                )
            ),
            r"\left\langle a\middle|b\right\rangle",
        ),
        ("box", v5_equation(v5_template(37, v5_line(v5_char("x")))), r"\boxed{x}"),
        ("future", v5_equation(v5_future_record(b"future"), v5_char("x")), "x"),
    ]


def v5_template_corpus() -> list[tuple[str, bytes, str]]:
    """返回覆盖 v5 各类标准 template selector 的精确语料。"""

    line_x = v5_line(v5_char("x"))
    line_a = v5_line(v5_char("a"))
    line_b = v5_line(v5_char("b"))
    line_i = v5_line(v5_char("i"))
    line_n = v5_line(v5_char("n"))
    null_line = v5_line(null=True)
    cases = [
        ("angle", v5_template(0, line_x, variation=3), r"\left\langle x\right\rangle"),
        ("parenthesis", v5_template(1, line_x, variation=3), r"\left(x\right)"),
        ("brace", v5_template(2, line_x, variation=3), r"\left\{x\right\}"),
        ("bracket", v5_template(3, line_x, variation=3), r"\left[x\right]"),
        ("bar", v5_template(4, line_x, variation=3), r"\left|x\right|"),
        ("double_bar", v5_template(5, line_x, variation=3), r"\left\|x\right\|"),
        ("floor", v5_template(6, line_x, variation=3), r"\left\lfloor x\right\rfloor"),
        ("ceiling", v5_template(7, line_x, variation=3), r"\left\lceil x\right\rceil"),
        ("white_bracket", v5_template(8, line_x, variation=3), r"\left\llbracket x\right\rrbracket"),
        ("slash_fraction", v5_template(11, line_a, line_b, variation=2), r"{a}/{b}"),
        ("double_underbar", v5_template(12, line_x, variation=1), r"\underline{\underline{x}}"),
        ("overbar", v5_template(13, line_x), r"\overline{x}"),
        ("arrow", v5_template(14, line_a, variation=0x20), r"\overset{a}{\rightarrow}"),
        ("product", v5_template(17, line_x, line_n, line_i, v5_char("∏"), variation=3), r"\prod_{i}^{n}{x}"),
        ("coproduct", v5_template(18, line_x, line_n, line_i, v5_char("∐"), variation=3), r"\coprod_{i}^{n}{x}"),
        ("union", v5_template(19, line_x, line_n, line_i, v5_char("∪"), variation=3), r"\bigcup_{i}^{n}{x}"),
        ("intersection", v5_template(20, line_x, line_n, line_i, v5_char("∩"), variation=3), r"\bigcap_{i}^{n}{x}"),
        ("custom_bigop", v5_template(21, line_x, line_n, line_i, v5_char("∫"), variation=3), r"\int_{i}^{n}{x}"),
        ("sum_style_bigop", v5_template(22, line_x, line_n, line_i, v5_char("∑"), variation=0x43), r"\sum_{i}^{n}{x}"),
        (
            "two_byte_variation",
            v5_template(
                15,
                line_x,
                null_line,
                null_line,
                v5_char("∫"),
                variation=0x100,
            ),
            r"\int{x}",
        ),
        ("limit", v5_template(23, line_x, v5_line(v5_char("0")), v5_line(v5_char("1"))), r"\lim_{0}^{1}{x}"),
        ("overbrace", v5_template(24, line_x, line_n, variation=1), r"\overbrace{x}^{n}"),
        ("underbracket", v5_template(25, line_x, line_n), r"\underbracket{x}_{n}"),
        ("long_division", v5_template(26, line_x, v5_line(v5_char("q")), variation=1), r"\overline{q}\smash{\big) x}"),
        ("subscript", v5_template(27, line_i, null_line), r"_{i}"),
        ("superscript", v5_template(28, null_line, v5_line(v5_char("2"))), r"^{2}"),
        ("prescripts", v5_template(29, line_i, v5_line(v5_char("2")), variation=1), r"{}_{i}^{2}"),
        ("under_vector", v5_template(31, line_x, variation=0x05), r"\underleftarrow{x}"),
        ("tilde", v5_template(32, line_x), r"\widetilde{x}"),
        ("hat", v5_template(33, line_x), r"\widehat{x}"),
        ("arc", v5_template(34, line_x), r"\overparen{x}"),
        ("cancel", v5_template(36, line_x, variation=2), r"\cancel{x}"),
        ("back_cancel", v5_template(36, line_x, variation=4), r"\bcancel{x}"),
        ("cross_cancel", v5_template(36, line_x, variation=6), r"\xcancel{x}"),
        ("box", v5_template(37, line_x, variation=0x1E), r"\boxed{x}"),
    ]
    return [
        (name, v5_equation(template), expected)
        for name, template, expected in cases
    ]
