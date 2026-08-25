from __future__ import annotations

import struct

import pytest

from mineru.model.flash.office.legacy import mtef_v5 as mtef_v5_module
from mineru.model.flash.office.legacy import mtef as mtef_module
from mineru.model.flash.office.legacy.errors import LegacyOfficeResourceLimitError
from mineru.model.flash.office.legacy.mtef import (
    decode_equation_native,
    decode_equation_object,
    decode_mtef,
    decode_mtef_v3,
    decode_mtef_v5,
)

from _mtef_test_utils import build_equation_object, equation_native, formula_corpus
from _mtef_v5_test_utils import (
    v5_char,
    v5_encoding_definition,
    v5_equation,
    v5_equation_preferences,
    v5_font_definition,
    v5_font_style_definition,
    v5_formula_corpus,
    v5_future_record,
    v5_line,
    v5_template,
    v5_template_corpus,
    v5_text,
)

# WIRIS MTEF v5 页面公布的 MathType 7 二次公式字节表。原网页第 133 行
# 被帮助文本覆盖，按 dimension nibble padding 归一化为 0；第 255 行误插入
# 的 HTML “&” 字节不属于 CHAR record，构造 golden 时予以移除。
_WIRIS_QUADRATIC_MTEF = bytes.fromhex(
    "050100070044534d543700001357696e416c6c4261736963436f646550616765"
    "7300110554696d6573204e657720526f6d616e00110353796d626f6c00110543"
    "6f7572696572204e65770011044d5420457874726100120008212f458f442f41"
    "50f4100f475f4150f21f1e4150f4150f4100f445f425f48f425f4100f4100f43"
    "5f4100f21f00a5f20a25f48f41f4100f4000f40f48f417f48f4000f21a5f445f"
    "45f45f45f45f410f0c0100010001020202020002000101010003000100040000"
    "0a010003000b0000010002048612222d0200836200020486b100b103000a0000"
    "0100020083620003001c00000b01010100020088320000000a02048612222d02"
    "0088340002008361000200836300000b010100000a0100020088320002008361"
    "0000000000"
)


@pytest.mark.parametrize(
    ("name", "mtef", "expected"),
    v5_formula_corpus(),
    ids=[name for name, _mtef, _expected in v5_formula_corpus()],
)
def test_mtef_v5_corpus_decodes_to_exact_latex(
    name: str,
    mtef: bytes,
    expected: str,
) -> None:
    """验证常见 MTEF v5 字符、模板、PILE 和 MATRIX 精确输出。"""

    assert name
    assert decode_mtef_v5(mtef) == expected
    assert decode_mtef(mtef) == expected


@pytest.mark.parametrize(
    ("name", "mtef", "expected"),
    v5_template_corpus(),
    ids=[name for name, _mtef, _expected in v5_template_corpus()],
)
def test_mtef_v5_template_selectors_decode_to_exact_latex(
    name: str,
    mtef: bytes,
    expected: str,
) -> None:
    """验证标准 fence、bar、big-op、script、vector 和 box templates。"""

    assert name
    assert decode_mtef_v5(mtef) == expected


def test_mtef_dispatch_keeps_v3_and_rejects_v4() -> None:
    """验证通用入口只按 header 分派 v3/v5，v3 输出保持不变。"""

    _name, v3, expected = formula_corpus()[1]

    assert decode_mtef(v3) == expected
    assert decode_mtef_v3(v3) == expected
    assert decode_mtef(bytes([4, 1, 0, 3, 5, 0])) is None


def test_wiris_mathtype_7_quadratic_golden_decodes_exactly() -> None:
    """验证 WIRIS 公布的真实 MathType 7 v5 字节流恢复二次公式。"""

    assert decode_mtef_v5(_WIRIS_QUADRATIC_MTEF) == (
        r"\frac{-\mathit{b}\pm \sqrt{\mathit{b}^{2}-4"
        r"\mathit{a}\mathit{c}}}{2\mathit{a}}"
    )


def test_equation_native_and_ole_object_decode_mtef_v5() -> None:
    """验证 EQNOLEFILEHDR 与独立 OLE 对象均按首字节解码 v5。"""

    _name, mtef, expected = v5_formula_corpus()[1]

    assert decode_equation_native(equation_native(mtef)) == expected
    assert decode_equation_object(build_equation_object(mtef)) == expected


def test_mtef_v5_known_symbol_font_position_and_style_definitions() -> None:
    """验证无 MTCode 的 Symbol position 及显式 FONT_STYLE_DEF 粗体。"""

    symbol = v5_char(
        "x",
        typeface=6,
        font_position=0x61,
        omit_mtcode=True,
    )
    definitions = (
        v5_encoding_definition("WinAllBasicCodePages"),
        v5_font_definition(5, "Times New Roman"),
        v5_font_style_definition(1, 1),
    )
    bold = v5_char("x", typeface=-1)
    mt_extra = v5_char(
        "x",
        typeface=11,
        font_position=0x68,
        omit_mtcode=True,
    )

    assert decode_mtef_v5(v5_equation(symbol)) == r"\alpha"
    assert decode_mtef_v5(v5_equation(bold, definitions=definitions)) == r"\mathbf{x}"
    assert decode_mtef_v5(v5_equation(mt_extra)) == r"\hbar"


@pytest.mark.parametrize(
    ("mtcode", "expected"),
    [
        (0xE930, r"\leqslant"),
        (0xF000, r"\mathfrak{A}"),
        (0xF01A, r"\mathfrak{a}"),
        (0xF080, r"\mathbb{A}"),
        (0xF100, r"\mathcal{A}"),
    ],
)
def test_mtef_v5_common_private_mtcode_mapping(
    mtcode: int,
    expected: str,
) -> None:
    """验证常见 MathType PUA 关系符和数学字母可稳定映射。"""

    record = b"\x02\x00\x83" + struct.pack("<H", mtcode)

    assert decode_mtef_v5(v5_equation(record)) == expected


@pytest.mark.parametrize(
    ("embellishment", "expected"),
    [
        (7, "{}'x"),
        (14, r"\overrightharpoon{x}"),
        (15, r"\overleftharpoon{x}"),
        (21, r"\xcancel{x}"),
        (29, r"\underline{x}"),
        (33, r"\underrightarrow{x}"),
    ],
)
def test_mtef_v5_extended_embellishment_mapping(
    embellishment: int,
    expected: str,
) -> None:
    """验证 v5 新增的反向 prime、harpoon、strike 和下方修饰符。"""

    assert decode_mtef_v5(
        v5_equation(v5_char("x", embellishments=(embellishment,)))
    ) == expected


def test_mtef_v5_16bit_font_position_and_large_future_record() -> None:
    """验证 16 位 font position 和三字节 future length 保持同步。"""

    definitions = (
        v5_font_definition(1, "MTCode Font"),
        v5_font_style_definition(1, 0),
    )
    alpha = v5_char(
        "x",
        typeface=-1,
        font_position=0x03B1,
        omit_mtcode=True,
    )
    mtef = v5_equation(
        v5_future_record(b"x" * 300),
        alpha,
        definitions=definitions,
    )

    assert decode_mtef_v5(mtef) == r"\alpha"


def test_mtef_v5_color_and_size_metadata_stays_synchronized() -> None:
    """验证 COLOR_DEF、COLOR 和 SIZE 只校验结构而不改变公式内容。"""

    color_definition = b"\x10\x00" + struct.pack("<HHH", 0, 0, 0)
    color_reference = b"\x0f\x01"
    size = b"\x09\x00\x80"

    assert decode_mtef_v5(
        v5_equation(
            v5_char("x"),
            definitions=(color_definition, color_reference, size),
        )
    ) == "x"
    assert decode_mtef_v5(
        v5_equation(v5_char("x"), definitions=(color_reference,))
    ) is None


def test_mtef_v5_function_start_groups_function_style_characters() -> None:
    """验证 FUNC_START 将连续 FUNCTION 字符恢复为 LaTeX operator。"""

    function = (
        v5_char("s", typeface=2, function_start=True)
        + v5_char("i", typeface=2)
        + v5_char("n", typeface=2)
        + v5_char("x")
    )

    assert decode_mtef_v5(v5_equation(function)) == r"\sin x"


def test_mtef_v5_text_style_groups_and_escapes_visible_text() -> None:
    """验证 TEXT/TEXT_FE 连续字符使用单一 text 节点并安全转义。"""

    assert decode_mtef_v5(
        v5_equation(v5_text("rate_1 & rate_2", typeface=1))
    ) == r"\text{rate\_1 \& rate\_2}"


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x05\x01\x00\x07\x00unterminated",
        v5_equation(v5_template(35, v5_line(v5_char("x")))),
        v5_equation(v5_template(37, v5_line(v5_char("x")), variation=1)),
        v5_equation(v5_char("\ue000")),
        v5_equation(v5_char("x", font_position=1, omit_mtcode=True, typeface=11)),
        v5_equation(b"\x14"),
        v5_equation(b"\x64\xff\xff\xff"),
        v5_equation(b"\x02\x14\x83\x78\x00\x01\x00"),
        v5_equation(b"\x02\x00\x80\x78\x00"),
        v5_equation(b"\x06\x00\x02"),
        v5_equation(b"\x05\x00\x00\x02\x00\x00\x01"),
        v5_equation(v5_char("x")) + b"\x00",
    ],
)
def test_mtef_v5_invalid_or_unsupported_payload_fails_closed(
    payload: bytes,
) -> None:
    """验证坏 header、未知语义、PUA、MTExtra 和尾随字节不生成残缺 LaTeX。"""

    assert decode_mtef_v5(payload) is None


def test_every_strict_prefix_of_mtef_v5_fails_closed() -> None:
    """验证任意位置截断的 v5 对象都不会输出部分公式。"""

    _name, mtef, _expected = v5_formula_corpus()[1]

    assert all(decode_mtef_v5(mtef[:end]) is None for end in range(len(mtef)))


def test_mtef_v5_invalid_equation_native_lengths_fail_closed() -> None:
    """验证 v5 仍受 EQNOLEFILEHDR 对象边界约束。"""

    _name, mtef, _expected = v5_formula_corpus()[0]
    native = bytearray(equation_native(mtef))
    struct.pack_into("<I", native, 8, len(mtef) + 1)

    assert decode_equation_native(bytes(native)) is None


def test_mtef_v5_record_and_depth_limits_raise_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 v5 record/depth 超限抛稳定 resource-limit 错误。"""

    monkeypatch.setattr(mtef_v5_module, "MAX_RECORDS", 2)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_records"):
        decode_mtef_v5(v5_equation(v5_char("x")))

    monkeypatch.setattr(mtef_v5_module, "MAX_RECORDS", 100)
    monkeypatch.setattr(mtef_v5_module, "MAX_RECORD_DEPTH", 1)
    nested = v5_equation(
        v5_template(11, v5_line(v5_char("a")), v5_line(v5_char("b")))
    )
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_record_depth"):
        decode_mtef_v5(nested)


def test_mtef_v3_record_and_depth_limits_also_raise_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证共享资源超限语义同时覆盖原有 v3 reader。"""

    _name, mtef, _expected = formula_corpus()[1]
    monkeypatch.setattr(mtef_module, "MAX_RECORDS", 2)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_records"):
        decode_mtef_v3(mtef)

    monkeypatch.setattr(mtef_module, "MAX_RECORDS", 100)
    monkeypatch.setattr(mtef_module, "MAX_RECORD_DEPTH", 0)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_record_depth"):
        decode_mtef_v3(mtef)


def test_mtef_v5_equation_preferences_style_reference_is_validated() -> None:
    """验证 EQN_PREFS style 引用不存在的 FONT_DEF 时整体回退。"""

    invalid_preferences = v5_equation_preferences([(1, 0)])

    assert decode_mtef_v5(
        v5_equation(v5_char("x"), definitions=(invalid_preferences,))
    ) is None
