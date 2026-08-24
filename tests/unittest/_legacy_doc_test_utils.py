"""构造不依赖 Word 的最小 Word 97–2003 测试文件。"""

from __future__ import annotations

import struct

from _legacy_ppt_test_utils import _build_cfb


def _piece_table(text_fc: int, cp_count: int, *, compressed: bool) -> bytes:
    """构造只含一个 piece 的 CLX。"""

    fc_raw = text_fc * 2 | 0x4000_0000 if compressed else text_fc
    plc = struct.pack("<II", 0, cp_count)
    plc += b"\x00\x00" + struct.pack("<I", fc_raw) + b"\x00\x00"
    return b"\x02" + struct.pack("<I", len(plc)) + plc


def _section_plc(section_ends: list[int]) -> bytes:
    """构造只提供 CP 边界、Sed 使用零值的 PlcfSed。"""

    starts = [0, *section_ends]
    return struct.pack(f"<{len(starts)}I", *starts) + b"\x00" * (12 * len(section_ends))


def build_doc(
    text: str,
    *,
    section_ends: list[int] | None = None,
    compressed: bool = False,
    flags_extra: int = 0,
    n_fib: int = 0x00C1,
    footnote_text: str | None = None,
    codec: str = "cp1252",
    lid: int = 0x0409,
) -> bytes:
    """构造带 CLX 和 PlcfSed 的最小 DOC OLE 文件。"""

    main_encoded = text.encode(codec) if compressed else text.encode("utf-16le")
    main_cp_count = len(main_encoded) if compressed else len(main_encoded) // 2
    footnote_value = footnote_text or ""
    footnote_encoded = footnote_value.encode(codec) if compressed else footnote_value.encode("utf-16le")
    footnote_cp_count = len(footnote_encoded) if compressed else len(footnote_encoded) // 2
    encoded = main_encoded + footnote_encoded
    cp_count = main_cp_count + footnote_cp_count
    text_fc = 1024
    clx = _piece_table(text_fc, cp_count, compressed=compressed)
    section_ends = section_ends or [main_cp_count]
    sections = _section_plc(section_ends)
    table = clx + sections

    pair_count = 75
    pairs = [(0, 0)] * pair_count
    pairs[6] = (len(clx), len(sections))
    pairs[33] = (0, len(clx))
    if footnote_text is not None:
        reference_cp = text.index("\x02")
        footnote_ref = struct.pack("<IIH", reference_cp, main_cp_count, 1)
        footnote_ranges = struct.pack("<II", 0, footnote_cp_count)
        pairs[2] = (len(table), len(footnote_ref))
        table += footnote_ref
        pairs[3] = (len(table), len(footnote_ranges))
        table += footnote_ranges
    else:
        # 合法空脚注 reference PLC：仅保留终止 CP。
        pairs[2] = (len(table), 4)
        table += struct.pack("<I", 0)

    base = bytearray(32)
    struct.pack_into("<HH", base, 0, 0xA5EC, n_fib)
    struct.pack_into("<H", base, 6, lid)
    flags = 0x0004 | 0x0200 | flags_extra
    struct.pack_into("<H", base, 10, flags)
    struct.pack_into("<II", base, 24, text_fc, text_fc + len(encoded))
    rgw = [0] * 14
    if flags & 0x4000:
        rgw[13] = lid
    rglw = [0] * 11
    rglw[0] = text_fc + len(encoded)
    rglw[3] = main_cp_count
    rglw[4] = footnote_cp_count
    fib = bytes(base)
    fib += struct.pack("<H", len(rgw)) + struct.pack(f"<{len(rgw)}H", *rgw)
    fib += struct.pack("<H", len(rglw)) + struct.pack(f"<{len(rglw)}I", *rglw)
    fib += struct.pack("<H", pair_count)
    fib += b"".join(struct.pack("<II", offset, size) for offset, size in pairs)
    fib += struct.pack("<H", 0)
    word_document = fib + b"\x00" * (text_fc - len(fib)) + encoded
    return _build_cfb(
        [
            ("1Table", table),
            ("Data", b""),
            ("WordDocument", word_document),
        ]
    )


def utf16_cp(text: str) -> int:
    """返回字符串占用的 UTF-16 code unit 数。"""

    return len(text.encode("utf-16le")) // 2
