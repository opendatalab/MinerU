# Copyright (c) Opendatalab. All rights reserved.

"""BIFF5–BIFF8 字符串、富文本区间与 codepage 解码。"""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct

from .models import XlsFontStyle, XlsRichRun, XlsRichText
from .records import SegmentReader


@dataclass(frozen=True, slots=True)
class DecodedString:
    """已解码文本及以 UTF-16 code unit 表示的字体切换点。"""

    text: str
    font_starts: tuple[tuple[int, int], ...] = ()


def clean_text(text: str) -> str:
    """规范换行、NUL 与不可见控制字符，同时保留制表符。"""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\x00")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", normalized)


def codepage_name(codepage: int) -> str:
    """把 BIFF CODEPAGE 值映射为 Python codec 名称。"""

    return {
        874: "cp874",
        932: "shift_jis",
        936: "gbk",
        949: "euc_kr",
        950: "big5",
        1200: "utf-16le",
        1250: "cp1250",
        1251: "cp1251",
        1252: "cp1252",
        1253: "cp1253",
        1254: "cp1254",
        1255: "cp1255",
        1256: "cp1256",
        1257: "cp1257",
        1258: "cp1258",
    }.get(int(codepage), "cp1252")


def _decode_utf16_units(units: list[int]) -> str:
    """容错解码 UTF-16 code units。"""

    payload = struct.pack(f"<{len(units)}H", *units) if units else b""
    return payload.decode("utf-16le", "replace")


def read_biff8_string(
    reader: SegmentReader,
    *,
    short: bool,
    rich: bool,
) -> DecodedString | None:
    """读取可跨 CONTINUE 且可切换压缩模式的 BIFF8 Unicode 字符串。"""

    character_count = reader.u8() if short else reader.u16()
    flags = reader.u8()
    if character_count is None or flags is None:
        return None
    wide = bool(flags & 0x01)
    run_count = reader.u16() if rich and flags & 0x08 else 0
    extension_size = reader.u32() if rich and flags & 0x04 else 0
    if run_count is None or extension_size is None:
        return None

    units: list[int] = []
    remaining = int(character_count)
    while remaining > 0:
        if reader.remaining_in_segment() == 0:
            if not reader.next_segment():
                return None
            repeated_flags = reader.u8()
            if repeated_flags is None:
                return None
            wide = bool(repeated_flags & 0x01)
        unit_size = 2 if wide else 1
        take = min(reader.remaining_in_segment() // unit_size, remaining)
        if take <= 0:
            return None
        payload = reader.read(take * unit_size)
        if payload is None:
            return None
        if wide:
            units.extend(struct.unpack(f"<{take}H", payload))
        else:
            units.extend(int(value) for value in payload)
        remaining -= take

    font_starts: list[tuple[int, int]] = []
    for _ in range(int(run_count)):
        run = reader.read_across(4)
        if run is None:
            break
        character_index, font_index = struct.unpack("<HH", run)
        if character_index <= character_count:
            font_starts.append((int(character_index), int(font_index)))
    if extension_size:
        reader.skip(int(extension_size))
    return DecodedString(
        text=clean_text(_decode_utf16_units(units)),
        font_starts=tuple(font_starts),
    )


def read_byte_string(
    reader: SegmentReader,
    *,
    short: bool,
    encoding: str,
) -> DecodedString | None:
    """按 workbook CODEPAGE 读取 BIFF5/BIFF7 单字节字符串。"""

    character_count = reader.u8() if short else reader.u16()
    if character_count is None:
        return None
    payload = reader.read_across(int(character_count))
    if payload is None:
        return None
    try:
        text = payload.decode(encoding, "replace")
    except LookupError:
        text = payload.decode("cp1252", "replace")
    return DecodedString(clean_text(text))


def utf16_unit_to_index(text: str, unit_offset: int) -> int:
    """把 UTF-16 code unit 偏移转换为 Python 字符索引。"""

    units = 0
    for index, char in enumerate(text):
        if units >= unit_offset:
            return index
        units += 2 if ord(char) > 0xFFFF else 1
    return len(text)


def to_rich_text(
    decoded: DecodedString,
    fonts: list[XlsFontStyle],
) -> XlsRichText:
    """把字体切换点解析成稳定的字符区间。"""

    if not decoded.font_starts or not decoded.text:
        return XlsRichText(decoded.text)
    starts = sorted(decoded.font_starts, key=lambda item: item[0])
    runs: list[XlsRichRun] = []
    for index, (unit_start, font_index) in enumerate(starts):
        unit_end = starts[index + 1][0] if index + 1 < len(starts) else 0x7FFF_FFFF
        start = utf16_unit_to_index(decoded.text, unit_start)
        end = utf16_unit_to_index(decoded.text, unit_end)
        resolved_font_index = font_index if font_index < 4 else font_index - 1
        style = (
            fonts[resolved_font_index]
            if 0 <= resolved_font_index < len(fonts)
            else XlsFontStyle()
        )
        if start < end and style != XlsFontStyle():
            runs.append(XlsRichRun(start=start, end=end, style=style))
    return XlsRichText(decoded.text, tuple(runs))


def read_txo_text(
    base_payload: bytes,
    continuation_segments: list[bytes],
    fonts: list[XlsFontStyle],
) -> XlsRichText | None:
    """读取 TXO 文本和 8 字节 formatting runs。"""

    if len(base_payload) < 14:
        return None
    character_count = int(struct.unpack_from("<H", base_payload, 10)[0])
    formatting_size = int(struct.unpack_from("<H", base_payload, 12)[0])
    if character_count <= 0:
        return XlsRichText("")
    units: list[int] = []
    consumed_segments = 0
    remaining = character_count
    for segment in continuation_segments:
        if remaining <= 0:
            break
        consumed_segments += 1
        if not segment:
            continue
        wide = bool(segment[0] & 0x01)
        payload = segment[1:]
        if wide:
            take = min(remaining, len(payload) // 2)
            units.extend(struct.unpack(f"<{take}H", payload[: take * 2]))
        else:
            take = min(remaining, len(payload))
            units.extend(int(value) for value in payload[:take])
        remaining -= take
    text = clean_text(_decode_utf16_units(units))
    formatting = b"".join(continuation_segments[consumed_segments:])[:formatting_size]
    starts: list[tuple[int, int]] = []
    for offset in range(0, len(formatting) - 7, 8):
        character_index, font_index = struct.unpack_from("<HH", formatting, offset)
        if character_index <= character_count:
            starts.append((int(character_index), int(font_index)))
    return to_rich_text(DecodedString(text, tuple(starts)), fonts)
