"""构造确定性 WMF/EMF 测试载荷。"""

from __future__ import annotations

import struct


def _pad4(data: bytes) -> bytes:
    """把 EMF payload 填充到 4 字节边界。"""
    return data + b"\x00" * (-len(data) % 4)


def emf_record(record_type: int, payload: bytes = b"") -> bytes:
    """构造带通用头部和 4 字节对齐的 EMF record。"""
    aligned = _pad4(payload)
    return struct.pack("<II", record_type, 8 + len(aligned)) + aligned


def _emf_header(
    total_bytes: int,
    record_count: int,
    *,
    bounds: tuple[int, int, int, int],
    frame: tuple[int, int, int, int],
) -> bytes:
    """构造满足 EMR_HEADER 基础版本的 88 字节头部。"""
    header = bytearray(88)
    struct.pack_into("<II", header, 0, 1, 88)
    struct.pack_into("<4i", header, 8, *bounds)
    struct.pack_into("<4i", header, 24, *frame)
    struct.pack_into("<I", header, 40, 0x464D4520)
    struct.pack_into("<I", header, 44, 0x00010000)
    struct.pack_into("<II", header, 48, total_bytes, record_count)
    struct.pack_into("<HH", header, 56, 16, 0)
    struct.pack_into("<III", header, 60, 0, 0, 0)
    struct.pack_into("<2i", header, 72, 1000, 1000)
    struct.pack_into("<2i", header, 80, 254, 254)
    return bytes(header)


def build_emf(
    records: list[bytes],
    *,
    bounds: tuple[int, int, int, int] = (0, 0, 100, 100),
    frame: tuple[int, int, int, int] = (0, 0, 2540, 2540),
) -> bytes:
    """为给定 records 补齐 EMR_HEADER 与 EMR_EOF。"""
    eof = emf_record(14, struct.pack("<III", 0, 0, 20))
    body = b"".join(records) + eof
    total_bytes = 88 + len(body)
    return _emf_header(total_bytes, len(records) + 2, bounds=bounds, frame=frame) + body


def emf_create_pen(handle: int, colorref: int, *, width: int = 1, style: int = 0) -> bytes:
    """构造 EMR_CREATEPEN。"""
    return emf_record(38, struct.pack("<IIiiI", handle, style, width, 0, colorref))


def emf_create_brush(handle: int, colorref: int, *, style: int = 0) -> bytes:
    """构造 EMR_CREATEBRUSHINDIRECT。"""
    return emf_record(39, struct.pack("<IIII", handle, style, colorref, 0))


def emf_select_object(handle: int) -> bytes:
    """构造 EMR_SELECTOBJECT。"""
    return emf_record(37, struct.pack("<I", handle))


def emf_rectangle(left: int, top: int, right: int, bottom: int) -> bytes:
    """构造 EMR_RECTANGLE。"""
    return emf_record(43, struct.pack("<4i", left, top, right, bottom))


def emf_savedc() -> bytes:
    """构造 EMR_SAVEDC。"""
    return emf_record(33)


def emf_restoredc(level: int = -1) -> bytes:
    """构造 EMR_RESTOREDC。"""
    return emf_record(34, struct.pack("<i", level))


def emf_set_world_transform(matrix: tuple[float, float, float, float, float, float]) -> bytes:
    """构造 EMR_SETWORLDTRANSFORM。"""
    return emf_record(35, struct.pack("<6f", *matrix))


def emf_move_to(x: int, y: int) -> bytes:
    """构造 EMR_MOVETOEX。"""
    return emf_record(27, struct.pack("<2i", x, y))


def emf_line_to(x: int, y: int) -> bytes:
    """构造 EMR_LINETO。"""
    return emf_record(54, struct.pack("<2i", x, y))


def emf_intersect_clip_rect(left: int, top: int, right: int, bottom: int) -> bytes:
    """构造 EMR_INTERSECTCLIPRECT。"""
    return emf_record(30, struct.pack("<4i", left, top, right, bottom))


def emf_angle_arc(center_x: int, center_y: int, radius: int, start_angle: float, sweep_angle: float) -> bytes:
    """构造 EMR_ANGLEARC。"""
    return emf_record(41, struct.pack("<iiIff", center_x, center_y, radius, start_angle, sweep_angle))


def emf_begin_path() -> bytes:
    """构造 EMR_BEGINPATH。"""
    return emf_record(59)


def emf_end_path() -> bytes:
    """构造 EMR_ENDPATH。"""
    return emf_record(60)


def emf_close_figure() -> bytes:
    """构造 EMR_CLOSEFIGURE。"""
    return emf_record(61)


def emf_fill_path() -> bytes:
    """构造 EMR_FILLPATH。"""
    return emf_record(62)


def emf_stroke_and_fill_path() -> bytes:
    """构造 EMR_STROKEANDFILLPATH。"""
    return emf_record(63)


def emf_stroke_path() -> bytes:
    """构造 EMR_STROKEPATH。"""
    return emf_record(64)


def _emf_compact_poly_record(record_type: int, points: list[tuple[int, int]]) -> bytes:
    """构造使用 PointS 数组的紧凑 EMF poly record。"""
    if points:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        bounds = min(xs), min(ys), max(xs), max(ys)
    else:
        bounds = (0, 0, 0, 0)
    payload = struct.pack("<4iI", *bounds, len(points))
    payload += b"".join(struct.pack("<2h", *point) for point in points)
    return emf_record(record_type, payload)


def emf_polyline_to(points: list[tuple[int, int]]) -> bytes:
    """构造 EMR_POLYLINETO16。"""
    return _emf_compact_poly_record(89, points)


def emf_polybezier(points: list[tuple[int, int]], *, to: bool) -> bytes:
    """构造 EMR_POLYBEZIER16 或 EMR_POLYBEZIERTO16。"""
    return _emf_compact_poly_record(88 if to else 85, points)


def emf_set_polyfill_mode(mode: int) -> bytes:
    """构造 EMR_SETPOLYFILLMODE。"""
    return emf_record(19, struct.pack("<I", mode))


def emf_set_miter_limit(value: float) -> bytes:
    """构造 EMR_SETMITERLIMIT。"""
    return emf_record(58, struct.pack("<f", value))


def emf_set_text_align(value: int) -> bytes:
    """构造 EMR_SETTEXTALIGN。"""
    return emf_record(22, struct.pack("<I", value))


def emf_font(handle: int, face_name: str = "DejaVu Sans", *, height: int = -14) -> bytes:
    """构造只填充常用 LOGFONTW 字段的 EMR_EXTCREATEFONTINDIRECTW。"""
    logfont = bytearray(92)
    struct.pack_into("<iiiii", logfont, 0, height, 0, 0, 0, 400)
    logfont[23] = 1
    encoded = face_name.encode("utf-16le")[:62]
    logfont[28 : 28 + len(encoded)] = encoded
    return emf_record(82, struct.pack("<I", handle) + bytes(logfont))


def emf_text(text: str, x: int, y: int, *, dx: int | None = 12) -> bytes:
    """构造可选显式 Dx 数组的 EMR_EXTTEXTOUTW。"""
    encoded = text.encode("utf-16le")
    record = bytearray(76)
    nominal_dx = dx if dx is not None else 12
    struct.pack_into("<II", record, 0, 84, 0)
    struct.pack_into("<4i", record, 8, x, y - 20, x + max(1, len(text)) * nominal_dx, y + 5)
    struct.pack_into("<Iff", record, 24, 1, 1.0, 1.0)
    struct.pack_into("<2i", record, 36, x, y)
    struct.pack_into("<I", record, 44, len(text))
    struct.pack_into("<I", record, 48, 76)
    struct.pack_into("<I", record, 52, 0)
    struct.pack_into("<4i", record, 56, 0, 0, -1, -1)
    string_and_padding = _pad4(encoded)
    record.extend(string_and_padding)
    if dx is not None:
        dx_offset = 76 + len(string_and_padding)
        struct.pack_into("<I", record, 72, dx_offset)
        record.extend(struct.pack(f"<{len(text)}i", *([dx] * len(text))))
    struct.pack_into("<I", record, 4, len(record))
    return bytes(record)


def emf_stretch_dib() -> bytes:
    """构造含红绿蓝白四像素的 EMR_STRETCHDIBITS。"""
    header = bytearray(40)
    struct.pack_into("<IiiHHIIiiII", header, 0, 40, 2, -2, 1, 32, 0, 16, 0, 0, 0, 0)
    bits = bytes(
        (
            0,
            0,
            255,
            255,
            0,
            255,
            0,
            255,
            255,
            0,
            0,
            255,
            255,
            255,
            255,
            255,
        )
    )
    record = bytearray(80)
    struct.pack_into("<II", record, 0, 81, 80 + len(header) + len(bits))
    struct.pack_into("<4i", record, 8, 10, 10, 90, 90)
    struct.pack_into("<6i", record, 24, 10, 10, 0, 0, 2, 2)
    struct.pack_into("<4I", record, 48, 80, len(header), 80 + len(header), len(bits))
    struct.pack_into("<IIii", record, 64, 0, 0x00CC0020, 80, 80)
    record.extend(header)
    record.extend(bits)
    return bytes(record)


def emfplus_comment(*, dual: bool) -> bytes:
    """构造只含 EMF+ Header 的 EMR_COMMENT。"""
    plus = struct.pack("<IHHII", 0x2B464D45, 0x4001, 1 if dual else 0, 12, 0)
    return emf_record(70, struct.pack("<I", len(plus)) + plus)


def wmf_record(function: int, payload: bytes = b"") -> bytes:
    """构造按 WORD 计长的 WMF record。"""
    if len(payload) % 2:
        payload += b"\x00"
    return struct.pack("<IH", (6 + len(payload)) // 2, function) + payload


def wmf_move_to(x: int, y: int) -> bytes:
    """构造 META_MOVETO。"""
    return wmf_record(0x0214, struct.pack("<hh", y, x))


def wmf_set_text_align(value: int) -> bytes:
    """构造 META_SETTEXTALIGN。"""
    return wmf_record(0x012E, struct.pack("<H", value))


def wmf_set_map_mode(value: int) -> bytes:
    """构造 META_SETMAPMODE。"""
    return wmf_record(0x0103, struct.pack("<h", value))


def wmf_rectangle(left: int, top: int, right: int, bottom: int) -> bytes:
    """构造 META_RECTANGLE。"""
    return wmf_record(0x041B, struct.pack("<hhhh", bottom, right, top, left))


def wmf_textout(text: str, x: int, y: int) -> bytes:
    """构造没有显式字符 spacing 的 META_TEXTOUT。"""
    encoded = text.encode("cp1252")
    payload = struct.pack("<H", len(encoded)) + encoded
    if len(encoded) & 1:
        payload += b"\x00"
    payload += struct.pack("<hh", y, x)
    return wmf_record(0x0521, payload)


def build_placeable_wmf(records: list[bytes], *, bbox: tuple[int, int, int, int] = (0, 0, 1000, 1000)) -> bytes:
    """构造带 Aldus placeable header 的标准 WMF。"""
    eof = wmf_record(0)
    body = b"".join(records) + eof
    standard = struct.pack("<HHHIHIH", 1, 9, 0x0300, (18 + len(body)) // 2, 16, max(3, len(body) // 2), 0)
    placeable = bytearray(struct.pack("<IHhhhhHI", 0x9AC6CDD7, 0, *bbox, 1000, 0))
    checksum = 0
    for offset in range(0, 20, 2):
        checksum ^= struct.unpack_from("<H", placeable, offset)[0]
    placeable.extend(struct.pack("<H", checksum))
    return bytes(placeable) + standard + body


def basic_wmf() -> bytes:
    """构造包含画笔、画刷、矩形与多边形的 placeable WMF。"""
    records = [
        wmf_record(0x02FA, struct.pack("<HhhI", 0, 8, 0, 0x000000FF)),
        wmf_record(0x02FC, struct.pack("<HIH", 0, 0x0000FF00, 0)),
        wmf_record(0x012D, struct.pack("<H", 0)),
        wmf_record(0x012D, struct.pack("<H", 1)),
        wmf_record(0x041B, struct.pack("<hhhh", 800, 800, 200, 200)),
        wmf_record(
            0x0324,
            struct.pack("<Hhhhhhh", 3, 100, 900, 500, 100, 900, 900),
        ),
    ]
    return build_placeable_wmf(records)


__all__ = [
    "basic_wmf",
    "build_emf",
    "build_placeable_wmf",
    "emf_angle_arc",
    "emf_create_brush",
    "emf_create_pen",
    "emf_begin_path",
    "emf_close_figure",
    "emf_end_path",
    "emf_fill_path",
    "emf_font",
    "emf_intersect_clip_rect",
    "emf_line_to",
    "emf_move_to",
    "emf_polybezier",
    "emf_polyline_to",
    "emf_record",
    "emf_rectangle",
    "emf_restoredc",
    "emf_savedc",
    "emf_select_object",
    "emf_set_miter_limit",
    "emf_set_polyfill_mode",
    "emf_set_text_align",
    "emf_set_world_transform",
    "emf_stroke_and_fill_path",
    "emf_stroke_path",
    "emf_stretch_dib",
    "emf_text",
    "emfplus_comment",
    "wmf_move_to",
    "wmf_rectangle",
    "wmf_record",
    "wmf_set_map_mode",
    "wmf_set_text_align",
    "wmf_textout",
]
