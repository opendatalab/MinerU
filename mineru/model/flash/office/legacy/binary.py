# Copyright (c) Opendatalab. All rights reserved.
"""旧版 Office 二进制格式复用的有界小端读取能力。"""

from __future__ import annotations

import struct


def get_u16(data: bytes, offset: int) -> int | None:
    """有界读取小端无符号 16 位整数。"""
    if offset < 0 or offset + 2 > len(data):
        return None
    return int(struct.unpack_from("<H", data, offset)[0])


def get_i16(data: bytes, offset: int) -> int | None:
    """有界读取小端有符号 16 位整数。"""
    if offset < 0 or offset + 2 > len(data):
        return None
    return int(struct.unpack_from("<h", data, offset)[0])


def get_u32(data: bytes, offset: int) -> int | None:
    """有界读取小端无符号 32 位整数。"""
    if offset < 0 or offset + 4 > len(data):
        return None
    return int(struct.unpack_from("<I", data, offset)[0])


def get_f64(data: bytes, offset: int) -> float | None:
    """有界读取小端 IEEE-754 双精度浮点数。"""
    if offset < 0 or offset + 8 > len(data):
        return None
    return float(struct.unpack_from("<d", data, offset)[0])


def bounded_slice(data: bytes, offset: int, size: int) -> bytes | None:
    """返回不越界且不允许负偏移或负长度的字节范围。"""
    if offset < 0 or size < 0 or offset > len(data) - size:
        return None
    return data[offset : offset + size]


__all__ = ["bounded_slice", "get_f64", "get_i16", "get_u16", "get_u32"]
