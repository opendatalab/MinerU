# Copyright (c) Opendatalab. All rights reserved.
"""WMF/EMF 使用的严格有界二进制读取器。"""

from __future__ import annotations

import struct

from .models import MetafileMalformedError


class BoundedReader:
    """在固定 memoryview 边界内读取小端标量和切片。"""

    def __init__(self, data: bytes | memoryview, *, base_offset: int = 0) -> None:
        """保存输入视图与用于诊断的绝对起始偏移。"""
        self.data = memoryview(data)
        self.base_offset = base_offset

    def _require(self, offset: int, size: int) -> None:
        """验证读取范围，不允许负数或越过当前视图。"""
        if offset < 0 or size < 0 or offset > len(self.data) - size:
            raise MetafileMalformedError(
                f"metafile field exceeds record boundary: offset={self.base_offset + offset}, size={size}"
            )

    def u8(self, offset: int) -> int:
        """读取小端无符号 8 位整数。"""
        self._require(offset, 1)
        return int(self.data[offset])

    def i16(self, offset: int) -> int:
        """读取小端有符号 16 位整数。"""
        self._require(offset, 2)
        return int(struct.unpack_from("<h", self.data, offset)[0])

    def u16(self, offset: int) -> int:
        """读取小端无符号 16 位整数。"""
        self._require(offset, 2)
        return int(struct.unpack_from("<H", self.data, offset)[0])

    def i32(self, offset: int) -> int:
        """读取小端有符号 32 位整数。"""
        self._require(offset, 4)
        return int(struct.unpack_from("<i", self.data, offset)[0])

    def u32(self, offset: int) -> int:
        """读取小端无符号 32 位整数。"""
        self._require(offset, 4)
        return int(struct.unpack_from("<I", self.data, offset)[0])

    def f32(self, offset: int) -> float:
        """读取小端 IEEE-754 单精度浮点数。"""
        self._require(offset, 4)
        return float(struct.unpack_from("<f", self.data, offset)[0])

    def bytes(self, offset: int, size: int) -> bytes:
        """返回经过边界校验的不可变字节切片。"""
        self._require(offset, size)
        return self.data[offset : offset + size].tobytes()

    def subreader(self, offset: int, size: int) -> BoundedReader:
        """返回继承绝对偏移信息的有界子读取器。"""
        self._require(offset, size)
        return BoundedReader(self.data[offset : offset + size], base_offset=self.base_offset + offset)

    def remaining(self, offset: int) -> int:
        """返回从指定位置到当前视图末尾的剩余字节数。"""
        self._require(offset, 0)
        return len(self.data) - offset

    def __len__(self) -> int:
        """返回当前视图的总字节数。"""
        return len(self.data)


__all__ = ["BoundedReader"]
