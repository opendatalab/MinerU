# Copyright (c) Opendatalab. All rights reserved.

"""有界读取 Excel 97–2003 BIFF 记录流。"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterator

from loguru import logger

from mineru.model.flash.legacy_office.errors import LegacyOfficeResourceLimitError
from mineru.model.flash.legacy_office.limits import MAX_RECORDS

BOF = 0x0809
EOF = 0x000A
CONTINUE = 0x003C


@dataclass(frozen=True, slots=True)
class BiffRecord:
    """一条已完成边界校验的 BIFF 记录。"""

    offset: int
    record_type: int
    payload: bytes
    next_offset: int


@dataclass(slots=True)
class RecordBudget:
    """跨 globals、worksheet 与 OfficeArt 共享记录访问预算。"""

    count: int = 0

    def charge(self) -> None:
        """计入一条记录并在超过固定上限时硬失败。"""

        self.count += 1
        if self.count > MAX_RECORDS:
            raise LegacyOfficeResourceLimitError(
                f"workbook stream exceeds max_records={MAX_RECORDS}"
            )


def get_u16(data: bytes, offset: int) -> int | None:
    """安全读取小端无符号 16 位整数。"""

    if offset < 0 or offset + 2 > len(data):
        return None
    return int(struct.unpack_from("<H", data, offset)[0])


def get_i16(data: bytes, offset: int) -> int | None:
    """安全读取小端有符号 16 位整数。"""

    if offset < 0 or offset + 2 > len(data):
        return None
    return int(struct.unpack_from("<h", data, offset)[0])


def get_u32(data: bytes, offset: int) -> int | None:
    """安全读取小端无符号 32 位整数。"""

    if offset < 0 or offset + 4 > len(data):
        return None
    return int(struct.unpack_from("<I", data, offset)[0])


def get_f64(data: bytes, offset: int) -> float | None:
    """安全读取小端 IEEE-754 双精度数。"""

    if offset < 0 or offset + 8 > len(data):
        return None
    return float(struct.unpack_from("<d", data, offset)[0])


def record_at(
    data: bytes,
    offset: int,
    *,
    budget: RecordBudget | None = None,
) -> BiffRecord | None:
    """读取指定偏移的 BIFF 记录，截断 header 或 body 返回空值。"""

    if offset < 0 or offset + 4 > len(data):
        return None
    record_type, length = struct.unpack_from("<HH", data, offset)
    body_start = offset + 4
    body_end = body_start + int(length)
    if body_end < body_start or body_end > len(data):
        return None
    if budget is not None:
        budget.charge()
    return BiffRecord(
        offset=offset,
        record_type=int(record_type),
        payload=data[body_start:body_end],
        next_offset=body_end,
    )


def iter_records(
    data: bytes,
    *,
    start: int = 0,
    stop_at_eof: bool = False,
    budget: RecordBudget | None = None,
) -> Iterator[BiffRecord]:
    """顺序遍历 BIFF 记录，并在截断尾部保留已经完成的记录。"""

    cursor = start
    while cursor < len(data):
        if data[cursor:] and not any(data[cursor:]):
            return
        record = record_at(data, cursor, budget=budget)
        if record is None:
            logger.warning(
                "XLS_TRUNCATED_RECORD: workbook stream ends mid-record at byte {}",
                cursor,
            )
            return
        yield record
        cursor = record.next_offset
        if stop_at_eof and record.record_type == EOF:
            return


def collect_continues(
    data: bytes,
    base: BiffRecord,
    *,
    budget: RecordBudget,
) -> tuple[list[bytes], int]:
    """收集紧随 base 的 CONTINUE bodies，并返回下一条非延续记录偏移。"""

    segments = [base.payload]
    cursor = base.next_offset
    while True:
        record = record_at(data, cursor)
        if record is None or record.record_type != CONTINUE:
            return segments, cursor
        budget.charge()
        segments.append(record.payload)
        cursor = record.next_offset


class SegmentReader:
    """在基础记录及其 CONTINUE segments 上执行有界顺序读取。"""

    def __init__(self, segments: list[bytes]) -> None:
        """保存 segment 列表并把游标置于首段开头。"""

        self.segments = segments
        self.segment_index = 0
        self.offset = 0

    def remaining_in_segment(self) -> int:
        """返回当前 segment 尚未消费的字节数。"""

        if self.segment_index >= len(self.segments):
            return 0
        return len(self.segments[self.segment_index]) - self.offset

    def normalize(self) -> None:
        """跳过已经耗尽的 segments。"""

        while (
            self.segment_index < len(self.segments)
            and self.offset >= len(self.segments[self.segment_index])
        ):
            self.segment_index += 1
            self.offset = 0

    def next_segment(self) -> bool:
        """显式移动到下一 segment，若不存在则返回 False。"""

        if self.segment_index + 1 >= len(self.segments):
            return False
        self.segment_index += 1
        self.offset = 0
        return True

    def read(self, size: int) -> bytes | None:
        """只在当前 segment 内读取固定长度字段。"""

        self.normalize()
        if size < 0 or self.segment_index >= len(self.segments):
            return None
        segment = self.segments[self.segment_index]
        end = self.offset + size
        if end > len(segment):
            return None
        output = segment[self.offset:end]
        self.offset = end
        return output

    def read_across(self, size: int) -> bytes | None:
        """跨 segment 读取普通非字符数据。"""

        if size < 0:
            return None
        output = bytearray()
        remaining = size
        while remaining:
            self.normalize()
            available = self.remaining_in_segment()
            if available <= 0:
                return None
            take = min(available, remaining)
            chunk = self.read(take)
            if chunk is None:
                return None
            output.extend(chunk)
            remaining -= take
        return bytes(output)

    def skip(self, size: int) -> bool:
        """跨 segments 跳过指定字节数。"""

        return self.read_across(size) is not None

    def u8(self) -> int | None:
        """读取一个无符号 8 位整数。"""

        value = self.read(1)
        return int(value[0]) if value is not None else None

    def u16(self) -> int | None:
        """读取一个小端无符号 16 位整数。"""

        value = self.read(2)
        return int(struct.unpack("<H", value)[0]) if value is not None else None

    def u32(self) -> int | None:
        """读取一个小端无符号 32 位整数。"""

        value = self.read(4)
        return int(struct.unpack("<I", value)[0]) if value is not None else None
