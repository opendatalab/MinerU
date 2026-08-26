# Copyright (c) Opendatalab. All rights reserved.

"""DOC 二进制结构使用的有界整数、PLC 和记录预算工具。"""

from __future__ import annotations

from dataclasses import dataclass
import struct

from ..errors import LegacyOfficeResourceLimitError
from ..limits import MAX_RECORDS


@dataclass(slots=True)
class DocBudget:
    """限制 DOC 解析累计访问的记录和文本单元数。"""

    visited: int = 0

    def charge(self, amount: int = 1) -> None:
        """计入本次访问量，超过统一上限时稳定失败。"""

        if amount < 0 or self.visited + amount > MAX_RECORDS:
            raise LegacyOfficeResourceLimitError(f"DOC records exceed max_records={MAX_RECORDS}")
        self.visited += amount


def parse_plc(data: bytes, *, item_size: int, budget: DocBudget) -> tuple[list[int], list[bytes]]:
    """解析由 CP 数组和定长数据项组成的通用 PLC。"""

    if item_size < 0 or len(data) < 4:
        return [], []
    denominator = 4 + item_size
    payload = len(data) - 4
    if denominator <= 0 or payload % denominator:
        return [], []
    count = payload // denominator
    budget.charge(count + 1)
    cp_bytes = (count + 1) * 4
    cps = [int(struct.unpack_from("<I", data, index * 4)[0]) for index in range(count + 1)]
    items = [
        data[cp_bytes + index * item_size : cp_bytes + (index + 1) * item_size]
        for index in range(count)
    ]
    return cps, items
