# Copyright (c) Opendatalab. All rights reserved.

"""遍历并应用 Word 二进制单属性修饰符 SPRM。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import struct
from typing import Callable

from ..legacy.binary import get_i16, get_u16, get_u32
from .models import DocCharStyle, DocTableCellFormat, DocTableFormat
from .records import DocBudget


def _operand_length(opcode: int, operand: bytes) -> int:
    """根据 SPRM 的 spra 字段计算 operand 字节数。"""

    spra = opcode >> 13
    if spra in {0, 1}:
        return 1
    if spra in {2, 4, 5}:
        return 2
    if spra == 3:
        return 4
    if spra == 7:
        return 3
    if opcode == 0xD608:
        return (get_u16(operand, 0) or -1) + 1
    return (operand[0] + 1) if operand else 0


def walk_sprms(
    grpprl: bytes,
    callback: Callable[[int, bytes], None],
    *,
    budget: DocBudget | None = None,
) -> None:
    """有界顺序遍历 grpprl，截断尾部按可恢复内容处理。"""

    cursor = 0
    while cursor + 2 <= len(grpprl):
        opcode = int(struct.unpack_from("<H", grpprl, cursor)[0])
        cursor += 2
        length = _operand_length(opcode, grpprl[cursor:])
        if length <= 0 or cursor + length > len(grpprl):
            return
        if budget is not None:
            budget.charge()
        callback(opcode, grpprl[cursor : cursor + length])
        cursor += length


def _toggle(operand: bytes, base: bool) -> bool | None:
    """把 Word ToggleOperand 解析为相对样式基值。"""

    if not operand:
        return None
    return {0: False, 1: True, 0x80: base, 0x81: not base}.get(operand[0])


def chpx_style_id(grpprl: bytes) -> int | None:
    """读取 CHPX 指定的字符样式 istd。"""

    result: int | None = None

    def consume(opcode: int, operand: bytes) -> None:
        """记录最后一个有效 sprmCIstd。"""

        nonlocal result
        if opcode == 0x4A30:
            result = get_u16(operand, 0)

    walk_sprms(grpprl, consume)
    return result


def chpx_picture_location(grpprl: bytes) -> int | None:
    """读取 CHPX 中的 sprmCPicLocation。"""

    result: int | None = None

    def consume(opcode: int, operand: bytes) -> None:
        """记录最后一个有效图片偏移。"""

        nonlocal result
        if opcode == 0x6A03:
            result = get_u32(operand, 0)

    walk_sprms(grpprl, consume)
    return result


def apply_character_sprms(
    grpprl: bytes,
    current: DocCharStyle,
    style_base: DocCharStyle,
    *,
    budget: DocBudget | None = None,
) -> DocCharStyle:
    """按 Word 样式覆盖顺序把 CHPX 应用到字符样式。"""

    style = current

    def consume(opcode: int, operand: bytes) -> None:
        """应用当前可表达的字符属性。"""

        nonlocal style
        toggle_field = {
            0x0800: "deleted",  # sprmCFRMarkDel
            0x0802: "hidden",  # sprmCFFldVanish
            0x0835: "bold",
            0x0836: "italic",
            0x0837: "strike",
            0x083C: "hidden",  # sprmCFVanish
        }.get(opcode)
        if toggle_field is not None:
            value = _toggle(operand, bool(getattr(style_base, toggle_field)))
            if value is not None:
                style = replace(style, **{toggle_field: value})
            return
        if opcode == 0x2A3E and operand:  # sprmCKul
            style = replace(style, underline=operand[0] not in {0, 5})
        elif opcode == 0x2A48 and operand:  # sprmCIss
            style = replace(
                style,
                superscript=operand[0] == 1,
                subscript=operand[0] == 2,
            )
        elif opcode == 0x2A53 and operand:  # sprmCFDStrike
            style = replace(style, strike=operand[0] != 0)
        elif opcode == 0x2A54 and operand:  # sprmCEm
            style = replace(style, emphasis=operand[0] != 0)

    walk_sprms(grpprl, consume, budget=budget)
    return style


@dataclass(frozen=True, slots=True)
class PapDelta:
    """PAPX 或段落样式对可见段落属性的增量。"""

    in_table: bool | None = None
    row_mark: bool | None = None
    outline_level: int | None | object = None
    ilfo: int | None = None
    ilvl: int | None = None
    table_depth: int | None = None
    inner_cell: bool | None = None
    inner_row: bool | None = None
    table: DocTableFormat | None = None

    def merge(self, over: PapDelta) -> PapDelta:
        """让后应用的段落属性覆盖当前增量。"""

        return PapDelta(
            in_table=over.in_table if over.in_table is not None else self.in_table,
            row_mark=over.row_mark if over.row_mark is not None else self.row_mark,
            outline_level=(
                over.outline_level if over.outline_level is not None else self.outline_level
            ),
            ilfo=over.ilfo if over.ilfo is not None else self.ilfo,
            ilvl=over.ilvl if over.ilvl is not None else self.ilvl,
            table_depth=over.table_depth if over.table_depth is not None else self.table_depth,
            inner_cell=over.inner_cell if over.inner_cell is not None else self.inner_cell,
            inner_row=over.inner_row if over.inner_row is not None else self.inner_row,
            table=over.table if over.table is not None else self.table,
        )


def _parse_tdef_table(operand: bytes) -> DocTableFormat | None:
    """解析 TDefTableOperand 中的边界和横纵向合并标志。"""

    if len(operand) < 3:
        return None
    columns = operand[2]
    if columns > 63:
        return None
    boundaries: list[int] = []
    for index in range(columns + 1):
        value = get_i16(operand, 3 + index * 2)
        if value is None:
            return None
        boundaries.append(value)
    cells: list[DocTableCellFormat] = []
    tc_base = 3 + (columns + 1) * 2
    for index in range(columns):
        flags = get_u16(operand, tc_base + index * 20) or 0
        horizontal = flags & 0x3
        vertical = (flags >> 5) & 0x3
        right = boundaries[index + 1]
        cells.append(
            DocTableCellFormat(
                right=right,
                horizontal_first=horizontal >= 2,
                horizontal_continue=horizontal == 1,
                vertical_first=vertical == 3,
                vertical_continue=vertical == 1,
            )
        )
    return DocTableFormat(boundaries=tuple(boundaries), cells=tuple(cells))


def apply_paragraph_sprms(
    grpprl: bytes,
    data_stream: bytes,
    initial: PapDelta | None = None,
    *,
    budget: DocBudget | None = None,
) -> PapDelta:
    """应用 PAPX SPRM，并解析 huge PAPX 与表格行属性。"""

    delta = initial or PapDelta()

    def consume(opcode: int, operand: bytes) -> None:
        """应用一个段落或表格属性。"""

        nonlocal delta
        if opcode == 0x2416 and operand:
            delta = replace(delta, in_table=operand[0] != 0)
        elif opcode == 0x2417 and operand:
            delta = replace(delta, row_mark=operand[0] != 0)
        elif opcode == 0x6646:
            offset = get_u32(operand, 0)
            length = get_u16(data_stream, offset) if offset is not None else None
            if offset is not None and length is not None and offset + 2 + length <= len(data_stream):
                delta = apply_paragraph_sprms(
                    data_stream[offset + 2 : offset + 2 + length],
                    b"",
                    delta,
                    budget=budget,
                )
        elif opcode == 0x2640 and operand:
            delta = replace(delta, outline_level=operand[0] + 1 if operand[0] < 9 else -1)
        elif opcode == 0x260A and operand:
            delta = replace(delta, ilvl=int(operand[0]))
        elif opcode == 0x460B:
            delta = replace(delta, ilfo=get_u16(operand, 0))
        elif opcode == 0x6649:
            depth = get_u32(operand, 0)
            if depth is not None:
                delta = replace(delta, table_depth=int(depth))
        elif opcode == 0x664A:
            raw = get_u32(operand, 0)
            if raw is not None:
                signed = struct.unpack("<i", struct.pack("<I", raw))[0]
                delta = replace(delta, table_depth=max(0, (delta.table_depth or 0) + signed))
        elif opcode == 0x244B and operand:
            delta = replace(delta, inner_cell=operand[0] != 0)
        elif opcode == 0x244C and operand:
            delta = replace(delta, inner_row=operand[0] != 0)
        elif opcode == 0xD608:
            table = _parse_tdef_table(operand)
            if table is not None:
                header = delta.table.header if delta.table is not None else False
                delta = replace(delta, table=replace(table, header=header))
        elif opcode == 0x3404 and operand:
            table = delta.table or DocTableFormat()
            delta = replace(delta, table=replace(table, header=operand[0] != 0))
        elif opcode == 0xD62B and len(operand) >= 3 and delta.table is not None:
            cell_index = operand[1]
            flag = operand[2]
            cells = list(delta.table.cells)
            if cell_index < len(cells):
                cells[cell_index] = replace(
                    cells[cell_index],
                    vertical_continue=flag == 1,
                    vertical_first=flag == 3,
                )
                delta = replace(delta, table=replace(delta.table, cells=tuple(cells)))

    walk_sprms(grpprl, consume, budget=budget)
    return delta
