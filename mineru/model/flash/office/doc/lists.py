# Copyright (c) Opendatalab. All rights reserved.

"""解析 Word PlfLst/PlfLfo 并维护九级列表编号状态。"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..legacy.binary import bounded_slice, get_u16, get_u32
from .models import DocListInfo
from .records import DocBudget

LEVELS = 9


@dataclass(frozen=True, slots=True)
class NumberToken:
    """编号模板中的层级引用或普通文本。"""

    level: int | None = None
    text: str = ""


@dataclass(frozen=True, slots=True)
class LevelDefinition:
    """一个 Word 列表层级的编号规则。"""

    marker: str = "bullet"
    start: int = 1
    restart: int | None = None
    tokens: tuple[NumberToken, ...] = ()


@dataclass(frozen=True, slots=True)
class ListDefinition:
    """一个 LFO 解析后的完整列表定义。"""

    lsid: int
    levels: tuple[LevelDefinition, ...]
    overrides: tuple[int | None, ...] = (None,) * LEVELS


@dataclass(slots=True)
class _ListCounter:
    """同一 lsid 的多级编号运行状态。"""

    values: list[int] = field(default_factory=lambda: [0] * LEVELS)
    started: list[bool] = field(default_factory=lambda: [False] * LEVELS)


class ListTables:
    """按 ilfo 查询列表定义并生成逐段落可见编号。"""

    def __init__(self, definitions: dict[int, ListDefinition] | None = None) -> None:
        """初始化列表定义及空计数器。"""

        self._definitions = definitions or {}
        self._counters: dict[int, _ListCounter] = {}
        self._override_used: set[tuple[int, int]] = set()

    def paragraph_info(self, ilfo: int, level: int) -> DocListInfo | None:
        """推进指定列表层级并返回当前段落的标签。"""

        if ilfo in {0, 0xF801}:
            return None
        definition = self._definitions.get(ilfo)
        if definition is None:
            fallback_levels = tuple(LevelDefinition() for _ in range(LEVELS))
            definition = ListDefinition(lsid=(0xFFFF_FFFF ^ ilfo), levels=fallback_levels)
        level = min(max(level, 0), LEVELS - 1)
        level_def = definition.levels[level]
        if level_def.marker == "none":
            return None
        if level_def.marker == "bullet":
            return DocListInfo(
                identity=definition.lsid,
                level=level,
                ordered=False,
            )
        counter = self._counters.setdefault(definition.lsid, _ListCounter())
        first_for_override = (ilfo, level) not in self._override_used
        self._override_used.add((ilfo, level))
        override = definition.overrides[level]
        if first_for_override and override is not None:
            value = override
        elif counter.started[level]:
            value = counter.values[level] + 1
        else:
            value = level_def.start
        counter.values[level] = value
        counter.started[level] = True
        for deeper in range(level + 1, LEVELS):
            restart = definition.levels[deeper].restart
            if restart is None or level < restart:
                counter.started[deeper] = False
        return DocListInfo(
            identity=definition.lsid,
            level=level,
            ordered=True,
            start=value,
            label=_render_label(definition, counter, level),
        )


def _roman(value: int) -> str:
    """把正整数格式化为常用 Roman 编号。"""

    if value <= 0:
        return str(value)
    pairs = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    result: list[str] = []
    remaining = value
    for number, label in pairs:
        while remaining >= number:
            result.append(label)
            remaining -= number
    return "".join(result)


def _alpha(value: int) -> str:
    """把正整数格式化为 Excel 风格字母序号。"""

    if value <= 0:
        return str(value)
    result: list[str] = []
    remaining = value
    while remaining:
        remaining, digit = divmod(remaining - 1, 26)
        result.append(chr(ord("A") + digit))
    return "".join(reversed(result))


def _format_marker(marker: str, value: int) -> str:
    """按 Word nfc 对应的 marker 类型格式化整数。"""

    if marker == "upper_roman":
        return _roman(value)
    if marker == "lower_roman":
        return _roman(value).lower()
    if marker == "upper_alpha":
        return _alpha(value)
    if marker == "lower_alpha":
        return _alpha(value).lower()
    return str(value)


def _render_label(definition: ListDefinition, counter: _ListCounter, level: int) -> str:
    """用当前各层编号替换 LVL 模板中的占位符。"""

    level_def = definition.levels[level]
    if not level_def.tokens:
        return f"{_format_marker(level_def.marker, counter.values[level])}."
    parts: list[str] = []
    for token in level_def.tokens:
        if token.level is None:
            parts.append(token.text)
            continue
        source_level = min(max(token.level, 0), LEVELS - 1)
        source = definition.levels[source_level]
        value = counter.values[source_level] if counter.started[source_level] else source.start
        parts.append(_format_marker(source.marker, value))
    return "".join(parts).strip()


def _marker_for_nfc(nfc: int) -> str:
    """把常见 MS-OSHARED 编号格式映射为内部 marker。"""

    return {
        0: "decimal",
        1: "upper_roman",
        2: "lower_roman",
        3: "upper_alpha",
        4: "lower_alpha",
        23: "bullet",
        0xFF: "none",
    }.get(nfc, "decimal")


def _parse_level(data: bytes, offset: int) -> tuple[LevelDefinition, int] | None:
    """解析一个 LVL 及其 PAPX/CHPX 后的编号文本。"""

    header = bounded_slice(data, offset, 28)
    if header is None:
        return None
    start = get_u32(header, 0)
    if start is None:
        return None
    nfc = header[4]
    flags = header[5]
    placeholder_offsets = [value for value in header[6:15] if value]
    chpx_size = header[24]
    papx_size = header[25]
    restart_limit = header[26]
    cursor = offset + 28 + papx_size + chpx_size
    char_count = get_u16(data, cursor)
    if char_count is None:
        return None
    cursor += 2
    text_bytes = bounded_slice(data, cursor, char_count * 2)
    if text_bytes is None:
        return None
    units = [get_u16(text_bytes, index * 2) or 0 for index in range(char_count)]
    tokens: list[NumberToken] = []
    marker = _marker_for_nfc(nfc)
    if marker not in {"bullet", "none"}:
        for index, unit in enumerate(units):
            if unit <= 8 and index + 1 in placeholder_offsets:
                tokens.append(NumberToken(level=unit))
            elif unit >= 0x20:
                char = chr(unit)
                if tokens and tokens[-1].level is None:
                    tokens[-1] = NumberToken(text=tokens[-1].text + char)
                else:
                    tokens.append(NumberToken(text=char))
    restart = restart_limit if flags & 0x08 else None
    return (
        LevelDefinition(
            marker=marker,
            start=max(int(start), 0),
            restart=restart,
            tokens=tuple(tokens),
        ),
        cursor + char_count * 2,
    )


def _parse_list_definitions(
    table_stream: bytes,
    *,
    offset: int,
    size: int,
    budget: DocBudget,
) -> dict[int, tuple[LevelDefinition, ...]]:
    """解析 PlfLst 的 LSTF 数组及后续 LVL。"""

    data = table_stream[offset:] if 0 <= offset < len(table_stream) else b""
    count = get_u16(data, 0)
    if count is None or size < 2 + count * 28:
        return {}
    cursor = 2
    descriptors: list[tuple[int, bool]] = []
    for _ in range(count):
        record = bounded_slice(data, cursor, 28)
        if record is None:
            return {}
        lsid = get_u32(record, 0)
        if lsid is None:
            return {}
        descriptors.append((lsid, bool(record[26] & 1)))
        cursor += 28
        budget.charge()
    cursor = size
    result: dict[int, tuple[LevelDefinition, ...]] = {}
    for lsid, simple in descriptors:
        count_levels = 1 if simple else LEVELS
        levels: list[LevelDefinition] = []
        for _ in range(count_levels):
            parsed = _parse_level(data, cursor)
            if parsed is None:
                return result
            level, cursor = parsed
            levels.append(level)
            budget.charge()
        if simple:
            levels = [levels[0]] * LEVELS
        while len(levels) < LEVELS:
            levels.append(LevelDefinition())
        result[lsid] = tuple(levels)
    return result


def parse_list_tables(
    table_stream: bytes,
    *,
    list_offset: int,
    list_size: int,
    override_offset: int,
    override_size: int,
    budget: DocBudget,
) -> ListTables:
    """解析 PlfLst/PlfLfo 并建立一基 ilfo 查找表。"""

    by_lsid = _parse_list_definitions(
        table_stream,
        offset=list_offset,
        size=list_size,
        budget=budget,
    )
    data = bounded_slice(table_stream, override_offset, override_size)
    if data is None or len(data) < 4:
        return ListTables()
    count = get_u32(data, 0)
    if count is None or 4 + count * 16 > len(data):
        return ListTables()
    cursor = 4
    descriptors: list[tuple[int, int]] = []
    for _ in range(count):
        record = bounded_slice(data, cursor, 16)
        if record is None:
            return ListTables()
        lsid = get_u32(record, 0)
        if lsid is None:
            return ListTables()
        descriptors.append((lsid, record[12]))
        cursor += 16
        budget.charge()
    definitions: dict[int, ListDefinition] = {}
    for index, (lsid, override_count) in enumerate(descriptors, start=1):
        levels = list(by_lsid.get(lsid, tuple(LevelDefinition() for _ in range(LEVELS))))
        overrides: list[int | None] = [None] * LEVELS
        for _ in range(override_count):
            record = bounded_slice(data, cursor, 8)
            if record is None:
                break
            start = get_u32(record, 0) or 0
            bits = record[4]
            level_index = bits & 0x0F
            has_start = bool(bits & 0x10)
            has_format = bool(bits & 0x20)
            cursor += 8
            budget.charge()
            if has_format:
                parsed = _parse_level(data, cursor)
                if parsed is None:
                    break
                level_def, cursor = parsed
                if level_index < LEVELS:
                    levels[level_index] = level_def
                    if has_start:
                        overrides[level_index] = level_def.start
            elif has_start and level_index < LEVELS:
                levels[level_index] = LevelDefinition(
                    marker=levels[level_index].marker,
                    start=int(start),
                    restart=levels[level_index].restart,
                    tokens=levels[level_index].tokens,
                )
                overrides[level_index] = int(start)
        definitions[index] = ListDefinition(
            lsid=lsid,
            levels=tuple(levels),
            overrides=tuple(overrides),
        )
    return ListTables(definitions)
