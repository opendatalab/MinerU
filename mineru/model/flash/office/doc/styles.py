# Copyright (c) Opendatalab. All rights reserved.

"""解析 Word STSH/STD 样式表并解析继承链。"""

from __future__ import annotations

from dataclasses import dataclass
import re

from loguru import logger

from ..legacy.binary import bounded_slice, get_u16
from .models import DocCharStyle
from .records import DocBudget
from .sprm import PapDelta, apply_character_sprms, apply_paragraph_sprms

ISTD_NIL = 0x0FFF


@dataclass(frozen=True, slots=True)
class ResolvedStyle:
    """一个样式继承完成后的可见属性。"""

    name: str = ""
    character: DocCharStyle = DocCharStyle()
    paragraph: PapDelta = PapDelta()
    heading_level: int | None = None
    is_title: bool = False
    toc_level: int | None = None
    is_code: bool = False


@dataclass(frozen=True, slots=True)
class _RawStyle:
    """STSH 内尚未解析继承的 STD。"""

    sti: int
    name: str
    base: int
    paragraph_grpprl: bytes
    character_grpprl: bytes
    paragraph_style: bool


class Stylesheet:
    """按 istd 提供解析后样式的只读集合。"""

    def __init__(self, styles: dict[int, ResolvedStyle] | None = None) -> None:
        """保存已解析样式并准备默认样式。"""

        self._styles = styles or {}
        self._default = ResolvedStyle()

    def get(self, style_id: int) -> ResolvedStyle:
        """返回指定样式，不存在时使用空默认值。"""

        return self._styles.get(style_id, self._default)


def _style_semantics(sti: int, name: str) -> tuple[int | None, bool, int | None, bool]:
    """从内建 ID 与样式名识别标题、目录和代码语义。"""

    normalized = re.sub(r"\s+", " ", name.strip()).casefold()
    heading_level = sti if 1 <= sti <= 9 else None
    if heading_level is None:
        match = re.match(r"^(?:heading|标题)\s*([1-9])$", normalized)
        if match:
            heading_level = int(match.group(1))
    is_title = normalized in {"title", "标题", "文档标题"} or sti == 15
    toc_level: int | None = None
    match = re.match(r"^(?:toc|目录)\s*([1-9])$", normalized)
    if match:
        toc_level = int(match.group(1)) - 1
    is_code = any(token in normalized for token in ("code", "preformatted", "source code", "代码"))
    return heading_level, is_title, toc_level, is_code


def _parse_std(record: bytes, base_size: int) -> _RawStyle | None:
    """解析一条 STD 的名称、继承和 UPX 内容。"""

    if len(record) < max(base_size, 10):
        return None
    first = get_u16(record, 0)
    second = get_u16(record, 2)
    third = get_u16(record, 4)
    if first is None or second is None or third is None:
        return None
    sti = first & 0x0FFF
    kind = second & 0x000F
    base = (second >> 4) & 0x0FFF
    upx_count = third & 0x000F
    name_offset = max(base_size, 10)
    name_length = get_u16(record, name_offset)
    if name_length is None:
        return None
    name_bytes = name_length * 2
    raw_name = bounded_slice(record, name_offset + 2, name_bytes)
    if raw_name is None:
        return None
    name = raw_name.decode("utf-16le", errors="replace")
    cursor = name_offset + 2 + name_bytes + 2
    upx: list[bytes] = []
    for _ in range(upx_count):
        if cursor % 2:
            cursor += 1
        length = get_u16(record, cursor)
        if length is None:
            break
        payload = bounded_slice(record, cursor + 2, length)
        if payload is None:
            break
        upx.append(payload)
        cursor += 2 + length
    paragraph_style = kind == 1
    if paragraph_style:
        paragraph_grpprl = upx[0][2:] if upx and len(upx[0]) >= 2 else b""
        character_grpprl = upx[1] if len(upx) > 1 else b""
    elif kind == 2:
        paragraph_grpprl = b""
        character_grpprl = upx[0] if upx else b""
    else:
        paragraph_grpprl = b""
        character_grpprl = b""
    return _RawStyle(
        sti=sti,
        name=name,
        base=base,
        paragraph_grpprl=paragraph_grpprl,
        character_grpprl=character_grpprl,
        paragraph_style=paragraph_style,
    )


def parse_stylesheet(
    table_stream: bytes,
    *,
    offset: int,
    size: int,
    budget: DocBudget,
) -> Stylesheet:
    """解析 STSH，并在循环样式链处安全截断继承。"""

    payload = bounded_slice(table_stream, offset, size)
    if payload is None or len(payload) < 8:
        return Stylesheet()
    header_size = get_u16(payload, 0)
    style_count = get_u16(payload, 2)
    base_size = get_u16(payload, 4)
    if header_size is None or style_count is None or base_size is None:
        return Stylesheet()
    cursor = 2 + header_size
    raw_styles: dict[int, _RawStyle] = {}
    for style_id in range(style_count):
        length = get_u16(payload, cursor)
        if length is None:
            break
        cursor += 2
        if length == 0:
            continue
        record = bounded_slice(payload, cursor, length)
        if record is None:
            break
        cursor += length
        budget.charge()
        parsed = _parse_std(record, base_size)
        if parsed is not None:
            raw_styles[style_id] = parsed

    memo: dict[int, ResolvedStyle] = {}

    def resolve(style_id: int) -> ResolvedStyle:
        """迭代解析一个样式的 based-on 链。"""

        if style_id in memo:
            return memo[style_id]
        chain: list[int] = []
        visiting: set[int] = set()
        cursor_id: int | None = style_id
        base = ResolvedStyle()
        while cursor_id is not None:
            if cursor_id in memo:
                base = memo[cursor_id]
                break
            if cursor_id in visiting:
                logger.warning(f"DOC style inheritance cycle at istd={cursor_id}")
                break
            raw = raw_styles.get(cursor_id)
            if raw is None:
                break
            visiting.add(cursor_id)
            chain.append(cursor_id)
            cursor_id = raw.base if raw.base not in {ISTD_NIL, cursor_id} else None
        for current_id in reversed(chain):
            raw = raw_styles[current_id]
            paragraph = apply_paragraph_sprms(raw.paragraph_grpprl, b"", base.paragraph, budget=budget)
            character = apply_character_sprms(
                raw.character_grpprl,
                base.character,
                base.character,
                budget=budget,
            )
            heading, is_title, toc_level, is_code = _style_semantics(raw.sti, raw.name)
            base = ResolvedStyle(
                name=raw.name or base.name,
                character=character,
                paragraph=paragraph,
                heading_level=heading if heading is not None else base.heading_level,
                is_title=is_title or base.is_title,
                toc_level=toc_level if toc_level is not None else base.toc_level,
                is_code=is_code or base.is_code,
            )
            memo[current_id] = base
        return base

    for style_id in raw_styles:
        resolve(style_id)
    return Stylesheet(memo)
