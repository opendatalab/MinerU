# Copyright (c) Opendatalab. All rights reserved.

"""解析 Word CHPX/PAPX FKP 页面并提供按 FC 查询的格式 run。"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from loguru import logger

from ..legacy.binary import bounded_slice, get_u32
from .records import DocBudget
from .sprm import PapDelta, apply_paragraph_sprms


@dataclass(frozen=True, slots=True)
class CharacterRun:
    """一个物理 FC 范围内的原始 CHPX。"""

    fc_start: int
    fc_end: int
    grpprl: bytes


@dataclass(frozen=True, slots=True)
class ParagraphRun:
    """一个物理 FC 范围内的段落样式和 PAPX。"""

    fc_start: int
    fc_end: int
    style_id: int
    delta: PapDelta


class FormattingRuns:
    """按起始 FC 排序的 CHPX/PAPX 查询索引。"""

    def __init__(self, characters: list[CharacterRun], paragraphs: list[ParagraphRun]) -> None:
        """排序 run 并缓存二分查询键。"""

        self.characters = sorted(characters, key=lambda run: run.fc_start)
        self.paragraphs = sorted(paragraphs, key=lambda run: run.fc_start)
        self._character_starts = [run.fc_start for run in self.characters]
        self._paragraph_starts = [run.fc_start for run in self.paragraphs]

    def character_at(self, fc: int) -> CharacterRun | None:
        """返回覆盖指定 FC 的最后一个 CHPX run。"""

        index = bisect_right(self._character_starts, fc) - 1
        if index < 0:
            return None
        run = self.characters[index]
        return run if fc < run.fc_end else None

    def paragraph_at(self, fc: int) -> ParagraphRun | None:
        """返回覆盖指定 FC 的最后一个 PAPX run。"""

        index = bisect_right(self._paragraph_starts, fc) - 1
        if index < 0:
            return None
        run = self.paragraphs[index]
        return run if fc < run.fc_end else None


def _parse_bte_pages(table_stream: bytes, offset: int, size: int) -> list[int]:
    """从 PlcBteChpx/PlcBtePapx 读取 FKP page number。"""

    plc = bounded_slice(table_stream, offset, size)
    if plc is None or len(plc) < 8 or (len(plc) - 4) % 8:
        return []
    count = (len(plc) - 4) // 8
    page_offset = (count + 1) * 4
    pages: list[int] = []
    for index in range(count):
        raw = get_u32(plc, page_offset + index * 4)
        if raw is not None:
            pages.append(raw & 0x003F_FFFF)
    return pages


def _parse_chpx_page(page: bytes, budget: DocBudget) -> list[CharacterRun]:
    """解析一个 512 字节 ChpxFkp。"""

    count = page[511]
    if count == 0 or (count + 1) * 4 + count > 511:
        return []
    result: list[CharacterRun] = []
    offset_base = (count + 1) * 4
    for index in range(count):
        fc_start = get_u32(page, index * 4)
        fc_end = get_u32(page, (index + 1) * 4)
        if fc_start is None or fc_end is None or fc_end <= fc_start:
            continue
        byte_offset = page[offset_base + index]
        grpprl = b""
        if byte_offset:
            payload_offset = byte_offset * 2
            if payload_offset < 511:
                length = page[payload_offset]
                grpprl = page[payload_offset + 1 : payload_offset + 1 + length]
        budget.charge()
        result.append(CharacterRun(fc_start, fc_end, grpprl))
    return result


def _parse_papx_page(page: bytes, data_stream: bytes, budget: DocBudget) -> list[ParagraphRun]:
    """解析一个 512 字节 PapxFkp。"""

    count = page[511]
    header_end = (count + 1) * 4 + count * 13
    if count == 0 or header_end > 511:
        return []
    result: list[ParagraphRun] = []
    bx_base = (count + 1) * 4
    for index in range(count):
        fc_start = get_u32(page, index * 4)
        fc_end = get_u32(page, (index + 1) * 4)
        if fc_start is None or fc_end is None or fc_end <= fc_start:
            continue
        byte_offset = page[bx_base + index * 13]
        style_id = 0
        delta = PapDelta()
        if byte_offset:
            payload_offset = byte_offset * 2
            if payload_offset < 511:
                first_length = page[payload_offset]
                if first_length:
                    content_offset = payload_offset + 1
                    content_length = first_length * 2 - 1
                elif payload_offset + 1 < 511:
                    content_offset = payload_offset + 2
                    content_length = page[payload_offset + 1] * 2
                else:
                    content_offset = 511
                    content_length = 0
                content = page[content_offset : min(content_offset + content_length, 511)]
                if len(content) >= 2:
                    style_id = int.from_bytes(content[:2], "little")
                    delta = apply_paragraph_sprms(content[2:], data_stream, budget=budget)
        budget.charge()
        result.append(ParagraphRun(fc_start, fc_end, style_id, delta))
    return result


def parse_formatting_runs(
    word_document: bytes,
    table_stream: bytes,
    data_stream: bytes,
    *,
    chpx_offset: int,
    chpx_size: int,
    papx_offset: int,
    papx_size: int,
    budget: DocBudget,
) -> FormattingRuns:
    """解析 FIB 指向的全部 CHPX/PAPX FKP；坏可选页仅告警跳过。"""

    characters: list[CharacterRun] = []
    paragraphs: list[ParagraphRun] = []
    for page_number in _parse_bte_pages(table_stream, chpx_offset, chpx_size):
        page = bounded_slice(word_document, page_number * 512, 512)
        if page is None:
            logger.warning(f"DOC ChpxFkp page is truncated: {page_number}")
            continue
        characters.extend(_parse_chpx_page(page, budget))
    for page_number in _parse_bte_pages(table_stream, papx_offset, papx_size):
        page = bounded_slice(word_document, page_number * 512, 512)
        if page is None:
            logger.warning(f"DOC PapxFkp page is truncated: {page_number}")
            continue
        paragraphs.extend(_parse_papx_page(page, data_stream, budget))
    return FormattingRuns(characters, paragraphs)
