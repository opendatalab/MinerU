# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ...types import Block, PageInfo
from ...utils.enum_class import BlockType
from ..utils.html_image_utils import replace_inline_table_images, save_span_image_if_needed
from .office_magic_model import MagicModel


def blocks_to_page_info(page_blocks: list[dict[str, Any]], image_writer: Any, page_index: int) -> PageInfo:
    """将blocks转换为页面信息"""

    magic_model = MagicModel(page_blocks)
    image_blocks = magic_model.get_image_blocks()
    table_blocks = magic_model.get_table_blocks()
    chart_blocks = magic_model.get_chart_blocks()

    if image_writer:
        # Write embedded images to local storage via image_writer
        for img_block in image_blocks:
            for sub_block in img_block.blocks:
                if sub_block.type != "image_body":
                    continue
                for line in sub_block.lines:
                    for span in line.spans:
                        save_span_image_if_needed(span, image_writer, page_index)

        replace_inline_table_images(table_blocks, image_writer, page_index)

        # Replace inline base64 images inside chart content with local paths
        for chart_block in chart_blocks:
            for sub_block in chart_block.blocks:
                if sub_block.type != "chart_body":
                    continue
                for line in sub_block.lines:
                    for span in line.spans:
                        if span.type != "chart":
                            continue
                        save_span_image_if_needed(span, image_writer, page_index)

    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    list_blocks = magic_model.get_list_blocks()
    index_blocks = magic_model.get_index_blocks()
    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    blocks = [
        *image_blocks,
        *chart_blocks,
        *table_blocks,
        *title_blocks,
        *text_blocks,
        *interline_equation_blocks,
        *list_blocks,
        *index_blocks,
    ]

    # 对page_blocks根据index的值进行排序
    blocks.sort(key=lambda x: x.index)

    page_info = PageInfo(
        para_blocks=blocks,
        discarded_blocks=discarded_blocks,
        page_idx=page_index,
    )
    return page_info


def _extract_section_parts_from_content(content: str, level: int) -> list[int] | None:
    """Try to extract a leading section number (e.g. '1.2.1') from title content.

    Returns a list of ints [n1, n2, ..., nL] when the number of parts equals
    `level`, otherwise None.  Handles formats like:
        '1心肌特异性...'       (no separator)
        '1.2.1建立...'         (Chinese text immediately after number)
        '2.2.1 ALKBH5 ...'    (space separator)
    """
    match = re.match(r"^(\d+(?:\.\d+)*)", content.strip())
    if match:
        parts = [int(p) for p in match.group(1).split(".")]
        if len(parts) == level:
            return parts
    return None


def _collect_index_text_blocks(index_block: Block, result: list[Block]) -> None:
    """Depth-first collect TOC leaf text blocks."""
    for child in index_block.blocks:
        if child.type == BlockType.INDEX:
            _collect_index_text_blocks(child, result)
        elif child.type == BlockType.TEXT:
            result.append(child)


def _link_index_entries_by_anchor(middle_json: list[PageInfo]) -> None:
    """Keep TOC anchors only when they exist on parsed body blocks."""
    valid_anchors: set[str] = set()

    for page_info in middle_json:
        for block in page_info.para_blocks:
            anchor = block.anchor
            if isinstance(anchor, str) and anchor.strip():
                valid_anchors.add(anchor.strip())

    if not valid_anchors:
        return

    for page_info in middle_json:
        for block in page_info.para_blocks:
            if block.type != BlockType.INDEX:
                continue
            toc_text_blocks: list[Block] = []
            _collect_index_text_blocks(block, toc_text_blocks)
            for text_block in toc_text_blocks:
                anchor = text_block.anchor
                if not isinstance(anchor, str):
                    text_block.anchor = ""
                    continue
                anchor = anchor.strip()
                if not anchor or anchor not in valid_anchors:
                    text_block.anchor = ""
                    continue
                text_block.anchor = anchor


def result_to_middle_json(
    model_output_blocks_list: list[list[dict[str, Any]]],
    image_writer: object,
) -> list[PageInfo]:
    middle_json: list[PageInfo] = []
    for index, page_blocks in enumerate(model_output_blocks_list):
        page_info = blocks_to_page_info(page_blocks, image_writer, index)
        middle_json.append(page_info)

    section_counters: dict[int, int] = defaultdict(int)
    for page_info in middle_json:
        for block in page_info.para_blocks:
            if block.type != BlockType.TITLE:
                continue
            level = block.level
            if level is None:
                level = 1
            if block.is_numbered_style:
                # Ensure all ancestor levels start at 1 (never 0)
                for ancestor in range(1, level):
                    if section_counters[ancestor] == 0:
                        section_counters[ancestor] = 1
                # Increment current level counter and reset all deeper levels
                section_counters[level] += 1
                for deeper in list(section_counters.keys()):
                    if deeper > level:
                        section_counters[deeper] = 0
                # Build section number string, e.g. "1.2.1"
                section_number = ".".join(str(section_counters[lvl]) for lvl in range(1, level + 1))
                block.section_number = section_number
            else:
                # Some documents embed the section number directly in the content
                # (is_numbered_style=False).  Parse it and sync the counters so
                # that subsequent numbered blocks continue from the right base.
                lines = block.lines
                content = ""
                if lines and lines[0].spans:
                    content = lines[0].spans[0].content
                parts = _extract_section_parts_from_content(content, level)
                if parts:
                    for k, v in enumerate(parts, start=1):
                        section_counters[k] = v
                    # Reset all deeper levels
                    for deeper in list(section_counters.keys()):
                        if deeper > level:
                            section_counters[deeper] = 0

    _link_index_entries_by_anchor(middle_json)
    return middle_json
