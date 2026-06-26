# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import copy
import re
from typing import Any

from loguru import logger
from PIL import Image

from ...types import EMPTY_BBOX, NOT_EXTRACT_TYPES, BBox, Block, BlockType, ContentType, IntBBox, Line, Span
from ...utils.guess_suffix_or_lang import guess_language_by_text
from ...utils.pdf_document import PDFPage
from ..utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from ..utils.content_block_draft import VlmContentBlockDraft
from ..utils.span_block_fix import fix_text_block
from ..utils.span_pre_proc import SpanBlockMatcher, txt_spans_extract
from ..utils.visual_magic_model_utils import (
    GENERIC_CHILD_TYPES,
    IMAGE_BLOCK_BODY,
    VISUAL_MAIN_TYPES,
    clean_content,
    code_content_clean,
    fallback_inline_caption_fragments,
    fallback_leading_table_continuation_captions,
    isolated_formula_clean,
    regroup_visual_blocks,
)

not_extract_list = [
    *NOT_EXTRACT_TYPES,
    BlockType.CAPTION,
    BlockType.FOOTNOTE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
]

OCR_DET_LINE_BLOCK_TYPES = {
    *not_extract_list,
    BlockType.LIST,
    BlockType.INDEX,
    BlockType.ABSTRACT,
    BlockType.ASIDE_TEXT,
    BlockType.PHONETIC,
    BlockType.CHART_CAPTION,
    BlockType.CHART_FOOTNOTE,
    BlockType.CODE_FOOTNOTE,
}


def _copy_raw_text_block_metadata(draft: VlmContentBlockDraft, block: Block) -> None:
    if draft.raw_type != BlockType.TEXT:
        return
    block.merge_prev = draft.merge_prev


class MagicModel:
    def __init__(
        self,
        page_model_list: list[dict[str, Any]],
        pdf_page: PDFPage,
        scale: float,
        page_pil_img: Image.Image,
        width: int,
        height: int,
        _ocr_enable: bool,
        _vlm_ocr_enable: bool,
    ) -> None:
        (
            page_blocks,
            self.page_inline_formula,
            self.page_ocr_res,
        ) = self._split_page_model_list(copy.deepcopy(page_model_list))

        self.width = width
        self.height = height

        blocks: list[Block] = []
        page_text_inline_formula_spans: list[Span] = []

        for inline_formula in self.page_inline_formula:
            inline_formula["bbox"] = self.cal_real_bbox(inline_formula["bbox"])
            inline_formula_latex = inline_formula.pop("latex", "")
            if inline_formula_latex:
                page_text_inline_formula_spans.append(
                    Span(
                        type=ContentType.INLINE_EQUATION,
                        bbox=inline_formula["bbox"],
                        content=inline_formula_latex,
                        score=inline_formula["score"],
                    )
                )
        for ocr_res in self.page_ocr_res:
            ocr_res["bbox"] = self.cal_real_bbox(ocr_res["bbox"])
            page_text_inline_formula_spans.append(
                Span(
                    type=ContentType.TEXT,
                    bbox=ocr_res["bbox"],
                    content=ocr_res.get("text", ""),
                    score=ocr_res["score"],
                )
            )
        if not _vlm_ocr_enable and not _ocr_enable:
            # Bad code
            virtual_block = (0, 0, width, height, None, None, None, "text")
            page_text_inline_formula_spans = txt_spans_extract(
                pdf_page,
                page_text_inline_formula_spans,
                page_pil_img,
                scale,
                [virtual_block],
                [],
            )
        span_matcher = SpanBlockMatcher(page_text_inline_formula_spans)

        # 解析每个块
        for index, block_info in enumerate(page_blocks):
            try:
                draft = VlmContentBlockDraft.from_content_block(block_info, width, height)
                block_bbox = draft.bbox
                block_type = draft.raw_type
                raw_block_type = draft.raw_type
                block_content = draft.content
                block_angle = draft.angle
                block_sub_type = draft.sub_type if raw_block_type in ["image", "chart"] else None
            except Exception as e:
                # 如果解析失败，可能是因为格式不正确，跳过这个块
                logger.warning(f"Invalid block format: {block_info}, error: {e}")
                continue

            span_type = "unknown"
            code_block_sub_type = None
            guess_lang = None

            if block_type in [
                "text",
                "title",
                "doc_title",
                "paragraph_title",
                "ref_text",
                "phonetic",
                "header",
                "footer",
                "page_number",
                "aside_text",
                "page_footnote",
                "list",
            ]:
                span_type = ContentType.TEXT
            elif block_type in ["image_caption", "table_caption", "code_caption"]:
                block_type = BlockType.CAPTION
                span_type = ContentType.TEXT
            elif block_type in ["image_footnote", "table_footnote"]:
                block_type = BlockType.FOOTNOTE
                span_type = ContentType.TEXT
            elif block_type == "image":
                block_type = BlockType.IMAGE_BODY
                span_type = ContentType.IMAGE
            elif block_type == "image_block":
                block_type = IMAGE_BLOCK_BODY
                span_type = ContentType.IMAGE
            elif block_type == "table":
                block_type = BlockType.TABLE_BODY
                span_type = ContentType.TABLE
            elif block_type == "chart":
                block_type = BlockType.CHART_BODY
                span_type = ContentType.CHART
            elif block_type in ["code", "algorithm"]:
                block_content = code_content_clean(block_content)
                code_block_sub_type = block_type
                block_type = BlockType.CODE_BODY
                span_type = ContentType.TEXT
                guess_lang = guess_language_by_text(block_content)
            elif block_type == "equation":
                block_type = BlockType.INTERLINE_EQUATION
                span_type = ContentType.INTERLINE_EQUATION

            if span_type == ContentType.TEXT and block_content is None:
                # 文本类块缺失 content 时按空文本处理，避免 VLM 渲染阶段遇到 None。
                block_content = ""

            # code 和 algorithm 类型的块，如果内容中包含行内公式，则需要将块类型切换为 algorithm
            switch_code_to_algorithm = False

            span = None
            if span_type in [ContentType.IMAGE, ContentType.TABLE, ContentType.CHART]:
                span = Span(type=span_type, bbox=block_bbox)
                if span_type == ContentType.TABLE and block_content is not None:
                    span.content = block_content
                elif raw_block_type in ["image", "chart"] and block_content is not None:
                    span.content = block_content
            elif span_type == ContentType.INTERLINE_EQUATION:
                span = Span(
                    type=span_type,
                    bbox=block_bbox,
                    content=isolated_formula_clean(block_content or ""),
                )
            elif _vlm_ocr_enable or block_type not in not_extract_list:
                # vlm_ocr_enable 模式下，所有文本块都直接使用 block 的内容
                # 非 vlm_ocr_enable 模式下，非提取块仍沿用直接内容模式
                if block_content:
                    block_content = clean_content(block_content)

                if block_type in [BlockType.TITLE, BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE] and block_content:
                    block_content = re.sub(r"\n\s*", " ", block_content).strip()

                if (
                    block_content
                    and block_content.count("\\(") == block_content.count("\\)")
                    and block_content.count("\\(") > 0
                ):
                    switch_code_to_algorithm = True

                    spans = []
                    last_end = 0
                    for match in re.finditer(r"\\\((.+?)\\\)", block_content):
                        start, end = match.span()

                        if start > last_end:
                            text_before = block_content[last_end:start]
                            if text_before.strip():
                                spans.append(
                                    Span(
                                        type=ContentType.TEXT,
                                        bbox=block_bbox,
                                        content=text_before,
                                    )
                                )

                        formula = match.group(1)
                        spans.append(
                            Span(
                                type=ContentType.INLINE_EQUATION,
                                bbox=block_bbox,
                                content=formula.strip(),
                            )
                        )

                        last_end = end

                    if last_end < len(block_content):
                        text_after = block_content[last_end:]
                        if text_after.strip():
                            spans.append(
                                Span(
                                    type=ContentType.TEXT,
                                    bbox=block_bbox,
                                    content=text_after,
                                )
                            )

                    span = spans
                else:
                    span = Span(
                        type=span_type,
                        bbox=block_bbox,
                        content=block_content or "",
                    )

            if span_type in [
                ContentType.IMAGE,
                ContentType.TABLE,
                ContentType.CHART,
                ContentType.INTERLINE_EQUATION,
            ] or (_vlm_ocr_enable or block_type not in not_extract_list):
                if span is None:
                    continue
                if isinstance(span, Span):
                    spans = [span]
                elif isinstance(span, list):
                    spans = span
                else:
                    raise ValueError(f"Invalid span type: {span_type}, expected dict or list, got {type(span)}")

                if block_type == BlockType.CODE_BODY:
                    if switch_code_to_algorithm and code_block_sub_type == "code":
                        code_block_sub_type = "algorithm"
                    line = Line(spans=spans, bbox=block_bbox, _code_type=code_block_sub_type, _code_guess_lang=guess_lang)
                else:
                    line = Line(spans=spans, bbox=block_bbox)

                block = Block(index=index, type=block_type, bbox=block_bbox, lines=[line], angle=block_angle)
                if block_sub_type:
                    block.sub_type = block_sub_type
                if raw_block_type == "table" and draft.cell_merge:
                    block._cell_merge = draft.cell_merge
                if _vlm_ocr_enable and self._supports_ocr_det_lines(block_type):
                    ocr_det_lines = self._build_ocr_det_lines(span_matcher.collect_for_block(block_bbox))
                    if ocr_det_lines:
                        block._ocr_det_lines = ocr_det_lines
                _copy_raw_text_block_metadata(draft, block)
            else:
                block_spans = span_matcher.collect_for_block(block_bbox)
                block = Block(index=index, type=block_type, bbox=block_bbox, angle=block_angle)
                block._fix_spans = block_spans
                block = fix_text_block(block)
                _copy_raw_text_block_metadata(draft, block)

            blocks.append(block)

        fallback_inline_caption_fragments(blocks, VISUAL_MAIN_TYPES)
        fallback_leading_table_continuation_captions(blocks, VISUAL_MAIN_TYPES)

        self.image_blocks: list[Block] = []
        self.table_blocks: list[Block] = []
        self.chart_blocks: list[Block] = []
        self.interline_equation_blocks: list[Block] = []
        self.text_blocks: list[Block] = []
        self.title_blocks: list[Block] = []
        self.code_blocks: list[Block] = []
        self.discarded_blocks: list[Block] = []
        self.ref_text_blocks: list[Block] = []
        self.phonetic_blocks: list[Block] = []
        self.list_blocks: list[Block] = []

        for block in blocks:
            if block.type in VISUAL_MAIN_TYPES or block.type in GENERIC_CHILD_TYPES:
                continue
            elif block.type == BlockType.INTERLINE_EQUATION:
                self.interline_equation_blocks.append(block)
            elif block.type == BlockType.TEXT:
                self.text_blocks.append(block)
            elif block.type in [
                BlockType.TITLE,
                BlockType.DOC_TITLE,
                BlockType.PARAGRAPH_TITLE,
            ]:
                self.title_blocks.append(block)
            elif block.type == BlockType.REF_TEXT:
                self.ref_text_blocks.append(block)
            elif block.type == BlockType.PHONETIC:
                self.phonetic_blocks.append(block)
            elif block.type in [
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
                BlockType.ASIDE_TEXT,
                BlockType.PAGE_FOOTNOTE,
            ]:
                self.discarded_blocks.append(block)
            elif block.type == BlockType.LIST:
                self.list_blocks.append(block)

        self.list_blocks, self.text_blocks, self.ref_text_blocks = fix_list_blocks(
            self.list_blocks,
            self.text_blocks,
            self.ref_text_blocks,
        )

        visual_groups, unmatched_child_blocks = regroup_visual_blocks(blocks)
        self.image_blocks = visual_groups[BlockType.IMAGE]
        self.table_blocks = visual_groups[BlockType.TABLE]
        self.chart_blocks = visual_groups[BlockType.CHART]
        self.code_blocks = visual_groups[BlockType.CODE]

        for code_block in self.code_blocks:
            for block in code_block.blocks:
                if block.type == BlockType.CODE_BODY:
                    if block.lines:
                        line = block.lines[0]
                        code_block.sub_type = line._code_type or ""
                        line._code_type = None
                        if code_block.sub_type == "code":
                            code_block.guess_lang = line._code_guess_lang or ""
                            line._code_guess_lang = None
                    else:
                        code_block.sub_type = "code"
                        code_block.guess_lang = "txt"

        for block in unmatched_child_blocks:
            block.type = BlockType.TEXT
            self.text_blocks.append(block)

    @staticmethod
    def _split_page_model_list(
        page_model_list: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        page_blocks = []
        page_inline_formula = []
        page_ocr_res = []

        for item in page_model_list:
            item_type = item.get("type") or item.get("label")
            if item_type == "inline_formula":
                page_inline_formula.append(item)
            elif item_type == "ocr_text":
                page_ocr_res.append(item)
            else:
                page_blocks.append(item)

        return page_blocks, page_inline_formula, page_ocr_res

    @staticmethod
    def _supports_ocr_det_lines(block_type: str) -> bool:
        """判断当前块类型是否需要保留 OCR det 行提示供 Hybrid 段落合并使用。"""
        return block_type in OCR_DET_LINE_BLOCK_TYPES

    @staticmethod
    def _build_ocr_det_lines(block_spans: list[Span]) -> list[Line]:
        """将 OCR det span 聚合成 line，但不改变 VLM-OCR 的 canonical 文本内容。"""
        if not block_spans:
            return []

        fix_block = Block(
            index=0,
            type="",
            bbox=EMPTY_BBOX,
            _fix_spans=copy.deepcopy(block_spans),
        )
        return fix_text_block(fix_block).lines

    def cal_real_bbox(self, bbox: BBox) -> IntBBox:
        x1, y1, x2, y2 = bbox
        x_1, y_1, x_2, y_2 = (
            int(x1 * self.width),
            int(y1 * self.height),
            int(x2 * self.width),
            int(y2 * self.height),
        )
        if x_2 < x_1:
            x_1, x_2 = x_2, x_1
        if y_2 < y_1:
            y_1, y_2 = y_2, y_1
        return (x_1, y_1, x_2, y_2)

    def get_list_blocks(self) -> list[Block]:
        return self.list_blocks

    def get_image_blocks(self) -> list[Block]:
        return self.image_blocks

    def get_table_blocks(self) -> list[Block]:
        return self.table_blocks

    def get_chart_blocks(self) -> list[Block]:
        return self.chart_blocks

    def get_code_blocks(self) -> list[Block]:
        return self.code_blocks

    def get_ref_text_blocks(self) -> list[Block]:
        return self.ref_text_blocks

    def get_phonetic_blocks(self) -> list[Block]:
        return self.phonetic_blocks

    def get_title_blocks(self) -> list[Block]:
        return self.title_blocks

    def get_text_blocks(self) -> list[Block]:
        return self.text_blocks

    def get_interline_equation_blocks(self) -> list[Block]:
        return self.interline_equation_blocks

    def get_discarded_blocks(self) -> list[Block]:
        return self.discarded_blocks


def fix_list_blocks(
    list_blocks: list[Block], text_blocks: list[Block], ref_text_blocks: list[Block]
) -> tuple[list[Block], list[Block], list[Block]]:
    for list_block in list_blocks:
        list_block.blocks = []
        if list_block.lines:
            list_block.lines = []

    temp_text_blocks = text_blocks + ref_text_blocks
    need_remove_blocks = []
    for block in temp_text_blocks:
        for list_block in list_blocks:
            if (
                calculate_overlap_area_in_bbox1_area_ratio(
                    block.bbox,
                    list_block.bbox,
                )
                >= 0.8
            ):
                list_block.blocks.append(block)
                need_remove_blocks.append(block)
                break

    for block in need_remove_blocks:
        if block in text_blocks:
            text_blocks.remove(block)
        elif block in ref_text_blocks:
            ref_text_blocks.remove(block)

    list_blocks = [lb for lb in list_blocks if lb.blocks]

    for list_block in list_blocks:
        type_count = {}
        for sub_block in list_block.blocks:
            sub_block_type = sub_block.type
            if sub_block_type not in type_count:
                type_count[sub_block_type] = 0
            type_count[sub_block_type] += 1

        if type_count:
            list_block.sub_type = max(type_count, key=type_count.get)  # type: ignore
        else:
            list_block.sub_type = "unknown"

    return list_blocks, text_blocks, ref_text_blocks
