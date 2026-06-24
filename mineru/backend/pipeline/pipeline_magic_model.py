# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from ...render.merge import _merge_para_text
from ...types import EMPTY_BBOX, BBox, Block, BlockType, ContentType, Line, Span
from ...utils.guess_suffix_or_lang import guess_language_by_text
from ..utils.boxbase import calculate_overlap_area_2_minbox_area_ratio, calculate_overlap_area_in_bbox1_area_ratio
from ..utils.span_block_fix import (
    is_vertical_text_block_by_spans,
    line_sort_spans_by_left_to_right,
    merge_spans_to_line,
    merge_spans_to_vertical_line,
    vertical_line_sort_spans_from_top_to_bottom,
)
from ..utils.span_pre_proc import SpanBlockMatcher, txt_spans_extract
from ..utils.visual_magic_model_utils import (
    fallback_inline_caption_fragments,
    fallback_leading_table_continuation_captions,
    find_best_visual_parent,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class MagicModel:
    PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES: dict[str, str] = {
        "abstract": BlockType.ABSTRACT,
        "algorithm": BlockType.CODE,
        "aside_text": BlockType.ASIDE_TEXT,
        "chart": BlockType.CHART,
        "content": BlockType.INDEX,
        "display_formula": BlockType.INTERLINE_EQUATION,
        "doc_title": BlockType.DOC_TITLE,
        "figure_title": BlockType.CAPTION,
        "footer": BlockType.FOOTER,
        "footer_image": BlockType.FOOTER,
        "footnote": BlockType.PAGE_FOOTNOTE,
        "formula_number": BlockType.FORMULA_NUMBER,
        "header": BlockType.HEADER,
        "header_image": BlockType.HEADER,
        "image": BlockType.IMAGE,
        "number": BlockType.PAGE_NUMBER,
        "paragraph_title": BlockType.PARAGRAPH_TITLE,
        "reference_content": BlockType.REF_TEXT,
        "seal": BlockType.IMAGE,
        "table": BlockType.TABLE,
        "text": BlockType.TEXT,
        "vertical_text": BlockType.VERTICAL_TEXT,
        "vision_footnote": BlockType.FOOTNOTE,
    }

    VISUAL_MAIN_TYPES: set[str] = {
        BlockType.IMAGE,
        BlockType.TABLE,
        BlockType.CHART,
        BlockType.CODE,
    }
    VISUAL_CHILD_TYPES: set[str] = {
        BlockType.CAPTION,
        BlockType.FOOTNOTE,
    }
    VISUAL_TYPE_MAPPING: dict[str, dict[str, str]] = {
        BlockType.IMAGE: {
            "body": BlockType.IMAGE_BODY,
            "caption": BlockType.IMAGE_CAPTION,
            "footnote": BlockType.IMAGE_FOOTNOTE,
        },
        BlockType.TABLE: {
            "body": BlockType.TABLE_BODY,
            "caption": BlockType.TABLE_CAPTION,
            "footnote": BlockType.TABLE_FOOTNOTE,
        },
        BlockType.CHART: {
            "body": BlockType.CHART_BODY,
            "caption": BlockType.CHART_CAPTION,
            "footnote": BlockType.CHART_FOOTNOTE,
        },
        BlockType.CODE: {
            "body": BlockType.CODE_BODY,
            "caption": BlockType.CODE_CAPTION,
            "footnote": BlockType.CODE_FOOTNOTE,
        },
    }

    def __init__(
        self,
        page_model_info: dict[str, Any],
        page: object = None,
        scale: float = 1,
        page_pil_img: PILImage | None = None,
        page_w: int = 0,
        page_h: int = 0,
        ocr_enable: bool = False,
    ) -> None:
        self.__page_model_info = page_model_info
        self.page_inline_formula: list[Span] = []
        self.page_ocr_res: list[Span] = []
        self.page_blocks: list[Block] = []
        self.image_groups: list[dict[str, Any]] = []
        self.table_groups: list[dict[str, Any]] = []
        self.chart_groups: list[dict[str, Any]] = []
        self.__layout_det_by_index: dict[int, dict[str, Any]] = {}
        self.__scale = scale
        self.__fix_axis()  # bbox坐标修正，删除高度或者宽度小于等于0的spans
        self.__post_process()  # index重排，填充行内公式和文本span
        if not ocr_enable:
            # Bad code
            virtual_block = (0, 0, page_w, page_h, None, None, None, "text")
            self.page_ocr_res = txt_spans_extract(
                page,
                self.page_ocr_res,
                page_pil_img,
                scale,
                [virtual_block],
                [],
            )
        self.page_text_inline_formula_spans = self.page_inline_formula + self.page_ocr_res

        for layout_det in self.__page_model_info["layout_dets"]:
            if layout_det.get("label") in self.PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES:
                block_bbox = layout_det["bbox"]
                block_type = self.PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES[layout_det["label"]]
                block_index = layout_det["index"]
                block_score = layout_det["score"]
                block = self.__copy_block_fields(
                    layout_det,
                    type=block_type,
                    bbox=block_bbox,
                    index=block_index,
                    score=block_score,
                )
                if self.__is_seal_layout_block(layout_det):
                    block.sub_type = "seal"
                self.page_blocks.append(block)

        self.page_blocks.sort(key=lambda x: x.index)
        self.__build_page_blocks()
        fallback_inline_caption_fragments(self.page_blocks, self.VISUAL_MAIN_TYPES)
        fallback_leading_table_continuation_captions(self.page_blocks, self.VISUAL_MAIN_TYPES)
        self.__classify_visual_blocks()
        self.__build_return_blocks()

    @staticmethod
    def __fix_text_block(block: Block) -> Block:
        if block.type == BlockType.TEXT and is_vertical_text_block_by_spans(block._fix_spans):
            # layout 偶发会把竖排正文识别为横排 text，这里用旧版 span 高宽比规则兜底。
            block.type = BlockType.VERTICAL_TEXT

        if block.type == BlockType.VERTICAL_TEXT:
            # 如果是纵向文本块，则按纵向lines处理
            block_lines = merge_spans_to_vertical_line(block._fix_spans)
            sort_block_lines = vertical_line_sort_spans_from_top_to_bottom(block_lines)
        else:
            block_lines = merge_spans_to_line(block._fix_spans)
            sort_block_lines = line_sort_spans_by_left_to_right(block_lines)

        if block.type == BlockType.CODE:
            for line in sort_block_lines:
                line._is_list_start = True
            _temp_block = Block(index=0, type="", bbox=EMPTY_BBOX, lines=sort_block_lines)
            code_content = _merge_para_text(_temp_block, escape_markdown=False, list_line_break="\n")
            guess_lang = guess_language_by_text(code_content)
            if guess_lang not in ["txt", "unknown"]:
                block.sub_type = "code"
                block.guess_lang = guess_lang

        block.lines = sort_block_lines
        block._fix_spans = []
        return block

    @staticmethod
    def __copy_block_fields(block: dict[str, Any], **overrides: Any) -> Block:
        kwargs = {k: v for k, v in block.items() if k not in {"cls_id", "label"}}
        kwargs = {**kwargs, **overrides}
        return Block(**kwargs)

    @staticmethod
    def __is_inline_formula_block(layout_det: dict[str, Any]) -> bool:
        return layout_det.get("label") == "inline_formula" or layout_det.get("cls_id") == 15

    @staticmethod
    def __is_ocr_text_block(layout_det: dict[str, Any]) -> bool:
        return layout_det.get("label") == "ocr_text"

    @staticmethod
    def __is_seal_layout_block(layout_det: dict[str, Any]) -> bool:
        """判断原始 layout 是否为印章，输出层会将其规范为 image 子类型。"""
        return layout_det.get("label") == "seal"

    @staticmethod
    def __normalize_seal_text(content: str | list[str]) -> str:
        """将 seal OCR 的列表或字符串结果规范为 VLM 一致的多行字符串。"""
        if isinstance(content, list):
            return "\n".join(str(item) for item in content if str(item).strip())
        if isinstance(content, str):
            return content.strip()
        return ""

    def __build_return_blocks(self) -> None:
        self.preproc_blocks: list[Block] = []
        self.discarded_blocks: list[Block] = []
        for block in self.page_blocks:
            if block.type in [
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
                BlockType.ASIDE_TEXT,
                BlockType.PAGE_FOOTNOTE,
            ]:
                self.discarded_blocks.append(block)
            else:
                # 单独处理code block
                if block.type in [BlockType.CODE]:
                    for sub_block in block.blocks:
                        if sub_block.type == BlockType.CODE_BODY:
                            block.sub_type = sub_block.sub_type or "algorithm"
                            sub_block.sub_type = ""
                            if block.sub_type == "code":
                                block.guess_lang = sub_block.guess_lang or "txt"
                                sub_block.guess_lang = ""

                self.preproc_blocks.append(block)

    def __build_page_blocks(self) -> None:
        span_type = "unknown"
        span_matcher = SpanBlockMatcher(self.page_text_inline_formula_spans)
        for block in self.page_blocks:
            if block.type in [
                BlockType.ABSTRACT,
                BlockType.CODE,
                BlockType.ASIDE_TEXT,
                BlockType.INDEX,
                BlockType.DOC_TITLE,
                BlockType.CAPTION,
                BlockType.FOOTER,
                BlockType.PAGE_FOOTNOTE,
                BlockType.FORMULA_NUMBER,
                BlockType.HEADER,
                BlockType.PAGE_NUMBER,
                BlockType.PARAGRAPH_TITLE,
                BlockType.REF_TEXT,
                BlockType.TEXT,
                BlockType.VERTICAL_TEXT,
                BlockType.FOOTNOTE,
            ]:
                span_type = ContentType.TEXT
            elif block.type in [BlockType.IMAGE]:
                span_type = ContentType.IMAGE
            elif block.type in [BlockType.TABLE]:
                span_type = ContentType.TABLE
            elif block.type in [BlockType.CHART]:
                span_type = ContentType.CHART
            elif block.type in [BlockType.INTERLINE_EQUATION]:
                span_type = ContentType.INTERLINE_EQUATION

            if span_type in [
                ContentType.IMAGE,
                ContentType.TABLE,
                ContentType.CHART,
                ContentType.INTERLINE_EQUATION,
            ]:
                span = Span(type=span_type, bbox=block.bbox)
                if span_type == ContentType.IMAGE and block.sub_type == "seal":
                    seal_text = self.__normalize_seal_text(block.text)
                    if seal_text:
                        span.content = seal_text
                    block.text = ""
                if span_type == ContentType.TABLE:
                    span.html = block.html
                    block.html = ""
                if span_type == ContentType.INTERLINE_EQUATION:
                    span.content = block.latex
                    block.latex = ""

                # 构造line对象
                spans = [span]
                line = Line(spans=spans, bbox=block.bbox)
                block.lines = [line]
            else:
                # span填充
                if block.type == BlockType.FORMULA_NUMBER:
                    block_spans = span_matcher.collect_for_block(
                        block.bbox,
                        overlap_ratio_getter=self.__formula_number_overlap_ratio,
                    )
                else:
                    block_spans = span_matcher.collect_for_block(block.bbox)

                block._fix_spans = block_spans
                block = self.__fix_text_block(block)
        self.page_text_inline_formula_spans = span_matcher.remaining_spans()

    @staticmethod
    def __formula_number_overlap_ratio(span: Span, block_bbox: BBox) -> float:
        """公式编号框较窄时，沿用最小框重叠比例提高回填召回。"""
        return max(
            calculate_overlap_area_in_bbox1_area_ratio(span.bbox, block_bbox),
            calculate_overlap_area_2_minbox_area_ratio(span.bbox, block_bbox),
        )

    def __fix_axis(self) -> None:
        need_remove_list: list[dict[str, Any]] = []
        layout_dets = self.__page_model_info["layout_dets"]
        for layout_det in layout_dets:
            x0, y0, x1, y1 = layout_det["bbox"]
            bbox = [
                int(x0 / self.__scale),
                int(y0 / self.__scale),
                int(x1 / self.__scale),
                int(y1 / self.__scale),
            ]
            layout_det["bbox"] = bbox
            # 删除高度或者宽度小于等于2的spans
            if bbox[2] - bbox[0] <= 2 or bbox[3] - bbox[1] <= 2:
                need_remove_list.append(layout_det)
        for need_remove in need_remove_list:
            layout_dets.remove(need_remove)

    def __post_process(self) -> None:
        next_index = 1
        layout_dets = self.__page_model_info["layout_dets"]
        for layout_det in layout_dets:
            if self.__is_inline_formula_block(layout_det):
                layout_det.pop("index", None)
                self.page_inline_formula.append(
                    Span(
                        type=ContentType.INLINE_EQUATION,
                        bbox=layout_det["bbox"],
                        content=layout_det["latex"],
                        score=layout_det["score"],
                    )
                )
                continue

            if self.__is_ocr_text_block(layout_det):
                self.page_ocr_res.append(
                    Span(
                        type=ContentType.TEXT,
                        bbox=layout_det["bbox"],
                        content=layout_det["text"],
                        score=layout_det["score"],
                    )
                )
                continue

            if "index" in layout_det:
                layout_det["index"] = next_index
                next_index += 1

    def __classify_visual_blocks(self) -> None:
        if not self.page_blocks:
            return

        ordered_blocks = sorted(self.page_blocks, key=lambda x: x.index)
        original_type_by_index = {block.index: block.type for block in ordered_blocks}
        position_by_index = {block.index: pos for pos, block in enumerate(ordered_blocks)}
        main_blocks = [block for block in ordered_blocks if original_type_by_index[block.index] in self.VISUAL_MAIN_TYPES]
        child_blocks = [block for block in ordered_blocks if original_type_by_index[block.index] in self.VISUAL_CHILD_TYPES]

        child_parent_map: dict[int, int | None] = {}
        grouped_children: dict[int, dict[str, list[Block]]] = {
            main_block.index: {"captions": [], "footnotes": []} for main_block in main_blocks
        }

        for child_block in child_blocks:
            parent_block = self.__find_best_visual_parent(
                child_block,
                main_blocks,
                ordered_blocks,
                original_type_by_index,
                position_by_index,
            )
            child_parent_map[child_block.index] = None if parent_block is None else parent_block.index

        for child_block in child_blocks:
            original_child_type = original_type_by_index[child_block.index]
            parent_index = child_parent_map[child_block.index]

            if parent_index is None:
                child_block.type = BlockType.TEXT
                self.__sync_layout_det_type(child_block.index, BlockType.TEXT)
                continue

            parent_type = original_type_by_index[parent_index]
            child_kind = self.__child_kind(original_child_type)
            mapped_type = self.VISUAL_TYPE_MAPPING[parent_type][child_kind]
            child_block.type = mapped_type
            self.__sync_layout_det_type(child_block.index, mapped_type)
            grouped_children[parent_index][f"{child_kind}s"].append(child_block)

        self.image_groups = []
        self.table_groups = []
        self.chart_groups = []

        rebuilt_page_blocks: list[Block] = []
        for block in ordered_blocks:
            original_block_type = original_type_by_index[block.index]

            if original_block_type in self.VISUAL_CHILD_TYPES:
                if child_parent_map[block.index] is None:
                    rebuilt_page_blocks.append(block)
                continue

            if original_block_type not in self.VISUAL_MAIN_TYPES:
                rebuilt_page_blocks.append(block)
                continue

            mapping = self.VISUAL_TYPE_MAPPING[original_block_type]
            body_block = self.__make_child_block(block, mapping["body"])
            if original_block_type in [BlockType.IMAGE, BlockType.CHART]:
                body_block.sub_type = ""
            captions = sorted(
                [self.__make_child_block(caption, mapping["caption"]) for caption in grouped_children[block.index]["captions"]],
                key=lambda x: x.index,
            )
            footnotes = sorted(
                [
                    self.__make_child_block(footnote, mapping["footnote"])
                    for footnote in grouped_children[block.index]["footnotes"]
                ],
                key=lambda x: x.index,
            )

            self.__sync_layout_det_type(block.index, mapping["body"])

            group_info = {
                f"{original_block_type}_body": body_block,
                f"{original_block_type}_caption_list": captions,
                f"{original_block_type}_footnote_list": footnotes,
            }
            if original_block_type == BlockType.IMAGE:
                self.image_groups.append(group_info)
            elif original_block_type == BlockType.TABLE:
                self.table_groups.append(group_info)
            else:
                self.chart_groups.append(group_info)

            two_layer_block = Block(
                index=block.index,
                type=original_block_type,
                bbox=block.bbox,
                blocks=[body_block, *captions, *footnotes],
                score=block.score,
            )
            if original_block_type in [BlockType.IMAGE, BlockType.CHART] and block.sub_type:
                two_layer_block.sub_type = block.sub_type
            # 对blocks按index排序
            two_layer_block.blocks.sort(key=lambda x: x.index)
            rebuilt_page_blocks.append(two_layer_block)

        self.page_blocks = rebuilt_page_blocks

    def __find_best_visual_parent(
        self,
        child_block: Block,
        main_blocks: list[Block],
        ordered_blocks: list[Block],
        original_type_by_index: dict[int, str],
        position_by_index: dict[int, int],
    ) -> Block | None:
        child_type = original_type_by_index[child_block.index]
        if child_type not in self.VISUAL_CHILD_TYPES:
            return None

        return find_best_visual_parent(
            child_block,
            main_blocks,
            ordered_blocks,
            position_by_index,
            main_type_to_visual_type={block_type: block_type for block_type in self.VISUAL_MAIN_TYPES},
            type_by_index=original_type_by_index,
        )

    @staticmethod
    def __child_kind(block_type: str) -> str:
        if block_type == BlockType.CAPTION:
            return "caption"
        return "footnote"

    @staticmethod
    def __make_child_block(block: Block, block_type: str) -> Block:
        new_block = deepcopy(block)
        new_block.type = block_type
        return new_block

    def __sync_layout_det_type(self, block_index: int, block_type: str) -> None:
        layout_det = self.__layout_det_by_index.get(block_index)
        if layout_det is not None:
            layout_det["type"] = block_type

    def get_page_blocks(self) -> list[Block]:
        return self.page_blocks

    def get_preproc_blocks(self) -> list[Block]:
        return self.preproc_blocks

    def get_discarded_blocks(self) -> list[Block]:
        return self.discarded_blocks

    def get_imgs(self) -> list[dict[str, Any]]:
        return self.image_groups

    def get_tables(self) -> list[dict[str, Any]]:
        return self.table_groups

    def get_charts(self) -> list[dict[str, Any]]:
        return self.chart_groups

    def get_image_blocks(self) -> list[Block]:
        return [block for block in self.page_blocks if block.type == BlockType.IMAGE]

    def get_table_blocks(self) -> list[Block]:
        return [block for block in self.page_blocks if block.type == BlockType.TABLE]

    def get_chart_blocks(self) -> list[Block]:
        return [block for block in self.page_blocks if block.type == BlockType.CHART]
