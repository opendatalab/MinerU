# Copyright (c) Opendatalab. All rights reserved.
import copy
import re

from loguru import logger

from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.enum_class import ContentType, BlockType, NotExtractType
from mineru.utils.guess_suffix_or_lang import guess_language_by_text
from mineru.utils.span_block_fix import fix_text_block
from mineru.utils.span_pre_proc import txt_spans_extract
from mineru.utils.visual_magic_model_utils import (
    GENERIC_CHILD_TYPES,
    IMAGE_BLOCK_BODY,
    VISUAL_MAIN_TYPES,
    clean_content,
    code_content_clean,
    isolated_formula_clean,
    regroup_visual_blocks,
)
not_extract_list = [item.value for item in NotExtractType] + [
    BlockType.CAPTION,
    BlockType.FOOTNOTE,
]


def _copy_raw_text_block_metadata(raw_block_type, block_info, block):
    if raw_block_type != BlockType.TEXT:
        return
    if "merge_prev" in block_info:
        block["merge_prev"] = block_info["merge_prev"]


class MagicModel:
    def __init__(
        self,
        page_model_list: list,
        page,
        scale,
        page_pil_img,
        width,
        height,
        _ocr_enable,
        _vlm_ocr_enable,
    ):
        (
            self.page_blocks,
            self.page_inline_formula,
            self.page_ocr_res,
        ) = self._split_page_model_list(copy.deepcopy(page_model_list))

        self.width = width
        self.height = height

        blocks = []
        self.all_spans = []

        page_text_inline_formula_spans = []
        if not _vlm_ocr_enable:
            for inline_formula in self.page_inline_formula:
                inline_formula["bbox"] = self.cal_real_bbox(inline_formula["bbox"])
                inline_formula_latex = inline_formula.pop("latex", "")
                if inline_formula_latex:
                    page_text_inline_formula_spans.append(
                        {
                            "bbox": inline_formula["bbox"],
                            "type": ContentType.INLINE_EQUATION,
                            "content": inline_formula_latex,
                            "score": inline_formula["score"],
                        }
                    )
            for ocr_res in self.page_ocr_res:
                ocr_res["bbox"] = self.cal_real_bbox(ocr_res["bbox"])
                page_text_inline_formula_spans.append(
                    {
                        "bbox": ocr_res["bbox"],
                        "type": ContentType.TEXT,
                        "content": ocr_res["text"],
                        "score": ocr_res["score"],
                    }
                )
            if not _ocr_enable:
                virtual_block = [0, 0, width, height, None, None, None, "text"]
                page_text_inline_formula_spans = txt_spans_extract(
                    page,
                    page_text_inline_formula_spans,
                    page_pil_img,
                    scale,
                    [virtual_block],
                    [],
                )

        # 解析每个块
        for index, block_info in enumerate(self.page_blocks):
            try:
                block_bbox = self.cal_real_bbox(block_info["bbox"])
                block_type = block_info["type"]
                raw_block_type = block_type
                block_content = block_info.get("content")
                block_angle = block_info.get("angle", 0)
                block_sub_type = (
                    block_info.get("sub_type")
                    if raw_block_type in ["image", "chart"]
                    else None
                )
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

            # code 和 algorithm 类型的块，如果内容中包含行内公式，则需要将块类型切换为 algorithm
            switch_code_to_algorithm = False

            span = None
            if span_type in [ContentType.IMAGE, ContentType.TABLE, ContentType.CHART]:
                span = {
                    "bbox": block_bbox,
                    "type": span_type,
                }
                if span_type == ContentType.TABLE:
                    span["html"] = block_content
                elif raw_block_type in ["image", "chart"] and block_content is not None:
                    span["content"] = block_content
            elif span_type == ContentType.INTERLINE_EQUATION:
                span = {
                    "bbox": block_bbox,
                    "type": span_type,
                    "content": isolated_formula_clean(block_content),
                }
            elif _vlm_ocr_enable or block_type not in not_extract_list:
                # vlm_ocr_enable 模式下，所有文本块都直接使用 block 的内容
                # 非 vlm_ocr_enable 模式下，非提取块仍沿用直接内容模式
                if block_content:
                    block_content = clean_content(block_content)

                if block_type == "title" and block_content:
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
                                    {
                                        "bbox": block_bbox,
                                        "type": ContentType.TEXT,
                                        "content": text_before,
                                    }
                                )

                        formula = match.group(1)
                        spans.append(
                            {
                                "bbox": block_bbox,
                                "type": ContentType.INLINE_EQUATION,
                                "content": formula.strip(),
                            }
                        )

                        last_end = end

                    if last_end < len(block_content):
                        text_after = block_content[last_end:]
                        if text_after.strip():
                            spans.append(
                                {
                                    "bbox": block_bbox,
                                    "type": ContentType.TEXT,
                                    "content": text_after,
                                }
                            )

                    span = spans
                else:
                    span = {
                        "bbox": block_bbox,
                        "type": span_type,
                        "content": block_content,
                    }

            if (
                span_type
                in [
                    ContentType.IMAGE,
                    ContentType.TABLE,
                    ContentType.CHART,
                    ContentType.INTERLINE_EQUATION,
                ]
                or (_vlm_ocr_enable or block_type not in not_extract_list)
            ):
                if span is None:
                    continue
                if isinstance(span, dict) and "bbox" in span:
                    self.all_spans.append(span)
                    spans = [span]
                elif isinstance(span, list):
                    self.all_spans.extend(span)
                    spans = span
                else:
                    raise ValueError(
                        f"Invalid span type: {span_type}, expected dict or list, got {type(span)}"
                    )

                if block_type == BlockType.CODE_BODY:
                    if switch_code_to_algorithm and code_block_sub_type == "code":
                        code_block_sub_type = "algorithm"
                    line = {
                        "bbox": block_bbox,
                        "spans": spans,
                        "extra": {
                            "type": code_block_sub_type,
                            "guess_lang": guess_lang,
                        },
                    }
                else:
                    line = {"bbox": block_bbox, "spans": spans}

                block = {
                    "bbox": block_bbox,
                    "type": block_type,
                    "angle": block_angle,
                    "lines": [line],
                    "index": index,
                }
                if block_sub_type:
                    block["sub_type"] = block_sub_type
                if raw_block_type == "table" and "cell_merge" in block_info:
                    block["cell_merge"] = block_info["cell_merge"]
                _copy_raw_text_block_metadata(raw_block_type, block_info, block)
            else:
                block_spans = []
                for span in page_text_inline_formula_spans:
                    if (
                        calculate_overlap_area_in_bbox1_area_ratio(
                            span["bbox"],
                            block_bbox,
                        )
                        > 0.5
                    ):
                        block_spans.append(span)

                if block_spans:
                    for span in block_spans:
                        page_text_inline_formula_spans.remove(span)

                block = {
                    "bbox": block_bbox,
                    "type": block_type,
                    "angle": block_angle,
                    "spans": block_spans,
                    "index": index,
                }
                block = fix_text_block(block)
                _copy_raw_text_block_metadata(raw_block_type, block_info, block)

            blocks.append(block)

        self.image_blocks = []
        self.table_blocks = []
        self.chart_blocks = []
        self.interline_equation_blocks = []
        self.text_blocks = []
        self.title_blocks = []
        self.code_blocks = []
        self.discarded_blocks = []
        self.ref_text_blocks = []
        self.phonetic_blocks = []
        self.list_blocks = []

        for block in blocks:
            if block["type"] in VISUAL_MAIN_TYPES or block["type"] in GENERIC_CHILD_TYPES:
                continue
            elif block["type"] == BlockType.INTERLINE_EQUATION:
                self.interline_equation_blocks.append(block)
            elif block["type"] == BlockType.TEXT:
                self.text_blocks.append(block)
            elif block["type"] == BlockType.TITLE:
                self.title_blocks.append(block)
            elif block["type"] == BlockType.REF_TEXT:
                self.ref_text_blocks.append(block)
            elif block["type"] == BlockType.PHONETIC:
                self.phonetic_blocks.append(block)
            elif block["type"] in [
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
                BlockType.ASIDE_TEXT,
                BlockType.PAGE_FOOTNOTE,
            ]:
                self.discarded_blocks.append(block)
            elif block["type"] == BlockType.LIST:
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
            for block in code_block["blocks"]:
                if block["type"] == BlockType.CODE_BODY:
                    if block["lines"]:
                        line = block["lines"][0]
                        code_block["sub_type"] = line["extra"]["type"]
                        if code_block["sub_type"] == "code":
                            code_block["guess_lang"] = line["extra"]["guess_lang"]
                        del line["extra"]
                    else:
                        code_block["sub_type"] = "code"
                        code_block["guess_lang"] = "txt"

        for block in unmatched_child_blocks:
            block["type"] = BlockType.TEXT
            self.text_blocks.append(block)

    @staticmethod
    def _split_page_model_list(page_model_list):
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

    def cal_real_bbox(self, bbox):
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

    def get_list_blocks(self):
        return self.list_blocks

    def get_image_blocks(self):
        return self.image_blocks

    def get_table_blocks(self):
        return self.table_blocks

    def get_chart_blocks(self):
        return self.chart_blocks

    def get_code_blocks(self):
        return self.code_blocks

    def get_ref_text_blocks(self):
        return self.ref_text_blocks

    def get_phonetic_blocks(self):
        return self.phonetic_blocks

    def get_title_blocks(self):
        return self.title_blocks

    def get_text_blocks(self):
        return self.text_blocks

    def get_interline_equation_blocks(self):
        return self.interline_equation_blocks

    def get_discarded_blocks(self):
        return self.discarded_blocks

    def get_all_spans(self):
        return self.all_spans
def fix_list_blocks(list_blocks, text_blocks, ref_text_blocks):
    for list_block in list_blocks:
        list_block["blocks"] = []
        if "lines" in list_block:
            del list_block["lines"]

    temp_text_blocks = text_blocks + ref_text_blocks
    need_remove_blocks = []
    for block in temp_text_blocks:
        for list_block in list_blocks:
            if (
                calculate_overlap_area_in_bbox1_area_ratio(
                    block["bbox"],
                    list_block["bbox"],
                )
                >= 0.8
            ):
                list_block["blocks"].append(block)
                need_remove_blocks.append(block)
                break

    for block in need_remove_blocks:
        if block in text_blocks:
            text_blocks.remove(block)
        elif block in ref_text_blocks:
            ref_text_blocks.remove(block)

    list_blocks = [lb for lb in list_blocks if lb["blocks"]]

    for list_block in list_blocks:
        type_count = {}
        for sub_block in list_block["blocks"]:
            sub_block_type = sub_block["type"]
            if sub_block_type not in type_count:
                type_count[sub_block_type] = 0
            type_count[sub_block_type] += 1

        if type_count:
            list_block["sub_type"] = max(type_count, key=type_count.get)
        else:
            list_block["sub_type"] = "unknown"

    return list_blocks, text_blocks, ref_text_blocks
