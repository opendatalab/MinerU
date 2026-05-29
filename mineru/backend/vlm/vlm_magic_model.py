# Copyright (c) Opendatalab. All rights reserved.
import re

from loguru import logger

from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.guess_suffix_or_lang import guess_language_by_text
from mineru.utils.visual_magic_model_utils import (
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


def _copy_raw_text_block_metadata(raw_block_type, block_info, block):
    if raw_block_type != BlockType.TEXT:
        return
    if "merge_prev" in block_info:
        block["merge_prev"] = block_info["merge_prev"]


class MagicModel:
    def __init__(self, page_blocks: list, width, height):
        self.page_blocks = page_blocks

        blocks = []
        self.all_spans = []
        # 解析每个块
        for index, block_info in enumerate(page_blocks):
            block_bbox = block_info["bbox"]
            try:
                x1, y1, x2, y2 = block_bbox
                x_1, y_1, x_2, y_2 = (
                    int(x1 * width),
                    int(y1 * height),
                    int(x2 * width),
                    int(y2 * height),
                )
                if x_2 < x_1:
                    x_1, x_2 = x_2, x_1
                if y_2 < y_1:
                    y_1, y_2 = y_2, y_1
                block_bbox = (x_1, y_1, x_2, y_2)
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

            if span_type == ContentType.TEXT and block_content is None:
                # 文本类块缺失 content 时按空文本处理，避免下游 mkcontent 遇到 None。
                block_content = ""

            # code 和 algorithm 类型的块，如果内容中包含行内公式，则需要将块类型切换为 algorithm
            switch_code_to_algorithm = False

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
            else:
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

                    # 生成包含文本和公式的span列表
                    spans = []
                    last_end = 0

                    # 查找所有公式
                    for match in re.finditer(r"\\\((.+?)\\\)", block_content):
                        start, end = match.span()

                        # 添加公式前的文本
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

                        # 添加公式（去除\(和\)）
                        formula = match.group(1)
                        spans.append(
                            {
                                "bbox": block_bbox,
                                "type": ContentType.INLINE_EQUATION,
                                "content": formula.strip(),
                            }
                        )

                        last_end = end

                    # 添加最后一个公式后的文本
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

            # 处理span类型并添加到all_spans
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

            # 构造 line 对象
            if block_type == BlockType.CODE_BODY:
                if switch_code_to_algorithm and code_block_sub_type == "code":
                    code_block_sub_type = "algorithm"
                line = {
                    "bbox": block_bbox,
                    "spans": spans,
                    "extra": {"type": code_block_sub_type, "guess_lang": guess_lang},
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

            blocks.append(block)

        fallback_inline_caption_fragments(blocks, VISUAL_MAIN_TYPES)
        fallback_leading_table_continuation_captions(blocks, VISUAL_MAIN_TYPES)

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

    # 移除blocks为空的list_block
    list_blocks = [lb for lb in list_blocks if lb["blocks"]]

    for list_block in list_blocks:
        # 统计list_block["blocks"]中所有block的type，用众数作为list_block的sub_type
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
