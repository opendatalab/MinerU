import re

from loguru import logger

from mineru.utils.boxbase import (
    bbox_center_distance,
    bbox_distance,
    calculate_overlap_area_in_bbox1_area_ratio,
)
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.guess_suffix_or_lang import guess_language_by_text


IMAGE_BLOCK_BODY = "image_block_body"
GENERIC_CHILD_TYPES = (BlockType.CAPTION, BlockType.FOOTNOTE)
VISUAL_MAIN_TYPES = {
    BlockType.IMAGE_BODY: BlockType.IMAGE,
    IMAGE_BLOCK_BODY: BlockType.IMAGE,
    BlockType.TABLE_BODY: BlockType.TABLE,
    BlockType.CHART_BODY: BlockType.CHART,
    BlockType.CODE_BODY: BlockType.CODE,
}
VISUAL_TYPE_MAPPING = {
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
                block_content = block_info.get("content")
                block_angle = block_info.get("angle", 0)
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

            if span_type in [ContentType.IMAGE, ContentType.TABLE, ContentType.CHART]:
                span = {
                    "bbox": block_bbox,
                    "type": span_type,
                }
                if span_type == ContentType.TABLE:
                    span["html"] = block_content
                elif span_type == ContentType.CHART and block_content:
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

            blocks.append(
                {
                    "bbox": block_bbox,
                    "type": block_type,
                    "angle": block_angle,
                    "lines": [line],
                    "index": index,
                }
            )

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


def isolated_formula_clean(txt):
    latex = txt[:]
    if latex.startswith("\\["):
        latex = latex[2:]
    if latex.endswith("\\]"):
        latex = latex[:-2]
    latex = latex.strip()
    return latex


def code_content_clean(content):
    """清理代码内容，移除Markdown代码块的开始和结束标记"""
    if not content:
        return ""

    lines = content.splitlines()
    start_idx = 0
    end_idx = len(lines)

    # 处理开头的三个反引号
    if lines and lines[0].startswith("```"):
        start_idx = 1

    # 处理结尾的三个反引号
    if lines and end_idx > start_idx and lines[end_idx - 1].strip() == "```":
        end_idx -= 1

    # 只有在有内容时才进行join操作
    if start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx]).strip()
    return ""


def clean_content(content):
    if content and content.count("\\[") == content.count("\\]") and content.count("\\[") > 0:
        # Function to handle each match
        def replace_pattern(match):
            # Extract content between \[ and \]
            inner_content = match.group(1)
            return f"[{inner_content}]"

        # Find all patterns of \[x\] and apply replacement
        pattern = r"\\\[(.*?)\\\]"
        content = re.sub(pattern, replace_pattern, content)

    return content


def regroup_visual_blocks(blocks):
    ordered_blocks = sorted(blocks, key=lambda x: x["index"])
    absorbed_member_indices, sub_images_by_index = absorb_image_block_members(ordered_blocks)
    effective_blocks = [
        block for block in ordered_blocks if block["index"] not in absorbed_member_indices
    ]
    position_by_index = {
        block["index"]: pos for pos, block in enumerate(effective_blocks)
    }
    main_blocks = [
        block for block in effective_blocks if block["type"] in VISUAL_MAIN_TYPES
    ]
    child_blocks = [
        block for block in effective_blocks if block["type"] in GENERIC_CHILD_TYPES
    ]

    grouped_children = {
        block["index"]: {"captions": [], "footnotes": []} for block in main_blocks
    }
    unmatched_child_blocks = []

    for main_block in main_blocks:
        if main_block["index"] in sub_images_by_index:
            main_block["sub_images"] = sub_images_by_index[main_block["index"]]

    for child_block in child_blocks:
        parent_block = find_best_visual_parent(
            child_block,
            main_blocks,
            effective_blocks,
            position_by_index,
        )
        if parent_block is None:
            unmatched_child_blocks.append(child_block)
            continue

        child_kind = child_kind_from_type(child_block["type"])
        grouped_children[parent_block["index"]][f"{child_kind}s"].append(child_block)

    grouped_blocks = {
        BlockType.IMAGE: [],
        BlockType.TABLE: [],
        BlockType.CHART: [],
        BlockType.CODE: [],
    }

    for main_block in main_blocks:
        visual_type = VISUAL_MAIN_TYPES[main_block["type"]]
        mapping = VISUAL_TYPE_MAPPING[visual_type]
        body_block = dict(main_block)
        body_block["type"] = mapping["body"]
        body_block.pop("sub_images", None)

        captions = []
        for caption in sorted(
            grouped_children[main_block["index"]]["captions"],
            key=lambda x: x["index"],
        ):
            child_block = dict(caption)
            child_block["type"] = mapping["caption"]
            captions.append(child_block)

        footnotes = []
        for footnote in sorted(
            grouped_children[main_block["index"]]["footnotes"],
            key=lambda x: x["index"],
        ):
            child_block = dict(footnote)
            child_block["type"] = mapping["footnote"]
            footnotes.append(child_block)

        two_layer_block = {
            "type": visual_type,
            "bbox": body_block["bbox"],
            "blocks": [body_block, *captions, *footnotes],
            "index": body_block["index"],
        }
        if visual_type == BlockType.IMAGE and main_block.get("sub_images"):
            two_layer_block["sub_images"] = main_block["sub_images"]
        two_layer_block["blocks"].sort(key=lambda x: x["index"])

        grouped_blocks[visual_type].append(two_layer_block)

    for blocks_of_type in grouped_blocks.values():
        blocks_of_type.sort(key=lambda x: x["index"])

    return grouped_blocks, unmatched_child_blocks


def absorb_image_block_members(blocks):
    image_block_bodies = [
        block for block in blocks if block["type"] == IMAGE_BLOCK_BODY
    ]
    member_candidates = [
        block
        for block in blocks
        if block["type"] in [BlockType.IMAGE_BODY, BlockType.CHART_BODY]
    ]

    assignments = {}
    for member in member_candidates:
        best_key = None
        best_parent_index = None
        for image_block in image_block_bodies:
            overlap_ratio = calculate_overlap_area_in_bbox1_area_ratio(
                member["bbox"],
                image_block["bbox"],
            )
            if overlap_ratio < 0.9:
                continue

            candidate_key = (
                -overlap_ratio,
                bbox_area(image_block["bbox"]),
                image_block["index"],
            )
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_parent_index = image_block["index"]

        if best_parent_index is not None:
            assignments[member["index"]] = best_parent_index

    absorbed_member_indices = set()
    sub_images_by_index = {}
    for image_block in image_block_bodies:
        members = [
            member
            for member in member_candidates
            if assignments.get(member["index"]) == image_block["index"]
        ]
        if not members:
            continue

        members.sort(key=lambda x: x["index"])
        absorbed_member_indices.update(member["index"] for member in members)
        sub_images_by_index[image_block["index"]] = [
            {
                "type": child_visual_type(member["type"]),
                "bbox": relative_bbox(member["bbox"], image_block["bbox"]),
            }
            for member in members
        ]

    return absorbed_member_indices, sub_images_by_index


def find_best_visual_parent(child_block, main_blocks, ordered_blocks, position_by_index):
    child_type = child_block["type"]
    best_parent = None
    best_key = None

    for main_block in main_blocks:
        if not is_visual_neighbor(
            child_block,
            main_block,
            ordered_blocks,
            position_by_index,
        ):
            continue

        candidate_key = (
            bbox_distance(child_block["bbox"], main_block["bbox"]),
            abs(child_block["index"] - main_block["index"]),
            bbox_center_distance(child_block["bbox"], main_block["bbox"]),
            main_block["index"],
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_parent = main_block

    return best_parent


def is_visual_neighbor(child_block, main_block, ordered_blocks, position_by_index):
    child_type = child_block["type"]
    if child_type == BlockType.FOOTNOTE and child_block["index"] < main_block["index"]:
        return False

    if child_type == BlockType.CAPTION:
        allowed_between_types = {BlockType.CAPTION}
    else:
        allowed_between_types = set(GENERIC_CHILD_TYPES)

    child_pos = position_by_index[child_block["index"]]
    main_pos = position_by_index[main_block["index"]]
    start_pos = min(child_pos, main_pos) + 1
    end_pos = max(child_pos, main_pos)

    for pos in range(start_pos, end_pos):
        between_block = ordered_blocks[pos]
        if between_block["type"] not in allowed_between_types:
            return False

    return True


def child_kind_from_type(block_type):
    if block_type == BlockType.CAPTION:
        return "caption"
    return "footnote"


def child_visual_type(block_type):
    if block_type == BlockType.CHART_BODY:
        return BlockType.CHART
    return BlockType.IMAGE


def bbox_area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def relative_bbox(child_bbox, parent_bbox):
    parent_x0, parent_y0, parent_x1, parent_y1 = parent_bbox
    parent_w = max(parent_x1 - parent_x0, 1)
    parent_h = max(parent_y1 - parent_y0, 1)
    relative = [
        clamp_and_round((child_bbox[0] - parent_x0) / parent_w),
        clamp_and_round((child_bbox[1] - parent_y0) / parent_h),
        clamp_and_round((child_bbox[2] - parent_x0) / parent_w),
        clamp_and_round((child_bbox[3] - parent_y0) / parent_h),
    ]
    return relative


def clamp_and_round(value):
    return round(min(max(value, 0.0), 1.0), 3)


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
