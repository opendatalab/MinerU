import re
from typing import Literal

from loguru import logger

from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.guess_suffix_or_lang import guess_language_by_text
from mineru.utils.magic_model_utils import reduct_overlap, tie_up_category_by_distance_v3


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
                block_content = block_info["content"]
                block_angle = block_info["angle"]

                # print(f"坐标: {block_bbox}")
                # print(f"类型: {block_type}")
                # print(f"内容: {block_content}")
                # print("-" * 50)
            except Exception as e:
                # 如果解析失败，可能是因为格式不正确，跳过这个块
                logger.warning(f"Invalid block format: {block_info}, error: {e}")
                continue

            span_type = "unknown"
            line_type = None
            guess_lang = None

            if block_type in [
                "text",
                "title",
                "image_caption",
                "image_footnote",
                "table_caption",
                "table_footnote",
                "code_caption",
                "ref_text",
                "phonetic",
                "header",
                "footer",
                "page_number",
                "aside_text",
                "page_footnote",
                "list"
            ]:
                span_type = ContentType.TEXT
            elif block_type in ["image"]:
                block_type = BlockType.IMAGE_BODY
                span_type = ContentType.IMAGE
            elif block_type in ["table"]:
                block_type = BlockType.TABLE_BODY
                span_type = ContentType.TABLE
            elif block_type in ["code", "algorithm"]:
                block_content = code_content_clean(block_content)
                line_type = block_type
                block_type = BlockType.CODE_BODY
                span_type = ContentType.TEXT
                guess_lang = guess_language_by_text(block_content)
            elif block_type in ["equation"]:
                block_type = BlockType.INTERLINE_EQUATION
                span_type = ContentType.INTERLINE_EQUATION

            if span_type in ["image", "table"]:
                span = {
                    "bbox": block_bbox,
                    "type": span_type,
                }
                if span_type == ContentType.TABLE:
                    span["html"] = block_content
            elif span_type in [ContentType.INTERLINE_EQUATION]:
                span = {
                    "bbox": block_bbox,
                    "type": span_type,
                    "content": isolated_formula_clean(block_content),
                }
            else:

                if block_content:
                    block_content = clean_content(block_content)

                if block_content and block_content.count("\\(") == block_content.count("\\)") and block_content.count("\\(") > 0:

                    # 生成包含文本和公式的span列表
                    spans = []
                    last_end = 0

                    # 查找所有公式
                    for match in re.finditer(r'\\\((.+?)\\\)', block_content):
                        start, end = match.span()

                        # 添加公式前的文本
                        if start > last_end:
                            text_before = block_content[last_end:start]
                            if text_before.strip():
                                spans.append({
                                    "bbox": block_bbox,
                                    "type": ContentType.TEXT,
                                    "content": text_before
                                })

                        # 添加公式（去除\(和\)）
                        formula = match.group(1)
                        spans.append({
                            "bbox": block_bbox,
                            "type": ContentType.INLINE_EQUATION,
                            "content": formula.strip()
                        })

                        last_end = end

                    # 添加最后一个公式后的文本
                    if last_end < len(block_content):
                        text_after = block_content[last_end:]
                        if text_after.strip():
                            spans.append({
                                "bbox": block_bbox,
                                "type": ContentType.TEXT,
                                "content": text_after
                            })

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
                raise ValueError(f"Invalid span type: {span_type}, expected dict or list, got {type(span)}")

            # 构造line对象
            if block_type in [BlockType.CODE_BODY]:
                line = {"bbox": block_bbox, "spans": spans, "extra": {"type": line_type, "guess_lang": guess_lang}}
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
        self.interline_equation_blocks = []
        self.text_blocks = []
        self.title_blocks = []
        self.code_blocks = []
        self.discarded_blocks = []
        self.ref_text_blocks = []
        self.phonetic_blocks = []
        self.list_blocks = []
        for block in blocks:
            if block["type"] in [BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]:
                self.image_blocks.append(block)
            elif block["type"] in [BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]:
                self.table_blocks.append(block)
            elif block["type"] in [BlockType.CODE_BODY, BlockType.CODE_CAPTION]:
                self.code_blocks.append(block)
            elif block["type"] == BlockType.INTERLINE_EQUATION:
                self.interline_equation_blocks.append(block)
            elif block["type"] == BlockType.TEXT:
                self.text_blocks.append(block)
            elif block["type"] == BlockType.TITLE:
                self.title_blocks.append(block)
            elif block["type"] in [BlockType.REF_TEXT]:
                self.ref_text_blocks.append(block)
            elif block["type"] in [BlockType.PHONETIC]:
                self.phonetic_blocks.append(block)
            elif block["type"] in [BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER, BlockType.ASIDE_TEXT, BlockType.PAGE_FOOTNOTE]:
                self.discarded_blocks.append(block)
            elif block["type"] == BlockType.LIST:
                self.list_blocks.append(block)
            else:
                continue

        self.list_blocks, self.text_blocks, self.ref_text_blocks = fix_list_blocks(self.list_blocks, self.text_blocks, self.ref_text_blocks)
        self.image_blocks, not_include_image_blocks = fix_two_layer_blocks(self.image_blocks, BlockType.IMAGE)
        self.table_blocks, not_include_table_blocks = fix_two_layer_blocks(self.table_blocks, BlockType.TABLE)
        self.code_blocks, not_include_code_blocks = fix_two_layer_blocks(self.code_blocks, BlockType.CODE)
        for code_block in self.code_blocks:
            for block in code_block['blocks']:
                if block['type'] == BlockType.CODE_BODY:
                    if len(block["lines"]) > 0:
                        line = block["lines"][0]
                        code_block["sub_type"] = line["extra"]["type"]
                        if code_block["sub_type"] in ["code"]:
                            code_block["guess_lang"] = line["extra"]["guess_lang"]
                        del line["extra"]
                    else:
                        code_block["sub_type"] = "code"
                        code_block["guess_lang"] = "txt"

        for block in not_include_image_blocks + not_include_table_blocks + not_include_code_blocks:
            block["type"] = BlockType.TEXT
            self.text_blocks.append(block)


    def get_list_blocks(self):
        return self.list_blocks

    def get_image_blocks(self):
        return self.image_blocks

    def get_table_blocks(self):
        return self.table_blocks

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
    if latex.startswith("\\["): latex = latex[2:]
    if latex.endswith("\\]"): latex = latex[:-2]
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
        pattern = r'\\\[(.*?)\\\]'
        content = re.sub(pattern, replace_pattern, content)

    return content


def __tie_up_category_by_distance_v3(blocks, subject_block_type, object_block_type):
    # 定义获取主体和客体对象的函数
    def get_subjects():
        return reduct_overlap(
            list(
                map(
                    lambda x: {"bbox": x["bbox"], "lines": x["lines"], "index": x["index"], "angle":x["angle"]},
                    filter(
                        lambda x: x["type"] == subject_block_type,
                        blocks,
                    ),
                )
            )
        )

    def get_objects():
        return reduct_overlap(
            list(
                map(
                    lambda x: {"bbox": x["bbox"], "lines": x["lines"], "index": x["index"], "angle":x["angle"]},
                    filter(
                        lambda x: x["type"] == object_block_type,
                        blocks,
                    ),
                )
            )
        )

    # 调用通用方法
    return tie_up_category_by_distance_v3(
        get_subjects,
        get_objects
    )


def get_type_blocks(blocks, block_type: Literal["image", "table", "code"]):
    with_captions = __tie_up_category_by_distance_v3(blocks, f"{block_type}_body", f"{block_type}_caption")
    with_footnotes = __tie_up_category_by_distance_v3(blocks, f"{block_type}_body", f"{block_type}_footnote")
    ret = []
    for v in with_captions:
        record = {
            f"{block_type}_body": v["sub_bbox"],
            f"{block_type}_caption_list": v["obj_bboxes"],
        }
        filter_idx = v["sub_idx"]
        d = next(filter(lambda x: x["sub_idx"] == filter_idx, with_footnotes))
        record[f"{block_type}_footnote_list"] = d["obj_bboxes"]
        ret.append(record)
    return ret


def fix_two_layer_blocks(blocks, fix_type: Literal["image", "table", "code"]):
    need_fix_blocks = get_type_blocks(blocks, fix_type)
    fixed_blocks = []
    not_include_blocks = []
    processed_indices = set()

    # 处理需要组织成two_layer结构的blocks
    for block in need_fix_blocks:
        body = block[f"{fix_type}_body"]
        caption_list = block[f"{fix_type}_caption_list"]
        footnote_list = block[f"{fix_type}_footnote_list"]

        body["type"] = f"{fix_type}_body"
        for caption in caption_list:
            caption["type"] = f"{fix_type}_caption"
            processed_indices.add(caption["index"])
        for footnote in footnote_list:
            footnote["type"] = f"{fix_type}_footnote"
            processed_indices.add(footnote["index"])

        processed_indices.add(body["index"])

        two_layer_block = {
            "type": fix_type,
            "bbox": body["bbox"],
            "blocks": [
                body,
            ],
            "index": body["index"],
        }
        two_layer_block["blocks"].extend([*caption_list, *footnote_list])

        fixed_blocks.append(two_layer_block)

    # 添加未处理的blocks
    for block in blocks:
        if block["index"] not in processed_indices:
            # 直接添加未处理的block
            not_include_blocks.append(block)

    return fixed_blocks, not_include_blocks


def fix_list_blocks(list_blocks, text_blocks, ref_text_blocks):
    for list_block in list_blocks:
        list_block["blocks"] = []
        if "lines" in list_block:
            del list_block["lines"]

    temp_text_blocks = text_blocks + ref_text_blocks
    need_remove_blocks = []
    for block in temp_text_blocks:
        for list_block in list_blocks:
            if calculate_overlap_area_in_bbox1_area_ratio(block["bbox"], list_block["bbox"]) >= 0.8:
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
        line_content = []
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