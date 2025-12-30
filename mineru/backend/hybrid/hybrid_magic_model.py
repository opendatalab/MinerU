import re
from typing import Literal

from loguru import logger

from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
from mineru.utils.enum_class import ContentType, BlockType, NotExtractType
from mineru.utils.guess_suffix_or_lang import guess_language_by_text
from mineru.utils.magic_model_utils import reduct_overlap, tie_up_category_by_index
from mineru.utils.span_block_fix import fix_text_block
from mineru.utils.span_pre_proc import txt_spans_extract

not_extract_list = [item.value for item in NotExtractType]

class MagicModel:
    def __init__(self,
        page_blocks: list,
        page_inline_formula,
        page_ocr_res,
        page,
        scale,
        page_pil_img,
        width,
        height,
        _ocr_enable,
        _vlm_ocr_enable,
    ):
        self.page_blocks = page_blocks
        self.page_inline_formula = page_inline_formula
        self.page_ocr_res = page_ocr_res

        self.width = width
        self.height = height

        blocks = []
        self.all_spans = []

        page_text_inline_formula_spans = []
        if not _vlm_ocr_enable:
            for inline_formula in page_inline_formula:
                inline_formula["bbox"] = self.cal_real_bbox(inline_formula["bbox"])
                inline_formula_latex = inline_formula.pop("latex", "")
                if inline_formula_latex:
                    page_text_inline_formula_spans.append({
                        "bbox": inline_formula["bbox"],
                        "type": ContentType.INLINE_EQUATION,
                        "content": inline_formula_latex,
                        "score": inline_formula["score"],
                    })
            for ocr_res in page_ocr_res:
                ocr_res["bbox"] = self.cal_real_bbox(ocr_res["bbox"])
                if ocr_res['category_id'] == 15:
                    page_text_inline_formula_spans.append({
                        "bbox": ocr_res["bbox"],
                        "type": ContentType.TEXT,
                        "content": ocr_res["text"],
                        "score": ocr_res["score"],
                    })
            if not _ocr_enable:
                virtual_block = [0, 0, width, height, None, None, None, "text"]
                page_text_inline_formula_spans = txt_spans_extract(page, page_text_inline_formula_spans, page_pil_img, scale, [virtual_block],[])

        # 解析每个块
        for index, block_info in enumerate(page_blocks):
            try:
                block_bbox = self.cal_real_bbox(block_info["bbox"])
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
            code_block_sub_type = None
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
                code_block_sub_type = block_type
                block_type = BlockType.CODE_BODY
                span_type = ContentType.TEXT
                guess_lang = guess_language_by_text(block_content)
            elif block_type in ["equation"]:
                block_type = BlockType.INTERLINE_EQUATION
                span_type = ContentType.INTERLINE_EQUATION

            #  code 和 algorithm 类型的块，如果内容中包含行内公式，则需要将块类型切换为algorithm
            switch_code_to_algorithm = False

            span = None
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
            elif _vlm_ocr_enable or block_type not in not_extract_list:
                #  vlm_ocr_enable模式下，所有文本块都直接使用block的内容
                #  非vlm_ocr_enable模式下，非提取块需要使用span填充方式
                if block_content:
                    block_content = clean_content(block_content)

                if block_content and block_content.count("\\(") == block_content.count("\\)") and block_content.count("\\(") > 0:

                    switch_code_to_algorithm = True

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

            if (
                    span_type in ["image", "table", ContentType.INTERLINE_EQUATION]
                    or (_vlm_ocr_enable or block_type not in not_extract_list)
            ):
                if span is None:
                    continue
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
                    if switch_code_to_algorithm and code_block_sub_type == "code":
                        code_block_sub_type = "algorithm"
                    line = {"bbox": block_bbox, "spans": spans,
                            "extra": {"type": code_block_sub_type, "guess_lang": guess_lang}}
                else:
                    line = {"bbox": block_bbox, "spans": spans}

                block = {
                    "bbox": block_bbox,
                    "type": block_type,
                    "angle": block_angle,
                    "lines": [line],
                    "index": index,
                }

            else:  #  使用span填充方式
                block_spans = []
                for span in page_text_inline_formula_spans:
                    if calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block_bbox) > 0.5:
                        block_spans.append(span)
                # 从spans删除已经放入block_spans中的span
                if len(block_spans) > 0:
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

            blocks.append(block)

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
        bbox = (x_1, y_1, x_2, y_2)
        return bbox

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


def __tie_up_category_by_index(blocks, subject_block_type, object_block_type):
    """基于index的主客体关联包装函数"""
    # 定义获取主体和客体对象的函数
    def get_subjects():
        return reduct_overlap(
            list(
                map(
                    lambda x: {"bbox": x["bbox"], "lines": x["lines"], "index": x["index"], "angle": x["angle"]},
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
                    lambda x: {"bbox": x["bbox"], "lines": x["lines"], "index": x["index"], "angle": x["angle"]},
                    filter(
                        lambda x: x["type"] == object_block_type,
                        blocks,
                    ),
                )
            )
        )

    # 调用通用方法
    return tie_up_category_by_index(
        get_subjects,
        get_objects
    )


def get_type_blocks_by_index(blocks, block_type: Literal["image", "table", "code"]):
    """使用基于index的匹配策略来组织blocks"""
    with_captions = __tie_up_category_by_index(blocks, f"{block_type}_body", f"{block_type}_caption")
    with_footnotes = __tie_up_category_by_index(blocks, f"{block_type}_body", f"{block_type}_footnote")
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
    need_fix_blocks = get_type_blocks_by_index(blocks, fix_type)
    fixed_blocks = []
    not_include_blocks = []
    processed_indices = set()

    # 特殊处理表格类型，确保标题在表格前，注脚在表格后
    if fix_type in ["table", "image"]:
        # 收集所有不合适的caption和footnote
        misplaced_footnotes = []  # 存储(footnote, 原始block索引)

        # 第一步：移除不符合位置要求的footnote
        for block_idx, block in enumerate(need_fix_blocks):
            body = block[f"{fix_type}_body"]
            body_index = body["index"]

            # 检查footnote应在body后或同位置
            valid_footnotes = []
            for footnote in block[f"{fix_type}_footnote_list"]:
                if footnote["index"] >= body_index:
                    valid_footnotes.append(footnote)
                else:
                    misplaced_footnotes.append((footnote, block_idx))
            block[f"{fix_type}_footnote_list"] = valid_footnotes

        # 第三步：重新分配不合规的footnote到合适的body
        for footnote, original_block_idx in misplaced_footnotes:
            footnote_index = footnote["index"]
            best_block_idx = None
            min_distance = float('inf')

            # 寻找索引小于等于footnote_index的最近body
            for idx, block in enumerate(need_fix_blocks):
                body_index = block[f"{fix_type}_body"]["index"]
                if body_index <= footnote_index and idx != original_block_idx:
                    distance = footnote_index - body_index
                    if distance < min_distance:
                        min_distance = distance
                        best_block_idx = idx

            if best_block_idx is not None:
                # 找到合适的body，添加到对应block的footnote_list
                need_fix_blocks[best_block_idx][f"{fix_type}_footnote_list"].append(footnote)
            else:
                # 没找到合适的body，作为普通block处理
                not_include_blocks.append(footnote)

        # 第四步:将每个block的caption_list和footnote_list中不连续index的元素提出来作为普通block处理
        for block in need_fix_blocks:
            caption_list = block[f"{fix_type}_caption_list"]
            footnote_list = block[f"{fix_type}_footnote_list"]
            body_index = block[f"{fix_type}_body"]["index"]

            # 处理caption_list (从body往前看,caption在body之前)
            if caption_list:
                # 按index降序排列,从最接近body的开始检查
                caption_list.sort(key=lambda x: x["index"], reverse=True)
                filtered_captions = [caption_list[0]]
                for i in range(1, len(caption_list)):
                    prev_index = caption_list[i - 1]["index"]
                    curr_index = caption_list[i]["index"]

                    # 检查是否连续
                    if curr_index == prev_index - 1:
                        filtered_captions.append(caption_list[i])
                    else:
                        # 检查gap中是否只有body_index
                        gap_indices = set(range(curr_index + 1, prev_index))
                        if gap_indices == {body_index}:
                            # gap中只有body_index,不算真正的gap
                            filtered_captions.append(caption_list[i])
                        else:
                            # 出现真正的gap,后续所有caption都作为普通block
                            not_include_blocks.extend(caption_list[i:])
                            break
                # 恢复升序
                filtered_captions.reverse()
                block[f"{fix_type}_caption_list"] = filtered_captions

            # 处理footnote_list (从body往后看,footnote在body之后)
            if footnote_list:
                # 按index升序排列,从最接近body的开始检查
                footnote_list.sort(key=lambda x: x["index"])
                filtered_footnotes = [footnote_list[0]]
                for i in range(1, len(footnote_list)):
                    # 检查是否与前一个footnote连续
                    if footnote_list[i]["index"] == footnote_list[i - 1]["index"] + 1:
                        filtered_footnotes.append(footnote_list[i])
                    else:
                        # 出现gap,后续所有footnote都作为普通block
                        not_include_blocks.extend(footnote_list[i:])
                        break
                block[f"{fix_type}_footnote_list"] = filtered_footnotes

    # 构建两层结构blocks
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
            "blocks": [body],
            "index": body["index"],
        }
        two_layer_block["blocks"].extend([*caption_list, *footnote_list])
        # 对blocks按index排序
        two_layer_block["blocks"].sort(key=lambda x: x["index"])

        fixed_blocks.append(two_layer_block)

    # 添加未处理的blocks
    for block in blocks:
        block.pop("type", None)
        if block["index"] not in processed_indices and block not in not_include_blocks:
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