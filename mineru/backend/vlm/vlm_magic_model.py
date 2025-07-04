import re
from typing import Literal

from loguru import logger

from mineru.utils.boxbase import bbox_distance, is_in
from mineru.utils.enum_class import ContentType, BlockType, SplitFlag
from mineru.backend.vlm.vlm_middle_json_mkcontent import merge_para_with_text
from mineru.utils.format_utils import convert_otsl_to_html


class MagicModel:
    def __init__(self, token: str, width, height):
        self.token = token

        # 使用正则表达式查找所有块
        pattern = (
            r"<\|box_start\|>(.*?)<\|box_end\|><\|ref_start\|>(.*?)<\|ref_end\|><\|md_start\|>(.*?)(?:<\|md_end\|>|<\|im_end\|>)"
        )
        block_infos = re.findall(pattern, token, re.DOTALL)

        blocks = []
        self.all_spans = []
        # 解析每个块
        for index, block_info in enumerate(block_infos):
            block_bbox = block_info[0].strip()
            try:
                x1, y1, x2, y2 = map(int, block_bbox.split())
                x_1, y_1, x_2, y_2 = (
                    int(x1 * width / 1000),
                    int(y1 * height / 1000),
                    int(x2 * width / 1000),
                    int(y2 * height / 1000),
                )
                if x_2 < x_1:
                    x_1, x_2 = x_2, x_1
                if y_2 < y_1:
                    y_1, y_2 = y_2, y_1
                block_bbox = (x_1, y_1, x_2, y_2)
                block_type = block_info[1].strip()
                block_content = block_info[2].strip()

                # print(f"坐标: {block_bbox}")
                # print(f"类型: {block_type}")
                # print(f"内容: {block_content}")
                # print("-" * 50)
            except Exception as e:
                # 如果解析失败，可能是因为格式不正确，跳过这个块
                logger.warning(f"Invalid block format: {block_info}, error: {e}")
                continue

            span_type = "unknown"
            if block_type in [
                "text",
                "title",
                "image_caption",
                "image_footnote",
                "table_caption",
                "table_footnote",
                "list",
                "index",
            ]:
                span_type = ContentType.TEXT
            elif block_type in ["image"]:
                block_type = BlockType.IMAGE_BODY
                span_type = ContentType.IMAGE
            elif block_type in ["table"]:
                block_type = BlockType.TABLE_BODY
                span_type = ContentType.TABLE
            elif block_type in ["equation"]:
                block_type = BlockType.INTERLINE_EQUATION
                span_type = ContentType.INTERLINE_EQUATION

            if span_type in ["image", "table"]:
                span = {
                    "bbox": block_bbox,
                    "type": span_type,
                }
                if span_type == ContentType.TABLE:
                    if "<fcel>" in block_content or "<ecel>" in block_content:
                        lines = block_content.split("\n\n")
                        new_lines = []
                        for line in lines:
                            if "<fcel>" in line or "<ecel>" in line:
                                line = convert_otsl_to_html(line)
                            new_lines.append(line)
                        span["html"] = "\n\n".join(new_lines)
                    else:
                        span["html"] = block_content
            elif span_type in [ContentType.INTERLINE_EQUATION]:
                span = {
                    "bbox": block_bbox,
                    "type": span_type,
                    "content": isolated_formula_clean(block_content),
                }
            else:
                if block_content.count("\\(") == block_content.count("\\)") and block_content.count("\\(") > 0:
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

            if isinstance(span, dict) and "bbox" in span:
                self.all_spans.append(span)
                line = {
                    "bbox": block_bbox,
                    "spans": [span],
                }
            elif isinstance(span, list):
                self.all_spans.extend(span)
                line = {
                    "bbox": block_bbox,
                    "spans": span,
                }
            else:
                raise ValueError(f"Invalid span type: {span_type}, expected dict or list, got {type(span)}")

            blocks.append(
                {
                    "bbox": block_bbox,
                    "type": block_type,
                    "lines": [line],
                    "index": index,
                }
            )

        self.image_blocks = []
        self.table_blocks = []
        self.interline_equation_blocks = []
        self.text_blocks = []
        self.title_blocks = []
        for block in blocks:
            if block["type"] in [BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]:
                self.image_blocks.append(block)
            elif block["type"] in [BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]:
                self.table_blocks.append(block)
            elif block["type"] == BlockType.INTERLINE_EQUATION:
                self.interline_equation_blocks.append(block)
            elif block["type"] == BlockType.TEXT:
                self.text_blocks.append(block)
            elif block["type"] == BlockType.TITLE:
                self.title_blocks.append(block)
            else:
                continue

    def get_image_blocks(self):
        return fix_two_layer_blocks(self.image_blocks, BlockType.IMAGE)

    def get_table_blocks(self):
        return fix_two_layer_blocks(self.table_blocks, BlockType.TABLE)

    def get_title_blocks(self):
        return fix_title_blocks(self.title_blocks)

    def get_text_blocks(self):
        return fix_text_blocks(self.text_blocks)

    def get_interline_equation_blocks(self):
        return self.interline_equation_blocks

    def get_all_spans(self):
        return self.all_spans


def isolated_formula_clean(txt):
    latex = txt[:]
    if latex.startswith("\\["): latex = latex[2:]
    if latex.endswith("\\]"): latex = latex[:-2]
    latex = latex_fix(latex.strip())
    return latex


def latex_fix(latex):
    # valid pairs:
    # \left\{ ... \right\}
    # \left( ... \right)
    # \left| ... \right|
    # \left\| ... \right\|
    # \left[ ... \right]

    LEFT_COUNT_PATTERN = re.compile(r'\\left(?![a-zA-Z])')
    RIGHT_COUNT_PATTERN = re.compile(r'\\right(?![a-zA-Z])')
    left_count = len(LEFT_COUNT_PATTERN.findall(latex))  # 不匹配\lefteqn等
    right_count = len(RIGHT_COUNT_PATTERN.findall(latex))  # 不匹配\rightarrow

    if left_count != right_count:
        for _ in range(2):
            # replace valid pairs
            latex = re.sub(r'\\left\\\{', "{", latex) # \left\{
            latex = re.sub(r"\\left\|", "|", latex) # \left|
            latex = re.sub(r"\\left\\\|", "|", latex) # \left\|
            latex = re.sub(r"\\left\(", "(", latex) # \left(
            latex = re.sub(r"\\left\[", "[", latex) # \left[

            latex = re.sub(r"\\right\\\}", "}", latex) # \right\}
            latex = re.sub(r"\\right\|", "|", latex) # \right|
            latex = re.sub(r"\\right\\\|", "|", latex) # \right\|
            latex = re.sub(r"\\right\)", ")", latex) # \right)
            latex = re.sub(r"\\right\]", "]", latex) # \right]
            latex = re.sub(r"\\right\.", "", latex) # \right.

            # replace invalid pairs first
            latex = re.sub(r'\\left\{', "{", latex)
            latex = re.sub(r'\\right\}', "}", latex) # \left{ ... \right}
            latex = re.sub(r'\\left\\\(', "(", latex)
            latex = re.sub(r'\\right\\\)', ")", latex) # \left\( ... \right\)
            latex = re.sub(r'\\left\\\[', "[", latex)
            latex = re.sub(r'\\right\\\]', "]", latex) # \left\[ ... \right\]

    return latex


def __reduct_overlap(bboxes):
    N = len(bboxes)
    keep = [True] * N
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if is_in(bboxes[i]["bbox"], bboxes[j]["bbox"]):
                keep[i] = False
    return [bboxes[i] for i in range(N) if keep[i]]


def __tie_up_category_by_distance_v3(
    blocks: list,
    subject_block_type: str,
    object_block_type: str,
):
    subjects = __reduct_overlap(
        list(
            map(
                lambda x: {"bbox": x["bbox"], "lines": x["lines"], "index": x["index"]},
                filter(
                    lambda x: x["type"] == subject_block_type,
                    blocks,
                ),
            )
        )
    )
    objects = __reduct_overlap(
        list(
            map(
                lambda x: {"bbox": x["bbox"], "lines": x["lines"], "index": x["index"]},
                filter(
                    lambda x: x["type"] == object_block_type,
                    blocks,
                ),
            )
        )
    )

    ret = []
    N, M = len(subjects), len(objects)
    subjects.sort(key=lambda x: x["bbox"][0] ** 2 + x["bbox"][1] ** 2)
    objects.sort(key=lambda x: x["bbox"][0] ** 2 + x["bbox"][1] ** 2)

    OBJ_IDX_OFFSET = 10000
    SUB_BIT_KIND, OBJ_BIT_KIND = 0, 1

    all_boxes_with_idx = [(i, SUB_BIT_KIND, sub["bbox"][0], sub["bbox"][1]) for i, sub in enumerate(subjects)] + [
        (i + OBJ_IDX_OFFSET, OBJ_BIT_KIND, obj["bbox"][0], obj["bbox"][1]) for i, obj in enumerate(objects)
    ]
    seen_idx = set()
    seen_sub_idx = set()

    while N > len(seen_sub_idx):
        candidates = []
        for idx, kind, x0, y0 in all_boxes_with_idx:
            if idx in seen_idx:
                continue
            candidates.append((idx, kind, x0, y0))

        if len(candidates) == 0:
            break
        left_x = min([v[2] for v in candidates])
        top_y = min([v[3] for v in candidates])

        candidates.sort(key=lambda x: (x[2] - left_x) ** 2 + (x[3] - top_y) ** 2)

        fst_idx, fst_kind, left_x, top_y = candidates[0]
        candidates.sort(key=lambda x: (x[2] - left_x) ** 2 + (x[3] - top_y) ** 2)
        nxt = None

        for i in range(1, len(candidates)):
            if candidates[i][1] ^ fst_kind == 1:
                nxt = candidates[i]
                break
        if nxt is None:
            break

        if fst_kind == SUB_BIT_KIND:
            sub_idx, obj_idx = fst_idx, nxt[0] - OBJ_IDX_OFFSET

        else:
            sub_idx, obj_idx = nxt[0], fst_idx - OBJ_IDX_OFFSET

        pair_dis = bbox_distance(subjects[sub_idx]["bbox"], objects[obj_idx]["bbox"])
        nearest_dis = float("inf")
        for i in range(N):
            if i in seen_idx or i == sub_idx:
                continue
            nearest_dis = min(nearest_dis, bbox_distance(subjects[i]["bbox"], objects[obj_idx]["bbox"]))

        if pair_dis >= 3 * nearest_dis:
            seen_idx.add(sub_idx)
            continue

        seen_idx.add(sub_idx)
        seen_idx.add(obj_idx + OBJ_IDX_OFFSET)
        seen_sub_idx.add(sub_idx)

        ret.append(
            {
                "sub_bbox": {
                    "bbox": subjects[sub_idx]["bbox"],
                    "lines": subjects[sub_idx]["lines"],
                    "index": subjects[sub_idx]["index"],
                },
                "obj_bboxes": [
                    {"bbox": objects[obj_idx]["bbox"], "lines": objects[obj_idx]["lines"], "index": objects[obj_idx]["index"]}
                ],
                "sub_idx": sub_idx,
            }
        )

    for i in range(len(objects)):
        j = i + OBJ_IDX_OFFSET
        if j in seen_idx:
            continue
        seen_idx.add(j)
        nearest_dis, nearest_sub_idx = float("inf"), -1
        for k in range(len(subjects)):
            dis = bbox_distance(objects[i]["bbox"], subjects[k]["bbox"])
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_sub_idx = k

        for k in range(len(subjects)):
            if k != nearest_sub_idx:
                continue
            if k in seen_sub_idx:
                for kk in range(len(ret)):
                    if ret[kk]["sub_idx"] == k:
                        ret[kk]["obj_bboxes"].append(
                            {"bbox": objects[i]["bbox"], "lines": objects[i]["lines"], "index": objects[i]["index"]}
                        )
                        break
            else:
                ret.append(
                    {
                        "sub_bbox": {
                            "bbox": subjects[k]["bbox"],
                            "lines": subjects[k]["lines"],
                            "index": subjects[k]["index"],
                        },
                        "obj_bboxes": [
                            {"bbox": objects[i]["bbox"], "lines": objects[i]["lines"], "index": objects[i]["index"]}
                        ],
                        "sub_idx": k,
                    }
                )
            seen_sub_idx.add(k)
            seen_idx.add(k)

    for i in range(len(subjects)):
        if i in seen_sub_idx:
            continue
        ret.append(
            {
                "sub_bbox": {
                    "bbox": subjects[i]["bbox"],
                    "lines": subjects[i]["lines"],
                    "index": subjects[i]["index"],
                },
                "obj_bboxes": [],
                "sub_idx": i,
            }
        )

    return ret


def get_type_blocks(blocks, block_type: Literal["image", "table"]):
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


def fix_two_layer_blocks(blocks, fix_type: Literal["image", "table"]):
    need_fix_blocks = get_type_blocks(blocks, fix_type)
    fixed_blocks = []
    for block in need_fix_blocks:
        body = block[f"{fix_type}_body"]
        caption_list = block[f"{fix_type}_caption_list"]
        footnote_list = block[f"{fix_type}_footnote_list"]

        body["type"] = f"{fix_type}_body"
        for caption in caption_list:
            caption["type"] = f"{fix_type}_caption"
        for footnote in footnote_list:
            footnote["type"] = f"{fix_type}_footnote"

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

    return fixed_blocks


def fix_title_blocks(blocks):
    for block in blocks:
        if block["type"] == BlockType.TITLE:
            title_content = merge_para_with_text(block)
            title_level = count_leading_hashes(title_content)
            block['level'] = title_level
            for line in block['lines']:
                for span in line['spans']:
                    span['content'] = strip_leading_hashes(span['content'])
                    break
                break
    return blocks


def count_leading_hashes(text):
    match = re.match(r'^(#+)', text)
    return len(match.group(1)) if match else 0


def strip_leading_hashes(text):
    # 去除开头的#和紧随其后的空格
    return re.sub(r'^#+\s*', '', text)


def fix_text_blocks(blocks):
    i = 0
    while i < len(blocks):
        block = blocks[i]
        last_line = block["lines"][-1]if block["lines"] else None
        if last_line:
            last_span = last_line["spans"][-1] if last_line["spans"] else None
            if last_span and last_span['content'].endswith('<|txt_contd|>'):
                last_span['content'] = last_span['content'][:-len('<|txt_contd|>')]

                # 查找下一个未被清空的块
                next_idx = i + 1
                while next_idx < len(blocks) and blocks[next_idx].get(SplitFlag.LINES_DELETED, False):
                    next_idx += 1

                # 如果找到下一个有效块，则合并
                if next_idx < len(blocks):
                    next_block = blocks[next_idx]
                    # 将下一个块的lines扩展到当前块的lines中
                    block["lines"].extend(next_block["lines"])
                    # 清空下一个块的lines
                    next_block["lines"] = []
                    # 在下一个块中添加标志
                    next_block[SplitFlag.LINES_DELETED] = True
                    # 不增加i，继续检查当前块（现在已包含下一个块的内容）
                    continue
        i += 1
    return blocks