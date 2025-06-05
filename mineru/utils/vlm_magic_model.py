import re
from typing import Literal

from .boxbase import bbox_distance, is_in
from .enum_class import BlockType
from ..api.vlm_middle_json_mkcontent import merge_para_with_text


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