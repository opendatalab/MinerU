# Copyright (c) Opendatalab. All rights reserved.
import re

from mineru.utils.boxbase import (
    bbox_center_distance,
    bbox_distance,
    calculate_overlap_area_in_bbox1_area_ratio,
)
from mineru.utils.enum_class import BlockType


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

    if lines and lines[0].startswith("```"):
        start_idx = 1

    if lines and end_idx > start_idx and lines[end_idx - 1].strip() == "```":
        end_idx -= 1

    if start_idx < end_idx:
        return "\n".join(lines[start_idx:end_idx]).strip()
    return ""


def clean_content(content):
    if content and content.count("\\[") == content.count("\\]") and content.count("\\[") > 0:
        def replace_pattern(match):
            inner_content = match.group(1)
            return f"[{inner_content}]"

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
        body_block.pop("sub_type", None)

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
        if visual_type in [BlockType.IMAGE, BlockType.CHART] and main_block.get("sub_type"):
            two_layer_block["sub_type"] = main_block["sub_type"]
        if visual_type == BlockType.IMAGE and main_block.get("sub_images"):
            two_layer_block["sub_images"] = main_block["sub_images"]
        if visual_type == BlockType.TABLE and main_block.get("cell_merge"):
            two_layer_block["cell_merge"] = main_block["cell_merge"]
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
    return [
        clamp_and_round((child_bbox[0] - parent_x0) / parent_w),
        clamp_and_round((child_bbox[1] - parent_y0) / parent_h),
        clamp_and_round((child_bbox[2] - parent_x0) / parent_w),
        clamp_and_round((child_bbox[3] - parent_y0) / parent_h),
    ]


def clamp_and_round(value):
    return round(min(max(value, 0.0), 1.0), 3)
