# Copyright (c) Opendatalab. All rights reserved.
"""把 OFD 页面场景投影为带 bbox 的有序 raw model-list。"""

from __future__ import annotations

import re
import statistics
from dataclasses import replace
from typing import Any

from .._shared.xycut import sort_entries
from ....types import BlockType
from .geometry import bbox_center, bbox_union, normalize_bbox
from .models import OfdPageScene, TextLine
from .table import OfdTableBudget, OfdTableRegion, recover_tables
from .text import format_line_spans

_PAGE_NUMBER_RE = re.compile(r"^(?:\d+|[IVXLCDM]+)$", re.IGNORECASE)
_ASCII_WORD_GAP_RATIO = 0.2


def _same_baseline(first: TextLine, second: TextLine) -> bool:
    """判断两个文字片段是否可在同一视觉基线上拼接。"""
    if first.angle != second.angle or first.styles != second.styles:
        return False
    if (
        min(first.font_size, second.font_size) <= 0
        or max(first.font_size, second.font_size) / min(first.font_size, second.font_size) > 1.35
    ):
        return False
    first_height = first.bbox[3] - first.bbox[1]
    second_height = second.bbox[3] - second.bbox[1]
    tolerance = 0.4 * max(first_height, second_height, 0.5)
    if second.bbox[0] + tolerance < first.bbox[0]:
        return False
    center_delta = abs(bbox_center(first.bbox)[1] - bbox_center(second.bbox)[1])
    gap = max(0.0, second.bbox[0] - first.bbox[2])
    return center_delta <= tolerance and gap <= 2.0 * max(first_height, second_height, 0.5)


def _join_text(first: TextLine, second: TextLine) -> str:
    """按语言字符边界和实际字形间距决定同行片段间是否补空格。"""
    if not first.text or not second.text:
        return first.text + second.text
    ascii_word_boundary = (
        first.text[-1].isascii() and second.text[0].isascii() and first.text[-1].isalnum() and second.text[0].isalnum()
    )
    if not ascii_word_boundary:
        return first.text + second.text
    first_glyph = first.glyphs[-1] if first.glyphs else None
    second_glyph = second.glyphs[0] if second.glyphs else None
    first_right = first_glyph.bbox[2] if first_glyph is not None else first.bbox[2]
    second_left = second_glyph.bbox[0] if second_glyph is not None else second.bbox[0]
    gap = second_left - first_right
    glyph_widths = [
        glyph.bbox[2] - glyph.bbox[0]
        for glyph in (first_glyph, second_glyph)
        if glyph is not None and glyph.bbox[2] > glyph.bbox[0]
    ]
    reference_width = statistics.median(glyph_widths) if glyph_widths else 0.0
    spacing_threshold = _ASCII_WORD_GAP_RATIO * max(min(first.font_size, second.font_size), reference_width, 0.5)
    needs_space = gap >= spacing_threshold
    return f"{first.text}{' ' if needs_space else ''}{second.text}"


def merge_same_baseline_lines(lines: list[TextLine]) -> list[TextLine]:
    """保守合并相邻 TextObject 形成的同基线文字片段。"""

    def sort_key(item: TextLine) -> tuple[int, int, float, float, int]:
        """按自适应基线带和横向位置排列候选片段。"""
        height = max(item.bbox[3] - item.bbox[1], 0.5)
        baseline_bucket = round(bbox_center(item.bbox)[1] / max(0.4 * height, 0.5))
        return item.angle, baseline_bucket, item.bbox[0], item.bbox[1], item.paint_order

    ordered = sorted(lines, key=sort_key)
    output: list[TextLine] = []
    for line in ordered:
        if output and _same_baseline(output[-1], line):
            previous = output[-1]
            merged_bbox = bbox_union((previous.bbox, line.bbox))
            if merged_bbox is not None:
                output[-1] = replace(
                    previous,
                    text=_join_text(previous, line),
                    bbox=merged_bbox,
                    glyphs=previous.glyphs + line.glyphs,
                    paint_order=min(previous.paint_order, line.paint_order),
                )
                continue
        output.append(line)
    return output


def _normalized_signature(line: TextLine, scene: OfdPageScene) -> tuple[str, int, int]:
    """构造跨页重复边缘文字的稳定签名。"""
    width = scene.physical_box[2] - scene.physical_box[0]
    height = scene.physical_box[3] - scene.physical_box[1]
    center_x, center_y = bbox_center(line.bbox)
    text = re.sub(r"\d+", "#", re.sub(r"\s+", "", line.text)).casefold()
    return (
        text,
        round((center_x - scene.physical_box[0]) / max(width, 1.0) * 20),
        round((center_y - scene.physical_box[1]) / max(height, 1.0) * 20),
    )


def repeated_edge_signatures(scenes: list[OfdPageScene]) -> frozenset[tuple[str, int, int]]:
    """识别至少跨两页重复出现的页边缘文字。"""
    pages_by_signature: dict[tuple[str, int, int], set[int]] = {}
    for scene in scenes:
        height = scene.physical_box[3] - scene.physical_box[1]
        for line in scene.text_lines:
            center_y = bbox_center(line.bbox)[1]
            relative_y = (center_y - scene.physical_box[1]) / max(height, 1.0)
            if 0.18 < relative_y < 0.82:
                continue
            signature = _normalized_signature(line, scene)
            if signature[0]:
                pages_by_signature.setdefault(signature, set()).add(scene.page_idx)
    return frozenset(signature for signature, pages in pages_by_signature.items() if len(pages) >= 2)


def _auxiliary_type(line: TextLine, scene: OfdPageScene, repeated: frozenset[tuple[str, int, int]]) -> BlockType | None:
    """利用 ContentBox、页边缘和跨页重复性识别辅助文字。"""
    physical = scene.physical_box
    width = physical[2] - physical[0]
    height = physical[3] - physical[1]
    center_x, center_y = bbox_center(line.bbox)
    relative_x = (center_x - physical[0]) / max(width, 1.0)
    relative_y = (center_y - physical[1]) / max(height, 1.0)
    normalized_text = line.text.strip()
    if len(normalized_text) <= 6 and _PAGE_NUMBER_RE.fullmatch(normalized_text) and (relative_y <= 0.08 or relative_y >= 0.92):
        return BlockType.PAGE_NUMBER
    outside_content = False
    if scene.content_box is not None:
        content = scene.content_box
        outside_content = not (content[0] <= center_x <= content[2] and content[1] <= center_y <= content[3])
    is_repeated = _normalized_signature(line, scene) in repeated
    if (outside_content and relative_y <= 0.2) or (is_repeated and relative_y <= 0.08):
        return BlockType.HEADER
    if (outside_content and relative_y >= 0.8) or (is_repeated and relative_y >= 0.92):
        return BlockType.FOOTER
    if outside_content and (relative_x <= 0.12 or relative_x >= 0.88):
        return BlockType.ASIDE_TEXT
    return None


def _upright_bbox(bbox: list[float], scene: OfdPageScene, angle: int) -> list[float]:
    """把直角旋转对象的 bbox 转到共享 XYCut 使用的正向坐标。"""
    physical = scene.physical_box
    width = physical[2] - physical[0]
    height = physical[3] - physical[1]
    x0, y0, x1, y1 = (
        bbox[0] - physical[0],
        bbox[1] - physical[1],
        bbox[2] - physical[0],
        bbox[3] - physical[1],
    )
    if angle == 90:
        return [y0, width - x1, y1, width - x0]
    if angle == 180:
        return [width - x1, height - y1, width - x0, height - y0]
    if angle == 270:
        return [height - y1, x0, height - y0, x1]
    return [x0, y0, x1, y1]


class OfdReadingOrderProjector:
    """执行表格认领、辅助类型判定和 OFD-aware XYCut++。"""

    def __init__(self, scenes: list[OfdPageScene]) -> None:
        """缓存跨页重复签名和正文字号基线。"""
        self.scenes = scenes
        self.repeated = repeated_edge_signatures(scenes)
        body_sizes = [line.font_size for scene in scenes for line in scene.text_lines if line.font_size > 0]
        self.body_size = statistics.median(body_sizes) if body_sizes else 1.0
        self.document_title_emitted = False
        self.table_budget = OfdTableBudget()

    def _text_block(self, line: TextLine, scene: OfdPageScene) -> dict[str, Any]:
        """把 TextLine 转换为尚未归一化的 raw block。"""
        auxiliary = _auxiliary_type(line, scene, self.repeated)
        block_type: BlockType = auxiliary or BlockType.TEXT
        level: int | None = None
        relative_top = (line.bbox[1] - scene.physical_box[1]) / max(scene.physical_box[3] - scene.physical_box[1], 1.0)
        if auxiliary is None and line.font_size >= 1.8 * self.body_size and relative_top <= 0.35:
            if not self.document_title_emitted:
                block_type = BlockType.DOC_TITLE
                level = 1
                self.document_title_emitted = True
            else:
                block_type = BlockType.PARAGRAPH_TITLE
                level = 2
        elif auxiliary is None and line.font_size >= 1.45 * self.body_size:
            block_type = BlockType.PARAGRAPH_TITLE
            level = 2
        block: dict[str, Any] = {
            "type": block_type,
            "content": format_line_spans(line.text, line.styles),
            "bbox_mm": line.bbox,
            "angle": line.angle,
            "paint_order": line.paint_order,
            "line_bboxes_mm": [line.bbox],
        }
        if level is not None:
            block["level"] = level
        return block

    def _table_block(self, table: OfdTableRegion) -> dict[str, Any]:
        """把表格区域转换为尚未归一化的 raw block。"""
        return {
            "type": BlockType.TABLE,
            "content": table.html,
            "bbox_mm": table.bbox,
            "angle": 0,
            "paint_order": table.paint_order,
        }

    def _image_block(self, image: object) -> dict[str, Any]:
        """把 ImageItem 转换为 raw block或内部占位原子。"""
        payload = getattr(image, "image_base64", None)
        block: dict[str, Any] = {
            "type": BlockType.IMAGE,
            "content": "",
            "bbox_mm": getattr(image, "bbox"),
            "angle": 0,
            "paint_order": getattr(image, "paint_order"),
            "drop_after_sort": payload is None,
        }
        if payload is not None:
            block["image_base64"] = payload
        return block

    def project_page(self, scene: OfdPageScene) -> list[dict[str, Any]]:
        """把一页场景投影为最终阅读顺序 raw model-list。"""
        merged_lines = merge_same_baseline_lines(scene.text_lines)
        tables = recover_tables(scene.axis_lines, merged_lines, self.table_budget)
        consumed = {line_id for table in tables for line_id in table.consumed_line_ids}
        blocks = [self._text_block(line, scene) for line in merged_lines if id(line) not in consumed]
        blocks.extend(self._table_block(table) for table in tables)
        blocks.extend(self._image_block(image) for image in scene.images)
        page_center_y = bbox_center(scene.physical_box)[1]
        top = [
            block
            for block in blocks
            if block["type"] == BlockType.HEADER
            or (block["type"] == BlockType.PAGE_NUMBER and bbox_center(block["bbox_mm"])[1] < page_center_y)
        ]
        top_ids = {id(block) for block in top}
        bottom = [
            block
            for block in blocks
            if block["type"] == BlockType.FOOTER or (block["type"] == BlockType.PAGE_NUMBER and id(block) not in top_ids)
        ]
        bottom_ids = {id(block) for block in bottom}
        body = [block for block in blocks if id(block) not in top_ids and id(block) not in bottom_ids]
        sortable: list[dict[str, Any]] = []
        for block in body:
            upright = _upright_bbox(list(block["bbox_mm"]), scene, int(block.get("angle", 0) or 0) % 360)
            sortable.append({"bbox": [value * 72.0 / 25.4 for value in upright], "payload": block})
        ordered_body = [entry["payload"] for entry in sort_entries(sortable)]
        ordered = sorted(top, key=lambda block: (block["bbox_mm"][1], block["bbox_mm"][0], block["paint_order"]))
        ordered.extend(ordered_body)
        ordered.extend(sorted(bottom, key=lambda block: (block["bbox_mm"][1], block["bbox_mm"][0], block["paint_order"])))
        output: list[dict[str, Any]] = []
        for block in ordered:
            if block.get("drop_after_sort"):
                continue
            normalized = normalize_bbox(block["bbox_mm"], scene.physical_box)
            if normalized is None:
                continue
            result: dict[str, Any] = {
                "type": block["type"],
                "content": block.get("content", ""),
                "bbox": normalized,
                "angle": block.get("angle", 0),
            }
            if "level" in block:
                result["level"] = block["level"]
            if "image_base64" in block:
                result["image_base64"] = block["image_base64"]
            line_bboxes = [normalize_bbox(item, scene.physical_box) for item in block.get("line_bboxes_mm", [])]
            valid_line_bboxes = [item for item in line_bboxes if item is not None]
            if valid_line_bboxes:
                result["lines"] = [{"bbox": item} for item in valid_line_bboxes]
            output.append(result)
        return output

    def project(self) -> list[list[dict[str, Any]]]:
        """投影全部页面并保持原始页树顺序。"""
        return [self.project_page(scene) for scene in self.scenes]


__all__ = ["OfdReadingOrderProjector", "merge_same_baseline_lines", "repeated_edge_signatures"]
