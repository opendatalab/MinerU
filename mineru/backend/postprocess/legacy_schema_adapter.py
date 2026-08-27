# Copyright (c) Opendatalab. All rights reserved.
"""schema 1.0 preproc_blocks → raw model_list 适配器。

pr-5415 重构后 PageInfo 收紧为 (page_idx, blocks)，但旧 schema 1.0 的 page 含
preproc_blocks/para_blocks/discarded_blocks（像素坐标 + lines/spans 嵌套）。
本模块把旧 page dict 回推成新代码 process_page_blocks 期望的 raw model_list
（归一化坐标 + 扁平 content），再由 model_list_to_pages 重走完整后处理。
"""

from __future__ import annotations

from typing import Any

# 顶层视觉块类型：含嵌套子块（body + caption + footnote），回推时展平子块。
_VISUAL_PARENT_TYPES: frozenset[str] = frozenset(
    {
        "image",
        "image_block",
        "table",
        "chart",
        "code",
        "algorithm",
    }
)

# flash 旧数据无版面坐标，bbox=[0,0,0,0]，用最小合法占位满足新 schema（x1>x0, y1>y0）
_PLACEHOLDER_BBOX: tuple[float, float, float, float] = (0.0, 0.0, 0.001, 0.001)


def legacy_page_to_model_list(page: dict[str, Any]) -> list[dict[str, Any]]:
    """把 1.0 schema 的 page dict 转换为 raw model_list。

    preproc_blocks 优先（更接近 raw model_list）；flash 旧数据只有 para_blocks 时回退。
    discarded_blocks 也参与回推（header/footer/page_number 等辅助块）。
    - 顶层视觉块（image/table/chart）展平子块为独立 block
    - 文本块从 lines[].spans[].content 提取 content
    - bbox 像素坐标 → 归一化坐标（用 page_size）；零面积 bbox 设为 None
    - image_path 从 span 提到 block 级
    """
    page_size = page.get("page_size")
    width, height = (float(page_size[0]), float(page_size[1])) if page_size and len(page_size) >= 2 else (0.0, 0.0)

    source_blocks = page.get("preproc_blocks")
    if source_blocks is None:
        # flash 旧数据只有 para_blocks（无 preproc_blocks），回退使用
        source_blocks = page.get("para_blocks", [])

    model_list: list[dict[str, Any]] = []
    for block in source_blocks:
        _collect_blocks(block, width, height, model_list)
    for block in page.get("discarded_blocks", []):
        _collect_blocks(block, width, height, model_list)
    return model_list


def _collect_blocks(block: dict[str, Any], width: float, height: float, out: list[dict[str, Any]]) -> None:
    """递归收集 block：视觉父块展平子块，叶子块输出为 raw dict。"""
    btype = _normalize_block_type(block)
    nested = block.get("blocks") or []

    if block.get("type", "") in _VISUAL_PARENT_TYPES and nested:
        # 视觉父块：展平子块（table_body/table_caption 等成为独立顶层 block）
        for child in nested:
            _collect_blocks(child, width, height, out)
        return

    bbox = _denormalize_bbox(block.get("bbox"), width, height)
    # 零面积 bbox（flash 无版面坐标）用最小合法占位，满足新 schema 的 PDF bbox 要求（x1>x0, y1>y0）
    if (bbox is None and _is_empty_bbox(block.get("bbox"))) or (
        bbox is not None and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1])
    ):
        bbox = _PLACEHOLDER_BBOX

    content, image_path = _extract_lines_content(block.get("lines") or [])
    lines = _denormalize_lines(block.get("lines") or [], width, height)

    raw: dict[str, Any] = {"type": btype, "bbox": bbox}
    if content:
        raw["content"] = content
    if image_path:
        raw["image_path"] = image_path
    # lines 供 merge_para_text_blocks 做跨页几何分析，用完由 _remove_private_metadata 清除
    if lines:
        raw["lines"] = lines
    if block.get("level") is not None:
        raw["level"] = block["level"]
    # code_body 必须带 sub_type（_annotate_code_languages 直接读 code_block["sub_type"]）
    if btype == "code_body":
        raw["sub_type"] = block.get("sub_type") or "code"
    elif block.get("sub_type"):
        raw["sub_type"] = block["sub_type"]
    if block.get("anchor"):
        raw["anchor"] = block["anchor"]
    out.append(raw)


def _is_empty_bbox(bbox: Any) -> bool:
    """判断 bbox 是否为空（None 或 [0,0,0,0]）。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return True
    return all(float(v) == 0.0 for v in bbox)


def _normalize_block_type(block: dict[str, Any]) -> str:
    """旧 VLM 标签 → 新 schema BlockType 映射。"""
    btype = block.get("type", "")
    if btype == "title":
        level = block.get("level")
        return "doc_title" if level == 1 else "paragraph_title"
    if btype == "interline_equation":
        return "equation"
    return btype


def _denormalize_bbox(bbox: Any, width: float, height: float) -> tuple[float, float, float, float] | None:
    """像素坐标 → 归一化坐标（0-1），page_size 缺失时返回 None。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return (x0 / width, y0 / height, x1 / width, y1 / height)


def _extract_lines_content(lines: list[dict[str, Any]]) -> tuple[str, str]:
    """从 lines[].spans[] 提取文本 content 和第一个 image_path。

    schema 1.0 的 span 是同一行内的水平切片（字体/样式变化就会切开），
    所以行内 span 必须无分隔符拼接，只有行与行之间才换行；
    否则一行里每个 span 都会变成独立的一行，把句子拆碎。
    """
    line_texts: list[str] = []
    image_path = ""
    for line in lines:
        if not isinstance(line, dict):
            continue
        parts: list[str] = []
        for span in line.get("spans") or []:
            if not isinstance(span, dict):
                continue
            text = span.get("content")
            if text:
                parts.append(text)
            if not image_path and span.get("image_path"):
                image_path = span["image_path"]
        if parts:
            line_texts.append("".join(parts))
    return "\n".join(line_texts), image_path


def _denormalize_lines(
    lines: list[dict[str, Any]],
    width: float,
    height: float,
) -> list[dict[str, Any]]:
    """保留 lines 供跨页几何分析，line bbox 像素→归一化。

    page_size 有值时 line bbox 是像素坐标，需反归一化；
    page_size=None 时 line bbox 可能已是归一化的，直接用。
    """
    result: list[dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        line_bbox = _denormalize_bbox(line.get("bbox"), width, height)
        if line_bbox is None:
            # page_size=None 时直接用原始 bbox（可能已是归一化的）
            raw_bbox = line.get("bbox")
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                continue
            line_bbox = tuple(float(v) for v in raw_bbox)  # type: ignore[assignment]
        result.append({"bbox": list(line_bbox)})
    return result


__all__ = ["legacy_page_to_model_list"]
