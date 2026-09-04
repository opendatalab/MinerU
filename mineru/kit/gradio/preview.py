"""基于 schema 2.0 Middle JSON 的 Gradio 文档预览和布局标注。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from pypdf import PageObject, PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from ...types import BlockBase, BlockType, MiddleJson, PageInfo

_VISUAL_PARENT_TYPES = {
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.CODE,
}

_BLOCK_COLORS: dict[str, tuple[float, float, float]] = {
    "text": (0.60, 0.05, 0.30),
    "ref_text": (0.45, 0.20, 0.65),
    "doc_title": (0.20, 0.20, 0.80),
    "paragraph_title": (0.20, 0.40, 0.90),
    "equation": (0.00, 0.60, 0.10),
    "list": (0.10, 0.60, 0.35),
    "index": (0.10, 0.60, 0.35),
    "image_body": (0.30, 0.85, 0.05),
    "image_caption": (0.10, 0.55, 0.90),
    "image_footnote": (0.95, 0.45, 0.10),
    "table_body": (0.80, 0.80, 0.00),
    "table_caption": (1.00, 0.85, 0.10),
    "table_footnote": (0.55, 0.90, 0.25),
    "chart_body": (0.30, 0.85, 0.05),
    "chart_caption": (0.10, 0.55, 0.90),
    "chart_footnote": (0.95, 0.45, 0.10),
    "code_body": (0.40, 0.00, 0.80),
    "algorithm_body": (0.40, 0.00, 0.80),
    "code_caption": (0.70, 0.35, 0.95),
    "code_footnote": (0.85, 0.70, 0.95),
}


def draw_layout_overlay(
    middle_json: MiddleJson,
    origin_pdf_path: Path,
    output_path: Path,
    *,
    page_indices: tuple[int, ...] = (),
) -> None:
    """将当前 Middle JSON 的语义 bbox 绘制到 origin PDF，生成布局预览。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("middle_json must be a MiddleJson instance")
    if not origin_pdf_path.is_file():
        raise FileNotFoundError(origin_pdf_path)

    reader = PdfReader(str(origin_pdf_path))
    writer = PdfWriter()
    pages_by_original_idx = {page.page_idx: page for page in middle_json.pages}
    for output_index, source_page in enumerate(reader.pages):
        middle_page = _page_for_output_index(
            output_index,
            pages_by_original_idx,
            middle_json.pages,
            page_indices,
        )
        overlay = _build_page_overlay(source_page, middle_page)
        if overlay is not None:
            page_copy = PageObject.create_blank_page(
                None,
                width=float(source_page.cropbox.width),
                height=float(source_page.cropbox.height),
            )
            page_copy.merge_page(source_page)
            page_copy.merge_page(overlay)
            writer.add_page(page_copy)
        else:
            writer.add_page(source_page)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        writer.write(output_file)


def _page_for_output_index(
    output_index: int,
    pages_by_original_idx: dict[int, PageInfo],
    pages: list[PageInfo],
    page_indices: tuple[int, ...],
) -> PageInfo | None:
    """按裁页映射优先、页面顺序回退选择 overlay 对应的 Middle 页面。"""
    if page_indices and output_index < len(page_indices):
        mapped = pages_by_original_idx.get(page_indices[output_index])
        if mapped is not None:
            return mapped
    if output_index < len(pages):
        return pages[output_index]
    return None


def _build_page_overlay(page: Any, middle_page: PageInfo | None) -> PageObject | None:
    """为单个 PDF 页面生成透明 bbox overlay 页面。"""
    if middle_page is None:
        return None
    width = float(page.cropbox.width)
    height = float(page.cropbox.height)
    if width <= 0 or height <= 0:
        return None
    boxes = list(_iter_overlay_boxes(middle_page.blocks))
    if not boxes:
        return None

    packet = BytesIO()
    painter = canvas.Canvas(packet, pagesize=(width, height))
    painter.setLineWidth(1.0)
    for block_type, bbox in boxes:
        x0, y0, x1, y1 = _normalized_bbox_to_pdf(bbox, width, height)
        color = _BLOCK_COLORS.get(block_type, (0.90, 0.10, 0.10))
        painter.setStrokeColorRGB(*color)
        painter.rect(x0, y0, max(0.5, x1 - x0), max(0.5, y1 - y0), stroke=1, fill=0)
    painter.save()
    packet.seek(0)
    return PdfReader(packet).pages[0]


def _iter_overlay_boxes(blocks: Iterable[BlockBase]) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    """遍历当前 block 树并为视觉父块选择子块 bbox，避免重复绘制。"""
    for block in blocks:
        block_type = _block_type_value(block)
        if block.bbox is not None and block.type not in _VISUAL_PARENT_TYPES:
            yield block_type, tuple(float(item) for item in block.bbox)
        content = getattr(block, "content", None)
        children = [child for child in content if isinstance(child, BlockBase)] if isinstance(content, list) else []
        if block.type in _VISUAL_PARENT_TYPES and children:
            for child in children:
                if child.bbox is not None:
                    yield _block_type_value(child), tuple(float(item) for item in child.bbox)
        elif children:
            yield from _iter_overlay_boxes(children)


def _normalized_bbox_to_pdf(
    bbox: tuple[float, float, float, float],
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """把左上原点的归一化 bbox 转换为 PDF 左下原点坐标。"""
    x0 = min(max(bbox[0], 0.0), 1.0) * width
    x1 = min(max(bbox[2], 0.0), 1.0) * width
    top = min(max(bbox[1], 0.0), 1.0) * height
    bottom = min(max(bbox[3], 0.0), 1.0) * height
    return x0, height - bottom, x1, height - top


def _block_type_value(block: BlockBase) -> str:
    """统一读取 Enum 或字符串 block 类型值。"""
    value = getattr(block.type, "value", block.type)
    return str(value)


__all__ = ["draw_layout_overlay"]
