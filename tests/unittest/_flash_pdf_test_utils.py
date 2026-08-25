from __future__ import annotations



from mineru.model.flash.pdf import (
    models,
)


def _text_line(
    text: str,
    bbox: tuple[float, float, float, float],
    source_index: int,
    *,
    angle: int = 0,
    visual_row_id: int | None = None,
    run_index: int = 0,
    split_from_row: bool = False,
    effective_height: float | None = None,
    font_signature: tuple[str, int] | None = None,
    font_coverage: float = 0.0,
    dominant_font_weight: float | None = None,
    median_glyph_width: float | None = None,
    leading_emphasis_width: float | None = None,
    preserve_split_boundary: bool = False,
    semantic_type: str | None = None,
) -> models._LineItem:
    """构造栏带、排版恢复与图形标签测试使用的原生文本行。"""

    return models._LineItem(
        text=text,
        bbox=bbox,
        angle=angle,
        source_index=source_index,
        visual_row_id=visual_row_id,
        run_index=run_index,
        effective_height=effective_height or (bbox[3] - bbox[1]),
        font_signature=font_signature,
        font_coverage=font_coverage,
        dominant_font_weight=dominant_font_weight,
        median_glyph_width=median_glyph_width,
        leading_emphasis_width=leading_emphasis_width,
        split_from_row=split_from_row,
        preserve_split_boundary=preserve_split_boundary,
        semantic_type=semantic_type,
    )


def _prepared_text_page(
    *lines: models._LineItem,
    page_size: tuple[float, float] = (100.0, 100.0),
) -> models._PreparedPage:
    """构造跨页边缘类型测试使用的无容器轻量页面。"""

    return models._PreparedPage(
        page_size=page_size,
        remaining_lines=list(lines),
        table_bboxes=[],
        drawing_lines=[],
        fixed_blocks=[],
    )
