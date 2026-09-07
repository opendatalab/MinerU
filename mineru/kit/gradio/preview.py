"""Gradio 布局预览的文件适配，绘制能力由 DocVortex 统一维护。"""

from __future__ import annotations

from pathlib import Path

from docvortex.visualization import render_layout_pdf

from ...types import MiddleJson


def draw_layout_overlay(
    middle_json: MiddleJson,
    origin_pdf_path: Path,
    output_path: Path,
    *,
    page_indices: tuple[int, ...] = (),
) -> None:
    """读取原始 PDF 并写出共享绘制结果，保留 Gradio 的调用和文件契约。"""
    if not isinstance(middle_json, MiddleJson):
        raise TypeError("middle_json must be a MiddleJson instance")
    if not origin_pdf_path.is_file():
        raise FileNotFoundError(origin_pdf_path)

    pdf_bytes = render_layout_pdf(origin_pdf_path.read_bytes(), middle_json.pages, page_indices=page_indices or None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(pdf_bytes)


__all__ = ["draw_layout_overlay"]
