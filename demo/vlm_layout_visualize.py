# Copyright (c) Opendatalab. All rights reserved.
"""将两阶段解耦的中间结果 layout.json 可视化：把分块框和类型标签画到页面图上。

用法：
  python demo/vlm_layout_visualize.py --pdf demo/pdfs/demo1.pdf \
      --layout output/vlm_two_stage_demo/demo1/layout-only/layout.json \
      --output output/vlm_two_stage_demo/demo1/layout-only/vis
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

# 块类型 -> RGB（视觉正文/标题/表格/公式等使用高区分度颜色）
TYPE_COLORS = {
    "text": (68, 114, 196),
    "title": (237, 125, 49),
    "table": (112, 173, 71),
    "table_caption": (84, 130, 53),
    "table_footnote": (84, 130, 53),
    "equation": (255, 0, 102),
    "formula_number": (192, 0, 96),
    "image": (0, 176, 240),
    "image_caption": (0, 130, 178),
    "image_footnote": (0, 130, 178),
    "chart": (255, 192, 0),
    "code": (128, 96, 0),
    "algorithm": (128, 96, 0),
    "ref_text": (146, 106, 166),
    "index": (146, 106, 166),
    "aside_text": (128, 128, 128),
    "header": (166, 166, 166),
    "footer": (166, 166, 166),
    "page_number": (191, 191, 191),
    "page_footnote": (191, 191, 191),
    "list": (46, 117, 182),
    "list_item": (46, 117, 182),
}
DEFAULT_COLOR = (64, 64, 64)


def _load_font(size=16):
    for name in ("arial.ttf", "msyh.ttc", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_page_blocks(page_image: Image.Image, blocks: list[dict], font) -> Image.Image:
    annotated = page_image.convert("RGB")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = annotated.size
    for order, block in enumerate(blocks):
        block_type = block.get("type", "unknown")
        color = TYPE_COLORS.get(block_type, DEFAULT_COLOR)
        x0, y0, x1, y1 = block["bbox"]
        rect = [x0 * width, y0 * height, x1 * width, y1 * height]
        draw.rectangle(rect, outline=color + (255,), width=3, fill=color + (28,))
        label = f"{order}.{block_type}"
        angle = block.get("angle")
        if angle:
            label += f" ∠{angle}"
        text_bbox = draw.textbbox((rect[0], rect[1]), label, font=font)
        draw.rectangle(text_bbox, fill=color + (220,))
        draw.text((rect[0], rect[1]), label, fill=(255, 255, 255, 255), font=font)
    annotated = Image.alpha_composite(annotated.convert("RGBA"), overlay)
    return annotated.convert("RGB")


def visualize(pdf_path: str, layout_json_path: str, output_dir: str, scale: float = 2.0):
    layout_doc = json.load(open(layout_json_path, encoding="utf-8"))
    pages_by_idx = {int(p["page_idx"]): p for p in layout_doc.get("pages", [])}
    os.makedirs(output_dir, exist_ok=True)
    font = _load_font()

    pdf_doc = pdfium.PdfDocument(pdf_path)
    type_counter = Counter()
    saved = []
    try:
        for page_idx in sorted(pages_by_idx):
            page_meta = pages_by_idx[page_idx]
            page_image = pdf_doc[page_idx].render(scale=scale).to_pil()
            annotated = draw_page_blocks(page_image, page_meta["blocks"], font)
            out_path = os.path.join(output_dir, f"page_{page_idx:03d}.png")
            annotated.save(out_path)
            saved.append(out_path)
            type_counter.update(b.get("type", "unknown") for b in page_meta["blocks"])
    finally:
        pdf_doc.close()

    logger.info(f"{len(saved)} annotated pages -> {output_dir}")
    logger.info(f"block type stats: {dict(type_counter.most_common())}")
    return saved, type_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--layout", required=True, help="layout.json 路径")
    parser.add_argument("--output", required=True, help="标注图输出目录")
    parser.add_argument("--scale", type=float, default=2.0, help="页面渲染倍率")
    args = parser.parse_args()
    visualize(args.pdf, args.layout, args.output, scale=args.scale)


if __name__ == "__main__":
    main()
