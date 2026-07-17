# Copyright (c) Opendatalab. All rights reserved.
"""PP-DocLayoutV3 第一阶段导出脚本：产出两阶段解耦的标准中间结果 layout.json。

使用 HuggingFace transformers 官方移植版 `PaddlePaddle/PP-DocLayoutV3_safetensors`
（PPDocLayoutV3ForObjectDetection，2026-01 并入 transformers 主干）：
输出按阅读顺序排序，无需 paddlepaddle。

本脚本**不依赖 mineru 包**，设计为在独立环境中运行——
移植版要求 transformers>=5.x，与 MinerU 锁定的 transformers<5 **冲突**，
两套环境通过 layout.json 文件交接，零依赖交集。

== 运行环境配置（独立环境，勿装进 MinerU 的 venv） ==

  python -m venv ~/hf5-env && source ~/hf5-env/bin/activate
  pip install "transformers>=5.0" torch pypdfium2 pillow
  # HF 访问受限时：export HF_ENDPOINT=https://hf-mirror.com

  python demo/pp_doclayoutv3_layout.py --pdf demo/pdfs/demo1.pdf \
      --output output/vlm_two_stage_demo/demo1/layout-only-v3/layout.json

之后回到 MinerU 环境，用该 layout.json 单独跑第二阶段（识别可选 MinerU VLM 或 OvisOCR2）：

  python demo/vlm_two_stage_demo.py --pdf demo/pdfs/demo1.pdf --mode c \
      --layout-json output/vlm_two_stage_demo/demo1/layout-only-v3/layout.json \
      --recognizer ovis --ovis-url http://<ovis-host>:8000

注意：HF 移植版的 id2label 把 display_formula/inline_formula 都改名为 "formula"、
header_image/footer_image 并入 header/footer，因此本脚本按 **cls_id** 映射
（类目顺序与 PP-DocLayoutV2/V3 的 25 类规范一致），不按 id2label 名字。
"""
import argparse
import json
import os

LAYOUT_DOC_VERSION = 1

# PP-DocLayout 系列 25 类规范顺序（与 mineru/model/layout/pp_doclayoutv2.py 一致）
PP_DOCLAYOUT_LABELS = [
    "abstract",           # 0
    "algorithm",          # 1
    "aside_text",         # 2
    "chart",              # 3
    "content",            # 4
    "display_formula",    # 5
    "doc_title",          # 6
    "figure_title",       # 7
    "footer",             # 8
    "footer_image",       # 9
    "footnote",           # 10
    "formula_number",     # 11
    "header",             # 12
    "header_image",       # 13
    "image",              # 14
    "inline_formula",     # 15  -> 不导出（行内公式属于text内部，由识别阶段处理）
    "number",             # 16
    "paragraph_title",    # 17
    "reference",          # 18  -> 不导出（list外框，与reference_content重复覆盖）
    "reference_content",  # 19
    "seal",               # 20
    "table",              # 21
    "text",               # 22
    "vertical_text",      # 23
    "vision_footnote",    # 24
]

# 标签 -> mineru-vl-utils 块类型
# 与 mineru/backend/hybrid/hybrid_analyze.py 的 MEDIUM_EFFORT_LAYOUT_LABEL_TO_VLM_TYPE 保持一致；
# 本脚本不依赖 mineru 包，故在此静态复制一份。
LABEL_TO_BLOCK_TYPE = {
    "abstract": "text",
    "algorithm": "code",
    "aside_text": "aside_text",
    "content": "index",
    "doc_title": "title",
    "footer": "footer",
    "footer_image": "footer",
    "footnote": "page_footnote",
    "formula_number": "formula_number",
    "header": "header",
    "header_image": "header",
    "number": "page_number",
    "paragraph_title": "title",
    "reference_content": "ref_text",
    "text": "text",
    "vertical_text": "text",
    "figure_title": "image_caption",
    "vision_footnote": "image_footnote",
    "image": "image",
    "chart": "chart",
    "seal": "image",
    "table": "table",
    "display_formula": "equation",
}


def detection_to_block(
    cls_id: int,
    box_xyxy,
    page_width: float,
    page_height: float,
) -> dict | None:
    """HF移植版检测结果 -> layout.json块；不导出的类目返回None。"""
    if not 0 <= cls_id < len(PP_DOCLAYOUT_LABELS):
        return None
    label = PP_DOCLAYOUT_LABELS[cls_id]
    block_type = LABEL_TO_BLOCK_TYPE.get(label)
    if block_type is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    bbox = [
        round(min(max(x0 / page_width, 0.0), 1.0), 4),
        round(min(max(y0 / page_height, 0.0), 1.0), 4),
        round(min(max(x1 / page_width, 0.0), 1.0), 4),
        round(min(max(y1 / page_height, 0.0), 1.0), 4),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    block = {
        "type": block_type,
        "bbox": bbox,
        # 移植版另有polygon_points可表达歪斜框，本版先取外接矩形、方向按0度
        "angle": 0,
        "content": None,
    }
    if label == "seal":
        block["sub_type"] = "seal"
    return block


def build_layout_doc(pages: list[dict]) -> dict:
    """组装 layout.json（与 mineru.backend.vlm.stages 的 schema 保持一致）。"""
    return {
        "version": LAYOUT_DOC_VERSION,
        "layout_backend": "pp_doclayout_v3",
        "page_count": len(pages),
        # PP-DocLayout 系列会输出 formula_number 块，识别侧需做公式编号合并
        "emits_formula_number": True,
        "pages": pages,
    }


def run(
    pdf_path: str,
    output_path: str,
    model_path: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
    scale: float = 2.0,
    threshold: float = 0.5,
    device: str | None = None,
):
    # 重依赖延迟到运行时导入，保证本模块的纯函数可在无transformers-5环境下被测试
    import pypdfium2 as pdfium
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForObjectDetection.from_pretrained(model_path).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_path)

    pdf_doc = pdfium.PdfDocument(pdf_path)
    pages = []
    try:
        for page_idx in range(len(pdf_doc)):
            page_image = pdf_doc[page_idx].render(scale=scale).to_pil().convert("RGB")
            width, height = page_image.size
            inputs = processor(images=[page_image], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=[(height, width)],
                threshold=threshold,
            )
            blocks = []
            # post_process 输出已按模型的阅读顺序头排序，保持原序即可
            for score, label_id, box in zip(
                results[0]["scores"], results[0]["labels"], results[0]["boxes"]
            ):
                block = detection_to_block(int(label_id), box.tolist(), width, height)
                if block is not None:
                    blocks.append(block)
            pages.append({
                "page_idx": page_idx,
                "page_size": [width, height],
                "blocks": blocks,
            })
            print(f"page {page_idx + 1}/{len(pdf_doc)}: {len(blocks)} blocks")
    finally:
        pdf_doc.close()

    layout_doc = build_layout_doc(pages)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(layout_doc, f, ensure_ascii=False, indent=2)
    print(f"layout doc written: {output_path} ({layout_doc['page_count']} pages)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True, help="layout.json 输出路径")
    parser.add_argument("--model", default="PaddlePaddle/PP-DocLayoutV3_safetensors")
    parser.add_argument("--scale", type=float, default=2.0, help="页面渲染倍率")
    parser.add_argument("--threshold", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--device", default=None, help="cpu/cuda，缺省自动检测")
    args = parser.parse_args()
    run(args.pdf, args.output, model_path=args.model, scale=args.scale,
        threshold=args.threshold, device=args.device)


if __name__ == "__main__":
    main()
