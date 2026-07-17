# Copyright (c) Opendatalab. All rights reserved.
"""PP-DocLayoutV3 第一阶段导出脚本：产出两阶段解耦的标准中间结果 layout.json。

本脚本**不依赖 mineru 包**，设计为在独立的 Paddle 环境中运行——
paddlepaddle 对 numpy/opencv 的版本要求与 MinerU 环境容易冲突，
两套环境通过 layout.json 文件交接，零依赖交集。

== 运行环境配置（独立环境，勿装进 MinerU 的 venv） ==

  python -m venv ~/paddle-env && source ~/paddle-env/bin/activate
  # GPU 版参照 https://www.paddlepaddle.org.cn/en/install 选择对应CUDA版本；CPU版：
  pip install paddlepaddle paddleocr pypdfium2 pillow

  python demo/pp_doclayoutv3_layout.py --pdf demo/pdfs/demo1.pdf \
      --output output/vlm_two_stage_demo/demo1/layout-only-v3/layout.json

之后回到 MinerU 环境，用该 layout.json 单独跑第二阶段（识别可选 MinerU VLM 或 OvisOCR2）：

  python demo/vlm_two_stage_demo.py --pdf demo/pdfs/demo1.pdf --mode c \
      --layout-json output/vlm_two_stage_demo/demo1/layout-only-v3/layout.json \
      --recognizer ovis --ovis-url http://<ovis-host>:8000
"""
import argparse
import json
import os

LAYOUT_DOC_VERSION = 1

# PP-DocLayout 系列标签 -> mineru-vl-utils 块类型
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


def box_to_block(box: dict, page_width: float, page_height: float) -> dict | None:
    """PaddleOCR LayoutDetection 的box -> layout.json块；不识别的标签返回None。"""
    block_type = LABEL_TO_BLOCK_TYPE.get(box.get("label"))
    if block_type is None:
        return None
    coordinate = box.get("coordinate")
    if not coordinate or len(coordinate) != 4:
        return None
    x0, y0, x1, y1 = [float(v) for v in coordinate]
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
        # PP-DocLayoutV3 输出为多点框/矩形框，方向信息暂不导出，识别侧按0度处理
        "angle": 0,
        "content": None,
    }
    if box.get("label") == "seal":
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


def run(pdf_path: str, output_path: str, model_name: str = "PP-DocLayoutV3", scale: float = 2.0):
    # 重依赖延迟到运行时导入，保证本模块的纯函数可在无paddle环境下被测试
    import numpy as np
    import pypdfium2 as pdfium
    from paddleocr import LayoutDetection

    model = LayoutDetection(model_name=model_name)
    pdf_doc = pdfium.PdfDocument(pdf_path)
    pages = []
    try:
        for page_idx in range(len(pdf_doc)):
            page_image = pdf_doc[page_idx].render(scale=scale).to_pil().convert("RGB")
            width, height = page_image.size
            results = model.predict(input=np.asarray(page_image), batch_size=1)
            blocks = []
            for res in results:
                res_data = res.json.get("res", res.json) if hasattr(res, "json") else res
                # 官方输出已按阅读顺序排序，保持原序即可
                for box in res_data.get("boxes", []):
                    block = box_to_block(box, width, height)
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
    parser.add_argument("--model", default="PP-DocLayoutV3", help="PaddleOCR layout模型名")
    parser.add_argument("--scale", type=float, default=2.0, help="页面渲染倍率")
    args = parser.parse_args()
    run(args.pdf, args.output, model_name=args.model, scale=args.scale)


if __name__ == "__main__":
    main()
