# Copyright (c) Opendatalab. All rights reserved.
"""VLM后端两阶段解耦用法演示。

三种运行模式：
  a: 全默认 —— VLM一体完成layout+识别（与原版行为一致）
  b: layout替换为PP-DocLayoutV2小模型（CPU可跑），VLM只做识别
  c: 两阶段完全分离 —— 先跑layout落盘layout.json，再单独从layout.json跑识别
  layout-only: 只跑第一阶段，产出layout.json后退出（无需VLM/GPU）

示例：
  # 无GPU机器上只跑layout（第一阶段）
  python demo/vlm_two_stage_demo.py --pdf demo/pdfs/demo1.pdf --mode layout-only

  # 连远端VLM服务跑模式b（layout用CPU小模型，识别用远端VLM）
  python demo/vlm_two_stage_demo.py --pdf demo/pdfs/demo1.pdf --mode b \
      --backend http-client --server-url http://<gpu-host>:30000

  # 识别阶段替换为OvisOCR2（需另起OvisOCR2服务，见ovis_ocr_recognizer.py顶部说明）
  python demo/vlm_two_stage_demo.py --pdf demo/pdfs/demo1.pdf --mode b \
      --recognizer ovis --ovis-url http://<ovis-host>:8000
"""
import argparse
import json
import os
from pathlib import Path

from loguru import logger


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", default="demo/pdfs/demo1.pdf", help="输入PDF路径")
    parser.add_argument("--mode", default="layout-only", choices=["a", "b", "c", "layout-only"])
    parser.add_argument("--backend", default="transformers",
                        help="VLM子后端: transformers/vllm-engine/http-client等")
    parser.add_argument("--server-url", default=None, help="http-client后端的服务地址")
    parser.add_argument("--device", default=None, help="PP-DocLayoutV2设备，如cpu/cuda")
    parser.add_argument("--output", default="output/vlm_two_stage_demo", help="输出目录")
    parser.add_argument("--recognizer", default="mineru", choices=["mineru", "ovis"],
                        help="第二阶段识别模型：mineru=默认VLM，ovis=OvisOCR2")
    parser.add_argument("--ovis-url", default=None, help="OvisOCR2 OpenAI兼容服务地址")
    parser.add_argument("--ovis-model", default="ATH-MaaS/OvisOCR2", help="OvisOCR2模型名")
    parser.add_argument("--layout-json", default=None,
                        help="模式c使用的外部layout.json（如pp_doclayoutv3_layout.py的产出）；"
                             "缺省则用PP-DocLayoutV2现场生成")
    return parser


def build_content_recognizer(args):
    """根据参数构造第二阶段识别器；mineru默认识别器返回None（由doc_analyze兜底）。"""
    if args.recognizer == "ovis":
        if not args.ovis_url:
            raise SystemExit("--recognizer ovis 需要提供 --ovis-url")
        from mineru.backend.vlm.ovis_ocr_recognizer import OvisOcrContentRecognizer
        return OvisOcrContentRecognizer(server_url=args.ovis_url, model_name=args.ovis_model)
    return None


def write_outputs(md_dir, name, middle_json):
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import MakeMode

    md_writer = FileBasedDataWriter(md_dir)
    pdf_info = middle_json["pdf_info"]
    md_writer.write_string(f"{name}.md", union_make(pdf_info, MakeMode.MM_MD, "images"))
    md_writer.write_string(
        f"{name}_middle.json",
        json.dumps(middle_json, ensure_ascii=False, indent=2),
    )
    logger.info(f"outputs written to {md_dir}")


def main():
    args = build_arg_parser().parse_args()

    from mineru.backend.vlm import vlm_analyze
    from mineru.backend.vlm.stages import DEFAULT_LAYOUT_DOC_FILENAME, PrecomputedLayoutDetector
    from mineru.cli.common import convert_pdf_bytes_to_bytes, read_fn
    from mineru.data.data_reader_writer import FileBasedDataWriter

    pdf_path = Path(args.pdf)
    pdf_bytes = convert_pdf_bytes_to_bytes(read_fn(pdf_path))
    out_dir = os.path.join(args.output, pdf_path.stem, args.mode)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    image_writer = FileBasedDataWriter(image_dir)
    layout_writer = FileBasedDataWriter(out_dir)
    layout_json_path = os.path.join(out_dir, DEFAULT_LAYOUT_DOC_FILENAME)

    vlm_kwargs = dict(backend=args.backend, server_url=args.server_url)
    content_recognizer = build_content_recognizer(args)

    if args.mode == "a":
        # 默认路径：与原版完全一致的一体两步调用（指定ovis时改走解耦识别）
        middle_json, _ = vlm_analyze.doc_analyze(
            pdf_bytes,
            image_writer,
            content_recognizer=content_recognizer,
            **vlm_kwargs,
        )
        write_outputs(out_dir, pdf_path.stem, middle_json)
    elif args.mode == "b":
        # layout换成pipeline小模型，识别交给VLM或OvisOCR2
        from mineru.backend.vlm.pipeline_layout_detector import PipelineLayoutDetector
        middle_json, _ = vlm_analyze.doc_analyze(
            pdf_bytes,
            image_writer,
            layout_detector=PipelineLayoutDetector(device=args.device),
            content_recognizer=content_recognizer,
            layout_writer=layout_writer,
            **vlm_kwargs,
        )
        write_outputs(out_dir, pdf_path.stem, middle_json)
    elif args.mode == "layout-only":
        # 只跑第一阶段（无需VLM），产出标准中间结果layout.json
        from mineru.backend.vlm.pipeline_layout_detector import PipelineLayoutDetector
        layout_doc = vlm_analyze.doc_layout_analyze(
            pdf_bytes,
            PipelineLayoutDetector(device=args.device),
            layout_writer=layout_writer,
        )
        logger.info(
            f"layout done: {layout_doc['page_count']} pages -> {layout_json_path}"
        )
    elif args.mode == "c":
        # 两阶段分离：阶段一落盘（或使用外部layout.json）-> 阶段二从文件恢复
        if args.layout_json:
            layout_json_path = args.layout_json
            if not os.path.exists(layout_json_path):
                raise SystemExit(f"--layout-json 不存在: {layout_json_path}")
            logger.info(f"using external layout doc: {layout_json_path}")
        elif not os.path.exists(layout_json_path):
            from mineru.backend.vlm.pipeline_layout_detector import PipelineLayoutDetector
            vlm_analyze.doc_layout_analyze(
                pdf_bytes,
                PipelineLayoutDetector(device=args.device),
                layout_writer=layout_writer,
            )
            logger.info(f"stage 1 done -> {layout_json_path}")
        middle_json, _ = vlm_analyze.doc_analyze(
            pdf_bytes,
            image_writer,
            layout_detector=PrecomputedLayoutDetector(layout_json_path),
            content_recognizer=content_recognizer,
            **vlm_kwargs,
        )
        write_outputs(out_dir, pdf_path.stem, middle_json)


if __name__ == "__main__":
    main()
