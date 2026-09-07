# Copyright (c) Opendatalab. All rights reserved.
"""使用 Python SDK 解析示例 PDF；在项目根目录运行 python -m demo.demo。"""

from pathlib import Path

from mineru.parser import parse
from mineru.parser.writer import FileBasedDataWriter


def main() -> None:
    """按文件名顺序解析 demo/pdfs 中的 PDF，并分别保存解析结果。"""
    demo_dir = Path(__file__).resolve().parent
    pdf_dir = demo_dir / "pdfs"
    pdf_paths = sorted(path for path in pdf_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdf_paths:
        print(f"未找到 PDF 文件：{pdf_dir}", flush=True)
        return

    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{index}/{len(pdf_paths)}] 正在解析：{pdf_path.name}", flush=True)
        result = parse(
            pdf_path,
            tier="standard",
            ocr_mode="auto",
            page_range="all",
        )

        # 通过 SDK 保存 Markdown 和 JSON，每份 PDF 使用独立的输出目录。
        output_dir = demo_dir / "output" / pdf_path.stem
        result.save(FileBasedDataWriter(str(output_dir)))
        print(f"解析完成，共 {len(result.pages)} 页；输出目录：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
