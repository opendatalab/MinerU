# Copyright (c) Opendatalab. All rights reserved.
"""ocr-mineru skill 命令行风格帮助脚本

用法：
    python -m mineru.skill.scripts.help
    python -m mineru.skill.scripts.help --help
"""

HELP_TEXT = """
ocr-mineru: 使用 MinerU 对 PDF/图片进行 OCR 和结构化解析

用法：
    from mineru.skill import parse_file_sync, ParseOptions

    result = parse_file_sync(
        "path/to/file.pdf",
        options=ParseOptions(
            backend="hybrid-engine",   # pipeline / vlm-engine / hybrid-engine / ...
            parse_method="auto",       # auto / txt / ocr
            language="ch",             # ch / en / japan / korean / ...
            formula_enable=True,       # 是否识别公式
            table_enable=True,         # 是否识别表格
            image_analysis=True,       # 是否分析图片/图表
            effort="medium",           # medium / high（仅 hybrid 后端）
        ),
    )

输出字段：
    result.markdown          # Markdown 全文
    result.content_list_v2   # 结构化内容列表
    result.middle_json       # 中间解析结果
    result.images            # {文件名: base64 data URL}
    result.output_dir        # 结果输出目录

常用方法：
    result.get_text()                    # 提取所有文本
    result.get_text(["title", "table"])  # 按类型提取文本
    result.get_tables()                  # 提取表格块
    result.get_images()                  # 获取图片列表
    result.save_markdown("out.md")       # 保存 markdown
    result.save_content_list("out.json") # 保存结构化内容
    result.save_all("output_dir")        # 保存全部产物

触发词：
    OCR 识别、解析 PDF、提取 PDF 内容、识别图片文字、mineru

注意：
    - 首次使用需要下载模型权重，请确保网络可访问。
    - 大文档默认超时 600 秒。
    - 显存不足时可切换 backend="pipeline" 在 CPU 运行。
"""


def print_help() -> None:
    """打印帮助信息"""
    print(HELP_TEXT.strip())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print_help()
    else:
        print_help()
