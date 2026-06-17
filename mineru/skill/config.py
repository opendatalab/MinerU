# Copyright (c) Opendatalab. All rights reserved.
"""ocr-mineru skill 默认配置与常量"""

from pathlib import Path

# 默认输出目录
DEFAULT_OUTPUT_DIR = Path("./mineru_skill_output")

# 支持的文件后缀
SUPPORTED_SUFFIXES = {
    "pdf",
    "png",
    "jpeg",
    "jpg",
    "jp2",
    "webp",
    "gif",
    "bmp",
    "tiff",
    "tif",
    "docx",
    "pptx",
    "xlsx",
}

# 默认解析配置
DEFAULT_BACKEND = "hybrid-engine"
DEFAULT_PARSE_METHOD = "auto"
DEFAULT_LANGUAGE = "ch"
DEFAULT_FORMULA_ENABLE = True
DEFAULT_TABLE_ENABLE = True
DEFAULT_IMAGE_ANALYSIS = True
DEFAULT_EFFORT = "medium"

# 默认解析超时时间（秒）
DEFAULT_PARSE_TIMEOUT = 600
