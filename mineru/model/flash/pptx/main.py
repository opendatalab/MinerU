# Copyright (c) Opendatalab. All rights reserved.
from typing import Any, BinaryIO

from mineru.model.flash import PptxModel


def convert_path(file_path: str) -> list[list[dict[str, Any]]]:
    """从 PPTX 文件路径调用统一模型入口。"""

    with open(file_path, "rb") as fh:
        return convert_binary(fh)


def convert_binary(file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
    """兼容旧二进制转换函数，并转发给 PptxModel。"""

    return PptxModel().predict(file_binary)


if __name__ == "__main__":
    print(convert_path("powerpoint_sample.pptx"))
