# Copyright (c) Opendatalab. All rights reserved.
from typing import Any, BinaryIO

from ... import XlsxModel


def convert_path(file_path: str) -> list[list[dict[str, Any]]]:
    """从 XLSX 文件路径调用统一模型入口。"""

    with open(file_path, "rb") as fh:
        return convert_binary(fh)


def convert_binary(file_binary: BinaryIO) -> list[list[dict[str, Any]]]:
    """兼容旧二进制转换函数，并转发给 XlsxModel。"""

    return XlsxModel().predict(file_binary)


if __name__ == "__main__":
    print(convert_path("test_xlsx/xlsx_01.xlsx"))
