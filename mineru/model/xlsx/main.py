# Copyright (c) Opendatalab. All rights reserved.
from typing import BinaryIO

from mineru.model.xlsx.xlsx_converter import XlsxConverter


def convert_path(file_path: str):
    with open(file_path, "rb") as fh:
        return convert_binary(fh)


def convert_binary(file_binary: BinaryIO):
    converter = XlsxConverter()
    converter.convert(file_binary)
    return converter.pages

if __name__ == "__main__":
    print(convert_path("test_xlsx/xlsx_01.xlsx"))
