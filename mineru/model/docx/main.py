from typing import BinaryIO
from docx_converter import DocxConverter


def convert_path(file_path: str):
    with open(file_path, "rb") as fh:
        return convert_binary(fh, file_path)


def convert_binary(file_binary: BinaryIO, file_path: str):
    converter = DocxConverter(file_path=file_path, output_path="./output")
    if not converter.accepts(file_binary):
        raise Exception("Not a docx file")
    return converter.convert(file_binary)


if __name__ == "__main__":
    print(convert_path("test.docx"))