from typing import BinaryIO

from mineru.model.pptx.pptx_converter import PptxConverter


def convert_path(file_path: str):
    with open(file_path, "rb") as fh:
        return convert_binary(fh)


def convert_binary(file_binary: BinaryIO):
    converter = PptxConverter()
    converter.convert(file_binary)
    return converter.pages


if __name__ == "__main__":
    print(convert_path("powerpoint_sample.pptx"))
