# Copyright (c) Opendatalab. All rights reserved.
from io import BytesIO

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlsImage
from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, OneCellAnchor
from openpyxl.drawing.xdr import XDRPositiveSize2D
from PIL import Image

from mineru.model.xlsx.xlsx_converter import XlsxConverter


def _image_stream(color: tuple[int, int, int]) -> BytesIO:
    stream = BytesIO()
    Image.new("RGB", (2, 2), color).save(stream, format="PNG")
    stream.seek(0)
    return stream


def test_collect_sheet_images_preserves_images_inside_merged_cell() -> None:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.merge_cells("B1:D1")

    for col, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), start=1):
        image = XlsImage(_image_stream(color))
        image.anchor = OneCellAnchor(
            _from=AnchorMarker(col=col, row=0),
            ext=XDRPositiveSize2D(cx=10, cy=10),
        )
        worksheet.add_image(image)

    converter = XlsxConverter()
    converter.workbook = workbook

    images = converter._collect_sheet_images(worksheet)

    assert len(images) == 3
    assert [image_info["anchor"] for image_info in images] == [(0, 1)] * 3
