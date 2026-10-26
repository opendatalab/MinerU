"""验证 Basic/Standard 使用共享原生恢复后保留旋转表头与真实上下标。"""

from __future__ import annotations

from io import BytesIO
from typing import Literal

import pytest
from bs4 import BeautifulSoup
from docvortex.document.pdf import PDFDocument
from PIL import Image
from reportlab.pdfgen import canvas

from mineru.backend.analysis.pdf import tables as pdf_tables
from mineru.types import BlockType


def _rotated_header_pdf() -> bytes:
    """生成横排表体与旋转表头混排的真实 PDF，不模拟 DocVortex 返回值。"""
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=(380, 380))
    for x in (20, 130, 240, 350):
        pdf.line(x, 30, x, 330)
    for y in (30, 130, 330):
        pdf.line(20, y, 350, y)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(30, 230, "Part No.")
    pdf.drawString(30, 70, "Chip-1")
    for x, value in ((170, "Core"), (280, "Main frequency (MHz)")):
        pdf.saveState()
        pdf.translate(x, 310)
        pdf.rotate(-90)
        pdf.drawString(0, 0, value)
        pdf.restoreState()
    pdf.saveState()
    pdf.translate(170, 110)
    pdf.rotate(-90)
    pdf.drawString(0, 0, "M0+")
    pdf.restoreState()
    pdf.saveState()
    pdf.translate(280, 110)
    pdf.rotate(-90)
    pdf.drawString(0, 0, "A")
    pdf.setFont("Helvetica", 7)
    pdf.drawString(8, 5, "2")
    pdf.restoreState()
    pdf.save()
    return output.getvalue()


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_native_priority_preserves_cell_writing_direction(effort: Literal["medium", "high"]) -> None:
    """两个档位应接收原生 HTML，正常表头无伪上下标且真实上标保持完整。"""
    table = {"type": BlockType.TABLE, "bbox": [20 / 380, 50 / 380, 350 / 380, 350 / 380], "angle": 0}
    with PDFDocument(_rotated_header_pdf()) as document, Image.new("RGB", (380, 380), "white") as image:
        summary = pdf_tables._apply_native_txt_table_priority(
            [[table]],
            [[]],
            [document[0]],
            [{"img_pil": image, "scale": 1.0}],
            effort=effort,
        )
    assert summary == pdf_tables._NativeTablePrioritySummary(total=1, accepted=1)
    soup = BeautifulSoup(table["content"], "html.parser")
    rows = soup.find_all("tr")
    assert [[cell.get_text() for cell in row.find_all("td")] for row in rows] == [
        ["Part No.", "Core", "Main frequency (MHz)"],
        ["Chip-1", "M0+", "A2"],
    ]
    assert rows[0].find_all(["sub", "sup"]) == []
    assert [tag.get_text() for tag in soup.find_all("sup")] == ["2"]
    assert soup.find_all("sub") == []
