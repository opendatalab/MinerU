"""为真实浏览器验收生成可重复的 PDF、图片和错误样本。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


def create_fixtures(root: Path) -> dict[str, str]:
    """生成中文、长文档、扫描页与异常 PDF，保存供浏览器上传的绝对路径。"""
    root.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    result: dict[str, str] = {}
    for name, filename, count in (("chinese", "中文 空格 # % 文档.pdf", 3), ("long", "long-document.pdf", 100)):
        path = root / filename
        painter = canvas.Canvas(str(path), pagesize=(480, 640))
        for page in range(1, count + 1):
            painter.setFillColorRGB(0.96, 0.97, 1)
            painter.rect(0, 0, 480, 640, fill=1, stroke=0)
            painter.setFillColorRGB(0.12, 0.2, 0.35)
            painter.setFont("Helvetica", 24)
            painter.drawString(35, 585, f"PDF preview - page {page} / {count}")
            painter.setFont("STSong-Light", 20)
            painter.drawString(35, 535, "中文预览：文字、表格和页面顺序")
            painter.setFont("Helvetica", 13)
            painter.drawString(35, 490, "MinerU / PDF.js / offline assets")
            painter.setStrokeColorRGB(0.25, 0.4, 0.6)
            for row in range(5):
                painter.line(35, 440 - row * 45, 440, 440 - row * 45)
            for column in range(4):
                painter.line(35 + column * 135, 260, 35 + column * 135, 440)
            painter.showPage()
        painter.save()
        result[name] = str(path.resolve())
    image = Image.new("RGB", (480, 640), "#e3f2e7")
    ImageDraw.Draw(image).text((35, 65), "SCANNED PAGE / IMAGE INPUT", fill="#125534", font_size=22)
    image.save(root / "photo.png")
    result["image"] = str((root / "photo.png").resolve())
    painter = canvas.Canvas(str(root / "scan.pdf"), pagesize=(480, 640))
    painter.drawImage(ImageReader(image), 0, 0, 480, 640)
    painter.save()
    result["scan"] = str((root / "scan.pdf").resolve())
    writer = PdfWriter()
    writer.append(PdfReader(result["chinese"]))
    writer.encrypt("preview-test-only")
    writer.write(root / "encrypted.pdf")
    result["encrypted"] = str((root / "encrypted.pdf").resolve())
    (root / "damaged.pdf").write_bytes(b"%PDF-1.7\ninvalid PDF contents")
    result["damaged"] = str((root / "damaged.pdf").resolve())
    (root / "FLASH.CSV").write_text("name,value\nPDF.js,6.3.289\n", encoding="utf-8")
    result["csv"] = str((root / "FLASH.CSV").resolve())
    (root / "fixtures.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(create_fixtures(Path(sys.argv[1])), ensure_ascii=False))
