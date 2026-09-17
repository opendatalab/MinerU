"""Minimal mixed-font PDF for MinerU #5517 digit-drop.

Glyph placement aims for CNKI-like geometry: CJK on the line mid-axis;
ASCII/fullwidth Nd digits from a smaller/offset font sit low enough that
calculate_char_in_span fails against a layout-like span that still contains
their centers (same scenario as tests/unittest/test_pdf_native_script_detection.py).
"""
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

pdfmetrics.registerFont(
    TTFont("MaShanZheng", "/usr/share/fonts/truetype/sand-box/google/Ma Shan Zheng/MaShanZheng-Regular.ttf")
)
pdfmetrics.registerFont(TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))


def draw_probe(c, y, left, digits, right, *, digit_font, digit_size, cjk_size, dy, label):
    x = 56.0
    c.setFont("MaShanZheng", cjk_size)
    c.drawString(x, y, left)
    x += c.stringWidth(left, "MaShanZheng", cjk_size) + 1.0
    c.setFont(digit_font, digit_size)
    c.drawString(x, y + dy, digits)
    x += c.stringWidth(digits, digit_font, digit_size) + 1.0
    c.setFont("MaShanZheng", cjk_size)
    c.drawString(x, y, right)
    c.setFont("DejaVuSans", 7)
    c.drawString(56, y - 20, label)


def main():
    out = Path("/workspace/MinerU/tests/assets/issue_5517_digit_drop_repro.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out), pagesize=(440, 380))
    c.setTitle("MinerU #5517 digit-drop minimal repro")
    c.setAuthor("Zhu Lei / kabishou11")
    c.setFont("DejaVuSans", 9)
    c.drawString(50, 350, "Minimal repro for #5517: native span fill drops off-axis mixed-font Nd digits")

    # dy negative in reportlab = lower on page. Extreme size/offset so Nd fails
    # mid-line check against a CJK-height layout span that still contains centers.
    draw_probe(
        c, 290, "至", "0.73", "个",
        digit_font="DejaVuSans", digit_size=4, cjk_size=18, dy=-8,
        label="L1 ASCII: DejaVu 4pt, baseline-8 (primary)",
    )
    draw_probe(
        c, 230, "至", "０．７３", "个",
        digit_font="MaShanZheng", digit_size=4, cjk_size=18, dy=-8,
        label="L2 fullwidth Nd: 4pt baseline-8",
    )
    draw_probe(
        c, 170, "大约", "0.73", "个百分点",
        digit_font="DejaVuSans", digit_size=4, cjk_size=18, dy=-8,
        label="L3 phrase ASCII baseline-8",
    )
    draw_probe(
        c, 110, "至", "0.73", "个",
        digit_font="DejaVuSans", digit_size=18, cjk_size=18, dy=0,
        label="L4 control same baseline/size",
    )
    c.showPage()
    c.save()
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
