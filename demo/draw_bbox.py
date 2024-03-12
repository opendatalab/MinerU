from pathlib import Path

from magic_pdf.libs.commons import fitz, join_path  # PyMuPDF
from magic_pdf.pdf_parse_by_ocr import parse_pdf_by_ocr
import json
import os




def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


# PDF文件路径
pdf_path = "D:\\projects\\Magic-PDF\\ocr_demo\\ocr_0_org.pdf"

doc = fitz.open(pdf_path)  # Open the PDF
# 你的数据
data = [[[-2, 0, 603, 80, 24]], [[-3, 0, 602, 80, 24]]]
ocr_json_file_path = r"D:\projects\Magic-PDF\ocr_demo\ocr_0.json"
ocr_pdf_info = read_json_file(ocr_json_file_path)

pth = Path(ocr_json_file_path)
book_name = pth.name
save_tmp_path = os.path.join(os.path.dirname(__file__), "../..", "tmp", "unittest")
save_path = join_path(save_tmp_path, "md")

pdf_info_dict = parse_pdf_by_ocr(
            pdf_path,
            None,
            ocr_pdf_info,
            save_path,
            book_name,
            debug_mode=True)
data_list = []
for page in pdf_info_dict.values():
    page_list = []
    blocks = page.get("preproc_blocks")
    for block in blocks:
        lines = block.get("lines")
        for line in lines:
            spans = line.get("spans")
            for span in spans:
                page_list.append(span["bbox"])
    data_list.append(page_list)
# 对每个页面进行处理
for i, page in enumerate(doc):
    # 获取当前页面的数据
    page_data = data_list[i]
    for img in page_data:
        x0, y0, x1, y1 = img
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # Define the rectangle
        page.draw_rect(rect_coords, color=(1, 0, 0), fill=None, width=1.5, overlay=True)  # Draw the rectangle

# Save the PDF
doc.save("D:\\projects\\Magic-PDF\\ocr_demo\\ocr_0_new1.pdf")