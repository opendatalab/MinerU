from libs.commons import fitz  # PyMuPDF

# PDF文件路径
pdf_path = "D:\\project\\20231108code-clean\\code-clean\\tmp\\unittest\\download-pdfs\\scihub\\scihub_53700000\\libgen.scimag53724000-53724999.zip_10.1097\\00129191-200509000-00018.pdf"

doc = fitz.open(pdf_path)  # Open the PDF
# 你的数据
data = [[[-2, 0, 603, 80, 24]], [[-3, 0, 602, 80, 24]]]

# 对每个页面进行处理
for i, page in enumerate(doc):
    # 获取当前页面的数据
    page_data = data[i]
    for img in page_data:
        x0, y0, x1, y1, _ = img
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # Define the rectangle
        page.draw_rect(rect_coords, color=(1, 0, 0), fill=None, width=1.5, overlay=True)  # Draw the rectangle

# Save the PDF
doc.save("D:\\project\\20231108code-clean\\code-clean\\tmp\\unittest\\download-pdfs\\scihub\\scihub_53700000\\libgen.scimag53724000-53724999.zip_10.1097\\00129191-200509000-00018_new.pdf")