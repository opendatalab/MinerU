from magic_pdf.libs.commons import fitz  # PyMuPDF

def draw_bbox(i, bbox_list, page, rgb_config):
    new_rgb = []
    for item in rgb_config:
        item = float(item) / 255
        new_rgb.append(item)
    page_data = bbox_list[i]
    for bbox in page_data:
        x0, y0, x1, y1 = bbox
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # Define the rectangle
        page.draw_rect(rect_coords, color=new_rgb, fill=None, width=0.5, overlay=True)  # Draw the rectangle


def draw_layout_bbox(pdf_info_dict, input_path, out_path):
    layout_bbox_list = []
    for page in pdf_info_dict.values():
        page_list = []
        for layout in page['layout_bboxes']:
            page_list.append(layout['layout_bbox'])
        layout_bbox_list.append(page_list)

    doc = fitz.open(input_path)
    for i, page in enumerate(doc):
        # 获取当前页面的数据
        page_data = layout_bbox_list[i]
        for j, bbox in enumerate(page_data):
            x0, y0, x1, y1 = bbox
            rect_coords = fitz.Rect(x0, y0, x1, y1)  # Define the rectangle
            page.draw_rect(rect_coords, color=(1, 0, 0), fill=None, width=0.5, overlay=True)  # Draw the rectangle
            page.insert_text((x0, y0), str(j + 1), fontsize=10, color=(1, 0, 0))  # Insert the index at the top left corner of the rectangle
    # Save the PDF
    doc.save(f"{out_path}/layout.pdf")

def draw_text_bbox(pdf_info_dict, input_path, out_path):
    text_list = []
    inline_equation_list = []
    displayed_equation_list = []
    for page in pdf_info_dict.values():
        page_text_list = []
        page_inline_equation_list = []
        page_displayed_equation_list = []
        for block in page['preproc_blocks']:
            for line in block['lines']:
                for span in line['spans']:
                    if span['type'] == 'text':
                        page_text_list.append(span['bbox'])
                    elif span['type'] == 'inline_equation':
                        page_inline_equation_list.append(span['bbox'])
                    elif span['type'] == 'displayed_equation':
                        page_displayed_equation_list.append(span['bbox'])
        text_list.append(page_text_list)
        inline_equation_list.append(page_inline_equation_list)
        displayed_equation_list.append(page_displayed_equation_list)

    doc = fitz.open(input_path)
    for i, page in enumerate(doc):
        # 获取当前页面的数据
        draw_bbox(i, text_list, page, [255, 0, 0])

        draw_bbox(i, inline_equation_list, page, [0, 255, 0])

        draw_bbox(i, displayed_equation_list, page, [0, 0, 255])

    # Save the PDF
    doc.save(f"{out_path}/text.pdf")
