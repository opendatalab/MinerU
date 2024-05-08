from magic_pdf.libs.Constants import CROSS_PAGE
from magic_pdf.libs.commons import fitz  # PyMuPDF
from magic_pdf.libs.ocr_content_type import ContentType, BlockType


def draw_bbox_without_number(i, bbox_list, page, rgb_config, fill_config):
    new_rgb = []
    for item in rgb_config:
        item = float(item) / 255
        new_rgb.append(item)
    page_data = bbox_list[i]
    for bbox in page_data:
        x0, y0, x1, y1 = bbox
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # Define the rectangle
        if fill_config:
            page.draw_rect(
                rect_coords,
                color=None,
                fill=new_rgb,
                fill_opacity=0.3,
                width=0.5,
                overlay=True,
            )  # Draw the rectangle
        else:
            page.draw_rect(
                rect_coords,
                color=new_rgb,
                fill=None,
                fill_opacity=1,
                width=0.5,
                overlay=True,
            )  # Draw the rectangle


def draw_bbox_with_number(i, bbox_list, page, rgb_config, fill_config):
    new_rgb = []
    for item in rgb_config:
        item = float(item) / 255
        new_rgb.append(item)
    page_data = bbox_list[i]
    for j, bbox in enumerate(page_data):
        x0, y0, x1, y1 = bbox
        rect_coords = fitz.Rect(x0, y0, x1, y1)  # Define the rectangle
        if fill_config:
            page.draw_rect(
                rect_coords,
                color=None,
                fill=new_rgb,
                fill_opacity=0.3,
                width=0.5,
                overlay=True,
            )  # Draw the rectangle
        else:
            page.draw_rect(
                rect_coords,
                color=new_rgb,
                fill=None,
                fill_opacity=1,
                width=0.5,
                overlay=True,
            )  # Draw the rectangle
        page.insert_text(
            (x0, y0 + 10), str(j + 1), fontsize=10, color=new_rgb
        )  # Insert the index in the top left corner of the rectangle


def draw_layout_bbox(pdf_info, pdf_bytes, out_path):
    layout_bbox_list = []
    dropped_bbox_list = []
    tables_list, tables_body_list, tables_caption_list, tables_footnote_list = [], [], [], []
    imgs_list, imgs_body_list, imgs_caption_list = [], [], []
    titles_list = []
    texts_list = []
    interequations_list = []
    for page in pdf_info:
        page_layout_list = []
        page_dropped_list = []
        tables, tables_body, tables_caption, tables_footnote = [], [], [], []
        imgs, imgs_body, imgs_caption = [], [], []
        titles = []
        texts = []
        interequations = []
        for layout in page["layout_bboxes"]:
            page_layout_list.append(layout["layout_bbox"])
        layout_bbox_list.append(page_layout_list)
        for dropped_bbox in page["discarded_blocks"]:
            page_dropped_list.append(dropped_bbox["bbox"])
        dropped_bbox_list.append(page_dropped_list)
        for block in page["para_blocks"]:
            bbox = block["bbox"]
            if block["type"] == BlockType.Table:
                tables.append(bbox)
                for nested_block in block["blocks"]:
                    bbox = nested_block["bbox"]
                    if nested_block["type"] == BlockType.TableBody:
                        tables_body.append(bbox)
                    elif nested_block["type"] == BlockType.TableCaption:
                        tables_caption.append(bbox)
                    elif nested_block["type"] == BlockType.TableFootnote:
                        tables_footnote.append(bbox)
            elif block["type"] == BlockType.Image:
                imgs.append(bbox)
                for nested_block in block["blocks"]:
                    bbox = nested_block["bbox"]
                    if nested_block["type"] == BlockType.ImageBody:
                        imgs_body.append(bbox)
                    elif nested_block["type"] == BlockType.ImageCaption:
                        imgs_caption.append(bbox)
            elif block["type"] == BlockType.Title:
                titles.append(bbox)
            elif block["type"] == BlockType.Text:
                texts.append(bbox)
            elif block["type"] == BlockType.InterlineEquation:
                interequations.append(bbox)
        tables_list.append(tables)
        tables_body_list.append(tables_body)
        tables_caption_list.append(tables_caption)
        tables_footnote_list.append(tables_footnote)
        imgs_list.append(imgs)
        imgs_body_list.append(imgs_body)
        imgs_caption_list.append(imgs_caption)
        titles_list.append(titles)
        texts_list.append(texts)
        interequations_list.append(interequations)

    pdf_docs = fitz.open("pdf", pdf_bytes)
    for i, page in enumerate(pdf_docs):
        draw_bbox_with_number(i, layout_bbox_list, page, [255, 0, 0], False)
        draw_bbox_without_number(i, dropped_bbox_list, page, [158, 158, 158], True)
        draw_bbox_without_number(i, tables_list, page, [153, 153, 0], True)  # color !
        draw_bbox_without_number(i, tables_body_list, page, [204, 204, 0], True)
        draw_bbox_without_number(i, tables_caption_list, page, [255, 255, 102], True)
        draw_bbox_without_number(i, tables_footnote_list, page, [229, 255, 204], True)
        draw_bbox_without_number(i, imgs_list, page, [51, 102, 0], True)
        draw_bbox_without_number(i, imgs_body_list, page, [153, 255, 51], True)
        draw_bbox_without_number(i, imgs_caption_list, page, [102, 178, 255], True)
        draw_bbox_without_number(i, titles_list, page, [102, 102, 255], True)
        draw_bbox_without_number(i, texts_list, page, [153, 0, 76], True)
        draw_bbox_without_number(i, interequations_list, page, [0, 255, 0], True)

    # Save the PDF
    pdf_docs.save(f"{out_path}/layout.pdf")


def draw_span_bbox(pdf_info, pdf_bytes, out_path):
    text_list = []
    inline_equation_list = []
    interline_equation_list = []
    image_list = []
    table_list = []
    dropped_list = []
    next_page_text_list = []
    next_page_inline_equation_list = []

    def get_span_info(span):
        if span["type"] == ContentType.Text:
            if span.get(CROSS_PAGE, False):
                next_page_text_list.append(span["bbox"])
            else:
                page_text_list.append(span["bbox"])
        elif span["type"] == ContentType.InlineEquation:
            if span.get(CROSS_PAGE, False):
                next_page_inline_equation_list.append(span["bbox"])
            else:
                page_inline_equation_list.append(span["bbox"])
        elif span["type"] == ContentType.InterlineEquation:
            page_interline_equation_list.append(span["bbox"])
        elif span["type"] == ContentType.Image:
            page_image_list.append(span["bbox"])
        elif span["type"] == ContentType.Table:
            page_table_list.append(span["bbox"])

    for page in pdf_info:
        page_text_list = []
        page_inline_equation_list = []
        page_interline_equation_list = []
        page_image_list = []
        page_table_list = []
        page_dropped_list = []

        # 将跨页的span放到移动到下一页的列表中
        if len(next_page_text_list) > 0:
            page_text_list.extend(next_page_text_list)
            next_page_text_list.clear()
        if len(next_page_inline_equation_list) > 0:
            page_inline_equation_list.extend(next_page_inline_equation_list)
            next_page_inline_equation_list.clear()

        # 构造dropped_list
        for block in page["discarded_blocks"]:
            if block["type"] == BlockType.Discarded:
                for line in block["lines"]:
                    for span in line["spans"]:
                        page_dropped_list.append(span["bbox"])
        dropped_list.append(page_dropped_list)
        # 构造其余useful_list
        for block in page["para_blocks"]:
            if block["type"] in [
                BlockType.Text,
                BlockType.Title,
                BlockType.InterlineEquation,
            ]:
                for line in block["lines"]:
                    for span in line["spans"]:
                        get_span_info(span)
            elif block["type"] in [BlockType.Image, BlockType.Table]:
                for sub_block in block["blocks"]:
                    for line in sub_block["lines"]:
                        for span in line["spans"]:
                            get_span_info(span)
        text_list.append(page_text_list)
        inline_equation_list.append(page_inline_equation_list)
        interline_equation_list.append(page_interline_equation_list)
        image_list.append(page_image_list)
        table_list.append(page_table_list)
    pdf_docs = fitz.open("pdf", pdf_bytes)
    for i, page in enumerate(pdf_docs):
        # 获取当前页面的数据
        draw_bbox_without_number(i, text_list, page, [255, 0, 0], False)
        draw_bbox_without_number(i, inline_equation_list, page, [0, 255, 0], False)
        draw_bbox_without_number(i, interline_equation_list, page, [0, 0, 255], False)
        draw_bbox_without_number(i, image_list, page, [255, 204, 0], False)
        draw_bbox_without_number(i, table_list, page, [204, 0, 255], False)
        draw_bbox_without_number(i, dropped_list, page, [158, 158, 158], False)

    # Save the PDF
    pdf_docs.save(f"{out_path}/spans.pdf")
