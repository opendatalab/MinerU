import json
from io import BytesIO

from loguru import logger
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas

from .enum_class import BlockType, ContentType


def draw_bbox_without_number(i, bbox_list, page, c, rgb_config, fill_config):
    new_rgb = [float(color) / 255 for color in rgb_config]
    page_data = bbox_list[i]
    page_width, page_height = page.cropbox[2], page.cropbox[3]

    for bbox in page_data:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        rect = [bbox[0], page_height - bbox[3], width, height]  # Define the rectangle

        if fill_config:  # filled rectangle
            c.setFillColorRGB(new_rgb[0], new_rgb[1], new_rgb[2], 0.3)
            c.rect(rect[0], rect[1], rect[2], rect[3], stroke=0, fill=1)
        else:  # bounding box
            c.setStrokeColorRGB(new_rgb[0], new_rgb[1], new_rgb[2])
            c.rect(rect[0], rect[1], rect[2], rect[3], stroke=1, fill=0)
    return c


def draw_bbox_with_number(i, bbox_list, page, c, rgb_config, fill_config, draw_bbox=True):
    new_rgb = [float(color) / 255 for color in rgb_config]
    page_data = bbox_list[i]
    # 强制转换为 float
    page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])

    for j, bbox in enumerate(page_data):
        # 确保bbox的每个元素都是float
        x0, y0, x1, y1 = map(float, bbox)
        width = x1 - x0
        height = y1 - y0
        rect = [x0, page_height - y1, width, height]
        if draw_bbox:
            if fill_config:
                c.setFillColorRGB(*new_rgb, 0.3)
                c.rect(rect[0], rect[1], rect[2], rect[3], stroke=0, fill=1)
            else:
                c.setStrokeColorRGB(*new_rgb)
                c.rect(rect[0], rect[1], rect[2], rect[3], stroke=1, fill=0)
        c.setFillColorRGB(*new_rgb, 1.0)
        c.setFontSize(size=10)
        # 这里也要用float
        c.drawString(x1 + 2, page_height - y0 - 10, str(j + 1))

    return c


def draw_layout_bbox(pdf_info, pdf_bytes, out_path, filename):
    dropped_bbox_list = []
    tables_list, tables_body_list = [], []
    tables_caption_list, tables_footnote_list = [], []
    imgs_list, imgs_body_list, imgs_caption_list = [], [], []
    imgs_footnote_list = []
    titles_list = []
    texts_list = []
    interequations_list = []
    lists_list = []
    indexs_list = []
    for page in pdf_info:
        page_dropped_list = []
        tables, tables_body, tables_caption, tables_footnote = [], [], [], []
        imgs, imgs_body, imgs_caption, imgs_footnote = [], [], [], []
        titles = []
        texts = []
        interequations = []
        lists = []
        indices = []

        for dropped_bbox in page['discarded_blocks']:
            page_dropped_list.append(dropped_bbox['bbox'])
        dropped_bbox_list.append(page_dropped_list)
        for block in page["para_blocks"]:
            bbox = block["bbox"]
            if block["type"] == BlockType.TABLE:
                tables.append(bbox)
                for nested_block in block["blocks"]:
                    bbox = nested_block["bbox"]
                    if nested_block["type"] == BlockType.TABLE_BODY:
                        tables_body.append(bbox)
                    elif nested_block["type"] == BlockType.TABLE_CAPTION:
                        tables_caption.append(bbox)
                    elif nested_block["type"] == BlockType.TABLE_FOOTNOTE:
                        tables_footnote.append(bbox)
            elif block["type"] == BlockType.IMAGE:
                imgs.append(bbox)
                for nested_block in block["blocks"]:
                    bbox = nested_block["bbox"]
                    if nested_block["type"] == BlockType.IMAGE_BODY:
                        imgs_body.append(bbox)
                    elif nested_block["type"] == BlockType.IMAGE_CAPTION:
                        imgs_caption.append(bbox)
                    elif nested_block["type"] == BlockType.IMAGE_FOOTNOTE:
                        imgs_footnote.append(bbox)
            elif block["type"] == BlockType.TITLE:
                titles.append(bbox)
            elif block["type"] == BlockType.TEXT:
                texts.append(bbox)
            elif block["type"] == BlockType.INTERLINE_EQUATION:
                interequations.append(bbox)
            elif block["type"] == BlockType.LIST:
                lists.append(bbox)
            elif block["type"] == BlockType.INDEX:
                indices.append(bbox)

        tables_list.append(tables)
        tables_body_list.append(tables_body)
        tables_caption_list.append(tables_caption)
        tables_footnote_list.append(tables_footnote)
        imgs_list.append(imgs)
        imgs_body_list.append(imgs_body)
        imgs_caption_list.append(imgs_caption)
        imgs_footnote_list.append(imgs_footnote)
        titles_list.append(titles)
        texts_list.append(texts)
        interequations_list.append(interequations)
        lists_list.append(lists)
        indexs_list.append(indices)

    layout_bbox_list = []

    table_type_order = {"table_caption": 1, "table_body": 2, "table_footnote": 3}
    for page in pdf_info:
        page_block_list = []
        for block in page["para_blocks"]:
            if block["type"] in [
                BlockType.TEXT,
                BlockType.TITLE,
                BlockType.INTERLINE_EQUATION,
                BlockType.LIST,
                BlockType.INDEX,
            ]:
                bbox = block["bbox"]
                page_block_list.append(bbox)
            elif block["type"] in [BlockType.IMAGE]:
                for sub_block in block["blocks"]:
                    bbox = sub_block["bbox"]
                    page_block_list.append(bbox)
            elif block["type"] in [BlockType.TABLE]:
                sorted_blocks = sorted(block["blocks"], key=lambda x: table_type_order[x["type"]])
                for sub_block in sorted_blocks:
                    bbox = sub_block["bbox"]
                    page_block_list.append(bbox)

        layout_bbox_list.append(page_block_list)

    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        # 获取原始页面尺寸
        page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        packet = BytesIO()
        # 使用原始PDF的尺寸创建canvas
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        c = draw_bbox_without_number(i, dropped_bbox_list, page, c, [158, 158, 158], True)
        c = draw_bbox_without_number(i, tables_body_list, page, c, [204, 204, 0], True)
        c = draw_bbox_without_number(i, tables_caption_list, page, c, [255, 255, 102], True)
        c = draw_bbox_without_number(i, tables_footnote_list, page, c, [229, 255, 204], True)
        c = draw_bbox_without_number(i, imgs_body_list, page, c, [153, 255, 51], True)
        c = draw_bbox_without_number(i, imgs_caption_list, page, c, [102, 178, 255], True)
        c = draw_bbox_without_number(i, imgs_footnote_list, page, c, [255, 178, 102], True)
        c = draw_bbox_without_number(i, titles_list, page, c, [102, 102, 255], True)
        c = draw_bbox_without_number(i, texts_list, page, c, [153, 0, 76], True)
        c = draw_bbox_without_number(i, interequations_list, page, c, [0, 255, 0], True)
        c = draw_bbox_without_number(i, lists_list, page, c, [40, 169, 92], True)
        c = draw_bbox_without_number(i, indexs_list, page, c, [40, 169, 92], True)
        c = draw_bbox_with_number(i, layout_bbox_list, page, c, [255, 0, 0], False, draw_bbox=False)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # 添加检查确保overlay_pdf.pages不为空
        if len(overlay_pdf.pages) > 0:
            page.merge_page(overlay_pdf.pages[0])
        else:
            # 记录日志并继续处理下一个页面
            # logger.warning(f"layout.pdf: 第{i + 1}页未能生成有效的overlay PDF")
            pass

        output_pdf.add_page(page)

    # 保存结果
    with open(f"{out_path}/{filename}", "wb") as f:
        output_pdf.write(f)


def draw_span_bbox(pdf_info, pdf_bytes, out_path, filename):
    text_list = []
    inline_equation_list = []
    interline_equation_list = []
    image_list = []
    table_list = []
    dropped_list = []
    next_page_text_list = []
    next_page_inline_equation_list = []

    def get_span_info(span):
        if span['type'] == ContentType.TEXT:
            if span.get('cross_page', False):
                next_page_text_list.append(span['bbox'])
            else:
                page_text_list.append(span['bbox'])
        elif span['type'] == ContentType.INLINE_EQUATION:
            if span.get('cross_page', False):
                next_page_inline_equation_list.append(span['bbox'])
            else:
                page_inline_equation_list.append(span['bbox'])
        elif span['type'] == ContentType.INTERLINE_EQUATION:
            page_interline_equation_list.append(span['bbox'])
        elif span['type'] == ContentType.IMAGE:
            page_image_list.append(span['bbox'])
        elif span['type'] == ContentType.TABLE:
            page_table_list.append(span['bbox'])

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
        for block in page['discarded_blocks']:
            if block['type'] == BlockType.DISCARDED:
                for line in block['lines']:
                    for span in line['spans']:
                        page_dropped_list.append(span['bbox'])
        dropped_list.append(page_dropped_list)
        # 构造其余useful_list
        # for block in page['para_blocks']:  # span直接用分段合并前的结果就可以
        for block in page['preproc_blocks']:
            if block['type'] in [
                BlockType.TEXT,
                BlockType.TITLE,
                BlockType.INTERLINE_EQUATION,
                BlockType.LIST,
                BlockType.INDEX,
            ]:
                for line in block['lines']:
                    for span in line['spans']:
                        get_span_info(span)
            elif block['type'] in [BlockType.IMAGE, BlockType.TABLE]:
                for sub_block in block['blocks']:
                    for line in sub_block['lines']:
                        for span in line['spans']:
                            get_span_info(span)
        text_list.append(page_text_list)
        inline_equation_list.append(page_inline_equation_list)
        interline_equation_list.append(page_interline_equation_list)
        image_list.append(page_image_list)
        table_list.append(page_table_list)

    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        # 获取原始页面尺寸
        page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        packet = BytesIO()
        # 使用原始PDF的尺寸创建canvas
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        # 获取当前页面的数据
        draw_bbox_without_number(i, text_list, page, c,[255, 0, 0], False)
        draw_bbox_without_number(i, inline_equation_list, page, c, [0, 255, 0], False)
        draw_bbox_without_number(i, interline_equation_list, page, c, [0, 0, 255], False)
        draw_bbox_without_number(i, image_list, page, c, [255, 204, 0], False)
        draw_bbox_without_number(i, table_list, page, c, [204, 0, 255], False)
        draw_bbox_without_number(i, dropped_list, page, c, [158, 158, 158], False)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # 添加检查确保overlay_pdf.pages不为空
        if len(overlay_pdf.pages) > 0:
            page.merge_page(overlay_pdf.pages[0])
        else:
            # 记录日志并继续处理下一个页面
            # logger.warning(f"span.pdf: 第{i + 1}页未能生成有效的overlay PDF")
            pass

        output_pdf.add_page(page)

    # Save the PDF
    with open(f"{out_path}/{filename}", "wb") as f:
        output_pdf.write(f)


if __name__ == "__main__":
    # 读取PDF文件
    pdf_path = "examples/demo1.pdf"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # 从json文件读取pdf_info

    json_path = "examples/demo1_1746005777.0863056_middle.json"
    with open(json_path, "r", encoding="utf-8") as f:
        pdf_ann = json.load(f)
    pdf_info = pdf_ann["pdf_info"]
    # 调用可视化函数,输出到examples目录
    draw_layout_bbox(pdf_info, pdf_bytes, "examples", "output_with_layout.pdf")
