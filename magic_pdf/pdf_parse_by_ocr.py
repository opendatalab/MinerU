import json
import os
import time

from loguru import logger

from demo.draw_bbox import draw_layout_bbox, draw_text_bbox
from magic_pdf.libs.commons import (
    read_file,
    join_path,
    fitz,
    get_img_s3_client,
    get_delta_time,
    get_docx_model_output,
)
from magic_pdf.libs.coordinate_transform import get_scale_ratio
from magic_pdf.libs.safe_filename import sanitize_filename
from magic_pdf.pre_proc.detect_footer_by_model import parse_footers
from magic_pdf.pre_proc.detect_footnote import parse_footnotes_by_model
from magic_pdf.pre_proc.detect_header import parse_headers
from magic_pdf.pre_proc.detect_page_number import parse_pageNos
from magic_pdf.pre_proc.ocr_cut_image import cut_image_and_table
from magic_pdf.pre_proc.ocr_detect_layout import layout_detect
from magic_pdf.pre_proc.ocr_dict_merge import (
    remove_overlaps_min_spans,
    merge_spans_to_line_by_layout,
)
from magic_pdf.pre_proc.ocr_remove_spans import remove_spans_by_bboxes
from magic_pdf.pre_proc.remove_bbox_overlap import remove_overlap_between_bbox


def construct_page_component(page_id, blocks, layout_bboxes):
    return_dict = {
        "preproc_blocks": blocks,
        "page_idx": page_id,
        "layout_bboxes": layout_bboxes,
    }
    return return_dict


def parse_pdf_by_ocr(
    pdf_path,
    s3_pdf_profile,
    pdf_model_output,
    save_path,
    book_name,
    pdf_model_profile=None,
    image_s3_config=None,
    start_page_id=0,
    end_page_id=None,
    debug_mode=False,
):
    pdf_bytes = read_file(pdf_path, s3_pdf_profile)
    save_tmp_path = os.path.join(os.path.dirname(__file__), "../..", "tmp", "unittest")
    book_name = sanitize_filename(book_name)
    md_bookname_save_path = ""
    if debug_mode:
        save_path = join_path(save_tmp_path, "md")
        pdf_local_path = join_path(save_tmp_path, "download-pdfs", book_name)

        if not os.path.exists(os.path.dirname(pdf_local_path)):
            # 如果目录不存在，创建它
            os.makedirs(os.path.dirname(pdf_local_path))

        md_bookname_save_path = join_path(save_tmp_path, "md", book_name)
        if not os.path.exists(md_bookname_save_path):
            # 如果目录不存在，创建它
            os.makedirs(md_bookname_save_path)

        with open(pdf_local_path + ".pdf", "wb") as pdf_file:
            pdf_file.write(pdf_bytes)

    pdf_docs = fitz.open("pdf", pdf_bytes)
    # 初始化空的pdf_info_dict
    pdf_info_dict = {}
    img_s3_client = get_img_s3_client(save_path, image_s3_config)

    start_time = time.time()

    remove_bboxes = []

    end_page_id = end_page_id if end_page_id else len(pdf_docs) - 1
    for page_id in range(start_page_id, end_page_id + 1):

        # 获取当前页的page对象
        page = pdf_docs[page_id]

        if debug_mode:
            time_now = time.time()
            logger.info(
                f"page_id: {page_id}, last_page_cost_time: {get_delta_time(start_time)}"
            )
            start_time = time_now

        # 获取当前页的模型数据
        ocr_page_info = get_docx_model_output(
            pdf_model_output, pdf_model_profile, page_id
        )

        """从json中获取每页的页码、页眉、页脚的bbox"""
        page_no_bboxes = parse_pageNos(page_id, page, ocr_page_info)
        header_bboxes = parse_headers(page_id, page, ocr_page_info)
        footer_bboxes = parse_footers(page_id, page, ocr_page_info)
        footnote_bboxes = parse_footnotes_by_model(
            page_id, page, ocr_page_info, md_bookname_save_path, debug_mode=debug_mode
        )

        # 构建需要remove的bbox列表
        need_remove_spans_bboxes = []
        need_remove_spans_bboxes.extend(page_no_bboxes)
        need_remove_spans_bboxes.extend(header_bboxes)
        need_remove_spans_bboxes.extend(footer_bboxes)
        need_remove_spans_bboxes.extend(footnote_bboxes)

        layout_dets = ocr_page_info["layout_dets"]
        spans = []

        # 计算模型坐标和pymu坐标的缩放比例
        horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(
            ocr_page_info, page
        )

        for layout_det in layout_dets:
            category_id = layout_det["category_id"]
            allow_category_id_list = [1, 7, 13, 14, 15]
            if category_id in allow_category_id_list:
                x0, y0, _, _, x1, y1, _, _ = layout_det["poly"]
                bbox = [
                    int(x0 / horizontal_scale_ratio),
                    int(y0 / vertical_scale_ratio),
                    int(x1 / horizontal_scale_ratio),
                    int(y1 / vertical_scale_ratio),
                ]
                """要删除的"""
                #  3: 'header',      # 页眉
                #  4: 'page number', # 页码
                #  5: 'footnote',    # 脚注
                #  6: 'footer',      # 页脚
                """当成span拼接的"""
                #  1: 'image', # 图片
                #  7: 'table',       # 表格
                #  13: 'inline_equation',     # 行内公式
                #  14: 'displayed_equation',      # 行间公式
                #  15: 'text',      # ocr识别文本
                """layout信息"""
                #  11: 'full column',   # 单栏
                #  12: 'sub column',    # 多栏
                span = {
                    "bbox": bbox,
                }
                if category_id == 1:
                    span["type"] = "image"

                elif category_id == 7:
                    span["type"] = "table"

                elif category_id == 13:
                    span["content"] = layout_det["latex"]
                    span["type"] = "inline_equation"
                elif category_id == 14:
                    span["content"] = layout_det["latex"]
                    span["type"] = "displayed_equation"
                elif category_id == 15:
                    span["content"] = layout_det["text"]
                    span["type"] = "text"
                # print(span)
                spans.append(span)
            else:
                continue

        # 删除重叠spans中较小的那些
        spans = remove_overlaps_min_spans(spans)

        # 删除remove_span_block_bboxes中的bbox
        spans = remove_spans_by_bboxes(spans, need_remove_spans_bboxes)

        # 对image和table截图
        spans = cut_image_and_table(spans, page, page_id, book_name, save_path)

        # 行内公式调整, 高度调整至与同行文字高度一致(优先左侧, 其次右侧)

        # 模型识别错误的行间公式, type类型转换成行内公式

        # bbox去除粘连
        spans = remove_overlap_between_bbox(spans)

        # 对tpye=["displayed_equation", "image", "table"]进行额外处理,如果左边有字的话,将该span的bbox中y0调整至不高于文字的y0

        # 从ocr_page_info中解析layout信息(按自然阅读方向排序,并修复重叠和交错的bad case)
        layout_bboxes = layout_detect(
            ocr_page_info["subfield_dets"], page, ocr_page_info
        )

        # 将spans合并成line(在layout内,从上到下,从左到右)
        lines = merge_spans_to_line_by_layout(spans, layout_bboxes)

        # 目前不做block拼接,先做个结构,每个block中只有一个line,block的bbox就是line的bbox
        blocks = []
        for line in lines:
            blocks.append(
                {
                    "bbox": line["bbox"],
                    "lines": [line],
                }
            )

        # 构造pdf_info_dict
        page_info = construct_page_component(page_id, blocks, layout_bboxes)
        pdf_info_dict[f"page_{page_id}"] = page_info

    # 在测试时,保存调试信息
    if debug_mode:
        params_file_save_path = join_path(
            save_tmp_path, "md", book_name, "preproc_out.json"
        )
        with open(params_file_save_path, "w", encoding="utf-8") as f:
            json.dump(pdf_info_dict, f, ensure_ascii=False, indent=4)
        # drow_bbox
        draw_layout_bbox(pdf_info_dict, pdf_path, md_bookname_save_path)
        draw_text_bbox(pdf_info_dict, pdf_path, md_bookname_save_path)

    return pdf_info_dict
