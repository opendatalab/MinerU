import time
from loguru import logger
from magic_pdf.libs.commons import (
    fitz,
    get_delta_time,
    get_docx_model_output,
)
from magic_pdf.libs.convert_utils import dict_to_list
from magic_pdf.libs.coordinate_transform import get_scale_ratio
from magic_pdf.libs.drop_tag import DropTag
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.para.para_split import para_split
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component
from magic_pdf.pre_proc.detect_footer_by_model import parse_footers
from magic_pdf.pre_proc.detect_footnote import parse_footnotes_by_model
from magic_pdf.pre_proc.detect_header import parse_headers
from magic_pdf.pre_proc.detect_page_number import parse_pageNos
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.ocr_detect_layout import layout_detect
from magic_pdf.pre_proc.ocr_dict_merge import (
    merge_spans_to_line_by_layout, merge_lines_to_block,
)
from magic_pdf.pre_proc.ocr_span_list_modify import remove_spans_by_bboxes, remove_overlaps_min_spans, \
    adjust_bbox_for_standalone_block, modify_y_axis, modify_inline_equation, get_qa_need_list, \
    remove_spans_by_bboxes_dict
from magic_pdf.pre_proc.remove_bbox_overlap import remove_overlap_between_bbox_for_span


def parse_pdf_by_ocr(
        pdf_bytes,
        pdf_model_output,
        imageWriter,
        start_page_id=0,
        end_page_id=None,
        debug_mode=False,
):
    pdf_bytes_md5 = compute_md5(pdf_bytes)

    pdf_docs = fitz.open("pdf", pdf_bytes)
    # 初始化空的pdf_info_dict
    pdf_info_dict = {}

    start_time = time.time()

    end_page_id = end_page_id if end_page_id else len(pdf_docs) - 1
    for page_id in range(start_page_id, end_page_id + 1):

        # 获取当前页的page对象
        page = pdf_docs[page_id]
        # 获取当前页的宽高
        page_w = page.rect.width
        page_h = page.rect.height

        if debug_mode:
            time_now = time.time()
            logger.info(
                f"page_id: {page_id}, last_page_cost_time: {get_delta_time(start_time)}"
            )
            start_time = time_now

        # 获取当前页的模型数据
        ocr_page_info = get_docx_model_output(
            pdf_model_output, page_id
        )

        """从json中获取每页的页码、页眉、页脚的bbox"""
        page_no_bboxes = parse_pageNos(page_id, page, ocr_page_info)
        header_bboxes = parse_headers(page_id, page, ocr_page_info)
        footer_bboxes = parse_footers(page_id, page, ocr_page_info)
        footnote_bboxes = parse_footnotes_by_model(page_id, page, ocr_page_info, debug_mode=debug_mode)

        # 构建需要remove的bbox字典
        need_remove_spans_bboxes_dict = {
            DropTag.PAGE_NUMBER: page_no_bboxes,
            DropTag.HEADER: header_bboxes,
            DropTag.FOOTER: footer_bboxes,
            DropTag.FOOTNOTE: footnote_bboxes,
        }

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
                # 删除高度或者宽度为0的spans
                if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
                    continue
                """要删除的"""
                #  3: 'header',      # 页眉
                #  4: 'page number', # 页码
                #  5: 'footnote',    # 脚注
                #  6: 'footer',      # 页脚
                """当成span拼接的"""
                #  1: 'image', # 图片
                #  7: 'table',       # 表格
                #  13: 'inline_equation',     # 行内公式
                #  14: 'interline_equation',      # 行间公式
                #  15: 'text',      # ocr识别文本
                """layout信息"""
                #  11: 'full column',   # 单栏
                #  12: 'sub column',    # 多栏
                span = {
                    "bbox": bbox,
                }
                if category_id == 1:
                    span["type"] = ContentType.Image

                elif category_id == 7:
                    span["type"] = ContentType.Table

                elif category_id == 13:
                    span["content"] = layout_det["latex"]
                    span["type"] = ContentType.InlineEquation
                elif category_id == 14:
                    span["content"] = layout_det["latex"]
                    span["type"] = ContentType.InterlineEquation
                elif category_id == 15:
                    span["content"] = layout_det["text"]
                    span["type"] = ContentType.Text
                # print(span)
                spans.append(span)
            else:
                continue

        '''删除重叠spans中较小的那些'''
        spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

        '''
        删除remove_span_block_bboxes中的bbox
        并增加drop相关数据
        '''
        spans, dropped_spans_by_removed_bboxes = remove_spans_by_bboxes_dict(spans, need_remove_spans_bboxes_dict)

        '''对image和table截图'''
        spans = ocr_cut_image_and_table(spans, page, page_id, pdf_bytes_md5, imageWriter)

        '''行内公式调整, 高度调整至与同行文字高度一致(优先左侧, 其次右侧)'''
        displayed_list = []
        text_inline_lines = []
        modify_y_axis(spans, displayed_list, text_inline_lines)

        '''模型识别错误的行间公式, type类型转换成行内公式'''
        spans = modify_inline_equation(spans, displayed_list, text_inline_lines)

        '''bbox去除粘连'''
        spans = remove_overlap_between_bbox_for_span(spans)
        '''
        对tpye=["interline_equation", "image", "table"]进行额外处理,
        如果左边有字的话,将该span的bbox中y0调整至不高于文字的y0
        '''
        spans = adjust_bbox_for_standalone_block(spans)

        '''从ocr_page_info中解析layout信息(按自然阅读方向排序,并修复重叠和交错的bad case)'''
        layout_bboxes, layout_tree = layout_detect(ocr_page_info['subfield_dets'], page, ocr_page_info)

        '''将spans合并成line(在layout内,从上到下,从左到右)'''
        lines, dropped_spans_by_layout = merge_spans_to_line_by_layout(spans, layout_bboxes)

        '''将lines合并成block'''
        blocks = merge_lines_to_block(lines)

        '''获取QA需要外置的list'''
        images, tables, interline_equations, inline_equations = get_qa_need_list(blocks)

        '''drop的span_list合并'''
        dropped_spans = []
        dropped_spans.extend(dropped_spans_by_span_overlap)
        dropped_spans.extend(dropped_spans_by_removed_bboxes)
        dropped_spans.extend(dropped_spans_by_layout)

        dropped_text_block = []
        dropped_image_block = []
        dropped_table_block = []
        dropped_equation_block = []
        for span in dropped_spans:
            # drop出的spans进行分类
            if span['type'] == ContentType.Text:
                dropped_text_block.append(span)
            elif span['type'] == ContentType.Image:
                dropped_image_block.append(span)
            elif span['type'] == ContentType.Table:
                dropped_table_block.append(span)
            elif span['type'] in [ContentType.InlineEquation, ContentType.InterlineEquation]:
                dropped_equation_block.append(span)

        '''构造pdf_info_dict'''
        page_info = ocr_construct_page_component(blocks, layout_bboxes, page_id, page_w, page_h, layout_tree,
                                                 images, tables, interline_equations, inline_equations,
                                                 dropped_text_block, dropped_image_block, dropped_table_block,
                                                 dropped_equation_block,
                                                 need_remove_spans_bboxes_dict)
        pdf_info_dict[f"page_{page_id}"] = page_info

    """分段"""

    para_split(pdf_info_dict, debug_mode=debug_mode)

    """dict转list"""
    pdf_info_list = dict_to_list(pdf_info_dict)
    new_pdf_info_dict = {
        "pdf_info": pdf_info_list,
    }

    return new_pdf_info_dict
