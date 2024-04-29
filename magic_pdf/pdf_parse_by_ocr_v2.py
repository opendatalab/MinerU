import time

from loguru import logger

from magic_pdf.layout.layout_sort import get_bboxes_layout, LAYOUT_UNPROC, get_columns_cnt_of_layout
from magic_pdf.libs.convert_utils import dict_to_list
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.libs.commons import fitz, get_delta_time
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component_v2
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split
from magic_pdf.pre_proc.ocr_dict_merge import sort_blocks_by_layout, fill_spans_in_blocks, fix_block_spans
from magic_pdf.pre_proc.ocr_span_list_modify import remove_overlaps_min_spans, get_qa_need_list_v2
# from magic_pdf.para.para_split import para_split
from magic_pdf.para.para_split_v2 import para_split
from magic_pdf.pre_proc.resolve_bbox_conflict import check_useful_block_horizontal_overlap


def parse_pdf_by_ocr(pdf_bytes,
                     model_list,
                     imageWriter,
                     start_page_id=0,
                     end_page_id=None,
                     debug_mode=False,
                     ):
    pdf_bytes_md5 = compute_md5(pdf_bytes)
    pdf_docs = fitz.open("pdf", pdf_bytes)

    '''初始化空的pdf_info_dict'''
    pdf_info_dict = {}

    '''用model_list和docs对象初始化magic_model'''
    magic_model = MagicModel(model_list, pdf_docs)

    '''根据输入的起始范围解析pdf'''
    end_page_id = end_page_id if end_page_id else len(pdf_docs) - 1

    '''初始化启动时间'''
    start_time = time.time()

    for page_id in range(start_page_id, end_page_id + 1):

        '''debug时输出每页解析的耗时'''
        if debug_mode:
            time_now = time.time()
            logger.info(
                f"page_id: {page_id}, last_page_cost_time: {get_delta_time(start_time)}"
            )
            start_time = time_now

        '''从magic_model对象中获取后面会用到的区块信息'''
        img_blocks = magic_model.get_imgs(page_id)
        table_blocks = magic_model.get_tables(page_id)
        discarded_blocks = magic_model.get_discarded(page_id)
        text_blocks = magic_model.get_text_blocks(page_id)
        title_blocks = magic_model.get_title_blocks(page_id)
        inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations(page_id)

        page_w, page_h = magic_model.get_page_size(page_id)

        '''将所有区块的bbox整理到一起'''
        all_bboxes = ocr_prepare_bboxes_for_layout_split(
            img_blocks, table_blocks, discarded_blocks, text_blocks, title_blocks,
            interline_equations, page_w, page_h)

        """在切分之前，先检查一下bbox是否有左右重叠的情况，如果有，那么就认为这个pdf暂时没有能力处理好，这种左右重叠的情况大概率是由于pdf里的行间公式、表格没有被正确识别出来造成的 """
        useful_blocks = []
        for bbox in all_bboxes:
            useful_blocks.append({
                "bbox": bbox[:4]
            })
        is_useful_block_horz_overlap = check_useful_block_horizontal_overlap(useful_blocks)
        if is_useful_block_horz_overlap:
            logger.warning(
                f"pdf: {pdf_bytes_md5}, skip this page, page_id: {page_id}, reason: {DropReason.TEXT_BLCOK_HOR_OVERLAP}")
            continue

        '''根据区块信息计算layout'''
        page_boundry = [0, 0, page_w, page_h]
        layout_bboxes, layout_tree = get_bboxes_layout(all_bboxes, page_boundry, page_id)

        if len(text_blocks) > 0 and len(all_bboxes) > 0 and len(layout_bboxes) == 0:
            logger.warning(
                f"pdf: {pdf_bytes_md5}, skip this page, page_id: {page_id}, reason: {DropReason.CAN_NOT_DETECT_PAGE_LAYOUT}")
            continue

        """以下去掉复杂的布局和超过2列的布局"""
        if any([lay["layout_label"] == LAYOUT_UNPROC for lay in layout_bboxes]):  # 复杂的布局
            logger.warning(
                f"pdf: {pdf_bytes_md5}, skip this page, page_id: {page_id}, reason: {DropReason.COMPLICATED_LAYOUT}")
            continue

        layout_column_width = get_columns_cnt_of_layout(layout_tree)
        if layout_column_width > 2:  # 去掉超过2列的布局pdf
            logger.warning(
                f"pdf: {pdf_bytes_md5}, skip this page, page_id: {page_id}, reason: {DropReason.TOO_MANY_LAYOUT_COLUMNS}")
            continue

        '''根据layout顺序，对当前页面所有需要留下的block进行排序'''
        sorted_blocks = sort_blocks_by_layout(all_bboxes, layout_bboxes)

        '''获取所有需要拼接的span资源'''
        spans = magic_model.get_all_spans(page_id)
        '''删除重叠spans中较小的那些'''
        spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)
        '''对image和table截图'''
        spans = ocr_cut_image_and_table(spans, pdf_docs[page_id], page_id, pdf_bytes_md5, imageWriter)

        '''将span填入排好序的blocks中'''
        block_with_spans = fill_spans_in_blocks(sorted_blocks, spans)

        '''对block进行fix操作'''
        fix_blocks = fix_block_spans(block_with_spans, img_blocks, table_blocks)

        '''获取QA需要外置的list'''
        images, tables, interline_equations = get_qa_need_list_v2(fix_blocks)

        '''构造pdf_info_dict'''
        page_info = ocr_construct_page_component_v2(fix_blocks, layout_bboxes, page_id, page_w, page_h, layout_tree,
                                                    images, tables, interline_equations, discarded_blocks)
        pdf_info_dict[f"page_{page_id}"] = page_info

    """分段"""
    try:
        para_split(pdf_info_dict, debug_mode=debug_mode)
    except Exception as e:
        logger.exception(e)
        raise e

    """dict转list"""
    pdf_info_list = dict_to_list(pdf_info_dict)
    new_pdf_info_dict = {
        "pdf_info": pdf_info_list,
    }

    return new_pdf_info_dict
