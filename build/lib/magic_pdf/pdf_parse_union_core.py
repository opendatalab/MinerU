import time

from loguru import logger

from magic_pdf.libs.commons import fitz, get_delta_time
from magic_pdf.layout.layout_sort import get_bboxes_layout, LAYOUT_UNPROC, get_columns_cnt_of_layout
from magic_pdf.libs.convert_utils import dict_to_list
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.libs.local_math import float_equal
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.para.para_split_v2 import para_split
from magic_pdf.pre_proc.citationmarker_remove import remove_citation_marker
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component_v2
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.equations_replace import remove_chars_in_text_blocks, replace_equations_in_textblock, \
    combine_chars_to_pymudict
from magic_pdf.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split
from magic_pdf.pre_proc.ocr_dict_merge import sort_blocks_by_layout, fill_spans_in_blocks, fix_block_spans, \
    fix_discarded_block
from magic_pdf.pre_proc.ocr_span_list_modify import remove_overlaps_min_spans, get_qa_need_list_v2, \
    remove_overlaps_low_confidence_spans
from magic_pdf.pre_proc.resolve_bbox_conflict import check_useful_block_horizontal_overlap


def remove_horizontal_overlap_block_which_smaller(all_bboxes):
    useful_blocks = []
    for bbox in all_bboxes:
        useful_blocks.append({
            "bbox": bbox[:4]
        })
    is_useful_block_horz_overlap, smaller_bbox, bigger_bbox = check_useful_block_horizontal_overlap(useful_blocks)
    if is_useful_block_horz_overlap:
        logger.warning(
            f"skip this page, reason: {DropReason.USEFUL_BLOCK_HOR_OVERLAP}, smaller bbox is {smaller_bbox}, bigger bbox is {bigger_bbox}")
        for bbox in all_bboxes.copy():
            if smaller_bbox == bbox[:4]:
                all_bboxes.remove(bbox)

    return is_useful_block_horz_overlap, all_bboxes


def txt_spans_extract(pdf_page, inline_equations, interline_equations):
    text_raw_blocks = pdf_page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]
    char_level_text_blocks = pdf_page.get_text("rawdict", flags=fitz.TEXTFLAGS_TEXT)[
        "blocks"
    ]
    text_blocks = combine_chars_to_pymudict(text_raw_blocks, char_level_text_blocks)
    text_blocks = replace_equations_in_textblock(
        text_blocks, inline_equations, interline_equations
    )
    text_blocks = remove_citation_marker(text_blocks)
    text_blocks = remove_chars_in_text_blocks(text_blocks)
    spans = []
    for v in text_blocks:
        for line in v["lines"]:
            for span in line["spans"]:
                bbox = span["bbox"]
                if float_equal(bbox[0], bbox[2]) or float_equal(bbox[1], bbox[3]):
                    continue
                if span.get('type') not in (ContentType.InlineEquation, ContentType.InterlineEquation):
                    spans.append(
                        {
                            "bbox": list(span["bbox"]),
                            "content": span["text"],
                            "type": ContentType.Text,
                            "score": 1.0,
                        }
                    )
    return spans


def replace_text_span(pymu_spans, ocr_spans):
    return list(filter(lambda x: x["type"] != ContentType.Text, ocr_spans)) + pymu_spans


def parse_page_core(pdf_docs, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode):
    need_drop = False
    drop_reason = []

    '''从magic_model对象中获取后面会用到的区块信息'''
    img_blocks = magic_model.get_imgs(page_id)
    table_blocks = magic_model.get_tables(page_id)
    discarded_blocks = magic_model.get_discarded(page_id)
    text_blocks = magic_model.get_text_blocks(page_id)
    title_blocks = magic_model.get_title_blocks(page_id)
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations(page_id)

    page_w, page_h = magic_model.get_page_size(page_id)

    spans = magic_model.get_all_spans(page_id)

    '''根据parse_mode，构造spans'''
    if parse_mode == "txt":
        """ocr 中文本类的 span 用 pymu spans 替换！"""
        pymu_spans = txt_spans_extract(
            pdf_docs[page_id], inline_equations, interline_equations
        )
        spans = replace_text_span(pymu_spans, spans)
    elif parse_mode == "ocr":
        pass
    else:
        raise Exception("parse_mode must be txt or ocr")

    '''删除重叠spans中置信度较低的那些'''
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    '''删除重叠spans中较小的那些'''
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)
    '''对image和table截图'''
    spans = ocr_cut_image_and_table(spans, pdf_docs[page_id], page_id, pdf_bytes_md5, imageWriter)

    '''将所有区块的bbox整理到一起'''
    # interline_equation_blocks参数不够准，后面切换到interline_equations上
    interline_equation_blocks = []
    if len(interline_equation_blocks) > 0:
        all_bboxes, all_discarded_blocks, drop_reasons = ocr_prepare_bboxes_for_layout_split(
            img_blocks, table_blocks, discarded_blocks, text_blocks, title_blocks,
            interline_equation_blocks, page_w, page_h)
    else:
        all_bboxes, all_discarded_blocks, drop_reasons = ocr_prepare_bboxes_for_layout_split(
            img_blocks, table_blocks, discarded_blocks, text_blocks, title_blocks,
            interline_equations, page_w, page_h)

    if len(drop_reasons) > 0:
        need_drop = True
        drop_reason.append(DropReason.OVERLAP_BLOCKS_CAN_NOT_SEPARATION)

    '''先处理不需要排版的discarded_blocks'''
    discarded_block_with_spans, spans = fill_spans_in_blocks(all_discarded_blocks, spans, 0.4)
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    '''如果当前页面没有bbox则跳过'''
    if len(all_bboxes) == 0:
        logger.warning(f"skip this page, not found useful bbox, page_id: {page_id}")
        return ocr_construct_page_component_v2([], [], page_id, page_w, page_h, [],
                                               [], [], interline_equations, fix_discarded_blocks,
                                               need_drop, drop_reason)

    """在切分之前，先检查一下bbox是否有左右重叠的情况，如果有，那么就认为这个pdf暂时没有能力处理好，这种左右重叠的情况大概率是由于pdf里的行间公式、表格没有被正确识别出来造成的 """

    while True:  # 循环检查左右重叠的情况，如果存在就删除掉较小的那个bbox，直到不存在左右重叠的情况
        is_useful_block_horz_overlap, all_bboxes = remove_horizontal_overlap_block_which_smaller(all_bboxes)
        if is_useful_block_horz_overlap:
            need_drop = True
            drop_reason.append(DropReason.USEFUL_BLOCK_HOR_OVERLAP)
        else:
            break

    '''根据区块信息计算layout'''
    page_boundry = [0, 0, page_w, page_h]
    layout_bboxes, layout_tree = get_bboxes_layout(all_bboxes, page_boundry, page_id)

    if len(text_blocks) > 0 and len(all_bboxes) > 0 and len(layout_bboxes) == 0:
        logger.warning(
            f"skip this page, page_id: {page_id}, reason: {DropReason.CAN_NOT_DETECT_PAGE_LAYOUT}")
        need_drop = True
        drop_reason.append(DropReason.CAN_NOT_DETECT_PAGE_LAYOUT)

    """以下去掉复杂的布局和超过2列的布局"""
    if any([lay["layout_label"] == LAYOUT_UNPROC for lay in layout_bboxes]):  # 复杂的布局
        logger.warning(
            f"skip this page, page_id: {page_id}, reason: {DropReason.COMPLICATED_LAYOUT}")
        need_drop = True
        drop_reason.append(DropReason.COMPLICATED_LAYOUT)

    layout_column_width = get_columns_cnt_of_layout(layout_tree)
    if layout_column_width > 2:  # 去掉超过2列的布局pdf
        logger.warning(
            f"skip this page, page_id: {page_id}, reason: {DropReason.TOO_MANY_LAYOUT_COLUMNS}")
        need_drop = True
        drop_reason.append(DropReason.TOO_MANY_LAYOUT_COLUMNS)

    '''根据layout顺序，对当前页面所有需要留下的block进行排序'''
    sorted_blocks = sort_blocks_by_layout(all_bboxes, layout_bboxes)

    '''将span填入排好序的blocks中'''
    block_with_spans, spans = fill_spans_in_blocks(sorted_blocks, spans, 0.6)

    '''对block进行fix操作'''
    fix_blocks = fix_block_spans(block_with_spans, img_blocks, table_blocks)

    '''获取QA需要外置的list'''
    images, tables, interline_equations = get_qa_need_list_v2(fix_blocks)

    '''构造pdf_info_dict'''
    page_info = ocr_construct_page_component_v2(fix_blocks, layout_bboxes, page_id, page_w, page_h, layout_tree,
                                                images, tables, interline_equations, fix_discarded_blocks,
                                                need_drop, drop_reason)
    return page_info


def pdf_parse_union(pdf_bytes,
                    model_list,
                    imageWriter,
                    parse_mode,
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

        '''解析pdf中的每一页'''
        page_info = parse_page_core(pdf_docs, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode)
        pdf_info_dict[f"page_{page_id}"] = page_info

    """分段"""
    para_split(pdf_info_dict, debug_mode=debug_mode)

    """dict转list"""
    pdf_info_list = dict_to_list(pdf_info_dict)
    new_pdf_info_dict = {
        "pdf_info": pdf_info_list,
    }

    return new_pdf_info_dict


if __name__ == '__main__':
    pass
