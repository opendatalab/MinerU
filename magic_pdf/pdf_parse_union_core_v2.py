import statistics
import time

from loguru import logger

from typing import List

import torch

from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.commons import fitz, get_delta_time
from magic_pdf.libs.convert_utils import dict_to_list
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.hash_utils import compute_md5
from magic_pdf.libs.local_math import float_equal
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.model.magic_model import MagicModel
from magic_pdf.pre_proc.citationmarker_remove import remove_citation_marker
from magic_pdf.pre_proc.construct_page_dict import ocr_construct_page_component_v2
from magic_pdf.pre_proc.cut_image import ocr_cut_image_and_table
from magic_pdf.pre_proc.equations_replace import remove_chars_in_text_blocks, replace_equations_in_textblock, \
    combine_chars_to_pymudict
from magic_pdf.pre_proc.ocr_detect_all_bboxes import ocr_prepare_bboxes_for_layout_split_v2
from magic_pdf.pre_proc.ocr_dict_merge import  fill_spans_in_blocks, fix_block_spans, fix_discarded_block
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


def __replace_STX_ETX(text_str:str):
    """ Replace \u0002 and \u0003, as these characters become garbled when extracted using pymupdf. In fact, they were originally quotation marks.
Drawback: This issue is only observed in English text; it has not been found in Chinese text so far.

    Args:
        text_str (str): raw text

    Returns:
        _type_: replaced text
    """
    if text_str:
        s = text_str.replace('\u0002', "'")
        s = s.replace("\u0003", "'")
        return s
    return text_str


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
                            "content": __replace_STX_ETX(span["text"]),
                            "type": ContentType.Text,
                            "score": 1.0,
                        }
                    )
    return spans


def replace_text_span(pymu_spans, ocr_spans):
    return list(filter(lambda x: x["type"] != ContentType.Text, ocr_spans)) + pymu_spans


def model_init(model_name: str, local_path=None):
    from transformers import LayoutLMv3ForTokenClassification
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.is_bf16_supported():
            supports_bfloat16 = True
        else:
            supports_bfloat16 = False
    else:
        device = torch.device("cpu")
        supports_bfloat16 = False

    if model_name == "layoutreader":
        if local_path:
            model = LayoutLMv3ForTokenClassification.from_pretrained(local_path)
        else:
            model = LayoutLMv3ForTokenClassification.from_pretrained("hantian/layoutreader")
        # 检查设备是否支持 bfloat16
        if supports_bfloat16:
            model.bfloat16()
        model.to(device).eval()
    else:
        logger.error("model name not allow")
        exit(1)
    return model


class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_name: str, local_path=None):
        if model_name not in self._models:
            if local_path:
                self._models[model_name] = model_init(model_name=model_name, local_path=local_path)
            else:
                self._models[model_name] = model_init(model_name=model_name)
        return self._models[model_name]


def do_predict(boxes: List[List[int]], model) -> List[int]:
    from magic_pdf.model.v3.helpers import prepare_inputs, boxes2inputs, parse_logits
    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, model)
    logits = model(**inputs).logits.cpu().squeeze(0)
    return parse_logits(logits, len(boxes))


def cal_block_index(fix_blocks, sorted_bboxes):
    for block in fix_blocks:
        if block['type'] in ['text', 'title', 'interline_equation']:
            line_index_list = []
            if len(block['lines']) == 0:
                block['index'] = sorted_bboxes.index(block['bbox'])
            else:
                for line in block['lines']:
                    line['index'] = sorted_bboxes.index(line['bbox'])
                    line_index_list.append(line['index'])
                median_value = statistics.median(line_index_list)
                block['index'] = median_value

        elif block['type'] in ['table', 'image']:
            block['index'] = sorted_bboxes.index(block['bbox'])

    return fix_blocks


def sort_lines_by_model(fix_blocks, page_w, page_h):
    page_line_list = []
    for block in fix_blocks:
        if block['type'] in ['text', 'title', 'interline_equation']:
            if len(block['lines']) == 0:  # 没有line的block(一般是图片形式的文本块)，就直接用block的bbox来排序
                bbox = block['bbox']
                page_line_list.append(bbox)
            else:
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_list.append(bbox)
        elif block['type'] in ['table', 'image']:  # 简单的把表和图都当成一个line处理
            bbox = block['bbox']
            page_line_list.append(bbox)

    # 使用layoutreader排序
    x_scale = 1000.0 / page_w
    y_scale = 1000.0 / page_h
    boxes = []
    # logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(page_line_list)}")
    for left, top, right, bottom in page_line_list:
        if left < 0:
            logger.warning(
                f"left < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}")
            left = 0
        if right > page_w:
            logger.warning(
                f"right > page_w, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}")
            right = page_w
        if top < 0:
            logger.warning(
                f"top < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}")
            top = 0
        if bottom > page_h:
            logger.warning(
                f"bottom > page_h, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}")
            bottom = page_h

        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        assert (
                1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f"Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}"
        boxes.append([left, top, right, bottom])
    model_manager = ModelSingleton()
    model = model_manager.get_model("layoutreader")
    with torch.no_grad():
        orders = do_predict(boxes, model)
    sorted_bboxes = [page_line_list[i] for i in orders]

    return sorted_bboxes


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
        all_bboxes, all_discarded_blocks = ocr_prepare_bboxes_for_layout_split_v2(
            img_blocks, table_blocks, discarded_blocks, text_blocks, title_blocks,
            interline_equation_blocks, page_w, page_h)
    else:
        all_bboxes, all_discarded_blocks = ocr_prepare_bboxes_for_layout_split_v2(
            img_blocks, table_blocks, discarded_blocks, text_blocks, title_blocks,
            interline_equations, page_w, page_h)

    '''先处理不需要排版的discarded_blocks'''
    discarded_block_with_spans, spans = fill_spans_in_blocks(all_discarded_blocks, spans, 0.4)
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    '''如果当前页面没有bbox则跳过'''
    if len(all_bboxes) == 0:
        logger.warning(f"skip this page, not found useful bbox, page_id: {page_id}")
        return ocr_construct_page_component_v2([], [], page_id, page_w, page_h, [],
                                               [], [], interline_equations, fix_discarded_blocks,
                                               need_drop, drop_reason)

    '''将span填入blocks中'''
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.3)

    '''对block进行fix操作'''
    fix_blocks = fix_block_spans(block_with_spans, img_blocks, table_blocks)

    '''获取所有line并对line排序'''
    sorted_bboxes = sort_lines_by_model(fix_blocks, page_w, page_h)

    '''根据line的中位数算block的序列关系'''
    fix_blocks = cal_block_index(fix_blocks, sorted_bboxes)

    '''重排block'''
    sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])

    '''获取QA需要外置的list'''
    images, tables, interline_equations = get_qa_need_list_v2(sorted_blocks)

    '''构造pdf_info_dict'''
    page_info = ocr_construct_page_component_v2(sorted_blocks, [], page_id, page_w, page_h, [],
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
    # end_page_id = end_page_id if end_page_id else len(pdf_docs) - 1
    end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else len(pdf_docs) - 1

    if end_page_id > len(pdf_docs) - 1:
        logger.warning("end_page_id is out of range, use pdf_docs length")
        end_page_id = len(pdf_docs) - 1

    '''初始化启动时间'''
    start_time = time.time()

    for page_id, page in enumerate(pdf_docs):
        '''debug时输出每页解析的耗时'''
        if debug_mode:
            time_now = time.time()
            logger.info(
                f"page_id: {page_id}, last_page_cost_time: {get_delta_time(start_time)}"
            )
            start_time = time_now

        '''解析pdf中的每一页'''
        if start_page_id <= page_id <= end_page_id:
            page_info = parse_page_core(pdf_docs, magic_model, page_id, pdf_bytes_md5, imageWriter, parse_mode)
        else:
            page_w = page.rect.width
            page_h = page.rect.height
            page_info = ocr_construct_page_component_v2([], [], page_id, page_w, page_h, [],
                                                [], [], [], [],
                                                True, "skip page")
        pdf_info_dict[f"page_{page_id}"] = page_info

    """分段"""
    # para_split(pdf_info_dict, debug_mode=debug_mode)
    for page_num, page in pdf_info_dict.items():
        page['para_blocks'] = page['preproc_blocks']

    """dict转list"""
    pdf_info_list = dict_to_list(pdf_info_dict)
    new_pdf_info_dict = {
        "pdf_info": pdf_info_list,
    }

    clean_memory()

    return new_pdf_info_dict


if __name__ == '__main__':
    pass
