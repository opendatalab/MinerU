import json

from magic_pdf.libs.boxbase import get_minbox_if_overlap_by_ratio
from magic_pdf.libs.ocr_dict_merge import merge_spans


def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def construct_page_component(page_id, text_blocks_preproc):
    return_dict = {
        'preproc_blocks': text_blocks_preproc,
        'page_idx': page_id
    }
    return return_dict


def parse_pdf_by_ocr(
    ocr_json_file_path,
    start_page_id=0,
    end_page_id=None,
):
    ocr_pdf_info = read_json_file(ocr_json_file_path)
    pdf_info_dict = {}
    end_page_id = end_page_id if end_page_id else len(ocr_pdf_info) - 1
    for page_id in range(start_page_id, end_page_id + 1):
        ocr_page_info = ocr_pdf_info[page_id]
        layout_dets = ocr_page_info['layout_dets']
        spans = []
        for layout_det in layout_dets:
            category_id = layout_det['category_id']
            allow_category_id_list = [13, 14, 15]
            if category_id in allow_category_id_list:
                x0, y0, _, _, x1, y1, _, _ = layout_det['poly']
                bbox = [int(x0), int(y0), int(x1), int(y1)]
                #  13: 'embedding',     # 嵌入公式
                #  14: 'isolated',      # 单行公式
                #  15: 'ocr_text',      # ocr识别文本
                span = {
                    'bbox': bbox,
                }
                if category_id == 13:
                    span['content'] = layout_det['latex']
                    span['type'] = 'inline_equation'
                elif category_id == 14:
                    span['content'] = layout_det['latex']
                    span['type'] = 'displayed_equation'
                elif category_id == 15:
                    span['content'] = layout_det['text']
                    span['type'] = 'text'
                # print(span)
                spans.append(span)
            else:
                continue

        # 合并重叠的spans
        for span1 in spans.copy():
            for span2 in spans.copy():
                if span1 != span2:
                    overlap_box = get_minbox_if_overlap_by_ratio(span1['bbox'], span2['bbox'], 0.8)
                    if overlap_box is not None:
                        bbox_to_remove = next((span for span in spans if span['bbox'] == overlap_box), None)
                        if bbox_to_remove is not None:
                            spans.remove(bbox_to_remove)

        # 将spans合并成line
        lines = merge_spans(spans)

        # 目前不做block拼接,先做个结构,每个block中只有一个line,block的bbox就是line的bbox
        blocks = []
        for line in lines:
            blocks.append({
                "bbox": line['bbox'],
                "lines": [line],
            })

        # 构造pdf_info_dict
        page_info = construct_page_component(page_id, blocks)
        pdf_info_dict[f"page_{page_id}"] = page_info

    return pdf_info_dict

