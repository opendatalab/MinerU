from loguru import logger

from magic_pdf.libs.ocr_dict_merge import merge_spans_to_line, remove_overlaps_min_spans, modify_y_axis


def construct_page_component(page_id, blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
    }
    return return_dict


def parse_pdf_by_ocr(
    ocr_pdf_info,
    start_page_id=0,
    end_page_id=None,
):

    pdf_info_dict = {}
    end_page_id = end_page_id if end_page_id else len(ocr_pdf_info) - 1
    for page_id in range(start_page_id, end_page_id + 1):
        ocr_page_info = ocr_pdf_info[page_id]
        layout_dets = ocr_page_info['layout_dets']
        spans = []
        for layout_det in layout_dets:
            category_id = layout_det['category_id']
            allow_category_id_list = [1, 7, 13, 14, 15]
            if category_id in allow_category_id_list:
                x0, y0, _, _, x1, y1, _, _ = layout_det['poly']
                bbox = [int(x0), int(y0), int(x1), int(y1)]
                '''要删除的'''
                #  3: 'header',      # 页眉
                #  4: 'page number', # 页码
                #  5: 'footnote',    # 脚注
                #  6: 'footer',      # 页脚
                '''当成span拼接的'''
                #  1: 'image', # 图片
                #  7: 'table',       # 表格
                #  13: 'inline_equation',     # 行内公式
                #  14: 'displayed_equation',      # 行间公式
                #  15: 'text',      # ocr识别文本
                '''layout信息'''
                #  11: 'full column',   # 单栏
                #  12: 'sub column',    # 多栏
                span = {
                    'bbox': bbox,
                }
                if category_id == 1:
                    span['type'] = 'image'
                elif category_id == 7:
                    span['type'] = 'table'
                elif category_id == 13:
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

        # 删除重叠spans中较小的那些
        spans = remove_overlaps_min_spans(spans)

        # 对tpye=["displayed_equation", "image", "table"]进行额外处理,如果左边有字的话,将该span的bbox中y0调整低于文字的y0
        #spans = modify_y_axis(spans)

        # 将spans合并成line(从上到下,从左到右)
        lines = merge_spans_to_line(spans)
        # logger.info(lines)

        # 从ocr_page_info中获取layout信息


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

