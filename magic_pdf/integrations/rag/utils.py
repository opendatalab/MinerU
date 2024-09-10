import json
import os
from pathlib import Path

from loguru import logger

import magic_pdf.model as model_config
from magic_pdf.dict2md.ocr_mkcontent import merge_para_with_text
from magic_pdf.integrations.rag.type import (CategoryType, ContentObject,
                                             ElementRelation, ElementRelType,
                                             LayoutElements,
                                             LayoutElementsExtra, PageInfo)
from magic_pdf.libs.ocr_content_type import BlockType, ContentType
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.tools.common import do_parse, prepare_env


def convert_middle_json_to_layout_elements(
    json_data: dict,
    output_dir: str,
) -> list[LayoutElements]:
    uniq_anno_id = 0

    res: list[LayoutElements] = []
    for page_no, page_data in enumerate(json_data['pdf_info']):
        order_id = 0
        page_info = PageInfo(
            height=int(page_data['page_size'][1]),
            width=int(page_data['page_size'][0]),
            page_no=page_no,
        )
        layout_dets: list[ContentObject] = []
        extra_element_relation: list[ElementRelation] = []

        for para_block in page_data['para_blocks']:
            para_text = ''
            para_type = para_block['type']

            if para_type == BlockType.Text:
                para_text = merge_para_with_text(para_block)
                x0, y0, x1, y1 = para_block['bbox']
                content = ContentObject(
                    anno_id=uniq_anno_id,
                    category_type=CategoryType.text,
                    text=para_text,
                    order=order_id,
                    poly=[x0, y0, x1, y0, x1, y1, x0, y1],
                )
                uniq_anno_id += 1
                order_id += 1
                layout_dets.append(content)

            elif para_type == BlockType.Title:
                para_text = merge_para_with_text(para_block)
                x0, y0, x1, y1 = para_block['bbox']
                content = ContentObject(
                    anno_id=uniq_anno_id,
                    category_type=CategoryType.title,
                    text=para_text,
                    order=order_id,
                    poly=[x0, y0, x1, y0, x1, y1, x0, y1],
                )
                uniq_anno_id += 1
                order_id += 1
                layout_dets.append(content)

            elif para_type == BlockType.InterlineEquation:
                para_text = merge_para_with_text(para_block)
                x0, y0, x1, y1 = para_block['bbox']
                content = ContentObject(
                    anno_id=uniq_anno_id,
                    category_type=CategoryType.interline_equation,
                    text=para_text,
                    order=order_id,
                    poly=[x0, y0, x1, y0, x1, y1, x0, y1],
                )
                uniq_anno_id += 1
                order_id += 1
                layout_dets.append(content)

            elif para_type == BlockType.Image:
                body_anno_id = -1
                caption_anno_id = -1

                for block in para_block['blocks']:
                    if block['type'] == BlockType.ImageBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Image:
                                    x0, y0, x1, y1 = block['bbox']
                                    content = ContentObject(
                                        anno_id=uniq_anno_id,
                                        category_type=CategoryType.image_body,
                                        image_path=os.path.join(
                                            output_dir, span['image_path']),
                                        order=order_id,
                                        poly=[x0, y0, x1, y0, x1, y1, x0, y1],
                                    )
                                    body_anno_id = uniq_anno_id
                                    uniq_anno_id += 1
                                    order_id += 1
                                    layout_dets.append(content)

                for block in para_block['blocks']:
                    if block['type'] == BlockType.ImageCaption:
                        para_text += merge_para_with_text(block)
                        x0, y0, x1, y1 = block['bbox']
                        content = ContentObject(
                            anno_id=uniq_anno_id,
                            category_type=CategoryType.image_caption,
                            text=para_text,
                            order=order_id,
                            poly=[x0, y0, x1, y0, x1, y1, x0, y1],
                        )
                        caption_anno_id = uniq_anno_id
                        uniq_anno_id += 1
                        order_id += 1
                        layout_dets.append(content)

                if body_anno_id > 0 and caption_anno_id > 0:
                    element_relation = ElementRelation(
                        relation=ElementRelType.sibling,
                        source_anno_id=body_anno_id,
                        target_anno_id=caption_anno_id,
                    )
                    extra_element_relation.append(element_relation)

            elif para_type == BlockType.Table:
                body_anno_id, caption_anno_id, footnote_anno_id = -1, -1, -1

                for block in para_block['blocks']:
                    if block['type'] == BlockType.TableCaption:
                        para_text += merge_para_with_text(block)
                        x0, y0, x1, y1 = block['bbox']
                        content = ContentObject(
                            anno_id=uniq_anno_id,
                            category_type=CategoryType.table_caption,
                            text=para_text,
                            order=order_id,
                            poly=[x0, y0, x1, y0, x1, y1, x0, y1],
                        )
                        caption_anno_id = uniq_anno_id
                        uniq_anno_id += 1
                        order_id += 1
                        layout_dets.append(content)

                for block in para_block['blocks']:
                    if block['type'] == BlockType.TableBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Table:
                                    x0, y0, x1, y1 = para_block['bbox']
                                    content = ContentObject(
                                        anno_id=uniq_anno_id,
                                        category_type=CategoryType.table_body,
                                        order=order_id,
                                        poly=[x0, y0, x1, y0, x1, y1, x0, y1],
                                    )
                                    body_anno_id = uniq_anno_id
                                    uniq_anno_id += 1
                                    order_id += 1
                                    # if processed by table model
                                    if span.get('latex', ''):
                                        content.latex = span['latex']
                                    else:
                                        content.image_path = os.path.join(
                                            output_dir, span['image_path'])
                                    layout_dets.append(content)

                for block in para_block['blocks']:
                    if block['type'] == BlockType.TableFootnote:
                        para_text += merge_para_with_text(block)
                        x0, y0, x1, y1 = block['bbox']
                        content = ContentObject(
                            anno_id=uniq_anno_id,
                            category_type=CategoryType.table_footnote,
                            text=para_text,
                            order=order_id,
                            poly=[x0, y0, x1, y0, x1, y1, x0, y1],
                        )
                        footnote_anno_id = uniq_anno_id
                        uniq_anno_id += 1
                        order_id += 1
                        layout_dets.append(content)

                if caption_anno_id != -1 and body_anno_id != -1:
                    element_relation = ElementRelation(
                        relation=ElementRelType.sibling,
                        source_anno_id=body_anno_id,
                        target_anno_id=caption_anno_id,
                    )
                    extra_element_relation.append(element_relation)

                if footnote_anno_id != -1 and body_anno_id != -1:
                    element_relation = ElementRelation(
                        relation=ElementRelType.sibling,
                        source_anno_id=body_anno_id,
                        target_anno_id=footnote_anno_id,
                    )
                    extra_element_relation.append(element_relation)

        res.append(
            LayoutElements(
                page_info=page_info,
                layout_dets=layout_dets,
                extra=LayoutElementsExtra(
                    element_relation=extra_element_relation),
            ))

    return res


def inference(path, output_dir, method):
    model_config.__use_inside_model__ = True
    model_config.__model_mode__ = 'full'
    if output_dir == '':
        if os.path.isdir(path):
            output_dir = os.path.join(path, 'output')
        else:
            output_dir = os.path.join(os.path.dirname(path), 'output')

    local_image_dir, local_md_dir = prepare_env(output_dir,
                                                str(Path(path).stem), method)

    def read_fn(path):
        disk_rw = DiskReaderWriter(os.path.dirname(path))
        return disk_rw.read(os.path.basename(path), AbsReaderWriter.MODE_BIN)

    def parse_doc(doc_path: str):
        try:
            file_name = str(Path(doc_path).stem)
            pdf_data = read_fn(doc_path)
            do_parse(
                output_dir,
                file_name,
                pdf_data,
                [],
                method,
                False,
                f_draw_span_bbox=False,
                f_draw_layout_bbox=False,
                f_dump_md=False,
                f_dump_middle_json=True,
                f_dump_model_json=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                f_draw_model_bbox=False,
            )

            middle_json_fn = os.path.join(local_md_dir,
                                          f'{file_name}_middle.json')
            with open(middle_json_fn) as fd:
                jso = json.load(fd)
            os.remove(middle_json_fn)
            return convert_middle_json_to_layout_elements(jso, local_image_dir)

        except Exception as e:
            logger.exception(e)

    return parse_doc(path)


if __name__ == '__main__':
    import pprint

    base_dir = '/opt/data/pdf/resources/samples/'
    if 0:
        with open(base_dir + 'json_outputs/middle.json') as f:
            d = json.load(f)
        result = convert_middle_json_to_layout_elements(d, '/tmp')
        pprint.pp(result)
    if 0:
        with open(base_dir + 'json_outputs/middle.3.json') as f:
            d = json.load(f)
        result = convert_middle_json_to_layout_elements(d, '/tmp')
        pprint.pp(result)

    if 1:
        res = inference(
            base_dir + 'samples/pdf/one_page_with_table_image.pdf',
            '/tmp/output',
            'ocr',
        )
        pprint.pp(res)
