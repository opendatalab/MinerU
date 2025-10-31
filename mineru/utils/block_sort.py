# Copyright (c) Opendatalab. All rights reserved.
import copy
import os
import statistics
import warnings
from typing import List
import torch
from loguru import logger

from mineru.utils.config_reader import get_device
from mineru.utils.enum_class import BlockType, ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


def sort_blocks_by_bbox(blocks, page_w, page_h, footnote_blocks):

    """获取所有line并计算正文line的高度"""
    line_height = get_line_height(blocks)

    """获取所有line并对line排序"""
    sorted_bboxes = sort_lines_by_model(blocks, page_w, page_h, line_height, footnote_blocks)

    """根据line的中位数算block的序列关系"""
    blocks = cal_block_index(blocks, sorted_bboxes)

    """将image和table的block还原回group形式参与后续流程"""
    blocks = revert_group_blocks(blocks)

    """重排block"""
    sorted_blocks = sorted(blocks, key=lambda b: b['index'])

    """block内重排(img和table的block内多个caption或footnote的排序)"""
    for block in sorted_blocks:
        if block['type'] in [BlockType.IMAGE, BlockType.TABLE]:
            block['blocks'] = sorted(block['blocks'], key=lambda b: b['index'])

    return sorted_blocks


def get_line_height(blocks):
    page_line_height_list = []
    for block in blocks:
        if block['type'] in [
            BlockType.TEXT, BlockType.TITLE,
            BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
        ]:
            for line in block['lines']:
                bbox = line['bbox']
                page_line_height_list.append(int(bbox[3] - bbox[1]))
    if len(page_line_height_list) > 0:
        return statistics.median(page_line_height_list)
    else:
        return 10


def sort_lines_by_model(fix_blocks, page_w, page_h, line_height, footnote_blocks):
    page_line_list = []

    def add_lines_to_block(b):
        line_bboxes = insert_lines_into_block(b['bbox'], line_height, page_w, page_h)
        b['lines'] = []
        for line_bbox in line_bboxes:
            b['lines'].append({'bbox': line_bbox, 'spans': []})
        page_line_list.extend(line_bboxes)

    for block in fix_blocks:
        if block['type'] in [
            BlockType.TEXT, BlockType.TITLE,
            BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE,
            BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE
        ]:
            if len(block['lines']) == 0:
                add_lines_to_block(block)
            elif block['type'] in [BlockType.TITLE] and len(block['lines']) == 1 and (block['bbox'][3] - block['bbox'][1]) > line_height * 2:
                block['real_lines'] = copy.deepcopy(block['lines'])
                add_lines_to_block(block)
            else:
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_list.append(bbox)
        elif block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.INTERLINE_EQUATION]:
            block['real_lines'] = copy.deepcopy(block['lines'])
            add_lines_to_block(block)

    for block in footnote_blocks:
        footnote_block = {'bbox': block[:4]}
        add_lines_to_block(footnote_block)

    if len(page_line_list) > 200:  # layoutreader最高支持512line
        return None

    # 使用layoutreader排序
    x_scale = 1000.0 / page_w
    y_scale = 1000.0 / page_h
    boxes = []
    # logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(page_line_list)}")
    for left, top, right, bottom in page_line_list:
        if left < 0:
            logger.warning(
                f'left < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            left = 0
        if right > page_w:
            logger.warning(
                f'right > page_w, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            right = page_w
        if top < 0:
            logger.warning(
                f'top < 0, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            top = 0
        if bottom > page_h:
            logger.warning(
                f'bottom > page_h, left: {left}, right: {right}, top: {top}, bottom: {bottom}, page_w: {page_w}, page_h: {page_h}'
            )  # noqa: E501
            bottom = page_h

        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        assert (
            1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f'Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}'  # noqa: E126, E121
        boxes.append([left, top, right, bottom])
    model_manager = ModelSingleton()
    model = model_manager.get_model('layoutreader')
    with torch.no_grad():
        orders = do_predict(boxes, model)
    sorted_bboxes = [page_line_list[i] for i in orders]

    return sorted_bboxes


def insert_lines_into_block(block_bbox, line_height, page_w, page_h):
    # block_bbox是一个元组(x0, y0, x1, y1)，其中(x0, y0)是左下角坐标，(x1, y1)是右上角坐标
    x0, y0, x1, y1 = block_bbox

    block_height = y1 - y0
    block_weight = x1 - x0

    # 如果block高度小于n行正文，则直接返回block的bbox
    if line_height * 2 < block_height:
        if (
            block_height > page_h * 0.25 and page_w * 0.5 > block_weight > page_w * 0.25
        ):  # 可能是双列结构，可以切细点
            lines = int(block_height / line_height)
        else:
            # 如果block的宽度超过0.4页面宽度，则将block分成3行(是一种复杂布局，图不能切的太细)
            if block_weight > page_w * 0.4:
                lines = 3
            elif block_weight > page_w * 0.25:  # （可能是三列结构，也切细点）
                lines = int(block_height / line_height)
            else:  # 判断长宽比
                if block_height / block_weight > 1.2:  # 细长的不分
                    return [[x0, y0, x1, y1]]
                else:  # 不细长的还是分成两行
                    lines = 2

        line_height = (y1 - y0) / lines

        # 确定从哪个y位置开始绘制线条
        current_y = y0

        # 用于存储线条的位置信息[(x0, y), ...]
        lines_positions = []

        for i in range(lines):
            lines_positions.append([x0, current_y, x1, current_y + line_height])
            current_y += line_height
        return lines_positions

    else:
        return [[x0, y0, x1, y1]]


def model_init(model_name: str):
    from transformers import LayoutLMv3ForTokenClassification
    device_name = get_device()
    device = torch.device(device_name)
    bf_16_support = False
    if device_name.startswith("cuda"):
        if torch.cuda.get_device_properties(device).major >= 8:
            bf_16_support = True
    elif device_name.startswith("mps"):
        bf_16_support = True

    if model_name == 'layoutreader':
        # 检测modelscope的缓存目录是否存在
        layoutreader_model_dir = os.path.join(auto_download_and_get_model_root_path(ModelPath.layout_reader), ModelPath.layout_reader)
        if os.path.exists(layoutreader_model_dir):
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                layoutreader_model_dir
            )
        else:
            logger.warning(
                'local layoutreader model not exists, use online model from huggingface'
            )
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                'hantian/layoutreader'
            )
        if bf_16_support:
            model.to(device).eval().bfloat16()
        else:
            model.to(device).eval()
    else:
        logger.error('model name not allow')
        exit(1)
    return model


class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, model_name: str):
        if model_name not in self._models:
            self._models[model_name] = model_init(model_name=model_name)
        return self._models[model_name]


def do_predict(boxes: List[List[int]], model) -> List[int]:
    from mineru.model.reading_order.layout_reader import (
        boxes2inputs, parse_logits, prepare_inputs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        inputs = boxes2inputs(boxes)
        inputs = prepare_inputs(inputs, model)
        logits = model(**inputs).logits.cpu().squeeze(0)
    return parse_logits(logits, len(boxes))


def cal_block_index(fix_blocks, sorted_bboxes):

    if sorted_bboxes is not None:
        # 使用layoutreader排序
        for block in fix_blocks:
            line_index_list = []
            if len(block['lines']) == 0:
                block['index'] = sorted_bboxes.index(block['bbox'])
            else:
                for line in block['lines']:
                    line['index'] = sorted_bboxes.index(line['bbox'])
                    line_index_list.append(line['index'])
                median_value = statistics.median(line_index_list)
                block['index'] = median_value

            # 删除图表body block中的虚拟line信息, 并用real_lines信息回填
            if block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.TITLE, BlockType.INTERLINE_EQUATION]:
                if 'real_lines' in block:
                    block['virtual_lines'] = copy.deepcopy(block['lines'])
                    block['lines'] = copy.deepcopy(block['real_lines'])
                    del block['real_lines']
    else:
        # 使用xycut排序
        block_bboxes = []
        for block in fix_blocks:
            # 如果block['bbox']任意值小于0，将其置为0
            block['bbox'] = [max(0, x) for x in block['bbox']]
            block_bboxes.append(block['bbox'])

            # 删除图表body block中的虚拟line信息, 并用real_lines信息回填
            if block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY, BlockType.TITLE, BlockType.INTERLINE_EQUATION]:
                if 'real_lines' in block:
                    block['virtual_lines'] = copy.deepcopy(block['lines'])
                    block['lines'] = copy.deepcopy(block['real_lines'])
                    del block['real_lines']

        import numpy as np
        from mineru.model.reading_order.xycut import recursive_xy_cut

        random_boxes = np.array(block_bboxes)
        np.random.shuffle(random_boxes)
        res = []
        recursive_xy_cut(np.asarray(random_boxes).astype(int), np.arange(len(block_bboxes)), res)
        assert len(res) == len(block_bboxes)
        sorted_boxes = random_boxes[np.array(res)].tolist()

        for i, block in enumerate(fix_blocks):
            block['index'] = sorted_boxes.index(block['bbox'])

        # 生成line index
        sorted_blocks = sorted(fix_blocks, key=lambda b: b['index'])
        line_inedx = 1
        for block in sorted_blocks:
            for line in block['lines']:
                line['index'] = line_inedx
                line_inedx += 1

    return fix_blocks


def revert_group_blocks(blocks):
    image_groups = {}
    table_groups = {}
    new_blocks = []
    for block in blocks:
        if block['type'] in [BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION, BlockType.IMAGE_FOOTNOTE]:
            group_id = block['group_id']
            if group_id not in image_groups:
                image_groups[group_id] = []
            image_groups[group_id].append(block)
        elif block['type'] in [BlockType.TABLE_BODY, BlockType.TABLE_CAPTION, BlockType.TABLE_FOOTNOTE]:
            group_id = block['group_id']
            if group_id not in table_groups:
                table_groups[group_id] = []
            table_groups[group_id].append(block)
        else:
            new_blocks.append(block)

    for group_id, blocks in image_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.IMAGE_BODY, BlockType.IMAGE))

    for group_id, blocks in table_groups.items():
        new_blocks.append(process_block_list(blocks, BlockType.TABLE_BODY, BlockType.TABLE))

    return new_blocks


def process_block_list(blocks, body_type, block_type):
    indices = [block['index'] for block in blocks]
    median_index = statistics.median(indices)

    body_bbox = next((block['bbox'] for block in blocks if block.get('type') == body_type), [])

    return {
        'type': block_type,
        'bbox': body_bbox,
        'blocks': blocks,
        'index': median_index,
    }