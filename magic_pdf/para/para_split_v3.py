import copy

from magic_pdf.libs.Constants import LINES_DELETED, CROSS_PAGE

LINE_STOP_FLAG = ('.', '!', '?', '。', '！', '？')


def __process_blocks(blocks):

    result = []
    current_group = []

    for i in range(len(blocks)):
        current_block = blocks[i]

        # 如果当前块是 text 类型
        if current_block['type'] == 'text':

            current_block["bbox_fs"] = copy.deepcopy(current_block["bbox"])
            if len(current_block["lines"]) > 0:
                current_block['bbox_fs'] = [min([line['bbox'][0] for line in current_block['lines']]),
                                            min([line['bbox'][1] for line in current_block['lines']]),
                                            max([line['bbox'][2] for line in current_block['lines']]),
                                            max([line['bbox'][3] for line in current_block['lines']])]

            current_group.append(current_block)

            # 检查下一个块是否存在
            if i + 1 < len(blocks):
                next_block = blocks[i + 1]
                # 如果下一个块不是 text 类型且是 title 或 interline_equation 类型
                if next_block['type'] in ['title', 'interline_equation']:
                    result.append(current_group)
                    current_group = []

    # 处理最后一个 group
    if current_group:
        result.append(current_group)

    return result


def __merge_2_blocks(block1, block2):
    if len(block1['lines']) > 0:
        first_line = block1['lines'][0]
        line_height = first_line['bbox'][3] - first_line['bbox'][1]
        if abs(block1['bbox_fs'][0] - first_line['bbox'][0]) < line_height/2:
            last_line = block2['lines'][-1]
            if len(last_line['spans']) > 0:
                last_span = last_line['spans'][-1]
                line_height = last_line['bbox'][3] - last_line['bbox'][1]
                if abs(block2['bbox_fs'][2] - last_line['bbox'][2]) < line_height and not last_span['content'].endswith(LINE_STOP_FLAG):
                    if block1['page_num'] != block2['page_num']:
                        for line in block1['lines']:
                            for span in line['spans']:
                                span[CROSS_PAGE] = True
                    block2['lines'].extend(block1['lines'])
                    block1['lines'] = []
                    block1[LINES_DELETED] = True

    return block1, block2


def __para_merge_page(blocks):
    page_text_blocks_groups = __process_blocks(blocks)
    for text_blocks_group in page_text_blocks_groups:
        if len(text_blocks_group) > 1:
            # 倒序遍历
            for i in range(len(text_blocks_group)-1, -1, -1):
                current_block = text_blocks_group[i]
                # 检查是否有前一个块
                if i - 1 >= 0:
                    prev_block = text_blocks_group[i - 1]
                    __merge_2_blocks(current_block, prev_block)
        else:
            continue


def para_split(pdf_info_dict, debug_mode=False):
    all_blocks = []
    for page_num, page in pdf_info_dict.items():
        blocks = copy.deepcopy(page['preproc_blocks'])
        for block in blocks:
            block['page_num'] = page_num
        all_blocks.extend(blocks)

    __para_merge_page(all_blocks)
    for page_num, page in pdf_info_dict.items():
        page['para_blocks'] = []
        for block in all_blocks:
            if block['page_num'] == page_num:
                page['para_blocks'].append(block)


if __name__ == '__main__':
    input_blocks = [
        {'type': 'text', 'content': '这是第一段'},
        {'type': 'text', 'content': '这是第二段'},
        {'type': 'title', 'content': '这是一个标题'},
        {'type': 'text', 'content': '这是第三段'},
        {'type': 'interline_equation', 'content': '这是一个公式'},
        {'type': 'text', 'content': '这是第四段'},
        {'type': 'image', 'content': '这是一张图片'},
        {'type': 'text', 'content': '这是第五段'},
        {'type': 'table', 'content': '这是一张表格'}
    ]

    # 调用函数
    for group_index, group in enumerate(__process_blocks(input_blocks)):
        print(f"Group {group_index}: {group}")
