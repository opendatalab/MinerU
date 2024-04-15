from loguru import logger

from magic_pdf.layout.layout_sort import get_columns_cnt_of_layout
from magic_pdf.libs.drop_reason import DropReason


def __is_pseudo_single_column(page_info) -> bool:
    """
    判断一个页面是否伪单列。

    Args:
        page_info (dict): 页面信息字典，包括'_layout_tree'和'preproc_blocks'。

    Returns:
        Tuple[bool, Optional[str]]: 如果页面伪单列返回(True, extra_info)，否则返回(False, None)。

    """
    layout_tree = page_info['_layout_tree']
    layout_column_width = get_columns_cnt_of_layout(layout_tree)
    if layout_column_width == 1:
        text_blocks = page_info['preproc_blocks']
        # 遍历每一个text_block
        for text_block in text_blocks:
            lines = text_block['lines']
            num_lines = len(lines)
            num_satisfying_lines = 0

            for i in range(num_lines - 1):
                current_line = lines[i]
                next_line = lines[i + 1]

                # 获取当前line和下一个line的bbox属性
                current_bbox = current_line['bbox']
                next_bbox = next_line['bbox']

                # 检查是否满足条件
                if next_bbox[0] > current_bbox[2] or next_bbox[2] < current_bbox[0]:
                    num_satisfying_lines += 1
            # 如果有一半以上的line满足条件，就drop
            # print("num_satisfying_lines:", num_satisfying_lines, "num_lines:", num_lines)
            if num_lines > 20:
                radio = num_satisfying_lines / num_lines
                if radio >= 0.5:
                    extra_info = f"{{num_lines: {num_lines}, num_satisfying_lines: {num_satisfying_lines}}}"
                    block_text = []
                    for line in lines:
                        if line['spans']:
                            for span in line['spans']:
                                block_text.append(span['text'])
                    logger.warning(f"pseudo_single_column block_text: {block_text}")
                    return True, extra_info

    return False, None


def pdf_post_filter(page_info) -> tuple:
    """
    return:(True|False, err_msg)
        True, 如果pdf符合要求
        False, 如果pdf不符合要求

    """
    bool_is_pseudo_single_column, extra_info = __is_pseudo_single_column(page_info)
    if bool_is_pseudo_single_column:
        return False, {"_need_drop": True, "_drop_reason": DropReason.PSEUDO_SINGLE_COLUMN, "extra_info": extra_info}

    return True, None