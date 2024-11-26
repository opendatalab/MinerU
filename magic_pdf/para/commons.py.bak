import sys

from magic_pdf.libs.commons import fitz
from termcolor import cprint


if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


def open_pdf(pdf_path):
    try:
        pdf_document = fitz.open(pdf_path)  # type: ignore
        return pdf_document
    except Exception as e:
        print(f"无法打开PDF文件：{pdf_path}。原因是：{e}")
        raise e


def print_green_on_red(text):
    cprint(text, "green", "on_red", attrs=["bold"], end="\n\n")


def print_green(text):
    print()
    cprint(text, "green", attrs=["bold"], end="\n\n")


def print_red(text):
    print()
    cprint(text, "red", attrs=["bold"], end="\n\n")


def print_yellow(text):
    print()
    cprint(text, "yellow", attrs=["bold"], end="\n\n")


def safe_get(dict_obj, key, default):
    val = dict_obj.get(key)
    if val is None:
        return default
    else:
        return val


def is_bbox_overlap(bbox1, bbox2):
    """
    This function checks if bbox1 and bbox2 overlap or not

    Parameters
    ----------
    bbox1 : list
        bbox1
    bbox2 : list
        bbox2

    Returns
    -------
    bool
        True if bbox1 and bbox2 overlap, else False
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    if x0_1 > x1_2 or x0_2 > x1_1:
        return False
    if y0_1 > y1_2 or y0_2 > y1_1:
        return False

    return True


def is_in_bbox(bbox1, bbox2):
    """
    This function checks if bbox1 is in bbox2

    Parameters
    ----------
    bbox1 : list
        bbox1
    bbox2 : list
        bbox2

    Returns
    -------
    bool
        True if bbox1 is in bbox2, else False
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    if x0_1 >= x0_2 and y0_1 >= y0_2 and x1_1 <= x1_2 and y1_1 <= y1_2:
        return True
    else:
        return False


def calculate_para_bbox(lines):
    """
    This function calculates the minimum bbox of the paragraph

    Parameters
    ----------
    lines : list
        lines

    Returns
    -------
    para_bbox : list
        bbox of the paragraph
    """
    x0 = min(line["bbox"][0] for line in lines)
    y0 = min(line["bbox"][1] for line in lines)
    x1 = max(line["bbox"][2] for line in lines)
    y1 = max(line["bbox"][3] for line in lines)
    return [x0, y0, x1, y1]


def is_line_right_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction=2):
    """
    This function checks if the line is right aligned from its neighbors

    Parameters
    ----------
    curr_line_bbox : list
        bbox of the current line
    prev_line_bbox : list
        bbox of the previous line
    next_line_bbox : list
        bbox of the next line
    avg_char_width : float
        average of char widths
    direction : int
        0 for prev, 1 for next, 2 for both

    Returns
    -------
    bool
        True if the line is right aligned from its neighbors, False otherwise.
    """
    horizontal_ratio = 0.5
    horizontal_thres = horizontal_ratio * avg_char_width

    _, _, x1, _ = curr_line_bbox
    _, _, prev_x1, _ = prev_line_bbox if prev_line_bbox else (0, 0, 0, 0)
    _, _, next_x1, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)

    if direction == 0:
        return abs(x1 - prev_x1) < horizontal_thres
    elif direction == 1:
        return abs(x1 - next_x1) < horizontal_thres
    elif direction == 2:
        return abs(x1 - prev_x1) < horizontal_thres and abs(x1 - next_x1) < horizontal_thres
    else:
        return False


def is_line_left_aligned_from_neighbors(curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, direction=2):
    """
    This function checks if the line is left aligned from its neighbors

    Parameters
    ----------
    curr_line_bbox : list
        bbox of the current line
    prev_line_bbox : list
        bbox of the previous line
    next_line_bbox : list
        bbox of the next line
    avg_char_width : float
        average of char widths
    direction : int
        0 for prev, 1 for next, 2 for both

    Returns
    -------
    bool
        True if the line is left aligned from its neighbors, False otherwise.
    """
    horizontal_ratio = 0.5
    horizontal_thres = horizontal_ratio * avg_char_width

    x0, _, _, _ = curr_line_bbox
    prev_x0, _, _, _ = prev_line_bbox if prev_line_bbox else (0, 0, 0, 0)
    next_x0, _, _, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)

    if direction == 0:
        return abs(x0 - prev_x0) < horizontal_thres
    elif direction == 1:
        return abs(x0 - next_x0) < horizontal_thres
    elif direction == 2:
        return abs(x0 - prev_x0) < horizontal_thres and abs(x0 - next_x0) < horizontal_thres
    else:
        return False


def end_with_punctuation(line_text):
    """
    This function checks if the line ends with punctuation marks
    """

    english_end_puncs = [".", "?", "!"]
    chinese_end_puncs = ["。", "？", "！"]
    end_puncs = english_end_puncs + chinese_end_puncs

    last_non_space_char = None
    for ch in line_text[::-1]:
        if not ch.isspace():
            last_non_space_char = ch
            break

    if last_non_space_char is None:
        return False

    return last_non_space_char in end_puncs


def is_nested_list(lst):
    if isinstance(lst, list):
        return any(isinstance(sub, list) for sub in lst)
    return False
