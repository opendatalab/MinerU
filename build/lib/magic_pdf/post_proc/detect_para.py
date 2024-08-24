import os
import sys
import json
import re
import math
import unicodedata
from collections import Counter


import numpy as np
from termcolor import cprint


from magic_pdf.libs.commons import fitz
from magic_pdf.libs.nlp_utils import NLPModels


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


class DenseSingleLineBlockException(Exception):
    """
    This class defines the exception type for dense single line-block.
    """

    def __init__(self, message="DenseSingleLineBlockException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class TitleDetectionException(Exception):
    """
    This class defines the exception type for title detection.
    """

    def __init__(self, message="TitleDetectionException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class TitleLevelException(Exception):
    """
    This class defines the exception type for title level.
    """

    def __init__(self, message="TitleLevelException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class ParaSplitException(Exception):
    """
    This class defines the exception type for paragraph splitting.
    """

    def __init__(self, message="ParaSplitException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class ParaMergeException(Exception):
    """
    This class defines the exception type for paragraph merging.
    """

    def __init__(self, message="ParaMergeException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class DiscardByException:
    """
    This class discards pdf files by exception
    """

    def __init__(self) -> None:
        pass

    def discard_by_single_line_block(self, pdf_dic, exception: DenseSingleLineBlockException):
        """
        This function discards pdf files by single line block exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        exception_page_nums = 0
        page_num = 0
        for page_id, page in pdf_dic.items():
            if page_id.startswith("page_"):
                page_num += 1
                if "preproc_blocks" in page.keys():
                    preproc_blocks = page["preproc_blocks"]

                    all_single_line_blocks = []
                    for block in preproc_blocks:
                        if len(block["lines"]) == 1:
                            all_single_line_blocks.append(block)

                    if len(preproc_blocks) > 0 and len(all_single_line_blocks) / len(preproc_blocks) > 0.9:
                        exception_page_nums += 1

        if page_num == 0:
            return None

        if exception_page_nums / page_num > 0.1:  # Low ratio means basically, whenever this is the case, it is discarded
            return exception.message

        return None

    def discard_by_title_detection(self, pdf_dic, exception: TitleDetectionException):
        """
        This function discards pdf files by title detection exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        # return exception.message
        return None

    def discard_by_title_level(self, pdf_dic, exception: TitleLevelException):
        """
        This function discards pdf files by title level exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        # return exception.message
        return None

    def discard_by_split_para(self, pdf_dic, exception: ParaSplitException):
        """
        This function discards pdf files by split para exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        # return exception.message
        return None

    def discard_by_merge_para(self, pdf_dic, exception: ParaMergeException):
        """
        This function discards pdf files by merge para exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        # return exception.message
        return None


class LayoutFilterProcessor:
    def __init__(self) -> None:
        pass

    def batch_process_blocks(self, pdf_dict):
        """
        This function processes the blocks in batch.

        Parameters
        ----------
        self : object
            The instance of the class.

        pdf_dict : dict
            pdf dictionary

        Returns
        -------
        pdf_dict : dict
            pdf dictionary
        """
        for page_id, blocks in pdf_dict.items():
            if page_id.startswith("page_"):
                if "layout_bboxes" in blocks.keys() and "para_blocks" in blocks.keys():
                    layout_bbox_objs = blocks["layout_bboxes"]
                    if layout_bbox_objs is None:
                        continue
                    layout_bboxes = [bbox_obj["layout_bbox"] for bbox_obj in layout_bbox_objs]

                    # Enlarge each value of x0, y0, x1, y1 for each layout_bbox to prevent loss of text.
                    layout_bboxes = [
                        [math.ceil(x0), math.ceil(y0), math.ceil(x1), math.ceil(y1)] for x0, y0, x1, y1 in layout_bboxes
                    ]

                    para_blocks = blocks["para_blocks"]
                    if para_blocks is None:
                        continue

                    for lb_bbox in layout_bboxes:
                        for i, para_block in enumerate(para_blocks):
                            para_bbox = para_block["bbox"]
                            para_blocks[i]["in_layout"] = 0
                            if is_in_bbox(para_bbox, lb_bbox):
                                para_blocks[i]["in_layout"] = 1

                    blocks["para_blocks"] = para_blocks

        return pdf_dict


class RawBlockProcessor:
    def __init__(self) -> None:
        self.y_tolerance = 2
        self.pdf_dic = {}

    def __span_flags_decomposer(self, span_flags):
        """
        Make font flags human readable.

        Parameters
        ----------
        self : object
            The instance of the class.

        span_flags : int
            span flags

        Returns
        -------
        l : dict
            decomposed flags
        """

        l = {
            "is_superscript": False,
            "is_italic": False,
            "is_serifed": False,
            "is_sans_serifed": False,
            "is_monospaced": False,
            "is_proportional": False,
            "is_bold": False,
        }

        if span_flags & 2**0:
            l["is_superscript"] = True  # 表示上标

        if span_flags & 2**1:
            l["is_italic"] = True  # 表示斜体

        if span_flags & 2**2:
            l["is_serifed"] = True  # 表示衬线字体
        else:
            l["is_sans_serifed"] = True  # 表示非衬线字体

        if span_flags & 2**3:
            l["is_monospaced"] = True  # 表示等宽字体
        else:
            l["is_proportional"] = True  # 表示比例字体

        if span_flags & 2**4:
            l["is_bold"] = True  # 表示粗体

        return l

    def __make_new_lines(self, raw_lines):
        """
        This function makes new lines.

        Parameters
        ----------
        self : object
            The instance of the class.

        raw_lines : list
            raw lines

        Returns
        -------
        new_lines : list
            new lines
        """
        new_lines = []
        new_line = None

        for raw_line in raw_lines:
            raw_line_bbox = raw_line["bbox"]
            raw_line_spans = raw_line["spans"]
            raw_line_text = "".join([span["text"] for span in raw_line_spans])
            raw_line_dir = raw_line.get("dir", None)

            decomposed_line_spans = []
            for span in raw_line_spans:
                raw_flags = span["flags"]
                decomposed_flags = self.__span_flags_decomposer(raw_flags)
                span["decomposed_flags"] = decomposed_flags
                decomposed_line_spans.append(span)

            if new_line is None:  # Handle the first line
                new_line = {
                    "bbox": raw_line_bbox,
                    "text": raw_line_text,
                    "dir": raw_line_dir if raw_line_dir else (0, 0),
                    "spans": decomposed_line_spans,
                }
            else:  # Handle the rest lines
                if (
                    abs(raw_line_bbox[1] - new_line["bbox"][1]) <= self.y_tolerance
                    and abs(raw_line_bbox[3] - new_line["bbox"][3]) <= self.y_tolerance
                ):
                    new_line["bbox"] = (
                        min(new_line["bbox"][0], raw_line_bbox[0]),  # left
                        new_line["bbox"][1],  # top
                        max(new_line["bbox"][2], raw_line_bbox[2]),  # right
                        raw_line_bbox[3],  # bottom
                    )
                    new_line["text"] += raw_line_text
                    new_line["spans"].extend(raw_line_spans)
                    new_line["dir"] = (
                        new_line["dir"][0] + raw_line_dir[0],
                        new_line["dir"][1] + raw_line_dir[1],
                    )
                else:
                    new_lines.append(new_line)
                    new_line = {
                        "bbox": raw_line_bbox,
                        "text": raw_line_text,
                        "dir": raw_line_dir if raw_line_dir else (0, 0),
                        "spans": raw_line_spans,
                    }
        if new_line:
            new_lines.append(new_line)

        return new_lines

    def __make_new_block(self, raw_block):
        """
        This function makes a new block.

        Parameters
        ----------
        self : object
            The instance of the class.
        ----------
        raw_block : dict
            a raw block

        Returns
        -------
        new_block : dict
        """
        new_block = {}

        block_id = raw_block["number"]
        block_bbox = raw_block["bbox"]
        block_text = "".join(span["text"] for line in raw_block["lines"] for span in line["spans"])
        raw_lines = raw_block["lines"]
        block_lines = self.__make_new_lines(raw_lines)

        new_block["block_id"] = block_id
        new_block["bbox"] = block_bbox
        new_block["text"] = block_text
        new_block["lines"] = block_lines

        return new_block

    def batch_process_blocks(self, pdf_dic):
        """
        This function processes the blocks in batch.

        Parameters
        ----------
        self : object
            The instance of the class.
        ----------
        blocks : list
            Input block is a list of raw blocks.

        Returns
        -------
        result_dict : dict
            result dictionary
        """

        for page_id, blocks in pdf_dic.items():
            if page_id.startswith("page_"):
                para_blocks = []
                if "preproc_blocks" in blocks.keys():
                    input_blocks = blocks["preproc_blocks"]
                    for raw_block in input_blocks:
                        new_block = self.__make_new_block(raw_block)
                        para_blocks.append(new_block)

                blocks["para_blocks"] = para_blocks

        return pdf_dic


class BlockStatisticsCalculator:
    """
    This class calculates the statistics of the block.
    """

    def __init__(self) -> None:
        pass

    def __calc_stats_of_new_lines(self, new_lines):
        """
        This function calculates the paragraph metrics

        Parameters
        ----------
        combined_lines : list
            combined lines

        Returns
        -------
        X0 : float
            Median of x0 values, which represents the left average boundary of the block
        X1 : float
            Median of x1 values, which represents the right average boundary of the block
        avg_char_width : float
            Average of char widths, which represents the average char width of the block
        avg_char_height : float
            Average of line heights, which represents the average line height of the block

        """
        x0_values = []
        x1_values = []
        char_widths = []
        char_heights = []

        block_font_types = []
        block_font_sizes = []
        block_directions = []

        if len(new_lines) > 0:
            for i, line in enumerate(new_lines):
                line_bbox = line["bbox"]
                line_text = line["text"]
                line_spans = line["spans"]

                num_chars = len([ch for ch in line_text if not ch.isspace()])

                x0_values.append(line_bbox[0])
                x1_values.append(line_bbox[2])

                if num_chars > 0:
                    char_width = (line_bbox[2] - line_bbox[0]) / num_chars
                    char_widths.append(char_width)

                for span in line_spans:
                    block_font_types.append(span["font"])
                    block_font_sizes.append(span["size"])

                if "dir" in line:
                    block_directions.append(line["dir"])

                # line_font_types = [span["font"] for span in line_spans]
                char_heights = [span["size"] for span in line_spans]

        X0 = np.median(x0_values) if x0_values else 0
        X1 = np.median(x1_values) if x1_values else 0
        avg_char_width = sum(char_widths) / len(char_widths) if char_widths else 0
        avg_char_height = sum(char_heights) / len(char_heights) if char_heights else 0

        # max_freq_font_type = max(set(block_font_types), key=block_font_types.count) if block_font_types else None

        max_span_length = 0
        max_span_font_type = None
        for line in new_lines:
            line_spans = line["spans"]
            for span in line_spans:
                span_length = span["bbox"][2] - span["bbox"][0]
                if span_length > max_span_length:
                    max_span_length = span_length
                    max_span_font_type = span["font"]

        max_freq_font_type = max_span_font_type

        avg_font_size = sum(block_font_sizes) / len(block_font_sizes) if block_font_sizes else None

        avg_dir_horizontal = sum([dir[0] for dir in block_directions]) / len(block_directions) if block_directions else 0
        avg_dir_vertical = sum([dir[1] for dir in block_directions]) / len(block_directions) if block_directions else 0

        median_font_size = float(np.median(block_font_sizes)) if block_font_sizes else None

        return (
            X0,
            X1,
            avg_char_width,
            avg_char_height,
            max_freq_font_type,
            avg_font_size,
            (avg_dir_horizontal, avg_dir_vertical),
            median_font_size,
        )

    def __make_new_block(self, input_block):
        new_block = {}

        raw_lines = input_block["lines"]
        stats = self.__calc_stats_of_new_lines(raw_lines)

        block_id = input_block["block_id"]
        block_bbox = input_block["bbox"]
        block_text = input_block["text"]
        block_lines = raw_lines
        block_avg_left_boundary = stats[0]
        block_avg_right_boundary = stats[1]
        block_avg_char_width = stats[2]
        block_avg_char_height = stats[3]
        block_font_type = stats[4]
        block_font_size = stats[5]
        block_direction = stats[6]
        block_median_font_size = stats[7]

        new_block["block_id"] = block_id
        new_block["bbox"] = block_bbox
        new_block["text"] = block_text
        new_block["dir"] = block_direction
        new_block["X0"] = block_avg_left_boundary
        new_block["X1"] = block_avg_right_boundary
        new_block["avg_char_width"] = block_avg_char_width
        new_block["avg_char_height"] = block_avg_char_height
        new_block["block_font_type"] = block_font_type
        new_block["block_font_size"] = block_font_size
        new_block["lines"] = block_lines
        new_block["median_font_size"] = block_median_font_size

        return new_block

    def batch_process_blocks(self, pdf_dic):
        """
        This function processes the blocks in batch.

        Parameters
        ----------
        self : object
            The instance of the class.
        ----------
        blocks : list
            Input block is a list of raw blocks.
            Schema can refer to the value of key ""preproc_blocks".

        Returns
        -------
        result_dict : dict
            result dictionary
        """

        for page_id, blocks in pdf_dic.items():
            if page_id.startswith("page_"):
                para_blocks = []
                if "para_blocks" in blocks.keys():
                    input_blocks = blocks["para_blocks"]
                    for input_block in input_blocks:
                        new_block = self.__make_new_block(input_block)
                        para_blocks.append(new_block)

                blocks["para_blocks"] = para_blocks

        return pdf_dic


class DocStatisticsCalculator:
    """
    This class calculates the statistics of the document.
    """

    def __init__(self) -> None:
        pass

    def calc_stats_of_doc(self, pdf_dict):
        """
        This function computes the statistics of the document

        Parameters
        ----------
        result_dict : dict
            result dictionary

        Returns
        -------
        statistics : dict
            statistics of the document
        """

        total_text_length = 0
        total_num_blocks = 0

        for page_id, blocks in pdf_dict.items():
            if page_id.startswith("page_"):
                if "para_blocks" in blocks.keys():
                    para_blocks = blocks["para_blocks"]
                    for para_block in para_blocks:
                        total_text_length += len(para_block["text"])
                        total_num_blocks += 1

        avg_text_length = total_text_length / total_num_blocks if total_num_blocks else 0

        font_list = []

        for page_id, blocks in pdf_dict.items():
            if page_id.startswith("page_"):
                if "para_blocks" in blocks.keys():
                    input_blocks = blocks["para_blocks"]
                    for input_block in input_blocks:
                        block_text_length = len(input_block.get("text", ""))
                        if block_text_length < avg_text_length * 0.5:
                            continue
                        block_font_type = safe_get(input_block, "block_font_type", "")
                        block_font_size = safe_get(input_block, "block_font_size", 0)
                        font_list.append((block_font_type, block_font_size))

        font_counter = Counter(font_list)
        most_common_font = font_counter.most_common(1)[0] if font_list else (("", 0), 0)
        second_most_common_font = font_counter.most_common(2)[1] if len(font_counter) > 1 else (("", 0), 0)

        statistics = {
            "num_pages": 0,
            "num_blocks": 0,
            "num_paras": 0,
            "num_titles": 0,
            "num_header_blocks": 0,
            "num_footer_blocks": 0,
            "num_watermark_blocks": 0,
            "num_vertical_margin_note_blocks": 0,
            "most_common_font_type": most_common_font[0][0],
            "most_common_font_size": most_common_font[0][1],
            "number_of_most_common_font": most_common_font[1],
            "second_most_common_font_type": second_most_common_font[0][0],
            "second_most_common_font_size": second_most_common_font[0][1],
            "number_of_second_most_common_font": second_most_common_font[1],
            "avg_text_length": avg_text_length,
        }

        for page_id, blocks in pdf_dict.items():
            if page_id.startswith("page_"):
                blocks = pdf_dict[page_id]["para_blocks"]
                statistics["num_pages"] += 1
                for block_id, block_data in enumerate(blocks):
                    statistics["num_blocks"] += 1

                    if "paras" in block_data.keys():
                        statistics["num_paras"] += len(block_data["paras"])

                    for line in block_data["lines"]:
                        if line.get("is_title", 0):
                            statistics["num_titles"] += 1

                    if block_data.get("is_header", 0):
                        statistics["num_header_blocks"] += 1
                    if block_data.get("is_footer", 0):
                        statistics["num_footer_blocks"] += 1
                    if block_data.get("is_watermark", 0):
                        statistics["num_watermark_blocks"] += 1
                    if block_data.get("is_vertical_margin_note", 0):
                        statistics["num_vertical_margin_note_blocks"] += 1

        pdf_dict["statistics"] = statistics

        return pdf_dict


class TitleProcessor:
    """
    This class processes the title.
    """

    def __init__(self, *doc_statistics) -> None:
        if len(doc_statistics) > 0:
            self.doc_statistics = doc_statistics[0]

        self.nlp_model = NLPModels()
        self.MAX_TITLE_LEVEL = 3
        self.numbered_title_pattern = r"""
            ^                                 # 行首
            (                                 # 开始捕获组
                [\(\（]\d+[\)\）]              # 括号内数字，支持中文和英文括号，例如：(1) 或 （1）
                |\d+[\)\）]\s                  # 数字后跟右括号和空格，支持中文和英文括号，例如：2) 或 2）
                |[\(\（][A-Z][\)\）]            # 括号内大写字母，支持中文和英文括号，例如：(A) 或 （A）
                |[A-Z][\)\）]\s                # 大写字母后跟右括号和空格，例如：A) 或 A）
                |[\(\（][IVXLCDM]+[\)\）]       # 括号内罗马数字，支持中文和英文括号，例如：(I) 或 （I）
                |[IVXLCDM]+[\)\）]\s            # 罗马数字后跟右括号和空格，例如：I) 或 I）
                |\d+(\.\d+)*\s                # 数字或复合数字编号后跟空格，例如：1. 或 3.2.1 
                |[一二三四五六七八九十百千]+[、\s]       # 中文序号后跟顿号和空格，例如：一、
                |[\（|\(][一二三四五六七八九十百千]+[\）|\)]\s*  # 中文括号内中文序号后跟空格，例如：（一）
                |[A-Z]\.\d+(\.\d+)?\s         # 大写字母后跟点和数字，例如：A.1 或 A.1.1
                |[\(\（][a-z][\)\）]            # 括号内小写字母，支持中文和英文括号，例如：(a) 或 （a）
                |[a-z]\)\s                    # 小写字母后跟右括号和空格，例如：a) 
                |[A-Z]-\s                     # 大写字母后跟短横线和空格，例如：A- 
                |\w+:\s                       # 英文序号词后跟冒号和空格，例如：First: 
                |第[一二三四五六七八九十百千]+[章节部分条款]\s # 以“第”开头的中文标题后跟空格
                |[IVXLCDM]+\.                 # 罗马数字后跟点，例如：I.
                |\d+\.\s                      # 单个数字后跟点和空格，例如：1. 
            )                                 # 结束捕获组
            .+                                # 标题的其余部分
        """

    def _is_potential_title(
        self,
        curr_line,
        prev_line,
        prev_line_is_title,
        next_line,
        avg_char_width,
        avg_char_height,
        median_font_size,
    ):
        """
        This function checks if the line is a potential title.

        Parameters
        ----------
        curr_line : dict
            current line
        prev_line : dict
            previous line
        next_line : dict
            next line
        avg_char_width : float
            average of char widths
        avg_char_height : float
            average of line heights

        Returns
        -------
        bool
            True if the line is a potential title, False otherwise.
        """

        def __is_line_centered(line_bbox, page_bbox, avg_char_width):
            """
            This function checks if the line is centered on the page

            Parameters
            ----------
            line_bbox : list
                bbox of the line
            page_bbox : list
                bbox of the page
            avg_char_width : float
                average of char widths

            Returns
            -------
            bool
                True if the line is centered on the page, False otherwise.
            """
            horizontal_ratio = 0.5
            horizontal_thres = horizontal_ratio * avg_char_width

            x0, _, x1, _ = line_bbox
            _, _, page_x1, _ = page_bbox

            return abs((x0 + x1) / 2 - page_x1 / 2) < horizontal_thres

        def __is_bold_font_line(line):
            """
            Check if a line contains any bold font style.
            """

            def _is_bold_span(span):
                # if span text is empty or only contains space, return False
                if not span["text"].strip():
                    return False

                return bool(span["flags"] & 2**4)  # Check if the font is bold

            for span in line["spans"]:
                if not _is_bold_span(span):
                    return False

            return True

        def __is_italic_font_line(line):
            """
            Check if a line contains any italic font style.
            """

            def __is_italic_span(span):
                return bool(span["flags"] & 2**1)  # Check if the font is italic

            for span in line["spans"]:
                if not __is_italic_span(span):
                    return False

            return True

        def __is_punctuation_heavy(line_text):
            """
            Check if the line contains a high ratio of punctuation marks, which may indicate
            that the line is not a title.

            Parameters:
            line_text (str): Text of the line.

            Returns:
            bool: True if the line is heavy with punctuation, False otherwise.
            """
            # Pattern for common title format like "X.Y. Title"
            pattern = r"\b\d+\.\d+\..*\b"

            # If the line matches the title format, return False
            if re.match(pattern, line_text.strip()):
                return False

            # Find all punctuation marks in the line
            punctuation_marks = re.findall(r"[^\w\s]", line_text)
            number_of_punctuation_marks = len(punctuation_marks)

            text_length = len(line_text)

            if text_length == 0:
                return False

            punctuation_ratio = number_of_punctuation_marks / text_length
            if punctuation_ratio >= 0.1:
                return True

            return False

        def __has_mixed_font_styles(spans, strict_mode=False):
            """
            This function checks if the line has mixed font styles, the strict mode will compare the font types

            Parameters
            ----------
            spans : list
                spans of the line
            strict_mode : bool
                True for strict mode, the font types will be fully compared
                False for non-strict mode, the font types will be compared by the most longest common prefix

            Returns
            -------
            bool
                True if the line has mixed font styles, False otherwise.
            """
            if strict_mode:
                font_styles = set()
                for span in spans:
                    font_style = span["font"].lower()
                    font_styles.add(font_style)

                return len(font_styles) > 1

            else:  # non-strict mode
                font_styles = []
                for span in spans:
                    font_style = span["font"].lower()
                    font_styles.append(font_style)

                if len(font_styles) > 1:
                    longest_common_prefix = os.path.commonprefix(font_styles)
                    if len(longest_common_prefix) > 0:
                        return False
                    else:
                        return True
                else:
                    return False

        def __is_different_font_type_from_neighbors(curr_line_font_type, prev_line_font_type, next_line_font_type):
            """
            This function checks if the current line has a different font type from the previous and next lines

            Parameters
            ----------
            curr_line_font_type : str
                font type of the current line
            prev_line_font_type : str
                font type of the previous line
            next_line_font_type : str
                font type of the next line

            Returns
            -------
            bool
                True if the current line has a different font type from the previous and next lines, False otherwise.
            """
            return all(
                curr_line_font_type != other_font_type.lower()
                for other_font_type in [prev_line_font_type, next_line_font_type]
                if other_font_type is not None
            )

        def __is_larger_font_size_from_neighbors(curr_line_font_size, prev_line_font_size, next_line_font_size):
            """
            This function checks if the current line has a larger font size than the previous and next lines

            Parameters
            ----------
            curr_line_font_size : float
                font size of the current line
            prev_line_font_size : float
                font size of the previous line
            next_line_font_size : float
                font size of the next line

            Returns
            -------
            bool
                True if the current line has a larger font size than the previous and next lines, False otherwise.
            """
            return all(
                curr_line_font_size > other_font_size * 1.2
                for other_font_size in [prev_line_font_size, next_line_font_size]
                if other_font_size is not None
            )

        def __is_similar_to_pre_line(curr_line_font_type, prev_line_font_type, curr_line_font_size, prev_line_font_size):
            """
            This function checks if the current line is similar to the previous line

            Parameters
            ----------
            curr_line : dict
                current line
            prev_line : dict
                previous line

            Returns
            -------
            bool
                True if the current line is similar to the previous line, False otherwise.
            """

            if curr_line_font_type == prev_line_font_type and curr_line_font_size == prev_line_font_size:
                return True
            else:
                return False

        def __is_same_font_type_of_docAvg(curr_line_font_type):
            """
            This function checks if the current line has the same font type as the document average font type

            Parameters
            ----------
            curr_line_font_type : str
                font type of the current line

            Returns
            -------
            bool
                True if the current line has the same font type as the document average font type, False otherwise.
            """
            doc_most_common_font_type = safe_get(self.doc_statistics, "most_common_font_type", "").lower()
            doc_second_most_common_font_type = safe_get(self.doc_statistics, "second_most_common_font_type", "").lower()

            return curr_line_font_type.lower() in [doc_most_common_font_type, doc_second_most_common_font_type]

        def __is_font_size_not_less_than_docAvg(curr_line_font_size, ratio: float = 1):
            """
            This function checks if the current line has a large enough font size

            Parameters
            ----------
            curr_line_font_size : float
                font size of the current line
            ratio : float
                ratio of the current line font size to the document average font size

            Returns
            -------
            bool
                True if the current line has a large enough font size, False otherwise.
            """
            doc_most_common_font_size = safe_get(self.doc_statistics, "most_common_font_size", 0)
            doc_second_most_common_font_size = safe_get(self.doc_statistics, "second_most_common_font_size", 0)
            doc_avg_font_size = min(doc_most_common_font_size, doc_second_most_common_font_size)

            return curr_line_font_size >= doc_avg_font_size * ratio

        def __is_sufficient_spacing_above_and_below(
            curr_line_bbox,
            prev_line_bbox,
            next_line_bbox,
            avg_char_height,
            median_font_size,
        ):
            """
            This function checks if the current line has sufficient spacing above and below

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
            avg_char_height : float
                average of line heights

            Returns
            -------
            bool
                True if the current line has sufficient spacing above and below, False otherwise.
            """
            vertical_ratio = 1.25
            vertical_thres = vertical_ratio * median_font_size

            _, y0, _, y1 = curr_line_bbox

            sufficient_spacing_above = False
            if prev_line_bbox:
                vertical_spacing_above = min(y0 - prev_line_bbox[1], y1 - prev_line_bbox[3])
                sufficient_spacing_above = vertical_spacing_above > vertical_thres
            else:
                sufficient_spacing_above = True

            sufficient_spacing_below = False
            if next_line_bbox:
                vertical_spacing_below = min(next_line_bbox[1] - y0, next_line_bbox[3] - y1)
                sufficient_spacing_below = vertical_spacing_below > vertical_thres
            else:
                sufficient_spacing_below = True

            return (sufficient_spacing_above, sufficient_spacing_below)

        def __is_word_list_line_by_rules(curr_line_text):
            """
            This function checks if the current line is a word list

            Parameters
            ----------
            curr_line_text : str
                text of the current line

            Returns
            -------
            bool
                True if the current line is a name list, False otherwise.
            """
            # name_list_pattern = r"([a-zA-Z][a-zA-Z\s]{0,20}[a-zA-Z]|[\u4e00-\u9fa5·]{2,16})(?=[，,;；\s]|$)"
            name_list_pattern = r"(?<![\u4e00-\u9fa5])([A-Z][a-z]{0,19}\s[A-Z][a-z]{0,19}|[\u4e00-\u9fa5]{2,6})(?=[，,;；\s]|$)"

            compiled_pattern = re.compile(name_list_pattern)

            if compiled_pattern.search(curr_line_text):
                return True
            else:
                return False

        def __get_text_catgr_by_nlp(curr_line_text):
            """
            This function checks if the current line is a name list using nlp model, such as spacy

            Parameters
            ----------
            curr_line_text : str
                text of the current line

            Returns
            -------
            bool
                True if the current line is a name list, False otherwise.
            """

            result = self.nlp_model.detect_entity_catgr_using_nlp(curr_line_text)

            return result

        def __is_numbered_title(curr_line_text):
            """
            This function checks if the current line is a numbered list

            Parameters
            ----------
            curr_line_text : str
                text of the current line

            Returns
            -------
            bool
                True if the current line is a numbered list, False otherwise.
            """

            compiled_pattern = re.compile(self.numbered_title_pattern, re.VERBOSE)

            if compiled_pattern.search(curr_line_text):
                return True
            else:
                return False

        def __is_end_with_ending_puncs(line_text):
            """
            This function checks if the current line ends with a ending punctuation mark

            Parameters
            ----------
            line_text : str
                text of the current line

            Returns
            -------
            bool
                True if the current line ends with a punctuation mark, False otherwise.
            """
            end_puncs = [".", "?", "!", "。", "？", "！", "…"]

            line_text = line_text.rstrip()
            if line_text[-1] in end_puncs:
                return True

            return False

        def __contains_only_no_meaning_symbols(line_text):
            """
            This function checks if the current line contains only symbols that have no meaning, if so, it is not a title.
            Situation contains:
            1. Only have punctuation marks
            2. Only have other non-meaning symbols

            Parameters
            ----------
            line_text : str
                text of the current line

            Returns
            -------
            bool
                True if the current line contains only symbols that have no meaning, False otherwise.
            """

            punctuation_marks = re.findall(r"[^\w\s]", line_text)  # find all punctuation marks
            number_of_punctuation_marks = len(punctuation_marks)

            text_length = len(line_text)

            if text_length == 0:
                return False

            punctuation_ratio = number_of_punctuation_marks / text_length
            if punctuation_ratio >= 0.9:
                return True

            return False

        def __is_equation(line_text):
            """
            This function checks if the current line is an equation.

            Parameters
            ----------
            line_text : str

            Returns
            -------
            bool
                True if the current line is an equation, False otherwise.
            """
            equation_reg = r"\$.*?\\overline.*?\$"  # to match interline equations

            if re.search(equation_reg, line_text):
                return True
            else:
                return False

        def __is_title_by_len(text, max_length=200):
            """
            This function checks if the current line is a title by length.

            Parameters
            ----------
            text : str
                text of the current line

            max_length : int
                max length of the title

            Returns
            -------
            bool
                True if the current line is a title, False otherwise.

            """
            text = text.strip()
            return len(text) <= max_length

        def __compute_line_font_type_and_size(curr_line):
            """
            This function computes the font type and font size of the line.

            Parameters
            ----------
            line : dict
                line

            Returns
            -------
            font_type : str
                font type of the line
            font_size : float
                font size of the line
            """
            spans = curr_line["spans"]
            max_accumulated_length = 0
            max_span_font_size = curr_line["spans"][0]["size"]  # default value, float type
            max_span_font_type = curr_line["spans"][0]["font"].lower()  # default value, string type
            for span in spans:
                if span["text"].isspace():
                    continue
                span_length = span["bbox"][2] - span["bbox"][0]
                if span_length > max_accumulated_length:
                    max_accumulated_length = span_length
                    max_span_font_size = span["size"]
                    max_span_font_type = span["font"].lower()

            return max_span_font_type, max_span_font_size

        def __is_a_consistent_sub_title(pre_line, curr_line):
            """
            This function checks if the current line is a consistent sub title.

            Parameters
            ----------
            pre_line : dict
                previous line
            curr_line : dict
                current line

            Returns
            -------
            bool
                True if the current line is a consistent sub title, False otherwise.
            """
            if pre_line is None:
                return False

            start_letter_of_pre_line = pre_line["text"][0]
            start_letter_of_curr_line = curr_line["text"][0]

            has_same_prefix_digit = (
                start_letter_of_pre_line.isdigit()
                and start_letter_of_curr_line.isdigit()
                and start_letter_of_pre_line == start_letter_of_curr_line
            )

            # prefix text of curr_line satisfies the following title format: x.x
            prefix_text_pattern = r"^\d+\.\d+"
            has_subtitle_format = re.match(prefix_text_pattern, curr_line["text"])

            if has_same_prefix_digit or has_subtitle_format:
                return True

        """
        Title detecting main Process.
        """

        """
        Basic features about the current line.
        """
        curr_line_bbox = curr_line["bbox"]
        curr_line_text = curr_line["text"]
        curr_line_font_type, curr_line_font_size = __compute_line_font_type_and_size(curr_line)

        if len(curr_line_text.strip()) == 0:  # skip empty lines
            return False, False

        prev_line_bbox = prev_line["bbox"] if prev_line else None
        if prev_line:
            prev_line_font_type, prev_line_font_size = __compute_line_font_type_and_size(prev_line)
        else:
            prev_line_font_type, prev_line_font_size = None, None

        next_line_bbox = next_line["bbox"] if next_line else None
        if next_line:
            next_line_font_type, next_line_font_size = __compute_line_font_type_and_size(next_line)
        else:
            next_line_font_type, next_line_font_size = None, None

        """
        Aggregated features about the current line.
        """
        is_italc_font = __is_italic_font_line(curr_line)
        is_bold_font = __is_bold_font_line(curr_line)

        is_font_size_little_less_than_doc_avg = __is_font_size_not_less_than_docAvg(curr_line_font_size, ratio=0.8)
        is_font_size_not_less_than_doc_avg = __is_font_size_not_less_than_docAvg(curr_line_font_size, ratio=1)
        is_much_larger_font_than_doc_avg = __is_font_size_not_less_than_docAvg(curr_line_font_size, ratio=1.6)

        is_not_same_font_type_of_docAvg = not __is_same_font_type_of_docAvg(curr_line_font_type)

        is_potential_title_font = is_bold_font or is_font_size_not_less_than_doc_avg or is_not_same_font_type_of_docAvg

        is_mix_font_styles_strict = __has_mixed_font_styles(curr_line["spans"], strict_mode=True)
        is_mix_font_styles_loose = __has_mixed_font_styles(curr_line["spans"], strict_mode=False)

        is_punctuation_heavy = __is_punctuation_heavy(curr_line_text)

        is_word_list_line_by_rules = __is_word_list_line_by_rules(curr_line_text)
        is_person_or_org_list_line_by_nlp = __get_text_catgr_by_nlp(curr_line_text) in ["PERSON", "GPE", "ORG"]

        is_font_size_larger_than_neighbors = __is_larger_font_size_from_neighbors(
            curr_line_font_size, prev_line_font_size, next_line_font_size
        )

        is_font_type_diff_from_neighbors = __is_different_font_type_from_neighbors(
            curr_line_font_type, prev_line_font_type, next_line_font_type
        )

        has_sufficient_spaces_above, has_sufficient_spaces_below = __is_sufficient_spacing_above_and_below(
            curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_height, median_font_size
        )

        is_similar_to_pre_line = __is_similar_to_pre_line(
            curr_line_font_type, prev_line_font_type, curr_line_font_size, prev_line_font_size
        )

        is_consis_sub_title = __is_a_consistent_sub_title(prev_line, curr_line)

        """
        Further aggregated features about the current line.
        
        Attention:
            Features that start with __ are for internal use.
        """

        __is_line_left_aligned_from_neighbors = is_line_left_aligned_from_neighbors(
            curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width
        )
        __is_font_diff_from_neighbors = is_font_size_larger_than_neighbors or is_font_type_diff_from_neighbors
        is_a_left_inline_title = (
            is_mix_font_styles_strict and __is_line_left_aligned_from_neighbors and __is_font_diff_from_neighbors
        )

        is_title_by_check_prev_line = prev_line is None and has_sufficient_spaces_above and is_potential_title_font
        is_title_by_check_next_line = next_line is None and has_sufficient_spaces_below and is_potential_title_font

        is_title_by_check_pre_and_next_line = (
            (prev_line is not None or next_line is not None)
            and has_sufficient_spaces_above
            and has_sufficient_spaces_below
            and is_potential_title_font
        )

        is_numbered_title = __is_numbered_title(curr_line_text) and (
            (has_sufficient_spaces_above or prev_line is None) and (has_sufficient_spaces_below or next_line is None)
        )

        is_not_end_with_ending_puncs = not __is_end_with_ending_puncs(curr_line_text)

        is_not_only_no_meaning_symbols = not __contains_only_no_meaning_symbols(curr_line_text)

        is_equation = __is_equation(curr_line_text)

        is_title_by_len = __is_title_by_len(curr_line_text)

        """
        Decide if the line is a title.
        """

        is_title = (
            is_not_end_with_ending_puncs  # not end with ending punctuation marks
            and is_not_only_no_meaning_symbols  # not only have no meaning symbols
            and is_title_by_len  # is a title by length, default max length is 200
            and not is_equation  # an interline equation should never be a title
            and is_potential_title_font  # is a potential title font, which is bold or larger than the document average font size or not the same font type as the document average font type
            and (
                (is_not_same_font_type_of_docAvg and is_font_size_not_less_than_doc_avg)
                or (is_bold_font and is_much_larger_font_than_doc_avg and is_not_same_font_type_of_docAvg)
                or (
                    is_much_larger_font_than_doc_avg
                    and (is_title_by_check_prev_line or is_title_by_check_next_line or is_title_by_check_pre_and_next_line)
                )
                or (
                    is_font_size_little_less_than_doc_avg
                    and is_bold_font
                    and (is_title_by_check_prev_line or is_title_by_check_next_line or is_title_by_check_pre_and_next_line)
                )
            )  # Consider the following situations: bold font, much larger font than doc avg, not same font type as doc avg, sufficient spacing above and below
            and (
                (
                    not is_person_or_org_list_line_by_nlp
                    and (
                        is_much_larger_font_than_doc_avg
                        or (is_not_same_font_type_of_docAvg and is_font_size_not_less_than_doc_avg)
                    )
                )
                or (
                    not (is_word_list_line_by_rules and is_person_or_org_list_line_by_nlp)
                    and not is_a_left_inline_title
                    and not is_punctuation_heavy
                    and (is_title_by_check_prev_line or is_title_by_check_next_line or is_title_by_check_pre_and_next_line)
                )
                or (
                    is_person_or_org_list_line_by_nlp
                    and (is_bold_font and is_much_larger_font_than_doc_avg and is_not_same_font_type_of_docAvg)
                    and (is_bold_font and is_much_larger_font_than_doc_avg and is_not_same_font_type_of_docAvg)
                )
                or (is_numbered_title and not is_a_left_inline_title)
            )  # Exclude the following situations: person/org list
        )
        # ) or (prev_line_is_title and is_consis_sub_title)

        is_name_or_org_list_to_be_removed = (
            (is_person_or_org_list_line_by_nlp)
            and is_punctuation_heavy
            and (is_title_by_check_prev_line or is_title_by_check_next_line or is_title_by_check_pre_and_next_line)
        ) and not is_title

        if is_name_or_org_list_to_be_removed:
            is_author_or_org_list = True
        else:
            is_author_or_org_list = False

        # return is_title, is_author_or_org_list

        """
        # print reason why the line is a title
        if is_title:
            print_green("This line is a title.")
            print_green("↓" * 10)
            print()
            print("curr_line_text: ", curr_line_text)
            print()

        # print reason why the line is not a title
        line_text = curr_line_text.strip()
        test_text = "Career/Personal Life"
        text_content_condition = line_text == test_text
        
        if not is_title and text_content_condition: # Print specific line
        # if not is_title: # Print each line
            print_red("This line is not a title.")
            print_red("↓" * 10)

            print()
            print("curr_line_text: ", curr_line_text)
            print()

            if is_not_end_with_ending_puncs:
                print_green(f"is_not_end_with_ending_puncs")
            else:
                print_red(f"is_end_with_ending_puncs")

            if is_not_only_no_meaning_symbols:
                print_green(f"is_not_only_no_meaning_symbols")
            else:
                print_red(f"is_only_no_meaning_symbols")

            if is_title_by_len:
                print_green(f"is_title_by_len: {is_title_by_len}")
            else:
                print_red(f"is_not_title_by_len: {is_title_by_len}")

            if is_equation:
                print_red(f"is_equation")
            else:
                print_green(f"is_not_equation")

            if is_potential_title_font:
                print_green(f"is_potential_title_font")
            else:
                print_red(f"is_not_potential_title_font")

            if is_punctuation_heavy:
                print_red("is_punctuation_heavy")
            else:
                print_green("is_not_punctuation_heavy")

            if is_bold_font:
                print_green(f"is_bold_font")
            else:
                print_red(f"is_not_bold_font")

            if is_font_size_not_less_than_doc_avg:
                print_green(f"is_larger_font_than_doc_avg")
            else:
                print_red(f"is_not_larger_font_than_doc_avg")

            if is_much_larger_font_than_doc_avg:
                print_green(f"is_much_larger_font_than_doc_avg")
            else:
                print_red(f"is_not_much_larger_font_than_doc_avg")

            if is_not_same_font_type_of_docAvg:
                print_green(f"is_not_same_font_type_of_docAvg")
            else:
                print_red(f"is_same_font_type_of_docAvg")

            if is_word_list_line_by_rules:
                print_red("is_word_list_line_by_rules")
            else:
                print_green("is_not_name_list_by_rules")

            if is_person_or_org_list_line_by_nlp:
                print_red("is_person_or_org_list_line_by_nlp")
            else:
                print_green("is_not_person_or_org_list_line_by_nlp")

            if not is_numbered_title:
                print_red("is_not_numbered_title")
            else:
                print_green("is_numbered_title")

            if is_a_left_inline_title:
                print_red("is_a_left_inline_title")
            else:
                print_green("is_not_a_left_inline_title")

            if not is_title_by_check_prev_line:
                print_red("is_not_title_by_check_prev_line")
            else:
                print_green("is_title_by_check_prev_line")

            if not is_title_by_check_next_line:
                print_red("is_not_title_by_check_next_line")
            else:
                print_green("is_title_by_check_next_line")

            if not is_title_by_check_pre_and_next_line:
                print_red("is_not_title_by_check_pre_and_next_line")
            else:
                print_green("is_title_by_check_pre_and_next_line")

        # print_green("Common features:")
        # print_green("↓" * 10)

        # print(f"    curr_line_font_type: {curr_line_font_type}")
        # print(f"    curr_line_font_size: {curr_line_font_size}")
        # print()

        """

        return is_title, is_author_or_org_list

    def _detect_title(self, input_block):
        """
        Use the functions 'is_potential_title' to detect titles of each paragraph block.
        If a line is a title, then the value of key 'is_title' of the line will be set to True.
        """

        raw_lines = input_block["lines"]

        prev_line_is_title_flag = False

        for i, curr_line in enumerate(raw_lines):
            prev_line = raw_lines[i - 1] if i > 0 else None
            next_line = raw_lines[i + 1] if i < len(raw_lines) - 1 else None

            blk_avg_char_width = input_block["avg_char_width"]
            blk_avg_char_height = input_block["avg_char_height"]
            blk_media_font_size = input_block["median_font_size"]

            is_title, is_author_or_org_list = self._is_potential_title(
                curr_line,
                prev_line,
                prev_line_is_title_flag,
                next_line,
                blk_avg_char_width,
                blk_avg_char_height,
                blk_media_font_size,
            )

            if is_title:
                curr_line["is_title"] = is_title
                prev_line_is_title_flag = True
            else:
                curr_line["is_title"] = False
                prev_line_is_title_flag = False

            # print(f"curr_line['text']: {curr_line['text']}")
            # print(f"curr_line['is_title']: {curr_line['is_title']}")
            # print(f"prev_line['text']: {prev_line['text'] if prev_line else None}")
            # print(f"prev_line_is_title_flag: {prev_line_is_title_flag}")
            # print()

            if is_author_or_org_list:
                curr_line["is_author_or_org_list"] = is_author_or_org_list
            else:
                curr_line["is_author_or_org_list"] = False

        return input_block

    def batch_detect_titles(self, pdf_dic):
        """
        This function batch process the blocks to detect titles.

        Parameters
        ----------
        pdf_dict : dict
            result dictionary

        Returns
        -------
        pdf_dict : dict
            result dictionary
        """
        num_titles = 0

        for page_id, blocks in pdf_dic.items():
            if page_id.startswith("page_"):
                para_blocks = []
                if "para_blocks" in blocks.keys():
                    para_blocks = blocks["para_blocks"]

                    all_single_line_blocks = []
                    for block in para_blocks:
                        if len(block["lines"]) == 1:
                            all_single_line_blocks.append(block)

                    new_para_blocks = []
                    if not len(all_single_line_blocks) == len(para_blocks):  # Not all blocks are single line blocks.
                        for para_block in para_blocks:
                            new_block = self._detect_title(para_block)
                            new_para_blocks.append(new_block)
                            num_titles += sum([line.get("is_title", 0) for line in new_block["lines"]])
                    else:  # All blocks are single line blocks.
                        for para_block in para_blocks:
                            new_para_blocks.append(para_block)
                            num_titles += sum([line.get("is_title", 0) for line in para_block["lines"]])
                    para_blocks = new_para_blocks

                blocks["para_blocks"] = para_blocks

                for para_block in para_blocks:
                    all_titles = all(safe_get(line, "is_title", False) for line in para_block["lines"])
                    para_text_len = sum([len(line["text"]) for line in para_block["lines"]])
                    if (
                        all_titles and para_text_len < 200
                    ):  # total length of the paragraph is less than 200, more than this should not be a title
                        para_block["is_block_title"] = 1
                    else:
                        para_block["is_block_title"] = 0

                    all_name_or_org_list_to_be_removed = all(
                        safe_get(line, "is_author_or_org_list", False) for line in para_block["lines"]
                    )
                    if all_name_or_org_list_to_be_removed and page_id == "page_0":
                        para_block["is_block_an_author_or_org_list"] = 1
                    else:
                        para_block["is_block_an_author_or_org_list"] = 0

        pdf_dic["statistics"]["num_titles"] = num_titles

        return pdf_dic

    def _recog_title_level(self, title_blocks):
        """
        This function determines the title level based on the font size of the title.

        Parameters
        ----------
        title_blocks : list

        Returns
        -------
        title_blocks : list
        """

        font_sizes = np.array([safe_get(tb["block"], "block_font_size", 0) for tb in title_blocks])

        # Use the mean and std of font sizes to remove extreme values
        mean_font_size = np.mean(font_sizes)
        std_font_size = np.std(font_sizes)
        min_extreme_font_size = mean_font_size - std_font_size  # type: ignore
        max_extreme_font_size = mean_font_size + std_font_size  # type: ignore

        # Compute the threshold for title level
        middle_font_sizes = font_sizes[(font_sizes > min_extreme_font_size) & (font_sizes < max_extreme_font_size)]
        if middle_font_sizes.size > 0:
            middle_mean_font_size = np.mean(middle_font_sizes)
            level_threshold = middle_mean_font_size
        else:
            level_threshold = mean_font_size

        for tb in title_blocks:
            title_block = tb["block"]
            title_font_size = safe_get(title_block, "block_font_size", 0)

            current_level = 1  # Initialize title level, the biggest level is 1

            # print(f"Before adjustment by font size, {current_level}")
            if title_font_size >= max_extreme_font_size:
                current_level = 1
            elif title_font_size <= min_extreme_font_size:
                current_level = 3
            elif float(title_font_size) >= float(level_threshold):
                current_level = 2
            else:
                current_level = 3
            # print(f"After adjustment by font size, {current_level}")

            title_block["block_title_level"] = current_level

        return title_blocks

    def batch_recog_title_level(self, pdf_dic):
        """
        This function batch process the blocks to recognize title level.

        Parameters
        ----------
        pdf_dict : dict
            result dictionary

        Returns
        -------
        pdf_dict : dict
            result dictionary
        """
        title_blocks = []

        # Collect all titles
        for page_id, blocks in pdf_dic.items():
            if page_id.startswith("page_"):
                para_blocks = blocks.get("para_blocks", [])
                for block in para_blocks:
                    if block.get("is_block_title"):
                        title_obj = {"page_id": page_id, "block": block}
                        title_blocks.append(title_obj)

        # Determine title level
        if title_blocks:
            # Determine title level based on font size
            title_blocks = self._recog_title_level(title_blocks)

        return pdf_dic


class BlockTerminationProcessor:
    """
    This class is used to process the block termination.
    """

    def __init__(self) -> None:
        pass

    def _is_consistent_lines(
        self,
        curr_line,
        prev_line,
        next_line,
        consistent_direction,  # 0 for prev, 1 for next, 2 for both
    ):
        """
        This function checks if the line is consistent with its neighbors

        Parameters
        ----------
        curr_line : dict
            current line
        prev_line : dict
            previous line
        next_line : dict
            next line
        consistent_direction : int
            0 for prev, 1 for next, 2 for both

        Returns
        -------
        bool
            True if the line is consistent with its neighbors, False otherwise.
        """

        curr_line_font_size = curr_line["spans"][0]["size"]
        curr_line_font_type = curr_line["spans"][0]["font"].lower()

        if consistent_direction == 0:
            if prev_line:
                prev_line_font_size = prev_line["spans"][0]["size"]
                prev_line_font_type = prev_line["spans"][0]["font"].lower()
                return curr_line_font_size == prev_line_font_size and curr_line_font_type == prev_line_font_type
            else:
                return False

        elif consistent_direction == 1:
            if next_line:
                next_line_font_size = next_line["spans"][0]["size"]
                next_line_font_type = next_line["spans"][0]["font"].lower()
                return curr_line_font_size == next_line_font_size and curr_line_font_type == next_line_font_type
            else:
                return False

        elif consistent_direction == 2:
            if prev_line and next_line:
                prev_line_font_size = prev_line["spans"][0]["size"]
                prev_line_font_type = prev_line["spans"][0]["font"].lower()
                next_line_font_size = next_line["spans"][0]["size"]
                next_line_font_type = next_line["spans"][0]["font"].lower()
                return (curr_line_font_size == prev_line_font_size and curr_line_font_type == prev_line_font_type) and (
                    curr_line_font_size == next_line_font_size and curr_line_font_type == next_line_font_type
                )
            else:
                return False

        else:
            return False

    def _is_regular_line(self, curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, X0, X1, avg_line_height):
        """
        This function checks if the line is a regular line

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
        X0 : float
            median of x0 values, which represents the left average boundary of the page
        X1 : float
            median of x1 values, which represents the right average boundary of the page
        avg_line_height : float
            average of line heights

        Returns
        -------
        bool
            True if the line is a regular line, False otherwise.
        """
        horizontal_ratio = 0.5
        vertical_ratio = 0.5
        horizontal_thres = horizontal_ratio * avg_char_width
        vertical_thres = vertical_ratio * avg_line_height

        x0, y0, x1, y1 = curr_line_bbox

        x0_near_X0 = abs(x0 - X0) < horizontal_thres
        x1_near_X1 = abs(x1 - X1) < horizontal_thres

        prev_line_is_end_of_para = prev_line_bbox and (abs(prev_line_bbox[2] - X1) > avg_char_width)

        sufficient_spacing_above = False
        if prev_line_bbox:
            vertical_spacing_above = y1 - prev_line_bbox[3]
            sufficient_spacing_above = vertical_spacing_above > vertical_thres

        sufficient_spacing_below = False
        if next_line_bbox:
            vertical_spacing_below = next_line_bbox[1] - y0
            sufficient_spacing_below = vertical_spacing_below > vertical_thres

        return (
            (sufficient_spacing_above or sufficient_spacing_below)
            or (not x0_near_X0 and not x1_near_X1)
            or prev_line_is_end_of_para
        )

    def _is_possible_start_of_para(self, curr_line, prev_line, next_line, X0, X1, avg_char_width, avg_font_size):
        """
        This function checks if the line is a possible start of a paragraph

        Parameters
        ----------
        curr_line : dict
            current line
        prev_line : dict
            previous line
        next_line : dict
            next line
        X0 : float
            median of x0 values, which represents the left average boundary of the page
        X1 : float
            median of x1 values, which represents the right average boundary of the page
        avg_char_width : float
            average of char widths
        avg_line_height : float
            average of line heights

        Returns
        -------
        bool
            True if the line is a possible start of a paragraph, False otherwise.
        """
        start_confidence = 0.5  # Initial confidence of the line being a start of a paragraph
        decision_path = []  # Record the decision path

        curr_line_bbox = curr_line["bbox"]
        prev_line_bbox = prev_line["bbox"] if prev_line else None
        next_line_bbox = next_line["bbox"] if next_line else None

        indent_ratio = 1

        vertical_ratio = 1.5
        vertical_thres = vertical_ratio * avg_font_size

        left_horizontal_ratio = 0.5
        left_horizontal_thres = left_horizontal_ratio * avg_char_width

        right_horizontal_ratio = 2.5
        right_horizontal_thres = right_horizontal_ratio * avg_char_width

        x0, y0, x1, y1 = curr_line_bbox

        indent_condition = x0 > X0 + indent_ratio * avg_char_width
        if indent_condition:
            start_confidence += 0.2
            decision_path.append("indent_condition_met")

        x0_near_X0 = abs(x0 - X0) < left_horizontal_thres
        if x0_near_X0:
            start_confidence += 0.1
            decision_path.append("x0_near_X0")

        x1_near_X1 = abs(x1 - X1) < right_horizontal_thres
        if x1_near_X1:
            start_confidence += 0.1
            decision_path.append("x1_near_X1")

        if prev_line is None:
            prev_line_is_end_of_para = True
            start_confidence += 0.2
            decision_path.append("no_prev_line")
        else:
            prev_line_is_end_of_para, _, _ = self._is_possible_end_of_para(prev_line, next_line, X0, X1, avg_char_width)
            if prev_line_is_end_of_para:
                start_confidence += 0.1
                decision_path.append("prev_line_is_end_of_para")

        sufficient_spacing_above = False
        if prev_line_bbox:
            vertical_spacing_above = y1 - prev_line_bbox[3]
            sufficient_spacing_above = vertical_spacing_above > vertical_thres
            if sufficient_spacing_above:
                start_confidence += 0.2
                decision_path.append("sufficient_spacing_above")

        sufficient_spacing_below = False
        if next_line_bbox:
            vertical_spacing_below = next_line_bbox[1] - y0
            sufficient_spacing_below = vertical_spacing_below > vertical_thres
            if sufficient_spacing_below:
                start_confidence += 0.2
                decision_path.append("sufficient_spacing_below")

        is_regular_line = self._is_regular_line(
            curr_line_bbox, prev_line_bbox, next_line_bbox, avg_char_width, X0, X1, avg_font_size
        )
        if is_regular_line:
            start_confidence += 0.1
            decision_path.append("is_regular_line")

        is_start_of_para = (
            (sufficient_spacing_above or sufficient_spacing_below)
            or (indent_condition)
            or (not indent_condition and x0_near_X0 and x1_near_X1 and not is_regular_line)
            or prev_line_is_end_of_para
        )
        return (is_start_of_para, start_confidence, decision_path)

    def _is_possible_end_of_para(self, curr_line, next_line, X0, X1, avg_char_width):
        """
        This function checks if the line is a possible end of a paragraph

        Parameters
        ----------
        curr_line : dict
            current line
        next_line : dict
            next line
        X0 : float
            median of x0 values, which represents the left average boundary of the page
        X1 : float
            median of x1 values, which represents the right average boundary of the page
        avg_char_width : float
            average of char widths

        Returns
        -------
        bool
            True if the line is a possible end of a paragraph, False otherwise.
        """

        end_confidence = 0.5  # Initial confidence of the line being a end of a paragraph
        decision_path = []  # Record the decision path

        curr_line_bbox = curr_line["bbox"]
        next_line_bbox = next_line["bbox"] if next_line else None

        left_horizontal_ratio = 0.5
        right_horizontal_ratio = 0.5

        x0, _, x1, y1 = curr_line_bbox
        next_x0, next_y0, _, _ = next_line_bbox if next_line_bbox else (0, 0, 0, 0)

        x0_near_X0 = abs(x0 - X0) < left_horizontal_ratio * avg_char_width
        if x0_near_X0:
            end_confidence += 0.1
            decision_path.append("x0_near_X0")

        x1_smaller_than_X1 = x1 < X1 - right_horizontal_ratio * avg_char_width
        if x1_smaller_than_X1:
            end_confidence += 0.1
            decision_path.append("x1_smaller_than_X1")

        next_line_is_start_of_para = (
            next_line_bbox
            and (next_x0 > X0 + left_horizontal_ratio * avg_char_width)
            and (not is_line_left_aligned_from_neighbors(curr_line_bbox, None, next_line_bbox, avg_char_width, direction=1))
        )
        if next_line_is_start_of_para:
            end_confidence += 0.2
            decision_path.append("next_line_is_start_of_para")

        is_line_left_aligned_from_neighbors_bool = is_line_left_aligned_from_neighbors(
            curr_line_bbox, None, next_line_bbox, avg_char_width
        )
        if is_line_left_aligned_from_neighbors_bool:
            end_confidence += 0.1
            decision_path.append("line_is_left_aligned_from_neighbors")

        is_line_right_aligned_from_neighbors_bool = is_line_right_aligned_from_neighbors(
            curr_line_bbox, None, next_line_bbox, avg_char_width
        )
        if not is_line_right_aligned_from_neighbors_bool:
            end_confidence += 0.1
            decision_path.append("line_is_not_right_aligned_from_neighbors")

        is_end_of_para = end_with_punctuation(curr_line["text"]) and (
            (x0_near_X0 and x1_smaller_than_X1)
            or (is_line_left_aligned_from_neighbors_bool and not is_line_right_aligned_from_neighbors_bool)
        )

        return (is_end_of_para, end_confidence, decision_path)

    def _cut_paras_per_block(
        self,
        block,
    ):
        """
        Processes a raw block from PyMuPDF and returns the processed block.

        Parameters
        ----------
        raw_block : dict
            A raw block from pymupdf.

        Returns
        -------
        processed_block : dict

        """

        def _construct_para(lines, is_block_title, para_title_level):
            """
            Construct a paragraph from given lines.
            """

            font_sizes = [span["size"] for line in lines for span in line["spans"]]
            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0

            font_colors = [span["color"] for line in lines for span in line["spans"]]
            most_common_font_color = max(set(font_colors), key=font_colors.count) if font_colors else None

            font_type_lengths = {}
            for line in lines:
                for span in line["spans"]:
                    font_type = span["font"]
                    bbox_width = span["bbox"][2] - span["bbox"][0]
                    if font_type in font_type_lengths:
                        font_type_lengths[font_type] += bbox_width
                    else:
                        font_type_lengths[font_type] = bbox_width

            # get the font type with the longest bbox width
            most_common_font_type = max(font_type_lengths, key=font_type_lengths.get) if font_type_lengths else None  # type: ignore

            para_bbox = calculate_para_bbox(lines)
            para_text = " ".join(line["text"] for line in lines)

            return {
                "para_bbox": para_bbox,
                "para_text": para_text,
                "para_font_type": most_common_font_type,
                "para_font_size": avg_font_size,
                "para_font_color": most_common_font_color,
                "is_para_title": is_block_title,
                "para_title_level": para_title_level,
            }

        block_bbox = block["bbox"]
        block_text = block["text"]
        block_lines = block["lines"]

        X0 = safe_get(block, "X0", 0)
        X1 = safe_get(block, "X1", 0)
        avg_char_width = safe_get(block, "avg_char_width", 0)
        avg_char_height = safe_get(block, "avg_char_height", 0)
        avg_font_size = safe_get(block, "avg_font_size", 0)

        is_block_title = safe_get(block, "is_block_title", False)
        para_title_level = safe_get(block, "block_title_level", 0)

        # Segment into paragraphs
        para_ranges = []
        in_paragraph = False
        start_idx_of_para = None

        # Create the processed paragraphs
        processed_paras = {}
        para_bboxes = []
        end_idx_of_para = 0

        for line_index, line in enumerate(block_lines):
            curr_line = line
            prev_line = block_lines[line_index - 1] if line_index > 0 else None
            next_line = block_lines[line_index + 1] if line_index < len(block_lines) - 1 else None

            """
            Start processing paragraphs.
            """

            # Check if the line is the start of a paragraph
            is_start_of_para, start_confidence, decision_path = self._is_possible_start_of_para(
                curr_line, prev_line, next_line, X0, X1, avg_char_width, avg_font_size
            )
            if not in_paragraph and is_start_of_para:
                in_paragraph = True
                start_idx_of_para = line_index

                # print_green(">>> Start of a paragraph")
                # print("    curr_line_text: ", curr_line["text"])
                # print("    start_confidence: ", start_confidence)
                # print("    decision_path: ", decision_path)

            # Check if the line is the end of a paragraph
            is_end_of_para, end_confidence, decision_path = self._is_possible_end_of_para(
                curr_line, next_line, X0, X1, avg_char_width
            )
            if in_paragraph and (is_end_of_para or not next_line):
                para_ranges.append((start_idx_of_para, line_index))
                start_idx_of_para = None
                in_paragraph = False

                # print_red(">>> End of a paragraph")
                # print("    curr_line_text: ", curr_line["text"])
                # print("    end_confidence: ", end_confidence)
                # print("    decision_path: ", decision_path)

        # Add the last paragraph if it is not added
        if in_paragraph and start_idx_of_para is not None:
            para_ranges.append((start_idx_of_para, len(block_lines) - 1))

        # Process the matched paragraphs
        for para_index, (start_idx, end_idx) in enumerate(para_ranges):
            matched_lines = block_lines[start_idx : end_idx + 1]
            para_properties = _construct_para(matched_lines, is_block_title, para_title_level)
            para_key = f"para_{len(processed_paras)}"
            processed_paras[para_key] = para_properties
            para_bboxes.append(para_properties["para_bbox"])
            end_idx_of_para = end_idx + 1

        # Deal with the remaining lines
        if end_idx_of_para < len(block_lines):
            unmatched_lines = block_lines[end_idx_of_para:]
            unmatched_properties = _construct_para(unmatched_lines, is_block_title, para_title_level)
            unmatched_key = f"para_{len(processed_paras)}"
            processed_paras[unmatched_key] = unmatched_properties
            para_bboxes.append(unmatched_properties["para_bbox"])

        block["paras"] = processed_paras

        return block

    def batch_process_blocks(self, pdf_dict):
        """
        Parses the blocks of all pages.

        Parameters
        ----------
        pdf_dict : dict
            PDF dictionary.
        filter_blocks : list
            List of bounding boxes to filter.

        Returns
        -------
        result_dict : dict
            Result dictionary.

        """

        num_paras = 0

        for page_id, page in pdf_dict.items():
            if page_id.startswith("page_"):
                para_blocks = []
                if "para_blocks" in page.keys():
                    input_blocks = page["para_blocks"]
                    for input_block in input_blocks:
                        new_block = self._cut_paras_per_block(input_block)
                        para_blocks.append(new_block)
                        num_paras += len(new_block["paras"])

                page["para_blocks"] = para_blocks

        pdf_dict["statistics"]["num_paras"] = num_paras
        return pdf_dict


class BlockContinuationProcessor:
    """
    This class is used to process the blocks to detect block continuations.
    """

    def __init__(self) -> None:
        pass

    def __is_similar_font_type(self, font_type_1, font_type_2, prefix_length_ratio=0.3):
        """
        This function checks if the two font types are similar.
        Definition of similar font types: the two font types have a common prefix,
        and the length of the common prefix is at least a certain ratio of the length of the shorter font type.

        Parameters
        ----------
        font_type1 : str
            font type 1
        font_type2 : str
            font type 2
        prefix_length_ratio : float
            minimum ratio of the common prefix length to the length of the shorter font type

        Returns
        -------
        bool
            True if the two font types are similar, False otherwise.
        """

        if isinstance(font_type_1, list):
            font_type_1 = font_type_1[0] if font_type_1 else ""
        if isinstance(font_type_2, list):
            font_type_2 = font_type_2[0] if font_type_2 else ""

        if font_type_1 == font_type_2:
            return True

        # Find the length of the common prefix
        common_prefix_length = len(os.path.commonprefix([font_type_1, font_type_2]))

        # Calculate the minimum prefix length based on the ratio
        min_prefix_length = int(min(len(font_type_1), len(font_type_2)) * prefix_length_ratio)

        return common_prefix_length >= min_prefix_length

    def __is_same_block_font(self, block_1, block_2):
        """
        This function compares the font of block1 and block2

        Parameters
        ----------
        block1 : dict
            block1
        block2 : dict
            block2

        Returns
        -------
        is_same : bool
            True if block1 and block2 have the same font, else False
        """
        block_1_font_type = safe_get(block_1, "block_font_type", "")
        block_1_font_size = safe_get(block_1, "block_font_size", 0)
        block_1_avg_char_width = safe_get(block_1, "avg_char_width", 0)

        block_2_font_type = safe_get(block_2, "block_font_type", "")
        block_2_font_size = safe_get(block_2, "block_font_size", 0)
        block_2_avg_char_width = safe_get(block_2, "avg_char_width", 0)

        if isinstance(block_1_font_size, list):
            block_1_font_size = block_1_font_size[0] if block_1_font_size else 0
        if isinstance(block_2_font_size, list):
            block_2_font_size = block_2_font_size[0] if block_2_font_size else 0

        block_1_text = safe_get(block_1, "text", "")
        block_2_text = safe_get(block_2, "text", "")

        if block_1_avg_char_width == 0 or block_2_avg_char_width == 0:
            return False

        if not block_1_text or not block_2_text:
            return False
        else:
            text_len_ratio = len(block_2_text) / len(block_1_text)
            if text_len_ratio < 0.2:
                avg_char_width_condition = (
                    abs(block_1_avg_char_width - block_2_avg_char_width) / min(block_1_avg_char_width, block_2_avg_char_width)
                    < 0.5
                )
            else:
                avg_char_width_condition = (
                    abs(block_1_avg_char_width - block_2_avg_char_width) / min(block_1_avg_char_width, block_2_avg_char_width)
                    < 0.2
                )

        block_font_size_condition = abs(block_1_font_size - block_2_font_size) < 1

        return (
            self.__is_similar_font_type(block_1_font_type, block_2_font_type)
            and avg_char_width_condition
            and block_font_size_condition
        )

    def _is_alphabet_char(self, char):
        if (char >= "\u0041" and char <= "\u005a") or (char >= "\u0061" and char <= "\u007a"):
            return True
        else:
            return False

    def _is_chinese_char(self, char):
        if char >= "\u4e00" and char <= "\u9fa5":
            return True
        else:
            return False

    def _is_other_letter_char(self, char):
        try:
            cat = unicodedata.category(char)
            if cat == "Lu" or cat == "Ll":
                return not self._is_alphabet_char(char) and not self._is_chinese_char(char)
        except TypeError:
            print("The input to the function must be a single character.")
        return False

    def _is_year(self, s: str):
        try:
            number = int(s)
            return 1900 <= number <= 2099
        except ValueError:
            return False

    def _match_brackets(self, text):
        # pattern = r"^[\(\)\[\]（）【】{}｛｝<>＜＞〔〕〘〙\"\'“”‘’]"
        pattern = r"^[\(\)\]（）】{}｛｝>＞〕〙\"\'“”‘’]"
        return bool(re.match(pattern, text))

    def _is_para_font_consistent(self, para_1, para_2):
        """
        This function compares the font of para1 and para2

        Parameters
        ----------
        para1 : dict
            para1
        para2 : dict
            para2

        Returns
        -------
        is_same : bool
            True if para1 and para2 have the same font, else False
        """
        if para_1 is None or para_2 is None:
            return False

        para_1_font_type = safe_get(para_1, "para_font_type", "")
        para_1_font_size = safe_get(para_1, "para_font_size", 0)
        para_1_font_color = safe_get(para_1, "para_font_color", "")

        para_2_font_type = safe_get(para_2, "para_font_type", "")
        para_2_font_size = safe_get(para_2, "para_font_size", 0)
        para_2_font_color = safe_get(para_2, "para_font_color", "")

        if isinstance(para_1_font_type, list):  # get the most common font type
            para_1_font_type = max(set(para_1_font_type), key=para_1_font_type.count)
        if isinstance(para_2_font_type, list):
            para_2_font_type = max(set(para_2_font_type), key=para_2_font_type.count)
        if isinstance(para_1_font_size, list):  # compute average font type
            para_1_font_size = sum(para_1_font_size) / len(para_1_font_size)
        if isinstance(para_2_font_size, list):  # compute average font type
            para_2_font_size = sum(para_2_font_size) / len(para_2_font_size)

        return (
            self.__is_similar_font_type(para_1_font_type, para_2_font_type)
            and abs(para_1_font_size - para_2_font_size) < 1.5
            # and para_font_color1 == para_font_color2
        )

    def _is_para_puncs_consistent(self, para_1, para_2):
        """
        This function determines whether para1 and para2 are originally from the same paragraph by checking the puncs of para1(former) and para2(latter)

        Parameters
        ----------
        para1 : dict
            para1
        para2 : dict
            para2

        Returns
        -------
        is_same : bool
            True if para1 and para2 are from the same paragraph by using the puncs, else False
        """
        para_1_text = safe_get(para_1, "para_text", "").strip()
        para_2_text = safe_get(para_2, "para_text", "").strip()

        para_1_bboxes = safe_get(para_1, "para_bbox", [])
        para_1_font_sizes = safe_get(para_1, "para_font_size", 0)

        para_2_bboxes = safe_get(para_2, "para_bbox", [])
        para_2_font_sizes = safe_get(para_2, "para_font_size", 0)

        # print_yellow("    Features of determine puncs_consistent:")
        # print(f"    para_1_text: {para_1_text}")
        # print(f"    para_2_text: {para_2_text}")
        # print(f"    para_1_bboxes: {para_1_bboxes}")
        # print(f"    para_2_bboxes: {para_2_bboxes}")
        # print(f"    para_1_font_sizes: {para_1_font_sizes}")
        # print(f"    para_2_font_sizes: {para_2_font_sizes}")

        if is_nested_list(para_1_bboxes):
            x0_1, y0_1, x1_1, y1_1 = para_1_bboxes[-1]
        else:
            x0_1, y0_1, x1_1, y1_1 = para_1_bboxes

        if is_nested_list(para_2_bboxes):
            x0_2, y0_2, x1_2, y1_2 = para_2_bboxes[0]
            para_2_font_sizes = para_2_font_sizes[0]  # type: ignore
        else:
            x0_2, y0_2, x1_2, y1_2 = para_2_bboxes

        right_align_threshold = 0.5 * (para_1_font_sizes + para_2_font_sizes) * 0.8
        are_two_paras_right_aligned = abs(x1_1 - x1_2) < right_align_threshold

        left_indent_threshold = 0.5 * (para_1_font_sizes + para_2_font_sizes) * 0.8
        is_para1_left_indent_than_papa2 = x0_1 - x0_2 > left_indent_threshold
        is_para2_left_indent_than_papa1 = x0_2 - x0_1 > left_indent_threshold

        # Check if either para_text1 or para_text2 is empty
        if not para_1_text or not para_2_text:
            return False

        # Define the end puncs for a sentence to end and hyphen
        end_puncs = [".", "?", "!", "。", "？", "！", "…"]
        hyphen = ["-", "—"]

        # Check if para_text1 ends with either hyphen or non-end punctuation or spaces
        para_1_end_with_hyphen = para_1_text and para_1_text[-1] in hyphen
        para_1_end_with_end_punc = para_1_text and para_1_text[-1] in end_puncs
        para_1_end_with_space = para_1_text and para_1_text[-1] == " "
        para_1_not_end_with_end_punc = para_1_text and para_1_text[-1] not in end_puncs

        # print_yellow(f"    para_1_end_with_hyphen: {para_1_end_with_hyphen}")
        # print_yellow(f"    para_1_end_with_end_punc: {para_1_end_with_end_punc}")
        # print_yellow(f"    para_1_not_end_with_end_punc: {para_1_not_end_with_end_punc}")
        # print_yellow(f"    para_1_end_with_space: {para_1_end_with_space}")

        if para_1_end_with_hyphen:  # If para_text1 ends with hyphen
            # print_red(f"para_1 is end with hyphen.")
            para_2_is_consistent = para_2_text and (
                para_2_text[0] in hyphen
                or (self._is_alphabet_char(para_2_text[0]) and para_2_text[0].islower())
                or (self._is_chinese_char(para_2_text[0]))
                or (self._is_other_letter_char(para_2_text[0]))
            )
            if para_2_is_consistent:
                # print(f"para_2 is consistent.\n")
                return True
            else:
                # print(f"para_2 is not consistent.\n")
                pass

        elif para_1_end_with_end_punc:  # If para_text1 ends with ending punctuations
            # print_red(f"para_1 is end with end_punc.")
            para_2_is_consistent = (
                para_2_text
                and (
                    para_2_text[0]
                    == " "
                    # or (self._is_alphabet_char(para_2_text[0]) and para_2_text[0].isupper())
                    # or (self._is_chinese_char(para_2_text[0]))
                    # or (self._is_other_letter_char(para_2_text[0]))
                )
                and not is_para2_left_indent_than_papa1
            )
            if para_2_is_consistent:
                # print(f"para_2 is consistent.\n")
                return True
            else:
                # print(f"para_2 is not consistent.\n")
                pass

        elif para_1_not_end_with_end_punc:  # If para_text1 is not end with ending punctuations
            # print_red(f"para_1 is NOT end with end_punc.")
            para_2_is_consistent = para_2_text and (
                para_2_text[0] == " "
                or (self._is_alphabet_char(para_2_text[0]) and para_2_text[0].islower())
                or (self._is_alphabet_char(para_2_text[0]))
                or (self._is_year(para_2_text[0:4]))
                or (are_two_paras_right_aligned or is_para1_left_indent_than_papa2)
                or (self._is_chinese_char(para_2_text[0]))
                or (self._is_other_letter_char(para_2_text[0]))
                or (self._match_brackets(para_2_text[0]))
            )
            if para_2_is_consistent:
                # print(f"para_2 is consistent.\n")
                return True
            else:
                # print(f"para_2 is not consistent.\n")
                pass

        elif para_1_end_with_space:  # If para_text1 ends with space
            # print_red(f"para_1 is end with space.")
            para_2_is_consistent = para_2_text and (
                para_2_text[0] == " "
                or (self._is_alphabet_char(para_2_text[0]) and para_2_text[0].islower())
                or (self._is_chinese_char(para_2_text[0]))
                or (self._is_other_letter_char(para_2_text[0]))
            )
            if para_2_is_consistent:
                # print(f"para_2 is consistent.\n")
                return True
            else:
                pass
                # print(f"para_2 is not consistent.\n")

        return False

    def _is_block_consistent(self, block_1, block_2):
        """
        This function determines whether block1 and block2 are originally from the same block

        Parameters
        ----------
        block1 : dict
            block1s
        block2 : dict
            block2

        Returns
        -------
        is_same : bool
            True if block1 and block2 are from the same block, else False
        """
        return self.__is_same_block_font(block_1, block_2)

    def _is_para_continued(self, para_1, para_2):
        """
        This function determines whether para1 and para2 are originally from the same paragraph

        Parameters
        ----------
        para1 : dict
            para1
        para2 : dict
            para2

        Returns
        -------
        is_same : bool
            True if para1 and para2 are from the same paragraph, else False
        """
        is_para_font_consistent = self._is_para_font_consistent(para_1, para_2)
        is_para_puncs_consistent = self._is_para_puncs_consistent(para_1, para_2)

        return is_para_font_consistent and is_para_puncs_consistent

    def _are_boundaries_of_block_consistent(self, block_1, block_2):
        """
        This function checks if the boundaries of block1 and block2 are consistent

        Parameters
        ----------
        block1 : dict
            block1

        block2 : dict
            block2

        Returns
        -------
        is_consistent : bool
            True if the boundaries of block1 and block2 are consistent, else False
        """

        last_line_of_block_1 = block_1["lines"][-1]
        first_line_of_block_2 = block_2["lines"][0]

        spans_of_last_line_of_block_1 = last_line_of_block_1["spans"]
        spans_of_first_line_of_block_2 = first_line_of_block_2["spans"]

        font_type_of_last_line_of_block_1 = spans_of_last_line_of_block_1[0]["font"].lower()
        font_size_of_last_line_of_block_1 = spans_of_last_line_of_block_1[0]["size"]
        font_color_of_last_line_of_block_1 = spans_of_last_line_of_block_1[0]["color"]
        font_flags_of_last_line_of_block_1 = spans_of_last_line_of_block_1[0]["flags"]

        font_type_of_first_line_of_block_2 = spans_of_first_line_of_block_2[0]["font"].lower()
        font_size_of_first_line_of_block_2 = spans_of_first_line_of_block_2[0]["size"]
        font_color_of_first_line_of_block_2 = spans_of_first_line_of_block_2[0]["color"]
        font_flags_of_first_line_of_block_2 = spans_of_first_line_of_block_2[0]["flags"]

        return (
            self.__is_similar_font_type(font_type_of_last_line_of_block_1, font_type_of_first_line_of_block_2)
            and abs(font_size_of_last_line_of_block_1 - font_size_of_first_line_of_block_2) < 1
            # and font_color_of_last_line_of_block1 == font_color_of_first_line_of_block2
            and font_flags_of_last_line_of_block_1 == font_flags_of_first_line_of_block_2
        )

    def should_merge_next_para(self, curr_para, next_para):
        """
        This function checks if the next_para should be merged into the curr_para.

        Parameters
        ----------
        curr_para : dict
            The current paragraph.
        next_para : dict
            The next paragraph.

        Returns
        -------
        bool
            True if the next_para should be merged into the curr_para, False otherwise.
        """
        if self._is_para_continued(curr_para, next_para):
            return True
        else:
            return False

    def batch_tag_paras(self, pdf_dict):
        """
        This function tags the paragraphs in the pdf_dict.

        Parameters
        ----------
        pdf_dict : dict
            PDF dictionary.

        Returns
        -------
        pdf_dict : dict
            PDF dictionary with tagged paragraphs.
        """
        the_last_page_id = len(pdf_dict) - 1

        for curr_page_idx, (curr_page_id, curr_page_content) in enumerate(pdf_dict.items()):
            if curr_page_id.startswith("page_") and curr_page_content.get("para_blocks", []):
                para_blocks_of_curr_page = curr_page_content["para_blocks"]
                next_page_idx = curr_page_idx + 1
                next_page_id = f"page_{next_page_idx}"
                next_page_content = pdf_dict.get(next_page_id, {})

                for i, current_block in enumerate(para_blocks_of_curr_page):
                    for para_id, curr_para in current_block["paras"].items():
                        curr_para["curr_para_location"] = [
                            curr_page_idx,
                            current_block["block_id"],
                            int(para_id.split("_")[-1]),
                        ]
                        curr_para["next_para_location"] = None  # 默认设置为None
                        curr_para["merge_next_para"] = False  # 默认设置为False

                    next_block = para_blocks_of_curr_page[i + 1] if i < len(para_blocks_of_curr_page) - 1 else None

                    if next_block:
                        curr_block_last_para_key = list(current_block["paras"].keys())[-1]
                        curr_blk_last_para = current_block["paras"][curr_block_last_para_key]

                        next_block_first_para_key = list(next_block["paras"].keys())[0]
                        next_blk_first_para = next_block["paras"][next_block_first_para_key]

                        if self.should_merge_next_para(curr_blk_last_para, next_blk_first_para):
                            curr_blk_last_para["next_para_location"] = [
                                curr_page_idx,
                                next_block["block_id"],
                                int(next_block_first_para_key.split("_")[-1]),
                            ]
                            curr_blk_last_para["merge_next_para"] = True
                    else:
                        # Handle the case where the next block is in a different page
                        curr_block_last_para_key = list(current_block["paras"].keys())[-1]
                        curr_blk_last_para = current_block["paras"][curr_block_last_para_key]

                        while not next_page_content.get("para_blocks", []) and next_page_idx <= the_last_page_id:
                            next_page_idx += 1
                            next_page_id = f"page_{next_page_idx}"
                            next_page_content = pdf_dict.get(next_page_id, {})

                        if next_page_content.get("para_blocks", []):
                            next_blk_first_para_key = list(next_page_content["para_blocks"][0]["paras"].keys())[0]
                            next_blk_first_para = next_page_content["para_blocks"][0]["paras"][next_blk_first_para_key]

                            if self.should_merge_next_para(curr_blk_last_para, next_blk_first_para):
                                curr_blk_last_para["next_para_location"] = [
                                    next_page_idx,
                                    next_page_content["para_blocks"][0]["block_id"],
                                    int(next_blk_first_para_key.split("_")[-1]),
                                ]
                                curr_blk_last_para["merge_next_para"] = True

        return pdf_dict

    def find_block_by_id(self, para_blocks, block_id):
        """
        This function finds a block by its id.

        Parameters
        ----------
        para_blocks : list
            List of blocks.
        block_id : int
            Id of the block to find.

        Returns
        -------
        block : dict
            The block with the given id.
        """
        for blk_idx, block in enumerate(para_blocks):
            if block.get("block_id") == block_id:
                return block
        return None

    def batch_merge_paras(self, pdf_dict):
        """
        This function merges the paragraphs in the pdf_dict.

        Parameters
        ----------
        pdf_dict : dict
            PDF dictionary.

        Returns
        -------
        pdf_dict : dict
            PDF dictionary with merged paragraphs.
        """
        for page_id, page_content in pdf_dict.items():
            if page_id.startswith("page_") and page_content.get("para_blocks", []):
                para_blocks_of_page = page_content["para_blocks"]

                for i in range(len(para_blocks_of_page)):
                    current_block = para_blocks_of_page[i]
                    paras = current_block["paras"]

                    for para_id, curr_para in list(paras.items()):
                        # print(f"current para_id: {para_id}")
                        # 跳过标题段落
                        if curr_para.get("is_para_title"):
                            continue

                        while curr_para.get("merge_next_para"):
                            curr_para_location = curr_para.get("curr_para_location")
                            next_para_location = curr_para.get("next_para_location")

                            # print(f"curr_para_location: {curr_para_location}, next_para_location: {next_para_location}")
                            
                            if not next_para_location:
                                break

                            if curr_para_location == next_para_location:
                                # print_red("The next para is in the same block as the current para.")
                                curr_para["merge_next_para"] = False
                                break

                            next_page_idx, next_block_id, next_para_id = next_para_location
                            next_page_id = f"page_{next_page_idx}"
                            next_page_content = pdf_dict.get(next_page_id)
                            if not next_page_content:
                                break

                            next_block = self.find_block_by_id(next_page_content.get("para_blocks", []), next_block_id)

                            if not next_block:
                                break

                            next_para = next_block["paras"].get(f"para_{next_para_id}")

                            if not next_para or next_para.get("is_para_title"):
                                break

                            # 合并段落文本
                            curr_para_text = curr_para.get("para_text", "")
                            next_para_text = next_para.get("para_text", "")
                            curr_para["para_text"] = curr_para_text + " " + next_para_text

                            # 更新 next_para_location
                            curr_para["next_para_location"] = next_para.get("next_para_location")

                            # 将下一个段落文本置为空，表示已被合并
                            next_para["para_text"] = ""

                            # 更新 merge_next_para 标记
                            curr_para["merge_next_para"] = next_para.get("merge_next_para", False)

        return pdf_dict


class DrawAnnos:
    """
    This class draws annotations on the pdf file

    ----------------------------------------
                Color Code
    ----------------------------------------
        Red: (1, 0, 0)
        Green: (0, 1, 0)
        Blue: (0, 0, 1)
        Yellow: (1, 1, 0) - mix of red and green
        Cyan: (0, 1, 1) - mix of green and blue
        Magenta: (1, 0, 1) - mix of red and blue
        White: (1, 1, 1) - red, green and blue full intensity
        Black: (0, 0, 0) - no color component whatsoever
        Gray: (0.5, 0.5, 0.5) - equal and medium intensity of red, green and blue color components
        Orange: (1, 0.65, 0) - maximum intensity of red, medium intensity of green, no blue component
    """

    def __init__(self) -> None:
        pass

    def __is_nested_list(self, lst):
        """
        This function returns True if the given list is a nested list of any degree.
        """
        if isinstance(lst, list):
            return any(self.__is_nested_list(i) for i in lst) or any(isinstance(i, list) for i in lst)
        return False

    def __valid_rect(self, bbox):
        # Ensure that the rectangle is not empty or invalid
        if isinstance(bbox[0], list):
            return False  # It's a nested list, hence it can't be valid rect
        else:
            return bbox[0] < bbox[2] and bbox[1] < bbox[3]

    def __draw_nested_boxes(self, page, nested_bbox, color=(0, 1, 1)):
        """
        This function draws the nested boxes

        Parameters
        ----------
        page : fitz.Page
            page
        nested_bbox : list
            nested bbox
        color : tuple
            color, by default (0, 1, 1)    # draw with cyan color for combined paragraph
        """
        if self.__is_nested_list(nested_bbox):  # If it's a nested list
            for bbox in nested_bbox:
                self.__draw_nested_boxes(page, bbox, color)  # Recursively call the function
        elif self.__valid_rect(nested_bbox):  # If valid rectangle
            para_rect = fitz.Rect(nested_bbox)
            para_anno = page.add_rect_annot(para_rect)
            para_anno.set_colors(stroke=color)  # draw with cyan color for combined paragraph
            para_anno.set_border(width=1)
            para_anno.update()

    def draw_annos(self, input_pdf_path, pdf_dic, output_pdf_path):
        """
        This function draws annotations on the pdf file.

        Parameters
        ----------
        input_pdf_path : str
            path to the input pdf file
        pdf_dic : dict
            pdf dictionary
        output_pdf_path : str
            path to the output pdf file

        pdf_dic : dict
            pdf dictionary
        """
        pdf_doc = open_pdf(input_pdf_path)

        if pdf_dic is None:
            pdf_dic = {}

        if output_pdf_path is None:
            output_pdf_path = input_pdf_path.replace(".pdf", "_anno.pdf")

        for page_id, page in enumerate(pdf_doc):  # type: ignore
            page_key = f"page_{page_id}"
            for ele_key, ele_data in pdf_dic[page_key].items():
                if ele_key == "para_blocks":
                    para_blocks = ele_data
                    for para_block in para_blocks:
                        if "paras" in para_block.keys():
                            paras = para_block["paras"]
                            for para_key, para_content in paras.items():
                                para_bbox = para_content["para_bbox"]
                                # print(f"para_bbox: {para_bbox}")
                                # print(f"is a nested list: {self.__is_nested_list(para_bbox)}")
                                if self.__is_nested_list(para_bbox) and len(para_bbox) > 1:
                                    color = (0, 1, 1)
                                    self.__draw_nested_boxes(
                                        page, para_bbox, color
                                    )  # draw with cyan color for combined paragraph
                                else:
                                    if self.__valid_rect(para_bbox):
                                        para_rect = fitz.Rect(para_bbox)
                                        para_anno = page.add_rect_annot(para_rect)
                                        para_anno.set_colors(stroke=(0, 1, 0))  # draw with green color for normal paragraph
                                        para_anno.set_border(width=0.5)
                                        para_anno.update()

                                is_para_title = para_content["is_para_title"]
                                if is_para_title:
                                    if self.__is_nested_list(para_content["para_bbox"]) and len(para_content["para_bbox"]) > 1:
                                        color = (0, 0, 1)
                                        self.__draw_nested_boxes(
                                            page, para_content["para_bbox"], color
                                        )  # draw with cyan color for combined title
                                    else:
                                        if self.__valid_rect(para_content["para_bbox"]):
                                            para_rect = fitz.Rect(para_content["para_bbox"])
                                            if self.__valid_rect(para_content["para_bbox"]):
                                                para_anno = page.add_rect_annot(para_rect)
                                                para_anno.set_colors(stroke=(0, 0, 1))  # draw with blue color for normal title
                                                para_anno.set_border(width=0.5)
                                                para_anno.update()

        pdf_doc.save(output_pdf_path)
        pdf_doc.close()


class ParaProcessPipeline:
    def __init__(self) -> None:
        pass

    def para_process_pipeline(self, pdf_info_dict, para_debug_mode=None, input_pdf_path=None, output_pdf_path=None):
        """
        This function processes the paragraphs, including:
        1. Read raw input json file into pdf_dic
        2. Detect and replace equations
        3. Combine spans into a natural line
        4. Check if the paragraphs are inside bboxes passed from "layout_bboxes" key
        5. Compute statistics for each block
        6. Detect titles in the document
        7. Detect paragraphs inside each block
        8. Divide the level of the titles
        9. Detect and combine paragraphs from different blocks into one paragraph
        10. Check whether the final results after checking headings, dividing paragraphs within blocks, and merging paragraphs between blocks are plausible and reasonable.
        11. Draw annotations on the pdf file

        Parameters
        ----------
        pdf_dic_json_fpath : str
            path to the pdf dictionary json file.
            Notice: data noises, including overlap blocks, header, footer, watermark, vertical margin note have been removed already.
        input_pdf_doc : str
            path to the input pdf file
        output_pdf_path : str
            path to the output pdf file

        Returns
        -------
        pdf_dict : dict
            result dictionary
        """

        error_info = None

        output_json_file = ""
        output_dir = ""

        if input_pdf_path is not None:
            input_pdf_path = os.path.abspath(input_pdf_path)

            # print_green_on_red(f">>>>>>>>>>>>>>>>>>> Process the paragraphs of {input_pdf_path}")

        if output_pdf_path is not None:
            output_dir = os.path.dirname(output_pdf_path)
            output_json_file = f"{output_dir}/pdf_dic.json"

        def __save_pdf_dic(pdf_dic, output_pdf_path, stage="0", para_debug_mode=para_debug_mode):
            """
            Save the pdf_dic to a json file
            """
            output_pdf_file_name = os.path.basename(output_pdf_path)
            # output_dir = os.path.dirname(output_pdf_path)
            output_dir = "\\tmp\\pdf_parse"
            output_pdf_file_name = output_pdf_file_name.replace(".pdf", f"_stage_{stage}.json")
            pdf_dic_json_fpath = os.path.join(output_dir, output_pdf_file_name)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if para_debug_mode == "full":
                with open(pdf_dic_json_fpath, "w", encoding="utf-8") as f:
                    json.dump(pdf_dic, f, indent=2, ensure_ascii=False)

            # Validate the output already exists
            if not os.path.exists(pdf_dic_json_fpath):
                print_red(f"Failed to save the pdf_dic to {pdf_dic_json_fpath}")
                return None
            else:
                print_green(f"Succeed to save the pdf_dic to {pdf_dic_json_fpath}")

            return pdf_dic_json_fpath

        """
        Preprocess the lines of block
        """
        # Combine spans into a natural line
        rawBlockProcessor = RawBlockProcessor()
        pdf_dic = rawBlockProcessor.batch_process_blocks(pdf_info_dict)
        # print(f"pdf_dic['page_0']['para_blocks'][0]: {pdf_dic['page_0']['para_blocks'][0]}", end="\n\n")

        # Check if the paragraphs are inside bboxes passed from "layout_bboxes" key
        layoutFilter = LayoutFilterProcessor()
        pdf_dic = layoutFilter.batch_process_blocks(pdf_dic)

        # Compute statistics for each block
        blockStatisticsCalculator = BlockStatisticsCalculator()
        pdf_dic = blockStatisticsCalculator.batch_process_blocks(pdf_dic)
        # print(f"pdf_dic['page_0']['para_blocks'][0]: {pdf_dic['page_0']['para_blocks'][0]}", end="\n\n")

        # Compute statistics for all blocks(namely this pdf document)
        docStatisticsCalculator = DocStatisticsCalculator()
        pdf_dic = docStatisticsCalculator.calc_stats_of_doc(pdf_dic)
        # print(f"pdf_dic['statistics']: {pdf_dic['statistics']}", end="\n\n")

        # Dump the first three stages of pdf_dic to a json file
        if para_debug_mode == "full":
            pdf_dic_json_fpath = __save_pdf_dic(pdf_dic, output_pdf_path, stage="0", para_debug_mode=para_debug_mode)

        """
        Detect titles in the document
        """
        doc_statistics = pdf_dic["statistics"]
        titleProcessor = TitleProcessor(doc_statistics)
        pdf_dic = titleProcessor.batch_detect_titles(pdf_dic)

        if para_debug_mode == "full":
            pdf_dic_json_fpath = __save_pdf_dic(pdf_dic, output_pdf_path, stage="1", para_debug_mode=para_debug_mode)

        """
        Detect and divide the level of the titles
        """
        titleProcessor = TitleProcessor()

        pdf_dic = titleProcessor.batch_recog_title_level(pdf_dic)

        if para_debug_mode == "full":
            pdf_dic_json_fpath = __save_pdf_dic(pdf_dic, output_pdf_path, stage="2", para_debug_mode=para_debug_mode)

        """
        Detect and split paragraphs inside each block
        """
        blockInnerParasProcessor = BlockTerminationProcessor()

        pdf_dic = blockInnerParasProcessor.batch_process_blocks(pdf_dic)

        if para_debug_mode == "full":
            pdf_dic_json_fpath = __save_pdf_dic(pdf_dic, output_pdf_path, stage="3", para_debug_mode=para_debug_mode)

        # pdf_dic_json_fpath = __save_pdf_dic(pdf_dic, output_pdf_path, stage="3", para_debug_mode="full")
        # print_green(f"pdf_dic_json_fpath: {pdf_dic_json_fpath}")

        """
        Detect and combine paragraphs from different blocks into one paragraph
        """
        blockContinuationProcessor = BlockContinuationProcessor()

        pdf_dic = blockContinuationProcessor.batch_tag_paras(pdf_dic)
        pdf_dic = blockContinuationProcessor.batch_merge_paras(pdf_dic)

        if para_debug_mode == "full":
            pdf_dic_json_fpath = __save_pdf_dic(pdf_dic, output_pdf_path, stage="4", para_debug_mode=para_debug_mode)

        # pdf_dic_json_fpath = __save_pdf_dic(pdf_dic, output_pdf_path, stage="4", para_debug_mode="full")
        # print_green(f"pdf_dic_json_fpath: {pdf_dic_json_fpath}")

        """
        Discard pdf files by checking exceptions and return the error info to the caller
        """
        discardByException = DiscardByException()

        is_discard_by_single_line_block = discardByException.discard_by_single_line_block(
            pdf_dic, exception=DenseSingleLineBlockException()
        )
        is_discard_by_title_detection = discardByException.discard_by_title_detection(
            pdf_dic, exception=TitleDetectionException()
        )
        is_discard_by_title_level = discardByException.discard_by_title_level(pdf_dic, exception=TitleLevelException())
        is_discard_by_split_para = discardByException.discard_by_split_para(pdf_dic, exception=ParaSplitException())
        is_discard_by_merge_para = discardByException.discard_by_merge_para(pdf_dic, exception=ParaMergeException())

        if is_discard_by_single_line_block is not None:
            error_info = is_discard_by_single_line_block
        elif is_discard_by_title_detection is not None:
            error_info = is_discard_by_title_detection
        elif is_discard_by_title_level is not None:
            error_info = is_discard_by_title_level
        elif is_discard_by_split_para is not None:
            error_info = is_discard_by_split_para
        elif is_discard_by_merge_para is not None:
            error_info = is_discard_by_merge_para

        if error_info is not None:
            return pdf_dic, error_info

        """
        Dump the final pdf_dic to a json file
        """
        if para_debug_mode is not None:
            with open(output_json_file, "w", encoding="utf-8") as f:
                json.dump(pdf_info_dict, f, ensure_ascii=False, indent=4)

        """
        Draw the annotations
        """
        if para_debug_mode is not None:
            drawAnnos = DrawAnnos()
            drawAnnos.draw_annos(input_pdf_path, pdf_dic, output_pdf_path)

        """
        Remove the intermediate files which are generated in the process of paragraph processing if debug_mode is simple
        """
        if para_debug_mode is not None:
            for fpath in os.listdir(output_dir):
                if fpath.endswith(".json") and "stage" in fpath:
                    os.remove(os.path.join(output_dir, fpath))

        return pdf_dic, error_info


"""
Run this script to test the function with Command: 

python detect_para.py [pdf_path] [output_pdf_path]

Params:
- pdf_path: the path of the pdf file
- output_pdf_path: the path of the output pdf file
"""

if __name__ == "__main__":
    DEFAULT_PDF_PATH = (
        "app/pdf_toolbox/tests/assets/paper/paper.pdf" if os.name != "nt" else "app\\pdf_toolbox\\tests\\assets\\paper\\paper.pdf"
    )
    input_pdf_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF_PATH
    output_pdf_path = sys.argv[2] if len(sys.argv) > 2 else input_pdf_path.split(".")[0] + "_recogPara.pdf"
    output_json_path = sys.argv[3] if len(sys.argv) > 3 else input_pdf_path.split(".")[0] + "_recogPara.json"

    import stat

    # Remove existing output file if it exists
    if os.path.exists(output_pdf_path):
        os.chmod(output_pdf_path, stat.S_IWRITE)
        os.remove(output_pdf_path)

    input_pdf_doc = open_pdf(input_pdf_path)

    # postprocess the paragraphs
    paraProcessPipeline = ParaProcessPipeline()

    # parse paragraph and save to json file
    pdf_dic = {}

    blockInnerParasProcessor = BlockTerminationProcessor()

    """
    Construct the pdf dictionary.
    """

    for page_id, page in enumerate(input_pdf_doc):  # type: ignore
        # print(f"Processing page {page_id}")
        # print(f"page: {page}")
        raw_blocks = page.get_text("dict")["blocks"]

        # Save text blocks to "preproc_blocks"
        preproc_blocks = []
        for block in raw_blocks:
            if block["type"] == 0:
                preproc_blocks.append(block)

        layout_bboxes = []

        # Construct the pdf dictionary as schema above
        page_dict = {
            "para_blocks": None,
            "preproc_blocks": preproc_blocks,
            "images": None,
            "tables": None,
            "interline_equations": None,
            "inline_equations": None,
            "layout_bboxes": None,
            "pymu_raw_blocks": None,
            "global_statistic": None,
            "droped_text_block": None,
            "droped_image_block": None,
            "droped_table_block": None,
            "image_backup": None,
            "table_backup": None,
        }

        pdf_dic[f"page_{page_id}"] = page_dict

    # print(f"pdf_dic: {pdf_dic}")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(pdf_dic, f, ensure_ascii=False, indent=4)

    pdf_dic = paraProcessPipeline.para_process_pipeline(output_json_path, input_pdf_doc, output_pdf_path)
