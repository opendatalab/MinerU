from collections import Counter
import numpy as np

from magic_pdf.para.commons import *


if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


class BlockStatisticsCalculator:
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
            Input block is a list of raw blocks. Schema can refer to the value of key ""preproc_blocks", demo file is app/pdf_toolbox/tests/preproc_2_parasplit_example.json

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


