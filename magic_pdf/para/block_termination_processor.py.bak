from magic_pdf.para.commons import *


if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore



class BlockTerminationProcessor:
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

            # font_types = [span["font"] for line in lines for span in line["spans"]]
            # most_common_font_type = max(set(font_types), key=font_types.count) if font_types else None

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
