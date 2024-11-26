import os
import unicodedata

from magic_pdf.para.commons import *


if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


class BlockContinuationProcessor:
    """
    This class is used to process the blocks to detect block continuations.
    """

    def __init__(self) -> None:
        pass

    def __is_similar_font_type(self, font_type1, font_type2, prefix_length_ratio=0.3):
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

        if isinstance(font_type1, list):
            font_type1 = font_type1[0] if font_type1 else ""
        if isinstance(font_type2, list):
            font_type2 = font_type2[0] if font_type2 else ""

        if font_type1 == font_type2:
            return True

        # Find the length of the common prefix
        common_prefix_length = len(os.path.commonprefix([font_type1, font_type2]))

        # Calculate the minimum prefix length based on the ratio
        min_prefix_length = int(min(len(font_type1), len(font_type2)) * prefix_length_ratio)

        return common_prefix_length >= min_prefix_length

    def __is_same_block_font(self, block1, block2):
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
        block_1_font_type = safe_get(block1, "block_font_type", "")
        block_1_font_size = safe_get(block1, "block_font_size", 0)
        block_1_avg_char_width = safe_get(block1, "avg_char_width", 0)

        block_2_font_type = safe_get(block2, "block_font_type", "")
        block_2_font_size = safe_get(block2, "block_font_size", 0)
        block_2_avg_char_width = safe_get(block2, "avg_char_width", 0)

        if isinstance(block_1_font_size, list):
            block_1_font_size = block_1_font_size[0] if block_1_font_size else 0
        if isinstance(block_2_font_size, list):
            block_2_font_size = block_2_font_size[0] if block_2_font_size else 0

        block_1_text = safe_get(block1, "text", "")
        block_2_text = safe_get(block2, "text", "")

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

        block_font_size_condtion = abs(block_1_font_size - block_2_font_size) < 1

        return (
            self.__is_similar_font_type(block_1_font_type, block_2_font_type)
            and avg_char_width_condition
            and block_font_size_condtion
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

    def __is_para_font_consistent(self, para_1, para_2):
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
                    para_2_text[0] == " "
                    or (self._is_alphabet_char(para_2_text[0]) and para_2_text[0].isupper())
                    or (self._is_chinese_char(para_2_text[0]))
                    or (self._is_other_letter_char(para_2_text[0]))
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

    def _is_block_consistent(self, block1, block2):
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
        return self.__is_same_block_font(block1, block2)

    def _is_para_continued(self, para1, para2):
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
        is_para_font_consistent = self.__is_para_font_consistent(para1, para2)
        is_para_puncs_consistent = self._is_para_puncs_consistent(para1, para2)

        return is_para_font_consistent and is_para_puncs_consistent

    def _are_boundaries_of_block_consistent(self, block1, block2):
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

        last_line_of_block1 = block1["lines"][-1]
        first_line_of_block2 = block2["lines"][0]

        spans_of_last_line_of_block1 = last_line_of_block1["spans"]
        spans_of_first_line_of_block2 = first_line_of_block2["spans"]

        font_type_of_last_line_of_block1 = spans_of_last_line_of_block1[0]["font"].lower()
        font_size_of_last_line_of_block1 = spans_of_last_line_of_block1[0]["size"]
        font_color_of_last_line_of_block1 = spans_of_last_line_of_block1[0]["color"]
        font_flags_of_last_line_of_block1 = spans_of_last_line_of_block1[0]["flags"]

        font_type_of_first_line_of_block2 = spans_of_first_line_of_block2[0]["font"].lower()
        font_size_of_first_line_of_block2 = spans_of_first_line_of_block2[0]["size"]
        font_color_of_first_line_of_block2 = spans_of_first_line_of_block2[0]["color"]
        font_flags_of_first_line_of_block2 = spans_of_first_line_of_block2[0]["flags"]

        return (
            self.__is_similar_font_type(font_type_of_last_line_of_block1, font_type_of_first_line_of_block2)
            and abs(font_size_of_last_line_of_block1 - font_size_of_first_line_of_block2) < 1
            # and font_color_of_last_line_of_block1 == font_color_of_first_line_of_block2
            and font_flags_of_last_line_of_block1 == font_flags_of_first_line_of_block2
        )

    def _get_last_paragraph(self, block):
        """
        Retrieves the last paragraph from a block.

        Parameters
        ----------
        block : dict
            The block from which to retrieve the paragraph.

        Returns
        -------
        dict
            The last paragraph of the block.
        """
        if block["paras"]:
            last_para_key = list(block["paras"].keys())[-1]
            return block["paras"][last_para_key]
        else:
            return None

    def _get_first_paragraph(self, block):
        """
        Retrieves the first paragraph from a block.

        Parameters
        ----------
        block : dict
            The block from which to retrieve the paragraph.

        Returns
        -------
        dict
            The first paragraph of the block.
        """
        if block["paras"]:
            first_para_key = list(block["paras"].keys())[0]
            return block["paras"][first_para_key]
        else:
            return None

    def should_merge_next_para(self, curr_para, next_para):
        if self._is_para_continued(curr_para, next_para):
            return True
        else:
            return False

    def batch_tag_paras(self, pdf_dict):
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
        for block in para_blocks:
            if block.get("block_id") == block_id:
                return block
        return None

    def batch_merge_paras(self, pdf_dict):
        for page_id, page_content in pdf_dict.items():
            if page_id.startswith("page_") and page_content.get("para_blocks", []):
                para_blocks_of_page = page_content["para_blocks"]

                for i in range(len(para_blocks_of_page)):
                    current_block = para_blocks_of_page[i]
                    paras = current_block["paras"]

                    for para_id, curr_para in list(paras.items()):
                        # 跳过标题段落
                        if curr_para.get("is_para_title"):
                            continue

                        while curr_para.get("merge_next_para"):
                            next_para_location = curr_para.get("next_para_location")
                            if not next_para_location:
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
