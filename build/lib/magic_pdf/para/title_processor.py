import os
import re
import numpy as np

from magic_pdf.libs.nlp_utils import NLPModels

from magic_pdf.para.commons import *

if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


class TitleProcessor:
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

        # """
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

        # """

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
            return False

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
        # is_title = False
        # if prev_line_is_title:

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
            )  # not the same font type as the document average font type, which includes the most common font type and the second most common font type
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
            )
        )
        # ) or (is_similar_to_pre_line and prev_line_is_title)

        is_name_or_org_list_to_be_removed = (
            (is_person_or_org_list_line_by_nlp)
            and is_punctuation_heavy
            and (is_title_by_check_prev_line or is_title_by_check_next_line or is_title_by_check_pre_and_next_line)
        ) and not is_title

        if is_name_or_org_list_to_be_removed:
            is_author_or_org_list = True
            # print curr_line_text to check
            # print_yellow(f"Text of is_author_or_org_list: {curr_line_text}")
        else:
            is_author_or_org_list = False
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

    def _detect_block_title(self, input_block):
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

            if is_author_or_org_list:
                curr_line["is_author_or_org_list"] = is_author_or_org_list
            else:
                curr_line["is_author_or_org_list"] = False

        return input_block

    def batch_process_blocks_detect_titles(self, pdf_dic):
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
                            new_block = self._detect_block_title(para_block)
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

    def __determine_size_based_level(self, title_blocks):
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

    def batch_process_blocks_recog_title_level(self, pdf_dic):
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
            title_blocks = self.__determine_size_based_level(title_blocks)

        return pdf_dic
