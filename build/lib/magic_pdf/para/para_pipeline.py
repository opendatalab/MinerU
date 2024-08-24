import os
import json

from magic_pdf.para.commons import *

from magic_pdf.para.raw_processor import RawBlockProcessor
from magic_pdf.para.layout_match_processor import LayoutFilterProcessor
from magic_pdf.para.stats import BlockStatisticsCalculator
from magic_pdf.para.stats import DocStatisticsCalculator
from magic_pdf.para.title_processor import TitleProcessor
from magic_pdf.para.block_termination_processor import BlockTerminationProcessor
from magic_pdf.para.block_continuation_processor import BlockContinuationProcessor
from magic_pdf.para.draw import DrawAnnos
from magic_pdf.para.exceptions import (
    DenseSingleLineBlockException,
    TitleDetectionException,
    TitleLevelException,
    ParaSplitException,
    ParaMergeException,
    DiscardByException,
)


if sys.version_info[0] >= 3:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


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
        # Find and replace the interline and inline equations, should be better done before the paragraph processing
        # Create "para_blocks" for each page.
        # equationProcessor = EquationsProcessor()
        # pdf_dic = equationProcessor.batch_process_blocks(pdf_info_dict)

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
        pdf_dic = titleProcessor.batch_process_blocks_detect_titles(pdf_dic)

        if para_debug_mode == "full":
            pdf_dic_json_fpath = __save_pdf_dic(pdf_dic, output_pdf_path, stage="1", para_debug_mode=para_debug_mode)

        """
        Detect and divide the level of the titles
        """
        titleProcessor = TitleProcessor()

        pdf_dic = titleProcessor.batch_process_blocks_recog_title_level(pdf_dic)

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

        """
        if any(
            info is not None
            for info in [
                is_discard_by_single_line_block,
                is_discard_by_title_detection,
                is_discard_by_title_level,
                is_discard_by_split_para,
                is_discard_by_merge_para,
            ]
        ):
            error_info = next(
                (
                    info
                    for info in [
                        is_discard_by_single_line_block,
                        is_discard_by_title_detection,
                        is_discard_by_title_level,
                        is_discard_by_split_para,
                        is_discard_by_merge_para,
                    ]
                    if info is not None
                ),
                None,
            )
            return pdf_dic, error_info

        if any(
            info is not None
            for info in [
                is_discard_by_single_line_block,
                is_discard_by_title_detection,
                is_discard_by_title_level,
                is_discard_by_split_para,
                is_discard_by_merge_para,
            ]
        ):
            error_info = next(
                (
                    info
                    for info in [
                        is_discard_by_single_line_block,
                        is_discard_by_title_detection,
                        is_discard_by_title_level,
                        is_discard_by_split_para,
                        is_discard_by_merge_para,
                    ]
                    if info is not None
                ),
                None,
            )
            return pdf_dic, error_info
        """

        """
        Dump the final pdf_dic to a json file
        """
        if para_debug_mode is not None:
            with open(output_json_file, "w", encoding="utf-8") as f:
                json.dump(pdf_info_dict, f, ensure_ascii=False, indent=4)

        """
        Draw the annotations
        """

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
