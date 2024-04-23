import time

# from anyio import Path

from magic_pdf.libs.commons import (
    fitz,
    get_delta_time,
    get_img_s3_client,
    get_docx_model_output,
)
import json
import os
from copy import deepcopy
import math
from loguru import logger
from magic_pdf.layout.bbox_sort import (
    prepare_bboxes_for_layout_split,
)
from magic_pdf.layout.layout_sort import (
    LAYOUT_UNPROC,
    get_bboxes_layout,
    get_columns_cnt_of_layout,
    sort_text_block,
)
from magic_pdf.libs.drop_reason import DropReason
from magic_pdf.libs.markdown_utils import escape_special_markdown_char
from magic_pdf.libs.safe_filename import sanitize_filename
from magic_pdf.libs.vis_utils import draw_bbox_on_page, draw_layout_bbox_on_page
from magic_pdf.pre_proc.cut_image import txt_save_images_by_bboxes
from magic_pdf.pre_proc.detect_images import parse_images
from magic_pdf.pre_proc.detect_tables import parse_tables  # 获取tables的bbox
from magic_pdf.pre_proc.detect_equation import parse_equations  # 获取equations的bbox
from magic_pdf.pre_proc.detect_header import parse_headers  # 获取headers的bbox
from magic_pdf.pre_proc.detect_page_number import parse_pageNos  # 获取pageNos的bbox
from magic_pdf.pre_proc.detect_footnote import (
    parse_footnotes_by_model,
    parse_footnotes_by_rule,
)  # 获取footnotes的bbox
from magic_pdf.pre_proc.detect_footer_by_model import parse_footers  # 获取footers的bbox

from magic_pdf.post_proc.detect_para import (
    ParaProcessPipeline,
    TitleDetectionException,
    TitleLevelException,
    ParaSplitException,
    ParaMergeException,
    DenseSingleLineBlockException,
)
from magic_pdf.pre_proc.main_text_font import get_main_text_font
from magic_pdf.pre_proc.remove_colored_strip_bbox import remove_colored_strip_textblock
from magic_pdf.pre_proc.remove_footer_header import remove_headder_footer_one_page
from magic_pdf.train_utils.extract_caption import extract_caption_bbox

"""
from para.para_pipeline import ParaProcessPipeline
from para.exceptions import (
    TitleDetectionException,
    TitleLevelException,
    ParaSplitException,
    ParaMergeException,
    DenseSingleLineBlockException,
)
"""

from magic_pdf.libs.commons import read_file, join_path
from magic_pdf.post_proc.remove_footnote import (
    merge_footnote_blocks,
    remove_footnote_blocks,
)
from magic_pdf.pre_proc.citationmarker_remove import remove_citation_marker
from magic_pdf.pre_proc.equations_replace import (
    combine_chars_to_pymudict,
    remove_chars_in_text_blocks,
    replace_equations_in_textblock,
)
from magic_pdf.pre_proc.pdf_pre_filter import pdf_filter
from magic_pdf.pre_proc.detect_footer_header_by_statistics import drop_footer_header
from magic_pdf.pre_proc.construct_page_dict import construct_page_component
from magic_pdf.pre_proc.fix_image import (
    combine_images,
    fix_image_vertical,
    fix_seperated_image,
    include_img_title,
)
from magic_pdf.post_proc.pdf_post_filter import pdf_post_filter
from magic_pdf.pre_proc.remove_rotate_bbox import (
    get_side_boundry,
    remove_rotate_side_textblock,
    remove_side_blank_block,
)
from magic_pdf.pre_proc.resolve_bbox_conflict import (
    check_text_block_horizontal_overlap,
    resolve_bbox_overlap_conflict,
)
from magic_pdf.pre_proc.fix_table import (
    fix_table_text_block,
    fix_tables,
    include_table_title,
)
from magic_pdf.pre_proc.solve_line_alien import solve_inline_too_large_interval

denseSingleLineBlockException_msg = DenseSingleLineBlockException().message
titleDetectionException_msg = TitleDetectionException().message
titleLevelException_msg = TitleLevelException().message
paraSplitException_msg = ParaSplitException().message
paraMergeException_msg = ParaMergeException().message


def parse_pdf_for_train(
    s3_pdf_path,
    s3_pdf_profile,
    pdf_model_output,
    save_path,
    book_name,
    image_s3_config=None,
    start_page_id=0,
    end_page_id=None,
    junk_img_bojids=[],
    debug_mode=False,
):
    pdf_bytes = read_file(s3_pdf_path, s3_pdf_profile)
    save_tmp_path = os.path.join(os.path.dirname(__file__), "../..", "tmp", "unittest")
    md_bookname_save_path = ""
    book_name = sanitize_filename(book_name)
    if debug_mode:
        save_path = join_path(save_tmp_path, "md")
        pdf_local_path = join_path(save_tmp_path, "download-pdfs", book_name)

        if not os.path.exists(os.path.dirname(pdf_local_path)):
            # 如果目录不存在，创建它
            os.makedirs(os.path.dirname(pdf_local_path))

        md_bookname_save_path = join_path(save_tmp_path, "md", book_name)
        if not os.path.exists(md_bookname_save_path):
            # 如果目录不存在，创建它
            os.makedirs(md_bookname_save_path)

        with open(pdf_local_path + ".pdf", "wb") as pdf_file:
            pdf_file.write(pdf_bytes)

    pdf_docs = fitz.open("pdf", pdf_bytes)
    pdf_info_dict = {}
    img_s3_client = get_img_s3_client(
        save_path, image_s3_config
    )  # 更改函数名和参数，避免歧义
    # img_s3_client = "img_s3_client"  #不创建这个对象，直接用字符串占位

    start_time = time.time()

    """通过统计pdf全篇文字,识别正文字体"""
    main_text_font = get_main_text_font(pdf_docs)

    end_page_id = end_page_id if end_page_id else len(pdf_docs) - 1
    for page_id in range(start_page_id, end_page_id + 1):
        page = pdf_docs[page_id]
        page_width = page.rect.width
        page_height = page.rect.height

        if debug_mode:
            time_now = time.time()
            logger.info(
                f"page_id: {page_id}, last_page_cost_time: {get_delta_time(start_time)}"
            )
            start_time = time_now
        """
        # 通过一个规则，过滤掉单页超过1500非junkimg的pdf
        # 对单页面非重复id的img数量做统计,如果当前页超过1500则直接return need_drop
        """
        page_imgs = page.get_images()
        img_counts = 0
        for img in page_imgs:
            img_bojid = img[0]
            if img_bojid in junk_img_bojids:  # 判断这个图片在不在junklist中
                continue  # 如果在junklist就不用管了，跳过
            else:
                recs = page.get_image_rects(img, transform=True)
                if recs:  # 如果这张图在当前页面有展示
                    img_counts += 1
        if (
            img_counts >= 1500
        ):  # 如果去除了junkimg的影响，单页img仍然超过1500的话，就排除当前pdf
            logger.warning(
                f"page_id: {page_id}, img_counts: {img_counts}, drop this pdf: {book_name}, drop_reason: {DropReason.HIGH_COMPUTATIONAL_lOAD_BY_IMGS}"
            )
            result = {
                "_need_drop": True,
                "_drop_reason": DropReason.HIGH_COMPUTATIONAL_lOAD_BY_IMGS,
            }
            if not debug_mode:
                return result

        """
        ==================================================================================================================================
        首先获取基本的block数据，对pdf进行分解，获取图片、表格、公式、text的bbox
        """
        # 解析pdf原始文本block
        text_raw_blocks = page.get_text(
            "dict",
            flags=fitz.TEXTFLAGS_TEXT,
        )["blocks"]
        model_output_json = get_docx_model_output(
            pdf_model_output, page_id
        )

        # 解析图片
        image_bboxes = parse_images(page_id, page, model_output_json, junk_img_bojids)
        image_bboxes = fix_image_vertical(
            image_bboxes, text_raw_blocks
        )  # 修正图片的位置
        image_bboxes = fix_seperated_image(image_bboxes)  # 合并有边重合的图片

        old_image_bboxes = deepcopy(image_bboxes)
        image_bboxes = include_img_title(
            text_raw_blocks, image_bboxes
        )  # 向图片上方和下方寻找title，使用规则进行匹配，暂时只支持英文规则
        """此时image_bboxes中可能出现这种情况，水平并列的2个图片，下方分别有各自的子标题，2个子标题下方又有大标题（形如Figxxx)，会出现2个图片的bbox都包含了这个大标题，这种情况需要把图片合并"""
        image_bboxes = combine_images(image_bboxes)  # 合并图片

        # 解析表格并对table_bboxes进行位置的微调,防止表格周围的文字被截断
        table_bboxes = parse_tables(page_id, page, model_output_json)
        table_bboxes = fix_tables(
            page, table_bboxes, include_table_title=False, scan_line_num=2
        )  # 修正
        table_bboxes = fix_table_text_block(
            text_raw_blocks, table_bboxes
        )  # 修正与text block的关系,某些table修正与pymupdf获取到的table内textblock没有完全包含，因此要进行一次修正。
        # debug_show_bbox(pdf_docs, page_id, table_bboxes, [], [b['bbox'] for b in text_raw_blocks], join_path(save_path, book_name, f"{book_name}_debug.pdf"), 7)

        old_table_bboxes = deepcopy(table_bboxes)
        table_bboxes = include_table_title(
            text_raw_blocks, table_bboxes
        )  # 向table上方和下方寻找title，使用规则进行匹配，暂时只支持英文规则

        # 解析公式
        equations_inline_bboxes, equations_interline_bboxes = parse_equations(
            page_id, page, model_output_json
        )

        # get image box and caption !
        image_bboxes_with_caption = extract_caption_bbox(image_bboxes, old_image_bboxes)

        # get table box and caption !
        table_bboxes_with_caption = extract_caption_bbox(table_bboxes, old_table_bboxes)

        """
        ==================================================================================================================================
        进入预处理-1阶段
        -------------------
        # # 解析标题
        # title_bboxs = parse_titles(page_id, page, model_output_json)
        # # 评估Layout是否规整、简单
        # isSimpleLayout_flag, fullColumn_cnt, subColumn_cnt, curPage_loss = evaluate_pdf_layout(page_id, page, model_output_json)
        接下来开始进行预处理过程
        """
        # title_bboxs = parse_titles(page_id, page, model_output_json)
        
        """去掉每页的页码、页眉、页脚"""
        page_no_bboxs = parse_pageNos(page_id, page, model_output_json)
        header_bboxs = parse_headers(page_id, page, model_output_json)
        footer_bboxs = parse_footers(page_id, page, model_output_json)
        (
            image_bboxes,
            table_bboxes,
            remain_text_blocks,
            removed_hdr_foot_txt_block,
            removed_hdr_foot_img_block,
            removed_hdr_foot_table,
        ) = remove_headder_footer_one_page(
            text_raw_blocks,
            image_bboxes,
            table_bboxes,
            header_bboxs,
            footer_bboxs,
            page_no_bboxs,
            page_width,
            page_height,
        )

        """去除页面上半部分长条色块内的文本块"""
        remain_text_blocks, removed_colored_narrow_strip_background_text_block = (
            remove_colored_strip_textblock(remain_text_blocks, page)
        )

        # debug_show_bbox(pdf_docs, page_id, footnote_bboxes_by_model, [b['bbox'] for b in remain_text_blocks], header_bboxs, join_path(save_path, book_name, f"{book_name}_debug.pdf"), 7)

        """去掉旋转的文字：水印、垂直排列的文字"""
        remain_text_blocks, removed_non_horz_text_block = remove_rotate_side_textblock(
            remain_text_blocks, page_width, page_height
        )  # 去掉水印，非水平文字
        remain_text_blocks, removed_empty_side_block = remove_side_blank_block(
            remain_text_blocks, page_width, page_height
        )  # 删除页面四周可能会留下的完全空白的textblock，这种block形成原因未知

        """出现在图片、表格上的文字块去掉，把层叠的图片单独分离出来，不参与layout的计算"""
        (
            image_bboxes,
            table_bboxes,
            equations_interline_bboxes,
            equations_inline_bboxes,
            remain_text_blocks,
            text_block_on_image_removed,
            images_overlap_backup,
            interline_eq_temp_text_block,
        ) = resolve_bbox_overlap_conflict(
            image_bboxes,
            table_bboxes,
            equations_interline_bboxes,
            equations_inline_bboxes,
            remain_text_blocks,
        )

        # """去掉footnote, 从文字和图片中"""
        # # 通过模型识别到的footnote
        # footnote_bboxes_by_model = parse_footnotes_by_model(page_id, page, model_output_json, md_bookname_save_path,
        #                                                     debug_mode=debug_mode)
        # # 通过规则识别到的footnote
        # footnote_bboxes_by_rule = parse_footnotes_by_rule(remain_text_blocks, page_height, page_id)
        """
        ==================================================================================================================================
        """
        if debug_mode:  # debugmode截图到本地
            save_path = join_path(save_tmp_path, "md")

        # 把图、表、公式都进行截图，保存到存储上，返回图片路径作为内容
        image_info, image_backup_info, table_info, inline_eq_info, interline_eq_info = (
            txt_save_images_by_bboxes(
                book_name,
                page_id,
                page,
                save_path,
                image_bboxes,
                images_overlap_backup,
                table_bboxes,
                equations_inline_bboxes,
                equations_interline_bboxes,
                # 传入img_s3_client
                img_s3_client,
            )
        )  # 只要表格和图片的截图

        """"以下进入到公式替换环节 """
        char_level_text_blocks = page.get_text("rawdict", flags=fitz.TEXTFLAGS_TEXT)[
            "blocks"
        ]
        remain_text_blocks = combine_chars_to_pymudict(
            remain_text_blocks, char_level_text_blocks
        )  # 合并chars
        remain_text_blocks = replace_equations_in_textblock(
            remain_text_blocks, inline_eq_info, interline_eq_info
        )
        remain_text_blocks = remove_citation_marker(
            remain_text_blocks
        )  # 公式替换之后去角标，防止公式无法替换成功。但是这样也会带来个问题就是把角标当公式。各有优劣。
        remain_text_blocks = remove_chars_in_text_blocks(
            remain_text_blocks
        )  # 减少中间态数据体积
        # debug_show_bbox(pdf_docs, page_id, [b['bbox'] for b in inline_eq_info], [b['bbox'] for b in interline_eq_info], [], join_path(save_path, book_name, f"{book_name}_debug.pdf"), 3)

        """去掉footnote, 从文字和图片中(先去角标再去footnote试试)"""
        # 通过模型识别到的footnote
        footnote_bboxes_by_model = parse_footnotes_by_model(
            page_id,
            page,
            model_output_json,
            md_bookname_save_path,
            debug_mode=debug_mode,
        )
        # 通过规则识别到的footnote
        footnote_bboxes_by_rule = parse_footnotes_by_rule(
            remain_text_blocks, page_height, page_id, main_text_font
        )
        """进入pdf过滤器，去掉一些不合理的pdf"""
        is_good_pdf, err = pdf_filter(
            page, remain_text_blocks, table_bboxes, image_bboxes
        )
        if not is_good_pdf:
            logger.warning(
                f"page_id: {page_id}, drop this pdf: {book_name}, reason: {err}"
            )
            if not debug_mode:
                return err

        """
        ==================================================================================================================================
        进行版面布局切分和过滤
        """
        """在切分之前，先检查一下bbox是否有左右重叠的情况，如果有，那么就认为这个pdf暂时没有能力处理好，这种左右重叠的情况大概率是由于pdf里的行间公式、表格没有被正确识别出来造成的 """

        is_text_block_horz_overlap = check_text_block_horizontal_overlap(
            remain_text_blocks, header_bboxs, footer_bboxs
        )

        if is_text_block_horz_overlap:
            # debug_show_bbox(pdf_docs, page_id, [b['bbox'] for b in remain_text_blocks], [], [], join_path(save_path, book_name, f"{book_name}_debug.pdf"), 0)
            logger.warning(
                f"page_id: {page_id}, drop this pdf: {book_name}, reason: {DropReason.TEXT_BLCOK_HOR_OVERLAP}"
            )
            result = {
                "_need_drop": True,
                "_drop_reason": DropReason.TEXT_BLCOK_HOR_OVERLAP,
            }
            if not debug_mode:
                return result

        """统一格式化成一个数据结构用于计算layout"""
        page_y0 = 0 if len(header_bboxs) == 0 else max([b[3] for b in header_bboxs])
        page_y1 = (
            page_height if len(footer_bboxs) == 0 else min([b[1] for b in footer_bboxs])
        )
        left_x, right_x = get_side_boundry(
            removed_non_horz_text_block, page_width, page_height
        )
        page_boundry = [
            math.floor(left_x),
            page_y0 + 1,
            math.ceil(right_x),
            page_y1 - 1,
        ]
        # 返回的是一个数组，每个元素[x0, y0, x1, y1, block_content, idx_x, idx_y], 初始时候idx_x, idx_y都是None. 对于图片、公式来说，block_content是图片的地址， 对于段落来说，block_content是段落的内容

        all_bboxes = prepare_bboxes_for_layout_split(
            image_info,
            image_backup_info,
            table_info,
            inline_eq_info,
            interline_eq_info,
            remain_text_blocks,
            page_boundry,
            page,
        )
        # debug_show_bbox(pdf_docs, page_id, [], [], all_bboxes, join_path(save_path, book_name, f"{book_name}_debug.pdf"), 1)
        """page_y0, page_y1能够过滤掉页眉和页脚，不会算作layout内"""
        layout_bboxes, layout_tree = get_bboxes_layout(
            all_bboxes, page_boundry, page_id
        )

        if (
            len(remain_text_blocks) > 0
            and len(all_bboxes) > 0
            and len(layout_bboxes) == 0
        ):
            logger.warning(
                f"page_id: {page_id}, drop this pdf: {book_name}, reason: {DropReason.CAN_NOT_DETECT_PAGE_LAYOUT}"
            )
            result = {
                "_need_drop": True,
                "_drop_reason": DropReason.CAN_NOT_DETECT_PAGE_LAYOUT,
            }
            if not debug_mode:
                return result

        """以下去掉复杂的布局和超过2列的布局"""
        if any(
            [lay["layout_label"] == LAYOUT_UNPROC for lay in layout_bboxes]
        ):  # 复杂的布局
            logger.warning(
                f"page_id: {page_id}, drop this pdf: {book_name}, reason: {DropReason.COMPLICATED_LAYOUT}"
            )
            result = {"_need_drop": True, "_drop_reason": DropReason.COMPLICATED_LAYOUT}
            if not debug_mode:
                return result

        layout_column_width = get_columns_cnt_of_layout(layout_tree)
        if layout_column_width > 2:  # 去掉超过2列的布局pdf
            logger.warning(
                f"page_id: {page_id}, drop this pdf: {book_name}, reason: {DropReason.TOO_MANY_LAYOUT_COLUMNS}"
            )
            result = {
                "_need_drop": True,
                "_drop_reason": DropReason.TOO_MANY_LAYOUT_COLUMNS,
                "extra_info": {"column_cnt": layout_column_width},
            }
            if not debug_mode:
                return result

        """
        ==================================================================================================================================
        构造出下游需要的数据结构
        """
        remain_text_blocks = (
            remain_text_blocks + interline_eq_temp_text_block
        )  # 把计算layout时候临时删除的行间公式再放回去，防止行间公式替换的时候丢失。
        removed_text_blocks = []
        removed_text_blocks.extend(removed_hdr_foot_txt_block)
        # removed_text_blocks.extend(removed_footnote_text_block)
        removed_text_blocks.extend(text_block_on_image_removed)
        removed_text_blocks.extend(removed_non_horz_text_block)
        removed_text_blocks.extend(removed_colored_narrow_strip_background_text_block)

        removed_images = []
        # removed_images.extend(footnote_imgs)
        removed_images.extend(removed_hdr_foot_img_block)

        images_backup = []
        images_backup.extend(image_backup_info)
        remain_text_blocks = escape_special_markdown_char(
            remain_text_blocks
        )  # 转义span里的text
        sorted_text_remain_text_block = sort_text_block(
            remain_text_blocks, layout_bboxes
        )

        footnote_bboxes_tmp = []
        footnote_bboxes_tmp.extend(footnote_bboxes_by_model)
        footnote_bboxes_tmp.extend(footnote_bboxes_by_rule)

        page_info = construct_page_component(
            page_id,
            image_info,
            table_info,
            sorted_text_remain_text_block,
            layout_bboxes,
            inline_eq_info,
            interline_eq_info,
            page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"],
            removed_text_blocks=removed_text_blocks,
            removed_image_blocks=removed_images,
            images_backup=images_backup,
            droped_table_block=[],
            table_backup=[],
            layout_tree=layout_tree,
            page_w=page.rect.width,
            page_h=page.rect.height,
            footnote_bboxes_tmp=footnote_bboxes_tmp,
        )

        page_info["image_bboxes_with_caption"] = image_bboxes_with_caption  # add by xr
        page_info["table_bboxes_with_caption"] = table_bboxes_with_caption

        page_info["bak_page_no_bboxes"] = page_no_bboxs
        page_info["bak_header_bboxes"] = header_bboxs
        page_info["bak_footer_bboxes"] = footer_bboxs
        page_info["bak_footer_note_bboxes"] = footnote_bboxes_tmp

        pdf_info_dict[f"page_{page_id}"] = page_info

    # end page for

    """计算后处理阶段耗时"""
    start_time = time.time()

    """
    ==================================================================================================================================
    去掉页眉和页脚，这里需要用到一定的统计量，所以放到最后
    页眉和页脚主要从文本box和图片box中去除，位于页面的四周。
    下面函数会直接修改pdf_info_dict,从文字块中、图片中删除属于页眉页脚的内容，删除内容做相对应记录
    """
    # 去页眉页脚
    header, footer = drop_footer_header(
        pdf_info_dict
    )  # TODO: using header and footer boxes here !

    """对单个layout内footnote和他下面的所有textbbox合并"""

    for page_key, page_info in pdf_info_dict.items():
        page_info = merge_footnote_blocks(page_info, main_text_font)
        page_info = remove_footnote_blocks(page_info)
        pdf_info_dict[page_key] = page_info

    """进入pdf后置过滤器，去掉一些不合理的pdf"""

    i = 0
    for page_info in pdf_info_dict.values():
        is_good_pdf, err = pdf_post_filter(page_info)
        if not is_good_pdf:
            logger.warning(f"page_id: {i}, drop this pdf: {book_name}, reason: {err}")
            if not debug_mode:
                return err
        i += 1

    if debug_mode:
        params_file_save_path = join_path(
            save_tmp_path, "md", book_name, "preproc_out.json"
        )
        page_draw_rect_save_path = join_path(
            save_tmp_path, "md", book_name, "layout.pdf"
        )
        # dir_path = os.path.dirname(page_draw_rect_save_path)
        # if not os.path.exists(dir_path):
        #     # 如果目录不存在，创建它
        #     os.makedirs(dir_path)

        with open(params_file_save_path, "w", encoding="utf-8") as f:
            json.dump(pdf_info_dict, f, ensure_ascii=False, indent=4)
        # 先检测本地 page_draw_rect_save_path 是否存在，如果存在则删除
        if os.path.exists(page_draw_rect_save_path):
            os.remove(page_draw_rect_save_path)
        # 绘制bbox和layout到pdf
        draw_bbox_on_page(pdf_docs, pdf_info_dict, page_draw_rect_save_path)
        draw_layout_bbox_on_page(
            pdf_docs, pdf_info_dict, header, footer, page_draw_rect_save_path
        )

    if debug_mode:
        # 打印后处理阶段耗时
        logger.info(f"post_processing_time: {get_delta_time(start_time)}")

    """
    ==================================================================================================================================
    进入段落处理-2阶段
    """

    # 处理行内文字间距较大问题
    pdf_info_dict = solve_inline_too_large_interval(pdf_info_dict)

    start_time = time.time()

    para_process_pipeline = ParaProcessPipeline()

    def _deal_with_text_exception(error_info):
        logger.warning(
            f"page_id: {page_id}, drop this pdf: {book_name}, reason: {error_info}"
        )
        if error_info == denseSingleLineBlockException_msg:
            logger.warning(
                f"Drop this pdf: {book_name}, reason: {DropReason.DENSE_SINGLE_LINE_BLOCK}"
            )
            result = {
                "_need_drop": True,
                "_drop_reason": DropReason.DENSE_SINGLE_LINE_BLOCK,
            }
            return result
        if error_info == titleDetectionException_msg:
            logger.warning(
                f"Drop this pdf: {book_name}, reason: {DropReason.TITLE_DETECTION_FAILED}"
            )
            result = {
                "_need_drop": True,
                "_drop_reason": DropReason.TITLE_DETECTION_FAILED,
            }
            return result
        elif error_info == titleLevelException_msg:
            logger.warning(
                f"Drop this pdf: {book_name}, reason: {DropReason.TITLE_LEVEL_FAILED}"
            )
            result = {"_need_drop": True, "_drop_reason": DropReason.TITLE_LEVEL_FAILED}
            return result
        elif error_info == paraSplitException_msg:
            logger.warning(
                f"Drop this pdf: {book_name}, reason: {DropReason.PARA_SPLIT_FAILED}"
            )
            result = {"_need_drop": True, "_drop_reason": DropReason.PARA_SPLIT_FAILED}
            return result
        elif error_info == paraMergeException_msg:
            logger.warning(
                f"Drop this pdf: {book_name}, reason: {DropReason.PARA_MERGE_FAILED}"
            )
            result = {"_need_drop": True, "_drop_reason": DropReason.PARA_MERGE_FAILED}
            return result

    if debug_mode:
        input_pdf_file = f"{pdf_local_path}.pdf"
        output_dir = f"{save_path}/{book_name}"
        output_pdf_file = f"{output_dir}/pdf_annos.pdf"

        """
        Call the para_process_pipeline function to process the pdf_info_dict.
        
        Parameters:
        para_debug_mode: str or None
            If para_debug_mode is None, the para_process_pipeline will not keep any intermediate results.
            If para_debug_mode is "simple", the para_process_pipeline will only keep the annos on the pdf and the final results as a json file.
            If para_debug_mode is "full", the para_process_pipeline will keep all the intermediate results generated during each step.
        """
        pdf_info_dict, error_info = para_process_pipeline.para_process_pipeline(
            pdf_info_dict,
            para_debug_mode="simple",
            input_pdf_path=input_pdf_file,
            output_pdf_path=output_pdf_file,
        )
        # 打印段落处理阶段耗时
        logger.info(f"para_process_time: {get_delta_time(start_time)}")

        # debug的时候不return drop信息
        if error_info is not None:
            _deal_with_text_exception(error_info)
        return pdf_info_dict
    else:
        pdf_info_dict, error_info = para_process_pipeline.para_process_pipeline(
            pdf_info_dict
        )
        if error_info is not None:
            return _deal_with_text_exception(error_info)

    return pdf_info_dict
