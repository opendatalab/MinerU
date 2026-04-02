from mineru.backend.pipeline.para_split import ListLineTag
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import _merge_para_text
from mineru.utils.boxbase import (
    bbox_center_distance,
    bbox_distance,
    calculate_overlap_area_2_minbox_area_ratio,
    calculate_overlap_area_in_bbox1_area_ratio,
)
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.guess_suffix_or_lang import guess_language_by_text
from mineru.utils.span_block_fix import merge_spans_to_vertical_line, vertical_line_sort_spans_from_top_to_bottom, \
    merge_spans_to_line, line_sort_spans_by_left_to_right
from mineru.utils.span_pre_proc import txt_spans_extract


class MagicModel:

    PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES = {
        "abstract": BlockType.ABSTRACT,
        "algorithm": BlockType.CODE,
        "aside_text": BlockType.ASIDE_TEXT,
        "chart": BlockType.CHART,
        "content": BlockType.INDEX,
        "display_formula": BlockType.INTERLINE_EQUATION,
        "doc_title": BlockType.DOC_TITLE,
        "figure_title": BlockType.CAPTION,
        "footer": BlockType.FOOTER,
        "footer_image": BlockType.FOOTER,
        "footnote": BlockType.PAGE_FOOTNOTE,
        "formula_number": BlockType.FORMULA_NUMBER,
        "header": BlockType.HEADER,
        "header_image": BlockType.HEADER,
        "image": BlockType.IMAGE,
        "number": BlockType.PAGE_NUMBER,
        "paragraph_title": BlockType.PARAGRAPH_TITLE,
        "reference_content": BlockType.REF_TEXT,
        "seal": BlockType.SEAL,
        "table": BlockType.TABLE,
        "text": BlockType.TEXT,
        "vertical_text": BlockType.VERTICAL_TEXT,
        "vision_footnote": BlockType.FOOTNOTE,
    }

    VISUAL_MAIN_TYPES = (BlockType.IMAGE, BlockType.TABLE, BlockType.CHART, BlockType.CODE)
    VISUAL_CHILD_TYPES = (BlockType.CAPTION, BlockType.FOOTNOTE)
    VISUAL_TYPE_MAPPING = {
        BlockType.IMAGE: {
            "body": BlockType.IMAGE_BODY,
            "caption": BlockType.IMAGE_CAPTION,
            "footnote": BlockType.IMAGE_FOOTNOTE,
        },
        BlockType.TABLE: {
            "body": BlockType.TABLE_BODY,
            "caption": BlockType.TABLE_CAPTION,
            "footnote": BlockType.TABLE_FOOTNOTE,
        },
        BlockType.CHART: {
            "body": BlockType.CHART_BODY,
            "caption": BlockType.CHART_CAPTION,
            "footnote": BlockType.CHART_FOOTNOTE,
        },
        BlockType.CODE: {
            "body": BlockType.CODE_BODY,
            "caption": BlockType.CODE_CAPTION,
            "footnote": BlockType.CODE_FOOTNOTE,
        }
    }

    def __init__(
        self,
        page_model_info: dict,
        page=None,
        scale=1,
        page_pil_img=None,
        page_w=None,
        page_h=None,
        ocr_enable=False,
    ):
        self.__page_model_info = page_model_info
        self.page_inline_formula = []
        self.page_ocr_res = []
        self.page_blocks = []
        self.image_groups = []
        self.table_groups = []
        self.chart_groups = []
        self.all_image_spans = []
        self.__layout_det_by_index = {}
        self.__scale = scale
        self.__fix_axis()  # bbox坐标修正，删除高度或者宽度小于等于0的spans
        self.__post_process()  # index重排，填充行内公式和文本span
        if not ocr_enable:
            virtual_block = [0, 0, page_w, page_h, None, None, None, "text"]
            self.page_ocr_res = txt_spans_extract(
                page,
                self.page_ocr_res,
                page_pil_img,
                scale,
                [virtual_block],
                [],
            )
        self.page_text_inline_formula_spans = self.page_inline_formula + self.page_ocr_res

        for layout_det in self.__page_model_info['layout_dets']:
            if layout_det.get('label') in self.PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES:
                block_bbox = layout_det['bbox']
                block_type = self.PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES[layout_det['label']]
                block_index = layout_det['index']
                block_score = layout_det['score']
                block = self.__copy_block_fields(
                    layout_det,
                    type=block_type,
                    bbox=block_bbox,
                    index=block_index,
                    score=block_score,
                )
                self.page_blocks.append(block)

        self.page_blocks.sort(key=lambda x: x["index"])
        self.__build_page_blocks()
        self.__classify_visual_blocks()
        self.__build_return_blocks()


    @staticmethod
    def __fix_text_block(block):
        if block["type"] == BlockType.VERTICAL_TEXT:
            # 如果是纵向文本块，则按纵向lines处理
            block_lines = merge_spans_to_vertical_line(block['spans'])
            sort_block_lines = vertical_line_sort_spans_from_top_to_bottom(block_lines)
        else:
            block_lines = merge_spans_to_line(block['spans'])
            sort_block_lines = line_sort_spans_by_left_to_right(block_lines)

        if block["type"] == BlockType.CODE:
            for line in sort_block_lines:
                line[ListLineTag.IS_LIST_START_LINE] = True
            code_content = _merge_para_text(
                {'lines': sort_block_lines},
                False,
                '\n'
            )
            guess_lang = guess_language_by_text(code_content)
            if guess_lang not in ["txt", "unknown"]:
                block["sub_type"] = "code"
                block["guess_lang"] = guess_lang

        block['lines'] = sort_block_lines
        del block['spans']
        return block

    @staticmethod
    def __copy_block_fields(block, **overrides):
        copied_block = {
            key: value
            for key, value in block.items()
            if key not in {"cls_id", "label"}
        }
        copied_block.update(overrides)
        return copied_block


    @staticmethod
    def __is_inline_formula_block(layout_det: dict) -> bool:
        return (
            layout_det.get("label") == "inline_formula"
            or layout_det.get("cls_id") == 15
        )

    @staticmethod
    def __is_ocr_text_block(layout_det: dict) -> bool:
        return layout_det.get("label") == "ocr_text"

    def __build_return_blocks(self):
        self.preproc_blocks = []
        self.discarded_blocks = []
        for block in self.page_blocks:
            if block["type"] in [
                BlockType.HEADER,
                BlockType.FOOTER,
                BlockType.PAGE_NUMBER,
                BlockType.ASIDE_TEXT,
                BlockType.PAGE_FOOTNOTE
            ]:
                self.discarded_blocks.append(block)
            else:
                # 单独处理code block
                if block["type"] in [BlockType.CODE]:
                    for sub_block in block["blocks"]:
                        if sub_block["type"] == BlockType.CODE_BODY:
                            block["sub_type"] = sub_block.pop("sub_type", "algorithm")
                            if block["sub_type"] == "code":
                                block["guess_lang"] = sub_block.pop("guess_lang", "txt")

                self.preproc_blocks.append(block)

    def __build_page_blocks(self):
        span_type = "unknown"
        for block in self.page_blocks:
            if block["type"] in [
                BlockType.ABSTRACT,
                BlockType.CODE,
                BlockType.ASIDE_TEXT,
                BlockType.INDEX,
                BlockType.DOC_TITLE,
                BlockType.CAPTION,
                BlockType.FOOTER,
                BlockType.PAGE_FOOTNOTE,
                BlockType.FORMULA_NUMBER,
                BlockType.HEADER,
                BlockType.PAGE_NUMBER,
                BlockType.PARAGRAPH_TITLE,
                BlockType.REF_TEXT,
                BlockType.TEXT,
                BlockType.VERTICAL_TEXT,
                BlockType.FOOTNOTE,
            ]:
                span_type = ContentType.TEXT
            elif block["type"] in [BlockType.IMAGE]:
                span_type = ContentType.IMAGE
            elif block["type"] in [BlockType.TABLE]:
                span_type = ContentType.TABLE
            elif block["type"] in [BlockType.CHART]:
                span_type = ContentType.CHART
            elif block["type"] in [BlockType.INTERLINE_EQUATION]:
                span_type = ContentType.INTERLINE_EQUATION
            elif block["type"] in [BlockType.SEAL]:
                span_type = ContentType.SEAL

            if span_type in [
                ContentType.IMAGE,
                ContentType.TABLE,
                ContentType.CHART,
                ContentType.INTERLINE_EQUATION,
                ContentType.SEAL
            ]:
                span = {
                    "bbox": block["bbox"],
                    "type": span_type,
                }
                if span_type == ContentType.TABLE:
                    span["html"] = block.get("html", "")
                    block.pop("html", None)
                if span_type == ContentType.INTERLINE_EQUATION:
                    span["content"] = block.get("latex", "")
                    block.pop("latex", None)
                if span_type == ContentType.SEAL:
                    span["content"] = block.get("text")
                    block.pop("text", None)

                self.all_image_spans.append(span)
                # 构造line对象
                spans = [span]
                line = {"bbox": block["bbox"], "spans": spans}
                block["lines"] = [line]
            else:
                # span填充
                block_spans = []
                for span in self.page_text_inline_formula_spans:
                    overlap_ratio = calculate_overlap_area_in_bbox1_area_ratio(
                        span['bbox'], block["bbox"]
                    )
                    if block["type"] == BlockType.FORMULA_NUMBER:
                        # OCR 检测框通常会比公式编号框更大，使用最小框重叠比避免编号文字无法回填。
                        overlap_ratio = max(
                            overlap_ratio,
                            calculate_overlap_area_2_minbox_area_ratio(
                                span['bbox'], block["bbox"]
                            ),
                        )
                    if overlap_ratio > 0.5:
                        block_spans.append(span)
                # 从spans删除已经放入block_spans中的span
                if len(block_spans) > 0:
                    for span in block_spans:
                        self.page_text_inline_formula_spans.remove(span)

                block["spans"] = block_spans
                block = self.__fix_text_block(block)

    def __fix_axis(self):
        need_remove_list = []
        layout_dets = self.__page_model_info["layout_dets"]
        for layout_det in layout_dets:
            x0, y0, x1, y1 = layout_det["bbox"]
            bbox = [
                int(x0 / self.__scale),
                int(y0 / self.__scale),
                int(x1 / self.__scale),
                int(y1 / self.__scale),
            ]
            layout_det["bbox"] = bbox
            # 删除高度或者宽度小于等于2的spans
            if bbox[2] - bbox[0] <= 2 or bbox[3] - bbox[1] <= 2:
                need_remove_list.append(layout_det)
        for need_remove in need_remove_list:
            layout_dets.remove(need_remove)

    def __post_process(self):
        next_index = 1
        layout_dets = self.__page_model_info["layout_dets"]
        for layout_det in layout_dets:
            if self.__is_inline_formula_block(layout_det):
                layout_det.pop("index", None)
                self.page_inline_formula.append({
                    "bbox": layout_det["bbox"],
                    "type": ContentType.INLINE_EQUATION,
                    "content": layout_det["latex"],
                    "score": layout_det["score"],
                })
                continue

            if self.__is_ocr_text_block(layout_det):
                self.page_ocr_res.append({
                    "bbox": layout_det["bbox"],
                    "type": ContentType.TEXT,
                    "content": layout_det["text"],
                    "score": layout_det["score"],
                })
                continue

            if "index" in layout_det:
                layout_det["index"] = next_index
                next_index += 1

    def __classify_visual_blocks(self):
        if not self.page_blocks:
            return

        ordered_blocks = sorted(self.page_blocks, key=lambda x: x["index"])
        original_type_by_index = {
            block["index"]: block["type"] for block in ordered_blocks
        }
        position_by_index = {
            block["index"]: pos for pos, block in enumerate(ordered_blocks)
        }
        main_blocks = [
            block
            for block in ordered_blocks
            if original_type_by_index[block["index"]] in self.VISUAL_MAIN_TYPES
        ]
        child_blocks = [
            block
            for block in ordered_blocks
            if original_type_by_index[block["index"]] in self.VISUAL_CHILD_TYPES
        ]

        child_parent_map = {}
        grouped_children = {
            main_block["index"]: {"captions": [], "footnotes": []}
            for main_block in main_blocks
        }

        for child_block in child_blocks:
            parent_block = self.__find_best_visual_parent(
                child_block,
                main_blocks,
                ordered_blocks,
                original_type_by_index,
                position_by_index,
            )
            child_parent_map[child_block["index"]] = (
                None if parent_block is None else parent_block["index"]
            )

        for child_block in child_blocks:
            original_child_type = original_type_by_index[child_block["index"]]
            parent_index = child_parent_map[child_block["index"]]

            if parent_index is None:
                child_block["type"] = BlockType.TEXT
                self.__sync_layout_det_type(child_block["index"], BlockType.TEXT)
                continue

            parent_type = original_type_by_index[parent_index]
            child_kind = self.__child_kind(original_child_type)
            mapped_type = self.VISUAL_TYPE_MAPPING[parent_type][child_kind]
            child_block["type"] = mapped_type
            self.__sync_layout_det_type(child_block["index"], mapped_type)
            grouped_children[parent_index][f"{child_kind}s"].append(child_block)

        self.image_groups = []
        self.table_groups = []
        self.chart_groups = []

        rebuilt_page_blocks = []
        for block in ordered_blocks:
            original_block_type = original_type_by_index[block["index"]]

            if original_block_type in self.VISUAL_CHILD_TYPES:
                if child_parent_map[block["index"]] is None:
                    rebuilt_page_blocks.append(block)
                continue

            if original_block_type not in self.VISUAL_MAIN_TYPES:
                rebuilt_page_blocks.append(block)
                continue

            mapping = self.VISUAL_TYPE_MAPPING[original_block_type]
            body_block = self.__make_child_block(block, mapping["body"])
            captions = sorted(
                [
                    self.__make_child_block(caption, mapping["caption"])
                    for caption in grouped_children[block["index"]]["captions"]
                ],
                key=lambda x: x["index"],
            )
            footnotes = sorted(
                [
                    self.__make_child_block(footnote, mapping["footnote"])
                    for footnote in grouped_children[block["index"]]["footnotes"]
                ],
                key=lambda x: x["index"],
            )

            self.__sync_layout_det_type(block["index"], mapping["body"])

            group_info = {
                f"{original_block_type}_body": body_block,
                f"{original_block_type}_caption_list": captions,
                f"{original_block_type}_footnote_list": footnotes,
            }
            if original_block_type == BlockType.IMAGE:
                self.image_groups.append(group_info)
            elif original_block_type == BlockType.TABLE:
                self.table_groups.append(group_info)
            else:
                self.chart_groups.append(group_info)

            two_layer_block = {
                "type": original_block_type,
                "bbox": block["bbox"],
                "blocks": [body_block, *captions, *footnotes],
                "index": block["index"],
                "score": block.get("score"),
            }
            # 对blocks按index排序
            two_layer_block["blocks"].sort(key=lambda x: x["index"])
            rebuilt_page_blocks.append(two_layer_block)

        self.page_blocks = rebuilt_page_blocks

    def __find_best_visual_parent(
        self,
        child_block,
        main_blocks,
        ordered_blocks,
        original_type_by_index,
        position_by_index,
    ):
        child_type = original_type_by_index[child_block["index"]]
        if child_type not in self.VISUAL_CHILD_TYPES:
            return None

        best_parent = None
        best_key = None
        for main_block in main_blocks:
            if not self.__is_visual_neighbor(
                child_block,
                main_block,
                ordered_blocks,
                original_type_by_index,
                position_by_index,
            ):
                continue

            candidate_key = (
                abs(child_block["index"] - main_block["index"]),
                bbox_distance(child_block["bbox"], main_block["bbox"]),
                bbox_center_distance(child_block["bbox"], main_block["bbox"]),
                main_block["index"],
            )

            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_parent = main_block

        return best_parent

    def __is_visual_neighbor(
        self,
        child_block,
        main_block,
        ordered_blocks,
        original_type_by_index,
        position_by_index,
    ):
        child_type = original_type_by_index[child_block["index"]]
        if child_type == BlockType.FOOTNOTE and child_block["index"] < main_block["index"]:
            return False

        child_pos = position_by_index[child_block["index"]]
        main_pos = position_by_index[main_block["index"]]
        start_pos = min(child_pos, main_pos) + 1
        end_pos = max(child_pos, main_pos)

        for pos in range(start_pos, end_pos):
            between_block = ordered_blocks[pos]
            if original_type_by_index[between_block["index"]] != child_type:
                return False

        return True

    @staticmethod
    def __child_kind(block_type):
        if block_type == BlockType.CAPTION:
            return "caption"
        return "footnote"

    @staticmethod
    def __make_child_block(block, block_type):
        return MagicModel.__copy_block_fields(block, type=block_type)

    def __sync_layout_det_type(self, block_index, block_type):
        layout_det = self.__layout_det_by_index.get(block_index)
        if layout_det is not None:
            layout_det["type"] = block_type

    def get_page_blocks(self):
        return self.page_blocks

    def get_all_image_spans(self):
        return self.all_image_spans

    def get_preproc_blocks(self):
        return self.preproc_blocks

    def get_discarded_blocks(self):
        return self.discarded_blocks

    def get_imgs(self):
        return self.image_groups

    def get_tables(self):
        return self.table_groups

    def get_charts(self):
        return self.chart_groups

    def get_image_blocks(self):
        return [
            block for block in self.page_blocks if block["type"] == BlockType.IMAGE
        ]

    def get_table_blocks(self):
        return [
            block for block in self.page_blocks if block["type"] == BlockType.TABLE
        ]

    def get_chart_blocks(self):
        return [
            block for block in self.page_blocks if block["type"] == BlockType.CHART
        ]
