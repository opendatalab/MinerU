from mineru.utils.boxbase import bbox_center_distance, bbox_distance
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.span_pre_proc import txt_spans_extract


class MagicModel:

    PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES = {
        "abstract": BlockType.ABSTRACT,
        "algorithm": BlockType.ALGORITHM,
        "aside_text": BlockType.ASIDE_TEXT,
        "chart": BlockType.CHART,
        "content": BlockType.INDEX,
        "display_formula": BlockType.INTERLINE_EQUATION,
        "doc_title": BlockType.DOC_TITLE,
        "figure_title": BlockType.CAPTION,
        "footer": BlockType.FOOTER,
        "footnote": BlockType.PAGE_FOOTNOTE,
        "formula_number": BlockType.FORMULA_NUMBER,
        "header": BlockType.HEADER,
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

    VISUAL_MAIN_TYPES = (BlockType.IMAGE, BlockType.TABLE, BlockType.CHART)
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
    }

    def __init__(
        self,
        page_model_info: dict,
        page=None,
        scale=1,
        page_pil_img=None,
        page_w=None,
        page_h=None,
        ocr_enable=True,
    ):
        self.__page_model_info = page_model_info
        self.page_inline_formula = []
        self.page_ocr_res = []
        self.page_blocks = []
        self.image_groups = []
        self.table_groups = []
        self.chart_groups = []
        self.__layout_det_by_index = {}
        self.__scale = scale
        self.__fix_axis()  # bbox坐标修正，删除高度或者宽度小于等于0的spans
        self.__post_process()  # index重排，填充行内公式和文本span
        self.page_text_inline_formula_spans = self.page_inline_formula + self.page_ocr_res

        if not ocr_enable:
            virtual_block = [0, 0, page_w, page_h, None, None, None, "text"]
            self.page_text_inline_formula_spans = txt_spans_extract(
                page,
                self.page_text_inline_formula_spans,
                page_pil_img,
                scale,
                [virtual_block],
                [],
            )

        for layout_det in self.__page_model_info['layout_dets']:
            if layout_det.get('label') in self.PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES:
                block_bbox = layout_det['bbox']
                block_type = self.PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES[layout_det['label']]
                block_index = layout_det['index']
                block_score = layout_det['score']
                self.page_blocks.append({
                    "bbox": block_bbox,
                    "type": block_type,
                    "index": block_index,
                    "score": block_score,
                })

        self.page_blocks.sort(key=lambda x: x["index"])
        self.__classify_visual_blocks()

        return None

    @staticmethod
    def __is_inline_formula_block(layout_det: dict) -> bool:
        return (
            layout_det.get("label") == "inline_formula"
            or layout_det.get("cls_id") == 15
        )

    @staticmethod
    def __is_ocr_text_block(layout_det: dict) -> bool:
        return layout_det.get("label") == "ocr_text"

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

            rebuilt_page_blocks.append(
                {
                    "type": original_block_type,
                    "bbox": block["bbox"],
                    "blocks": [body_block, *captions, *footnotes],
                    "index": block["index"],
                    "score": block.get("score"),
                }
            )

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
        return {
            "type": block_type,
            "bbox": block["bbox"],
            "index": block["index"],
            "score": block.get("score"),
        }

    def __sync_layout_det_type(self, block_index, block_type):
        layout_det = self.__layout_det_by_index.get(block_index)
        if layout_det is not None:
            layout_det["type"] = block_type

    def get_page_blocks(self):
        return self.page_blocks

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
