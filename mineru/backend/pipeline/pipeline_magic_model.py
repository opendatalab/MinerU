from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.span_pre_proc import txt_spans_extract


class MagicModel:

    PP_DOCLAYOUT_V2_LABELS_TO_BLOCK_TYPES = {
        "abstract": BlockType.ABSTRACT,
        "algorithm": BlockType.ALGORITHM,
        "aside_text": BlockType.ASIDE_TEXT,
        "chart": BlockType.CHART,
        "content": BlockType.TEXT,
        "display_formula": BlockType.INTERLINE_EQUATION,
        "doc_title": BlockType.DOC_TITLE,
        "figure_title": BlockType.CAPTION,
        "footer": BlockType.FOOTER,
        "footer_image": BlockType.FOOTER_IMAGE,
        "footnote": BlockType.PAGE_FOOTNOTE,
        "formula_number":BlockType.FORMULA_NUMBER,
        "header": BlockType.HEADER,
        "header_image": BlockType.HEADER_IMAGE,
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

    def __init__(self,
                 page_model_info: dict,
                 page,
                 scale,
                 page_pil_img,
                 page_w,
                 page_h,
                 ocr_enable
    ):
        self.__page_model_info = page_model_info
        self.page_inline_formula = []
        self.page_ocr_res = []
        self.page_blocks = []
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
                []
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


    @staticmethod
    def __is_inline_formula_block(layout_det: dict) -> bool:
        return (
            layout_det.get('label') == 'inline_formula'
            or layout_det.get('cls_id') == 15
        )

    @staticmethod
    def __is_ocr_text_block(layout_det: dict) -> bool:
        return layout_det.get('label') == 'ocr_text'

    def __fix_axis(self):
        need_remove_list = []
        layout_dets = self.__page_model_info['layout_dets']
        for layout_det in layout_dets:
            x0, y0, x1, y1 = layout_det['bbox']
            bbox = [
                int(x0 / self.__scale),
                int(y0 / self.__scale),
                int(x1 / self.__scale),
                int(y1 / self.__scale),
            ]
            layout_det['bbox'] = bbox
            # 删除高度或者宽度小于等于2的spans
            if bbox[2] - bbox[0] <= 2 or bbox[3] - bbox[1] <= 2:
                need_remove_list.append(layout_det)
        for need_remove in need_remove_list:
            layout_dets.remove(need_remove)

    def __post_process(self):
        next_index = 1
        layout_dets = self.__page_model_info['layout_dets']
        for layout_det in layout_dets:
            if self.__is_inline_formula_block(layout_det):
                layout_det.pop('index', None)
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

            if 'index' in layout_det:
                layout_det['index'] = next_index
                next_index += 1