from mineru.utils.boxbase import bbox_relative_pos, calculate_iou, bbox_distance, get_minbox_if_overlap_by_ratio
from mineru.utils.enum_class import CategoryId, ContentType
from mineru.utils.magic_model_utils import tie_up_category_by_distance_v3, reduct_overlap


class MagicModel:
    """每个函数没有得到元素的时候返回空list."""
    def __init__(self, page_model_info: dict, scale: float):
        self.__page_model_info = page_model_info
        self.__scale = scale
        """为所有模型数据添加bbox信息(缩放，poly->bbox)"""
        self.__fix_axis()

    @staticmethod
    def __is_inline_formula_block(layout_det: dict) -> bool:
        return (
            layout_det.get('label') == 'inline_formula'
            or layout_det.get('cls_id') == 15
        )

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
            # 删除高度或者宽度小于等于0的spans
            if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                need_remove_list.append(layout_det)
        for need_remove in need_remove_list:
            layout_dets.remove(need_remove)

        next_index = 1
        for layout_det in layout_dets:
            if self.__is_inline_formula_block(layout_det):
                layout_det.pop('index', None)
                continue

            if 'index' in layout_det:
                layout_det['index'] = next_index
                next_index += 1


    def __tie_up_category_by_distance_v3(self, subject_category_id, object_category_id):
        # 定义获取主体和客体对象的函数
        def get_subjects():
            return reduct_overlap(
                list(
                    map(
                        lambda x: {'bbox': x['bbox'], 'score': x['score']},
                        filter(
                            lambda x: x['category_id'] == subject_category_id,
                            self.__page_model_info['layout_dets'],
                        ),
                    )
                )
            )

        def get_objects():
            return reduct_overlap(
                list(
                    map(
                        lambda x: {'bbox': x['bbox'], 'score': x['score']},
                        filter(
                            lambda x: x['category_id'] == object_category_id,
                            self.__page_model_info['layout_dets'],
                        ),
                    )
                )
            )

        # 调用通用方法
        return tie_up_category_by_distance_v3(
            get_subjects,
            get_objects
        )

    def get_imgs(self):
        with_captions = self.__tie_up_category_by_distance_v3(
            CategoryId.ImageBody, CategoryId.ImageCaption
        )
        with_footnotes = self.__tie_up_category_by_distance_v3(
            CategoryId.ImageBody, CategoryId.ImageFootnote
        )
        ret = []
        for v in with_captions:
            record = {
                'image_body': v['sub_bbox'],
                'image_caption_list': v['obj_bboxes'],
            }
            filter_idx = v['sub_idx']
            d = next(filter(lambda x: x['sub_idx'] == filter_idx, with_footnotes))
            record['image_footnote_list'] = d['obj_bboxes']
            ret.append(record)
        return ret

    def get_tables(self) -> list:
        with_captions = self.__tie_up_category_by_distance_v3(
            CategoryId.TableBody, CategoryId.TableCaption
        )
        with_footnotes = self.__tie_up_category_by_distance_v3(
            CategoryId.TableBody, CategoryId.TableFootnote
        )
        ret = []
        for v in with_captions:
            record = {
                'table_body': v['sub_bbox'],
                'table_caption_list': v['obj_bboxes'],
            }
            filter_idx = v['sub_idx']
            d = next(filter(lambda x: x['sub_idx'] == filter_idx, with_footnotes))
            record['table_footnote_list'] = d['obj_bboxes']
            ret.append(record)
        return ret

    def get_equations(self) -> tuple[list, list, list]:  # 有坐标，也有字
        inline_equations = self.__get_blocks_by_type(
            CategoryId.InlineEquation, ['latex']
        )
        interline_equations = self.__get_blocks_by_type(
            CategoryId.InterlineEquation_YOLO, ['latex']
        )
        interline_equations_blocks = self.__get_blocks_by_type(
            CategoryId.InterlineEquation_Layout
        )
        return inline_equations, interline_equations, interline_equations_blocks

    def get_discarded(self) -> list:  # 自研模型，只有坐标
        blocks = self.__get_blocks_by_type(CategoryId.Abandon)
        return blocks

    def get_text_blocks(self) -> list:  # 自研模型搞的，只有坐标，没有字
        blocks = self.__get_blocks_by_type(CategoryId.Text)
        return blocks

    def get_title_blocks(self) -> list:  # 自研模型，只有坐标，没字
        blocks = self.__get_blocks_by_type(CategoryId.Title)
        return blocks

    def get_all_spans(self) -> list:

        def remove_duplicate_spans(spans):
            new_spans = []
            for span in spans:
                if not any(span == existing_span for existing_span in new_spans):
                    new_spans.append(span)
            return new_spans

        all_spans = []
        layout_dets = self.__page_model_info['layout_dets']
        allow_category_id_list = [
            CategoryId.ImageBody,
            CategoryId.TableBody,
            CategoryId.InlineEquation,
            CategoryId.InterlineEquation_YOLO,
            CategoryId.OcrText,
        ]
        """当成span拼接的"""
        for layout_det in layout_dets:
            category_id = layout_det['category_id']
            if category_id in allow_category_id_list:
                span = {'bbox': layout_det['bbox'], 'score': layout_det['score']}
                if category_id == CategoryId.ImageBody:
                    span['type'] = ContentType.IMAGE
                elif category_id == CategoryId.TableBody:
                    # 获取table模型结果
                    latex = layout_det.get('latex', None)
                    html = layout_det.get('html', None)
                    if latex:
                        span['latex'] = latex
                    elif html:
                        span['html'] = html
                    span['type'] = ContentType.TABLE
                elif category_id == CategoryId.InlineEquation:
                    span['content'] = layout_det['latex']
                    span['type'] = ContentType.INLINE_EQUATION
                elif category_id == CategoryId.InterlineEquation_YOLO:
                    span['content'] = layout_det['latex']
                    span['type'] = ContentType.INTERLINE_EQUATION
                elif category_id == CategoryId.OcrText:
                    span['content'] = layout_det['text']
                    span['type'] = ContentType.TEXT
                all_spans.append(span)
        return remove_duplicate_spans(all_spans)

    def __get_blocks_by_type(
        self, category_type: int, extra_col=None
    ) -> list:
        if extra_col is None:
            extra_col = []
        blocks = []
        layout_dets = self.__page_model_info.get('layout_dets', [])
        for item in layout_dets:
            category_id = item.get('category_id', -1)
            bbox = item.get('bbox', None)

            if category_id == category_type:
                block = {
                    'bbox': bbox,
                    'score': item.get('score'),
                }
                for col in extra_col:
                    block[col] = item.get(col, None)
                blocks.append(block)
        return blocks
