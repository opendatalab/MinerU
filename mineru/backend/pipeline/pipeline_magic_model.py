from mineru.utils.boxbase import bbox_relative_pos, get_minbox_if_overlap_by_ratio
from mineru.utils.enum_class import CategoryId, ContentType
import mineru.utils.magic_model_utils as magic_model_utils


class MagicModel:
    """每个函数没有得到元素的时候返回空list."""
    def __init__(self, page_model_info: dict, scale: float):
        self.__page_model_info = page_model_info
        self.__scale = scale
        """为所有模型数据添加bbox信息(缩放，poly->bbox)"""
        self.__fix_axis()
        """删除置信度特别低的模型数据(<0.05),提高质量"""
        self.__fix_by_remove_low_confidence()
        """删除高iou(>0.9)数据中置信度较低的那个"""
        self.__fix_by_remove_high_iou_and_low_confidence()
        """将部分tbale_footnote修正为image_footnote"""
        self.__fix_footnote()
        """处理重叠的image_body和table_body"""
        self.__fix_by_remove_overlap_image_table_body()

    def __fix_by_remove_overlap_image_table_body(self):
        need_remove_list = []
        layout_dets = self.__page_model_info['layout_dets']
        image_blocks = list(filter(
            lambda x: x['category_id'] == CategoryId.ImageBody, layout_dets
        ))
        table_blocks = list(filter(
            lambda x: x['category_id'] == CategoryId.TableBody, layout_dets
        ))

        def add_need_remove_block(blocks):
            for i in range(len(blocks)):
                for j in range(i + 1, len(blocks)):
                    block1 = blocks[i]
                    block2 = blocks[j]
                    overlap_box = get_minbox_if_overlap_by_ratio(
                        block1['bbox'], block2['bbox'], 0.8
                    )
                    if overlap_box is not None:
                        # 判断哪个区块的面积更小，移除较小的区块
                        area1 = (block1['bbox'][2] - block1['bbox'][0]) * (block1['bbox'][3] - block1['bbox'][1])
                        area2 = (block2['bbox'][2] - block2['bbox'][0]) * (block2['bbox'][3] - block2['bbox'][1])

                        if area1 <= area2:
                            block_to_remove = block1
                            large_block = block2
                        else:
                            block_to_remove = block2
                            large_block = block1

                        if block_to_remove not in need_remove_list:
                            # 扩展大区块的边界框
                            x1, y1, x2, y2 = large_block['bbox']
                            sx1, sy1, sx2, sy2 = block_to_remove['bbox']
                            x1 = min(x1, sx1)
                            y1 = min(y1, sy1)
                            x2 = max(x2, sx2)
                            y2 = max(y2, sy2)
                            large_block['bbox'] = [x1, y1, x2, y2]
                            need_remove_list.append(block_to_remove)

        # 处理图像-图像重叠
        add_need_remove_block(image_blocks)
        # 处理表格-表格重叠
        add_need_remove_block(table_blocks)

        # 从布局中移除标记的区块
        for need_remove in need_remove_list:
            if need_remove in layout_dets:
                layout_dets.remove(need_remove)


    def __fix_axis(self):
        need_remove_list = []
        layout_dets = self.__page_model_info['layout_dets']
        for layout_det in layout_dets:
            x0, y0, _, _, x1, y1, _, _ = layout_det['poly']
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

    def __fix_by_remove_low_confidence(self):
        magic_model_utils.remove_low_confidence(self.__page_model_info['layout_dets'])

    def __fix_by_remove_high_iou_and_low_confidence(self):
        layout_dets = list(filter(
            lambda x: x['category_id'] in [
                    CategoryId.Title,
                    CategoryId.Text,
                    CategoryId.ImageBody,
                    CategoryId.ImageCaption,
                    CategoryId.TableBody,
                    CategoryId.TableCaption,
                    CategoryId.TableFootnote,
                    CategoryId.InterlineEquation_Layout,
                    CategoryId.InterlineEquationNumber_Layout,
                ], self.__page_model_info['layout_dets']
            )
        )
        magic_model_utils.remove_high_iou_low_confidence(layout_dets)

    def __fix_footnote(self):
        footnotes = []
        figures = []
        tables = []

        for obj in self.__page_model_info['layout_dets']:
            if obj['category_id'] == CategoryId.TableFootnote:
                footnotes.append(obj)
            elif obj['category_id'] == CategoryId.ImageBody:
                figures.append(obj)
            elif obj['category_id'] == CategoryId.TableBody:
                tables.append(obj)
            if len(footnotes) * len(figures) == 0:
                continue
        dis_figure_footnote = {}
        dis_table_footnote = {}

        for i in range(len(footnotes)):
            for j in range(len(figures)):
                pos_flag_count = sum(
                    list(
                        map(
                            lambda x: 1 if x else 0,
                            bbox_relative_pos(
                                footnotes[i]['bbox'], figures[j]['bbox']
                            ),
                        )
                    )
                )
                if pos_flag_count > 1:
                    continue
                dis_figure_footnote[i] = min(
                    magic_model_utils.bbox_distance_with_relative_check(figures[j]['bbox'], footnotes[i]['bbox']),
                    dis_figure_footnote.get(i, float('inf')),
                )
        for i in range(len(footnotes)):
            for j in range(len(tables)):
                pos_flag_count = sum(
                    list(
                        map(
                            lambda x: 1 if x else 0,
                            bbox_relative_pos(
                                footnotes[i]['bbox'], tables[j]['bbox']
                            ),
                        )
                    )
                )
                if pos_flag_count > 1:
                    continue

                dis_table_footnote[i] = min(
                    magic_model_utils.bbox_distance_with_relative_check(tables[j]['bbox'], footnotes[i]['bbox']),
                    dis_table_footnote.get(i, float('inf')),
                )
        for i in range(len(footnotes)):
            if i not in dis_figure_footnote:
                continue
            if dis_table_footnote.get(i, float('inf')) > dis_figure_footnote[i]:
                footnotes[i]['category_id'] = CategoryId.ImageFootnote

    def __tie_up_category_by_distance_v3(
        self,
        subject_category_id: int,
        object_category_id: int,
    ):
        return magic_model_utils.tie_up_category_by_distance_v3(
            self.__page_model_info,
            subject_category_id,
            object_category_id,
            extract_bbox_func=lambda x: x['bbox'],
            extract_score_func=lambda x: x['score'],
            create_item_func=lambda x: {'bbox': x['bbox'], 'score': x['score']}
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