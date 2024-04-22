import json
import math

from magic_pdf.libs.commons import fitz
from loguru import logger

from magic_pdf.libs.commons import join_path
from magic_pdf.libs.coordinate_transform import get_scale_ratio
from magic_pdf.libs.ocr_content_type import ContentType
from magic_pdf.rw.AbsReaderWriter import AbsReaderWriter
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from magic_pdf.libs.math import float_gt
from magic_pdf.libs.boxbase import _is_in, bbox_relative_pos, bbox_distance
from magic_pdf.libs.ModelBlockTypeEnum import ModelBlockTypeEnum


class MagicModel:
    """
    每个函数没有得到元素的时候返回空list

    """

    def __fix_axis(self):
        need_remove_list = []
        for model_page_info in self.__model_list:
            page_no = model_page_info["page_info"]["page_no"]
            horizontal_scale_ratio, vertical_scale_ratio = get_scale_ratio(
                model_page_info, self.__docs[page_no]
            )
            layout_dets = model_page_info["layout_dets"]
            for layout_det in layout_dets:
                x0, y0, _, _, x1, y1, _, _ = layout_det["poly"]
                bbox = [
                    int(x0 / horizontal_scale_ratio),
                    int(y0 / vertical_scale_ratio),
                    int(x1 / horizontal_scale_ratio),
                    int(y1 / vertical_scale_ratio),
                ]
                layout_det["bbox"] = bbox
                # 删除高度或者宽度为0的spans
                if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
                    need_remove_list.append(layout_det)
            for need_remove in need_remove_list:
                layout_dets.remove(need_remove)


    def __init__(self, model_list: list, docs: fitz.Document):
        self.__model_list = model_list
        self.__docs = docs
        self.__fix_axis()

    def __reduct_overlap(self, bboxes):
        N = len(bboxes)
        keep = [True] * N
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if _is_in(bboxes[i], bboxes[j]):
                    keep[i] = False

        return [bboxes[i] for i in range(N) if keep[i]]

    def __tie_up_category_by_distance(
        self, page_no, subject_category_id, object_category_id
    ):
        """
        假定每个 subject 最多有一个 object (可以有多个相邻的 object 合并为单个 object)，每个 object 只能属于一个 subject
        """
        ret = []
        MAX_DIS_OF_POINT = 10**9 + 7

        subjects = self.__reduct_overlap(
            list(
                map(
                    lambda x: x["bbox"],
                    filter(
                        lambda x: x["category_id"] == subject_category_id,
                        self.__model_list[page_no]["layout_dets"],
                    ),
                )
            )
        )

        objects = self.__reduct_overlap(
            list(
                map(
                    lambda x: x["bbox"],
                    filter(
                        lambda x: x["category_id"] == object_category_id,
                        self.__model_list[page_no]["layout_dets"],
                    ),
                )
            )
        )
        subject_object_relation_map = {}

        subjects.sort(key=lambda x: x[0] ** 2 + x[1] ** 2)  # get the distance !

        all_bboxes = []

        for v in subjects:
            all_bboxes.append({"category_id": subject_category_id, "bbox": v})

        for v in objects:
            all_bboxes.append({"category_id": object_category_id, "bbox": v})

        N = len(all_bboxes)
        dis = [[MAX_DIS_OF_POINT] * N for _ in range(N)]

        for i in range(N):
            for j in range(i):
                if (
                    all_bboxes[i]["category_id"] == subject_category_id
                    and all_bboxes[j]["category_id"] == subject_category_id
                ):
                    continue

                dis[i][j] = bbox_distance(all_bboxes[i]["bbox"], all_bboxes[j]["bbox"])
                dis[j][i] = dis[i][j]

        used = set()
        for i in range(N):
            # 求第 i 个 subject 所关联的 object
            if all_bboxes[i]["category_id"] != subject_category_id:
                continue
            seen = set()
            candidates = []
            arr = []
            for j in range(N):

                pos_flag_count = sum(
                    list(
                        map(
                            lambda x: 1 if x else 0,
                            bbox_relative_pos(
                                all_bboxes[i]["bbox"], all_bboxes[j]["bbox"]
                            ),
                        )
                    )
                )
                if pos_flag_count > 1:
                    continue
                if (
                    all_bboxes[j]["category_id"] != object_category_id
                    or j in used
                    or dis[i][j] == MAX_DIS_OF_POINT
                ):
                    continue
                arr.append((dis[i][j], j))

            arr.sort(key=lambda x: x[0])
            if len(arr) > 0:
                candidates.append(arr[0][1])
                seen.add(arr[0][1])

            # 已经获取初始种子
            for j in set(candidates):
                tmp = []
                for k in range(i + 1, N):
                    pos_flag_count = sum(
                        list(
                            map(
                                lambda x: 1 if x else 0,
                                bbox_relative_pos(
                                    all_bboxes[j]["bbox"], all_bboxes[k]["bbox"]
                                ),
                            )
                        )
                    )

                    if pos_flag_count > 1:
                        continue

                    if (
                        all_bboxes[k]["category_id"] != object_category_id
                        or k in used
                        or k in seen
                        or dis[j][k] == MAX_DIS_OF_POINT
                    ):
                        continue
                    is_nearest = True
                    for l in range(i + 1, N):
                        if l in (j, k) or l in used or l in seen:
                            continue


                        if not float_gt(dis[l][k], dis[j][k]):
                            is_nearest = False
                            break


                    if is_nearest:
                        tmp.append(k)
                        seen.add(k)

                candidates = tmp
                if len(candidates) == 0:
                    break

            # 已经获取到某个 figure 下所有的最靠近的 captions，以及最靠近这些 captions 的 captions 。
            # 先扩一下 bbox，
            x0s = [all_bboxes[idx]["bbox"][0] for idx in seen] + [
                all_bboxes[i]["bbox"][0]
            ]
            y0s = [all_bboxes[idx]["bbox"][1] for idx in seen] + [
                all_bboxes[i]["bbox"][1]
            ]
            x1s = [all_bboxes[idx]["bbox"][2] for idx in seen] + [
                all_bboxes[i]["bbox"][2]
            ]
            y1s = [all_bboxes[idx]["bbox"][3] for idx in seen] + [
                all_bboxes[i]["bbox"][3]
            ]

            ox0, oy0, ox1, oy1 = min(x0s), min(y0s), max(x1s), max(y1s)
            ix0, iy0, ix1, iy1 = all_bboxes[i]["bbox"]

            # 分成了 4 个截取空间，需要计算落在每个截取空间下 objects 合并后占据的矩形面积
            caption_poses = [
                [ox0, oy0, ix0, oy1],
                [ox0, oy0, ox1, iy0],
                [ox0, iy1, ox1, oy1],
                [ix1, oy0, ox1, oy1],
            ]

            caption_areas = []
            for bbox in caption_poses:
                embed_arr = []
                for idx in seen:
                    if _is_in(all_bboxes[idx]["bbox"], bbox):
                        embed_arr.append(idx)

                if len(embed_arr) > 0:
                    embed_x0 = min([all_bboxes[idx]["bbox"][0] for idx in embed_arr])
                    embed_y0 = min([all_bboxes[idx]["bbox"][1] for idx in embed_arr])
                    embed_x1 = max([all_bboxes[idx]["bbox"][2] for idx in embed_arr])
                    embed_y1 = max([all_bboxes[idx]["bbox"][3] for idx in embed_arr])
                    caption_areas.append(
                        int(abs(embed_x1 - embed_x0) * abs(embed_y1 - embed_y0))
                    )
                else:
                    caption_areas.append(0)

            subject_object_relation_map[i] = []
            if max(caption_areas) > 0:
                max_area_idx = caption_areas.index(max(caption_areas))
                caption_bbox = caption_poses[max_area_idx]

                for j in seen:
                    if _is_in(all_bboxes[j]["bbox"], caption_bbox):
                        used.add(j)
                        subject_object_relation_map[i].append(j)

        for i in sorted(subject_object_relation_map.keys()):
            result = {
                "subject_body": all_bboxes[i]["bbox"],
                "all": all_bboxes[i]["bbox"],
            }

            if len(subject_object_relation_map[i]) > 0:
                x0 = min(
                    [all_bboxes[j]["bbox"][0] for j in subject_object_relation_map[i]]
                )
                y0 = min(
                    [all_bboxes[j]["bbox"][1] for j in subject_object_relation_map[i]]
                )
                x1 = max(
                    [all_bboxes[j]["bbox"][2] for j in subject_object_relation_map[i]]
                )
                y1 = max(
                    [all_bboxes[j]["bbox"][3] for j in subject_object_relation_map[i]]
                )
                result["object_body"] = [x0, y0, x1, y1]
                result["all"] = [
                    min(x0, all_bboxes[i]["bbox"][0]),
                    min(y0, all_bboxes[i]["bbox"][1]),
                    max(x1, all_bboxes[i]["bbox"][2]),
                    max(y1, all_bboxes[i]["bbox"][3]),
                ]
            ret.append(result)

        total_subject_object_dis = 0
        # 计算已经配对的 distance 距离
        for i in subject_object_relation_map.keys():
            for j in subject_object_relation_map[i]:
                total_subject_object_dis += bbox_distance(
                    all_bboxes[i]["bbox"], all_bboxes[j]["bbox"]
                )

        # 计算未匹配的 subject 和 object 的距离（非精确版）
        with_caption_subject = set(
            [
                key
                for key in subject_object_relation_map.keys()
                if len(subject_object_relation_map[i]) > 0
            ]
        )
        for i in range(N):
            if all_bboxes[i]["category_id"] != object_category_id or i in used:
                continue
            candidates = []
            for j in range(N):
                if (
                    all_bboxes[j]["category_id"] != subject_category_id
                    or j in with_caption_subject
                ):
                    continue
                candidates.append((dis[i][j], j))
            if len(candidates) > 0:
                candidates.sort(key=lambda x: x[0])
                total_subject_object_dis += candidates[0][1]
                with_caption_subject.add(j)
        return ret, total_subject_object_dis

    def get_imgs(self, page_no: int):  # @许瑞
        records, _ = self.__tie_up_category_by_distance(page_no, 3, 4)
        return [
            {
                "bbox": record["all"],
                "img_body_bbox": record["subject_body"],
                "img_caption_bbox": record.get("object_body", None),
            }
            for record in records
        ]

    def get_tables(
        self, page_no: int
    ) -> list:  # 3个坐标， caption, table主体，table-note
        with_captions, _ = self.__tie_up_category_by_distance(page_no, 5, 6)
        with_footnotes, _ = self.__tie_up_category_by_distance(page_no, 5, 7)
        ret = []
        N, M = len(with_captions), len(with_footnotes)
        assert N == M
        for i in range(N):
            record = {
                "table_caption_bbox": with_captions[i].get("object_body", None),
                "table_body_bbox": with_captions[i]["subject_body"],
                "table_footnote_bbox": with_footnotes[i].get("object_body", None),
            }

            x0 = min(with_captions[i]["all"][0], with_footnotes[i]["all"][0])
            y0 = min(with_captions[i]["all"][1], with_footnotes[i]["all"][1])
            x1 = max(with_captions[i]["all"][2], with_footnotes[i]["all"][2])
            y1 = max(with_captions[i]["all"][3], with_footnotes[i]["all"][3])
            record["bbox"] = [x0, y0, x1, y1]
            ret.append(record)
        return ret

    def get_equations(self, page_no: int) -> list:  # 有坐标，也有字
        inline_equations = self.__get_blocks_by_type(ModelBlockTypeEnum.EMBEDDING.value, page_no, ["latex"])
        interline_equations = self.__get_blocks_by_type(ModelBlockTypeEnum.ISOLATED.value, page_no, ["latex"])
        interline_equations_blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.ISOLATE_FORMULA.value, page_no)
        return inline_equations, interline_equations, interline_equations_blocks

    def get_discarded(self, page_no: int) -> list:  # 自研模型，只有坐标
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.ABANDON.value, page_no)
        return blocks

    def get_text_blocks(self, page_no: int) -> list:  # 自研模型搞的，只有坐标，没有字
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.PLAIN_TEXT.value, page_no)
        return blocks

    def get_title_blocks(self, page_no: int) -> list:  # 自研模型，只有坐标，没字
        blocks = self.__get_blocks_by_type(ModelBlockTypeEnum.TITLE.value, page_no)
        return blocks

    def get_ocr_text(self, page_no: int) -> list:  # paddle 搞的，有字也有坐标
        text_spans = []
        model_page_info = self.__model_list[page_no]
        layout_dets = model_page_info["layout_dets"]
        for layout_det in layout_dets:
            if layout_det["category_id"] == "15":
                span = {
                    "bbox": layout_det['bbox'],
                    "content": layout_det["text"],
                }
                text_spans.append(span)
        return text_spans

    def get_all_spans(self, page_no: int) -> list:
        all_spans = []
        model_page_info = self.__model_list[page_no]
        layout_dets = model_page_info["layout_dets"]
        allow_category_id_list = [3, 5, 13, 14, 15]
        """当成span拼接的"""
        #  3: 'image', # 图片
        #  5: 'table',       # 表格
        #  13: 'inline_equation',     # 行内公式
        #  14: 'interline_equation',      # 行间公式
        #  15: 'text',      # ocr识别文本
        for layout_det in layout_dets:
            category_id = layout_det["category_id"]
            if category_id in allow_category_id_list:
                span = {
                    "bbox": layout_det['bbox']
                }
                if category_id == 3:
                    span["type"] = ContentType.Image
                elif category_id == 5:
                    span["type"] = ContentType.Table
                elif category_id == 13:
                    span["content"] = layout_det["latex"]
                    span["type"] = ContentType.InlineEquation
                elif category_id == 14:
                    span["content"] = layout_det["latex"]
                    span["type"] = ContentType.InterlineEquation
                elif category_id == 15:
                    span["content"] = layout_det["text"]
                    span["type"] = ContentType.Text
                all_spans.append(span)
        return all_spans

    def get_page_size(self, page_no: int):  # 获取页面宽高
        # 获取当前页的page对象
        page = self.__docs[page_no]
        # 获取当前页的宽高
        page_w = page.rect.width
        page_h = page.rect.height
        return page_w, page_h

    def __get_blocks_by_type(self, type: int, page_no: int, extra_col: list[str] = []) -> list:
        blocks = []
        for page_dict in self.__model_list:
            layout_dets = page_dict.get("layout_dets", [])
            page_info = page_dict.get("page_info", {})
            page_number = page_info.get("page_no", -1)
            if page_no != page_number:
                continue
            for item in layout_dets:
                category_id = item.get("category_id", -1)
                bbox = item.get("bbox", None)

                if category_id == type:
                    block = {
                        "bbox": bbox
                    }
                    for col in extra_col:
                        block[col] = item.get(col, None)
                    blocks.append(block)
        return blocks

if __name__ == "__main__":
    drw = DiskReaderWriter(r"D:/project/20231108code-clean")
    if 0:
        pdf_file_path = r"linshixuqiu\19983-00.pdf"
        model_file_path = r"linshixuqiu\19983-00_new.json"
        pdf_bytes = drw.read(pdf_file_path, AbsReaderWriter.MODE_BIN)
        model_json_txt = drw.read(model_file_path, AbsReaderWriter.MODE_TXT)
        model_list = json.loads(model_json_txt)
        write_path = r"D:\project\20231108code-clean\linshixuqiu\19983-00"
        img_bucket_path = "imgs"
        img_writer = DiskReaderWriter(join_path(write_path, img_bucket_path))
        pdf_docs = fitz.open("pdf", pdf_bytes)
        magic_model = MagicModel(model_list, pdf_docs)

    if 1:
        model_list = json.loads(
            drw.read("/opt/data/pdf/20240418/j.chroma.2009.03.042.json")
        )
        pdf_bytes = drw.read(
            "/opt/data/pdf/20240418/j.chroma.2009.03.042.pdf", AbsReaderWriter.MODE_BIN
        )
        pdf_docs = fitz.open("pdf", pdf_bytes)
        magic_model = MagicModel(model_list, pdf_docs)
        for i in range(7):
            print(magic_model.get_imgs(i))
