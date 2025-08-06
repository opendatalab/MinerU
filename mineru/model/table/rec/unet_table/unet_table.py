import html
import os
import time
import traceback
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict, Any

import cv2
import numpy as np
from loguru import logger
from rapid_table import RapidTableInput, RapidTable

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from .table_structure_unet import TSRUnet
from .table_recover import TableRecover
from .wired_table_rec_utils import InputType, LoadImage
from .table_recover_utils import (
    match_ocr_cell,
    plot_html_table,
    box_4_2_poly_to_box_4_1,
    sorted_ocr_boxes,
    gather_ocr_list_by_row,
)


@dataclass
class UnetTableInput:
    model_path: str
    device: str = "cpu"


@dataclass
class UnetTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None


class UnetTableRecognition:
    def __init__(self, config: UnetTableInput):
        self.table_structure = TSRUnet(asdict(config))
        self.load_img = LoadImage()
        self.table_recover = TableRecover()

    def __call__(
        self,
        img: InputType,
        ocr_result: Optional[List[Union[List[List[float]], str, str]]] = None,
        ocr_engine = None,
        **kwargs,
    ) -> UnetTableOutput:
        s = time.perf_counter()
        need_ocr = True
        col_threshold = 15
        row_threshold = 10
        if kwargs:
            need_ocr = kwargs.get("need_ocr", True)
            col_threshold = kwargs.get("col_threshold", 15)
            row_threshold = kwargs.get("row_threshold", 10)
        img = self.load_img(img)
        polygons, rotated_polygons = self.table_structure(img, **kwargs)
        if polygons is None:
            logger.warning("polygons is None.")
            return UnetTableOutput("", None, None, 0.0)

        try:
            table_res, logi_points = self.table_recover(
                rotated_polygons, row_threshold, col_threshold
            )
            # 将坐标由逆时针转为顺时针方向，后续处理与无线表格对齐
            polygons[:, 1, :], polygons[:, 3, :] = (
                polygons[:, 3, :].copy(),
                polygons[:, 1, :].copy(),
            )
            if not need_ocr:
                sorted_polygons, idx_list = sorted_ocr_boxes(
                    [box_4_2_poly_to_box_4_1(box) for box in polygons]
                )
                return UnetTableOutput(
                    "",
                    sorted_polygons,
                    logi_points[idx_list],
                    time.perf_counter() - s,
                )
            cell_box_det_map, not_match_orc_boxes = match_ocr_cell(ocr_result, polygons)
            # 如果有识别框没有ocr结果，直接进行rec补充
            cell_box_det_map = self.fill_blank_rec(img, polygons, cell_box_det_map, ocr_engine)
            # 转换为中间格式，修正识别框坐标,将物理识别框，逻辑识别框，ocr识别框整合为dict，方便后续处理
            t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
            # 将每个单元格中的ocr识别结果排序和同行合并，输出的html能完整保留文字的换行格式
            t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)
            # cell_box_map =
            logi_points = [t_box_ocr["t_logic_box"] for t_box_ocr in t_rec_ocr_list]
            cell_box_det_map = {
                i: [ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]]
                for i, t_box_ocr in enumerate(t_rec_ocr_list)
            }
            pred_html = plot_html_table(logi_points, cell_box_det_map)
            polygons = np.array(polygons).reshape(-1, 8)
            logi_points = np.array(logi_points)
            elapse = time.perf_counter() - s

        except Exception:
            logger.warning(traceback.format_exc())
            return UnetTableOutput("", None, None, 0.0)
        return UnetTableOutput(pred_html, polygons, logi_points, elapse)

    def transform_res(
        self,
        cell_box_det_map: Dict[int, List[any]],
        polygons: np.ndarray,
        logi_points: List[np.ndarray],
    ) -> List[Dict[str, any]]:
        res = []
        for i in range(len(polygons)):
            ocr_res_list = cell_box_det_map.get(i)
            if not ocr_res_list:
                continue
            xmin = min([ocr_box[0][0][0] for ocr_box in ocr_res_list])
            ymin = min([ocr_box[0][0][1] for ocr_box in ocr_res_list])
            xmax = max([ocr_box[0][2][0] for ocr_box in ocr_res_list])
            ymax = max([ocr_box[0][2][1] for ocr_box in ocr_res_list])
            dict_res = {
                # xmin,xmax,ymin,ymax
                "t_box": [xmin, ymin, xmax, ymax],
                # row_start,row_end,col_start,col_end
                "t_logic_box": logi_points[i].tolist(),
                # [[xmin,xmax,ymin,ymax], text]
                "t_ocr_res": [
                    [box_4_2_poly_to_box_4_1(ocr_det[0]), ocr_det[1]]
                    for ocr_det in ocr_res_list
                ],
            }
            res.append(dict_res)
        return res

    def sort_and_gather_ocr_res(self, res):
        for i, dict_res in enumerate(res):
            _, sorted_idx = sorted_ocr_boxes(
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threshold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], threshold=0.3
            )
        return res

    def fill_blank_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
        ocr_engine
    ) -> Dict[int, List[Any]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        img_crop_info_list = []
        img_crop_list = []
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            box = sorted_polygons[i]
            if ocr_engine is None:
                logger.warning(f"No OCR engine provided for box {i}: {box}")
                continue
            # 从img中截取对应的区域
            x1, y1, x2, y2 = box[0][0], box[0][1], box[2][0], box[2][1]
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid box coordinates: {box}")
                continue
            img_crop = img[int(y1):int(y2), int(x1):int(x2)]
            img_crop_list.append(img_crop)
            img_crop_info_list.append([i, box])
            continue

        if len(img_crop_list) > 0:
            # 进行ocr识别
            ocr_res_list = ocr_engine.ocr(img_crop_list, det=False)[0]
            assert len(ocr_res_list) == len(img_crop_list)
            for j, ocr_res in enumerate(ocr_res_list):
                img_crop_info_list[j].append(ocr_res)


            for i, box, ocr_res in img_crop_info_list:
                # 处理ocr结果
                ocr_text, ocr_score = ocr_res
                # logger.debug(f"OCR result for box {i}: {ocr_text} with score {ocr_score}")
                if ocr_score < 0.9:
                    # logger.warning(f"Low confidence OCR result for box {i}: {ocr_text} with score {ocr_score}")
                    continue
                cell_box_map[i] = [[box, ocr_text, ocr_score]]

        return cell_box_map


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class UnetTableModel:
    def __init__(self, ocr_engine):
        model_path = os.path.join(auto_download_and_get_model_root_path(ModelPath.unet_structure), ModelPath.unet_structure)
        wired_input_args = UnetTableInput(model_path=model_path)
        self.wired_table_model = UnetTableRecognition(wired_input_args)
        slanet_plus_model_path = os.path.join(auto_download_and_get_model_root_path(ModelPath.slanet_plus), ModelPath.slanet_plus)
        wireless_input_args = RapidTableInput(model_type='slanet_plus', model_path=slanet_plus_model_path)
        self.wireless_table_model = RapidTable(wireless_input_args)
        self.ocr_engine = ocr_engine

    def predict(self, img, table_cls_score):
        bgr_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ocr_result = self.ocr_engine.ocr(bgr_img)[0]

        if ocr_result:
            ocr_result = [
                [item[0], escape_html(item[1][0]), item[1][1]]
                for item in ocr_result
                if len(item) == 2 and isinstance(item[1], tuple)
            ]
        else:
            ocr_result = None
        if ocr_result:
            try:
                wired_table_results = self.wired_table_model(np.asarray(img), ocr_result, self.ocr_engine)
                wired_html_code = wired_table_results.pred_html
                wired_table_cell_bboxes = wired_table_results.cell_bboxes
                wired_logic_points = wired_table_results.logic_points
                wired_elapse = wired_table_results.elapse

                wireless_table_results = self.wireless_table_model(np.asarray(img), ocr_result)
                wireless_html_code = wireless_table_results.pred_html
                wireless_table_cell_bboxes = wireless_table_results.cell_bboxes
                wireless_logic_points = wireless_table_results.logic_points
                wireless_elapse = wireless_table_results.elapse

                wired_len = len(wired_table_cell_bboxes) if wired_table_cell_bboxes is not None else 0
                wireless_len = len(wireless_table_cell_bboxes) if wireless_table_cell_bboxes is not None else 0
                # logger.debug(f"wired table cell bboxes: {wired_len}, wireless table cell bboxes: {wireless_len}")
                # 计算两种模型检测的单元格数量差异
                gap_of_len = wireless_len - wired_len
                # 判断是否使用无线表格模型的结果
                if (
                    wired_len <= round(wireless_len * 0.5)  # 有线模型检测到的单元格数太少（低于无线模型的50%）
                    or (wireless_len < wired_len < (2 * wireless_len) and table_cls_score <= 0.949)  # 有线模型检测到的单元格数反而更多
                    or (0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75))  # 两者相差不大但有线模型结果较少
                    or (gap_of_len == 0 and wired_len <= 4)  # 单元格数量完全相等且总量小于等于4
                ):
                    # logger.debug("fall back to wireless table model")
                    html_code = wireless_html_code
                    table_cell_bboxes = wireless_table_cell_bboxes
                    logic_points = wireless_logic_points
                else:
                    html_code = wired_html_code
                    table_cell_bboxes = wired_table_cell_bboxes
                    logic_points = wired_logic_points

                elapse = wired_elapse + wireless_elapse
                return html_code, table_cell_bboxes, logic_points, elapse
            except Exception as e:
                logger.exception(e)
        return None, None, None, None
