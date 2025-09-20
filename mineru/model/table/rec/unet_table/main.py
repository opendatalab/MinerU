import html
import logging
import os
import time
import traceback
from dataclasses import dataclass, asdict

from typing import List, Optional, Union, Dict, Any
import numpy as np
import cv2
from PIL import Image
from loguru import logger
from bs4 import BeautifulSoup

from mineru.utils.span_pre_proc import calculate_contrast
from .table_structure_unet import TSRUnet

from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from .table_recover import TableRecover
from .utils import InputType, LoadImage, VisTable
from .utils_table_recover import (
    match_ocr_cell,
    plot_html_table,
    box_4_2_poly_to_box_4_1,
    sorted_ocr_boxes,
    gather_ocr_list_by_row,
)


@dataclass
class WiredTableInput:
    model_path: str
    device: str = "cpu"


@dataclass
class WiredTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None


class WiredTableRecognition:
    def __init__(self, config: WiredTableInput, ocr_engine=None):
        self.table_structure = TSRUnet(asdict(config))
        self.load_img = LoadImage()
        self.table_recover = TableRecover()
        self.ocr_engine = ocr_engine

    def __call__(
        self,
        img: InputType,
        ocr_result: Optional[List[Union[List[List[float]], str, str]]] = None,
        **kwargs,
    ) -> WiredTableOutput:
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
            # logging.warning("polygons is None.")
            return WiredTableOutput("", None, None, 0.0)

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
                return WiredTableOutput(
                    "",
                    sorted_polygons,
                    logi_points[idx_list],
                    time.perf_counter() - s,
                )
            cell_box_det_map, not_match_orc_boxes = match_ocr_cell(ocr_result, polygons)
            # 如果有识别框没有ocr结果，直接进行rec补充
            cell_box_det_map = self.fill_blank_rec(img, polygons, cell_box_det_map)
            # 转换为中间格式，修正识别框坐标,将物理识别框，逻辑识别框，ocr识别框整合为dict，方便后续处理
            t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
            # 将每个单元格中的ocr识别结果排序和同行合并，输出的html能完整保留文字的换行格式
            t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)

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
            logging.warning(traceback.format_exc())
            return WiredTableOutput("", None, None, 0.0)
        return WiredTableOutput(pred_html, polygons, logi_points, elapse)

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
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threhold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], threhold=0.3
            )
        return res

    # def fill_blank_rec(
    #     self,
    #     img: np.ndarray,
    #     sorted_polygons: np.ndarray,
    #     cell_box_map: Dict[int, List[str]],
    # ) -> Dict[int, List[Any]]:
    #     """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
    #     for i in range(sorted_polygons.shape[0]):
    #         if cell_box_map.get(i):
    #             continue
    #         box = sorted_polygons[i]
    #         cell_box_map[i] = [[box, "", 1]]
    #         continue
    #     return cell_box_map
    def fill_blank_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
    ) -> Dict[int, List[Any]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_crop_info_list = []
        img_crop_list = []
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            box = sorted_polygons[i]
            if self.ocr_engine is None:
                logger.warning(f"No OCR engine provided for box {i}: {box}")
                continue
            # 从img中截取对应的区域
            x1, y1, x2, y2 = int(box[0][0])+1, int(box[0][1])+1, int(box[2][0])-1, int(box[2][1])-1
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                # logger.warning(f"Invalid box coordinates: {x1, y1, x2, y2}")
                continue
            # 判断长宽比
            if (x2 - x1) / (y2 - y1) > 20 or (y2 - y1) / (x2 - x1) > 20:
                # logger.warning(f"Box {i} has invalid aspect ratio: {x1, y1, x2, y2}")
                continue
            img_crop = bgr_img[int(y1):int(y2), int(x1):int(x2)]

            # 计算span的对比度，低于0.20的span不进行ocr
            if calculate_contrast(img_crop, img_mode='bgr') <= 0.17:
                cell_box_map[i] = [[box, "", 0.1]]
                # logger.debug(f"Box {i} skipped due to low contrast.")
                continue

            img_crop_list.append(img_crop)
            img_crop_info_list.append([i, box])

        if len(img_crop_list) > 0:
            # 进行ocr识别
            ocr_result = self.ocr_engine.ocr(img_crop_list, det=False)
            # ocr_result = [[]]
            # for crop_img in img_crop_list:
            #     tmp_ocr_result = self.ocr_engine.ocr(crop_img)
            #     if tmp_ocr_result[0] and len(tmp_ocr_result[0]) > 0 and isinstance(tmp_ocr_result[0], list) and len(tmp_ocr_result[0][0]) == 2:
            #         ocr_result[0].append(tmp_ocr_result[0][0][1])
            #     else:
            #         ocr_result[0].append(("", 0.0))

            if not ocr_result or not isinstance(ocr_result, list) or len(ocr_result) == 0:
                logger.warning("OCR engine returned no results or invalid result for image crops.")
                return cell_box_map
            ocr_res_list = ocr_result[0]
            if not isinstance(ocr_res_list, list) or len(ocr_res_list) != len(img_crop_list):
                logger.warning("OCR result list length does not match image crop list length.")
                return cell_box_map
            for j, ocr_res in enumerate(ocr_res_list):
                img_crop_info_list[j].append(ocr_res)

            for i, box, ocr_res in img_crop_info_list:
                # 处理ocr结果
                ocr_text, ocr_score = ocr_res
                # logger.debug(f"OCR result for box {i}: {ocr_text} with score {ocr_score}")
                if ocr_score < 0.6 or ocr_text in ['1','口','■','（204号', '（20', '（2', '（2号', '（20号', '号', '（204']:
                    # logger.warning(f"Low confidence OCR result for box {i}: {ocr_text} with score {ocr_score}")
                    box = sorted_polygons[i]
                    cell_box_map[i] = [[box, "", 0.1]]
                    continue
                cell_box_map[i] = [[box, ocr_text, ocr_score]]

        return cell_box_map


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


def count_table_cells_physical(html_code):
    """计算表格的物理单元格数量（合并单元格算一个）"""
    if not html_code:
        return 0

    # 简单计数td和th标签的数量
    html_lower = html_code.lower()
    td_count = html_lower.count('<td')
    th_count = html_lower.count('<th')
    return td_count + th_count


class UnetTableModel:
    def __init__(self, ocr_engine):
        model_path = os.path.join(auto_download_and_get_model_root_path(ModelPath.unet_structure), ModelPath.unet_structure)
        wired_input_args = WiredTableInput(model_path=model_path)
        self.wired_table_model = WiredTableRecognition(wired_input_args, ocr_engine)
        self.ocr_engine = ocr_engine

    def predict(self, input_img, ocr_result, wireless_html_code):
        if isinstance(input_img, Image.Image):
            np_img = np.asarray(input_img)
        elif isinstance(input_img, np.ndarray):
            np_img = input_img
        else:
            raise ValueError("Input must be a pillow object or a numpy array.")
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        if ocr_result is None:
            ocr_result = self.ocr_engine.ocr(bgr_img)[0]
            ocr_result = [
                [item[0], escape_html(item[1][0]), item[1][1]]
                for item in ocr_result
                if len(item) == 2 and isinstance(item[1], tuple)
            ]

        try:
            wired_table_results = self.wired_table_model(np_img, ocr_result)

            # viser = VisTable()
            # save_html_path = f"outputs/output.html"
            # save_drawed_path = f"outputs/output_table_vis.jpg"
            # save_logic_path = (
            #     f"outputs/output_table_vis_logic.jpg"
            # )
            # vis_imged = viser(
            #     np_img, wired_table_results, save_html_path, save_drawed_path, save_logic_path
            # )

            wired_html_code = wired_table_results.pred_html
            wired_len = count_table_cells_physical(wired_html_code)
            wireless_len = count_table_cells_physical(wireless_html_code)
            # 计算两种模型检测的单元格数量差异
            gap_of_len = wireless_len - wired_len
            # logger.debug(f"wired table cell bboxes: {wired_len}, wireless table cell bboxes: {wireless_len}")

            # 使用OCR结果计算两种模型填入的文字数量
            wireless_text_count = 0
            wired_text_count = 0
            for ocr_res in ocr_result:
                if ocr_res[1] in wireless_html_code:
                    wireless_text_count += 1
                if ocr_res[1] in wired_html_code:
                    wired_text_count += 1
            # logger.debug(f"wireless table ocr text count: {wireless_text_count}, wired table ocr text count: {wired_text_count}")

            # 使用HTML解析器计算空单元格数量
            wireless_soup = BeautifulSoup(wireless_html_code, 'html.parser') if wireless_html_code else BeautifulSoup("", 'html.parser')
            wired_soup = BeautifulSoup(wired_html_code, 'html.parser') if wired_html_code else BeautifulSoup("", 'html.parser')
            # 计算空单元格数量(没有文本内容或只有空白字符)
            wireless_blank_count = sum(1 for cell in wireless_soup.find_all(['td', 'th']) if not cell.text.strip())
            wired_blank_count = sum(1 for cell in wired_soup.find_all(['td', 'th']) if not cell.text.strip())
            # logger.debug(f"wireless table blank cell count: {wireless_blank_count}, wired table blank cell count: {wired_blank_count}")

            # 计算非空单元格数量
            wireless_non_blank_count = wireless_len - wireless_blank_count
            wired_non_blank_count = wired_len - wired_blank_count
            # 无线表非空格数量大于有线表非空格数量时，才考虑切换
            switch_flag = False
            if wireless_non_blank_count > wired_non_blank_count:
                # 假设非空表格是接近正方表，使用非空单元格数量开平方作为表格规模的估计
                wired_table_scale = round(wired_non_blank_count ** 0.5)
                # logger.debug(f"wireless non-blank cell count: {wireless_non_blank_count}, wired non-blank cell count: {wired_non_blank_count}, wired table scale: {wired_table_scale}")
                # 如果无线表非空格的数量比有线表多一列或以上，需要切换到无线表
                wired_scale_plus_2_cols = wired_non_blank_count + (wired_table_scale * 2)
                wired_scale_squared_plus_2_rows = wired_table_scale * (wired_table_scale + 2)
                if (wireless_non_blank_count + 3) >= max(wired_scale_plus_2_cols, wired_scale_squared_plus_2_rows):
                    switch_flag = True

            # 判断是否使用无线表格模型的结果
            if (
                switch_flag
                or (0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75))  # 两者相差不大但有线模型结果较少
                or (gap_of_len == 0 and wired_len <= 4)  # 单元格数量完全相等且总量小于等于4
                or (wired_text_count <= wireless_text_count * 0.6 and  wireless_text_count >=10) # 有线模型填入的文字明显少于无线模型
            ):
                # logger.debug("fall back to wireless table model")
                html_code = wireless_html_code
            else:
                html_code = wired_html_code

            return html_code
        except Exception as e:
            logger.warning(e)
            return wireless_html_code
