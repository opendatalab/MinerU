import copy
import math
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
from skimage import measure

from mineru.utils.os_env_config import get_op_num_threads
from .utils import OrtInferSession, resize_img
from .utils_table_line_rec import (
    get_table_line,
    final_adjust_lines,
    min_area_rect_box,
    draw_lines,
    adjust_lines,
)
from.utils_table_recover import (
    sorted_ocr_boxes,
    box_4_2_poly_to_box_4_1,
)


class TSRUnet:
    def __init__(self, config: Dict):
        self.K = 1000
        self.MK = 4000
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.inp_height = 1024
        self.inp_width = 1024

        config["intra_op_num_threads"] = get_op_num_threads("MINERU_INTRA_OP_NUM_THREADS")
        config["inter_op_num_threads"] = get_op_num_threads("MINERU_INTER_OP_NUM_THREADS")

        self.session = OrtInferSession(config)

    def __call__(
        self, img: np.ndarray, **kwargs
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        img_info = self.preprocess(img)
        pred = self.infer(img_info)
        polygons, rotated_polygons = self.postprocess(img, pred, **kwargs)
        if polygons.size == 0:
            return None, None
        polygons = polygons.reshape(polygons.shape[0], 4, 2)
        polygons[:, 3, :], polygons[:, 1, :] = (
            polygons[:, 1, :].copy(),
            polygons[:, 3, :].copy(),
        )
        rotated_polygons = rotated_polygons.reshape(rotated_polygons.shape[0], 4, 2)
        rotated_polygons[:, 3, :], rotated_polygons[:, 1, :] = (
            rotated_polygons[:, 1, :].copy(),
            rotated_polygons[:, 3, :].copy(),
        )
        _, idx = sorted_ocr_boxes(
            [box_4_2_poly_to_box_4_1(poly_box) for poly_box in rotated_polygons],
            threhold=0.4,
        )
        polygons = polygons[idx]
        rotated_polygons = rotated_polygons[idx]
        return polygons, rotated_polygons

    def preprocess(self, img) -> Dict[str, Any]:
        scale = (self.inp_height, self.inp_width)
        img, _, _ = resize_img(img, scale, True)
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(self.mean.reshape(1, -1))
        stdinv = 1 / np.float64(self.std.reshape(1, -1))
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        img = img.transpose(2, 0, 1)
        images = img[None, :]
        return {"img": images}

    def infer(self, input):
        result = self.session(input["img"][None, ...])[0][0]
        result = result[0].astype(np.uint8)
        return result

    def postprocess(self, img, pred, **kwargs):
        row = kwargs.get("row", 50) if kwargs else 50
        col = kwargs.get("col", 30) if kwargs else 30
        h_lines_threshold = kwargs.get("h_lines_threshold", 100) if kwargs else 100
        v_lines_threshold = kwargs.get("v_lines_threshold", 15) if kwargs else 15
        angle = kwargs.get("angle", 50) if kwargs else 50
        enhance_box_line = kwargs.get("enhance_box_line", True) if kwargs else True
        morph_close = (
            kwargs.get("morph_close", enhance_box_line) if kwargs else enhance_box_line
        )  # 是否进行闭合运算以找到更多小的框
        more_h_lines = (
            kwargs.get("more_h_lines", enhance_box_line) if kwargs else enhance_box_line
        )  # 是否调整以找到更多的横线
        more_v_lines = (
            kwargs.get("more_v_lines", enhance_box_line) if kwargs else enhance_box_line
        )  # 是否调整以找到更多的横线
        extend_line = (
            kwargs.get("extend_line", enhance_box_line) if kwargs else enhance_box_line
        )  # 是否进行线段延长使得端点连接
        # 是否进行旋转修正
        rotated_fix = kwargs.get("rotated_fix") if kwargs else True
        ori_shape = img.shape
        pred = np.uint8(pred)
        hpred = copy.deepcopy(pred)  # 横线
        vpred = copy.deepcopy(pred)  # 竖线
        whereh = np.where(hpred == 1)
        wherev = np.where(vpred == 2)
        hpred[wherev] = 0
        vpred[whereh] = 0

        hpred = cv2.resize(hpred, (ori_shape[1], ori_shape[0]))
        vpred = cv2.resize(vpred, (ori_shape[1], ori_shape[0]))

        h, w = pred.shape
        hors_k = int(math.sqrt(w) * 1.2)
        vert_k = int(math.sqrt(h) * 1.2)
        hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
        vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
        vpred = cv2.morphologyEx(
            vpred, cv2.MORPH_CLOSE, vkernel, iterations=1
        )  # 先膨胀后腐蚀的过程
        if morph_close:
            hpred = cv2.morphologyEx(hpred, cv2.MORPH_CLOSE, hkernel, iterations=1)
        colboxes = get_table_line(vpred, axis=1, lineW=col)  # 竖线
        rowboxes = get_table_line(hpred, axis=0, lineW=row)  # 横线
        rboxes_row_, rboxes_col_ = [], []
        if more_h_lines:
            rboxes_row_ = adjust_lines(rowboxes, alph=h_lines_threshold, angle=angle)
        if more_v_lines:
            rboxes_col_ = adjust_lines(colboxes, alph=v_lines_threshold, angle=angle)
        rowboxes += rboxes_row_
        colboxes += rboxes_col_
        if extend_line:
            rowboxes, colboxes = final_adjust_lines(rowboxes, colboxes)
        line_img = np.zeros(img.shape[:2], dtype="uint8")
        line_img = draw_lines(line_img, rowboxes + colboxes, color=255, lineW=2)
        rotated_angle = self.cal_rotate_angle(line_img)
        if rotated_fix and abs(rotated_angle) > 0.3:
            rotated_line_img = self.rotate_image(line_img, rotated_angle)
            rotated_polygons = self.cal_region_boxes(rotated_line_img)
            polygons = self.unrotate_polygons(
                rotated_polygons, rotated_angle, line_img.shape
            )
        else:
            polygons = self.cal_region_boxes(line_img)
            rotated_polygons = polygons.copy()
        return polygons, rotated_polygons

    def cal_region_boxes(self, tmp):
        labels = measure.label(tmp < 255, connectivity=2)  # 8连通区域标记
        regions = measure.regionprops(labels)
        ceilboxes = min_area_rect_box(
            regions,
            False,
            tmp.shape[1],
            tmp.shape[0],
            filtersmall=True,
            adjust_box=False,
        )  # 最后一个参数改为False
        return np.array(ceilboxes)

    def cal_rotate_angle(self, tmp):
        # 计算最外侧的旋转框
        contours, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        # 计算旋转角度
        angle = rect[2]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        return angle

    def rotate_image(self, image, angle):
        # 获取图像的中心点
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 进行旋转
        rotated_image = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
        )

        return rotated_image

    def unrotate_polygons(
        self, polygons: np.ndarray, angle: float, img_shape: tuple
    ) -> np.ndarray:
        # 将多边形旋转回原始位置
        (h, w) = img_shape
        center = (w // 2, h // 2)
        M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # 将 (N, 8) 转换为 (N, 4, 2)
        polygons_reshaped = polygons.reshape(-1, 4, 2)

        # 批量逆旋转
        unrotated_polygons = cv2.transform(polygons_reshaped, M_inv)

        # 将 (N, 4, 2) 转换回 (N, 8)
        unrotated_polygons = unrotated_polygons.reshape(-1, 8)

        return unrotated_polygons
