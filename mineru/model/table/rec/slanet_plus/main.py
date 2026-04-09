import os
import copy
import time
import html
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from .matcher import TableMatch
from .table_structure import TableStructurer
from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


@dataclass
class PaddleTableInput:
    model_type: Optional[str] = "slanet_plus"
    model_path: Union[str, Path, None, Dict[str, str]] = None
    use_cuda: bool = False
    device: str = "cpu"


@dataclass
class PaddleTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None


class PaddleTable:
    def __init__(self, config: PaddleTableInput):
        self.table_structure = TableStructurer(asdict(config))
        self.table_matcher = TableMatch()

    def predict(
        self,
        img: np.ndarray,
        ocr_result: List[Union[List[List[float]], str, str]] = None,
    ) -> PaddleTableOutput:
        if ocr_result is None:
            raise ValueError("OCR result is None")

        s = time.perf_counter()
        h, w = img.shape[:2]

        dt_boxes, rec_res = self.get_boxes_recs(ocr_result, h, w)

        pred_structures, cell_bboxes, _ = self.table_structure.process(
            copy.deepcopy(img)
        )

        # 适配slanet-plus模型输出的box缩放还原
        cell_bboxes = self.adapt_slanet_plus(img, cell_bboxes)

        pred_html = self.table_matcher(pred_structures, cell_bboxes, dt_boxes, rec_res)

        # 过滤掉占位的bbox
        mask = ~np.all(cell_bboxes == 0, axis=1)
        cell_bboxes = cell_bboxes[mask]

        logic_points = self.table_matcher.decode_logic_points(pred_structures)
        elapse = time.perf_counter() - s
        return PaddleTableOutput(pred_html, cell_bboxes, logic_points, elapse)

    def batch_predict(
        self,
        images: List[np.ndarray],
        ocr_results: List[List[Union[List[List[float]], str, str]]],
        batch_size: int = 4,
    ) -> List[PaddleTableOutput]:
        """批量处理图像"""
        s = time.perf_counter()

        batch_dt_boxes = []
        batch_rec_res = []

        for i, img in enumerate(images):
            h, w = img.shape[:2]
            dt_boxes, rec_res = self.get_boxes_recs(ocr_results[i], h, w)
            batch_dt_boxes.append(dt_boxes)
            batch_rec_res.append(rec_res)

        # 批量表格结构识别
        batch_results = self.table_structure.batch_process(images)

        output_results = []
        for i, (img, ocr_result, (pred_structures, cell_bboxes, _)) in enumerate(
            zip(images, ocr_results, batch_results)
        ):
            # 适配slanet-plus模型输出的box缩放还原
            cell_bboxes = self.adapt_slanet_plus(img, cell_bboxes)
            pred_html = self.table_matcher(
                pred_structures, cell_bboxes, batch_dt_boxes[i], batch_rec_res[i]
            )
            # 过滤掉占位的bbox
            mask = ~np.all(cell_bboxes == 0, axis=1)
            cell_bboxes = cell_bboxes[mask]

            logic_points = self.table_matcher.decode_logic_points(pred_structures)
            result = PaddleTableOutput(pred_html, cell_bboxes, logic_points, 0)
            output_results.append(result)

        total_elapse = time.perf_counter() - s
        for result in output_results:
            result.elapse = total_elapse / len(output_results)

        return output_results

    def get_boxes_recs(
        self, ocr_result: List[Union[List[List[float]], str, str]], h: int, w: int
    ) -> Tuple[np.ndarray, Tuple[str, str]]:
        if not ocr_result:
            return np.empty((0, 4), dtype=np.float32), []

        dt_boxes, rec_res, scores = list(zip(*ocr_result))
        rec_res = list(zip(rec_res, scores))

        r_boxes = []
        for box in dt_boxes:
            box = np.array(box)
            x_min = max(0, box[:, 0].min() - 1)
            x_max = min(w, box[:, 0].max() + 1)
            y_min = max(0, box[:, 1].min() - 1)
            y_max = min(h, box[:, 1].max() + 1)
            box = [x_min, y_min, x_max, y_max]
            r_boxes.append(box)
        dt_boxes = np.array(r_boxes)
        return dt_boxes, rec_res

    def adapt_slanet_plus(self, img: np.ndarray, cell_bboxes: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        resized = 488
        ratio = min(resized / h, resized / w)
        w_ratio = resized / (w * ratio)
        h_ratio = resized / (h * ratio)
        cell_bboxes[:, 0::2] *= w_ratio
        cell_bboxes[:, 1::2] *= h_ratio
        return cell_bboxes


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class PaddleTableModel(object):
    def __init__(self, ocr_engine):
        slanet_plus_model_path = os.path.join(
            auto_download_and_get_model_root_path(ModelPath.slanet_plus),
            ModelPath.slanet_plus,
        )
        input_args = PaddleTableInput(
            model_type="slanet_plus", model_path=slanet_plus_model_path
        )
        self.table_model = PaddleTable(input_args)
        self.ocr_engine = ocr_engine

    def predict(self, image, ocr_result=None):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # Continue with OCR on potentially rotated image

        if ocr_result is None:
            ocr_result = self.ocr_engine.ocr(bgr_image)[0]
            ocr_result = [
                [item[0], escape_html(item[1][0]), item[1][1]]
                for item in ocr_result
                if len(item) == 2 and isinstance(item[1], tuple)
            ]

        try:
            table_results = self.table_model.predict(np.asarray(image), ocr_result or [])
            html_code = table_results.pred_html
            table_cell_bboxes = table_results.cell_bboxes
            logic_points = table_results.logic_points
            elapse = table_results.elapse
            return html_code, table_cell_bboxes, logic_points, elapse
        except Exception as e:
            logger.exception(e)

        return None, None, None, None

    def batch_predict(self, table_res_list: List[Dict], batch_size: int = 4) -> None:
        """对传入的字典列表进行批量预测，无返回值"""

        with tqdm(total=len(table_res_list), desc="Table-wireless Predict") as pbar:
            for index in range(0, len(table_res_list), batch_size):
                batch_imgs = [
                    cv2.cvtColor(np.asarray(table_res_list[i]["table_img"]), cv2.COLOR_RGB2BGR)
                    for i in range(index, min(index + batch_size, len(table_res_list)))
                ]
                batch_ocrs = [
                    table_res_list[i].get("ocr_result") or []
                    for i in range(index, min(index + batch_size, len(table_res_list)))
                ]
                results = self.table_model.batch_predict(
                    batch_imgs, batch_ocrs, batch_size=batch_size
                )
                for i, result in enumerate(results):
                    table_res = table_res_list[index + i]
                    table_res["wireless_cell_bboxes"] = result.cell_bboxes
                    table_res["wireless_logic_points"] = result.logic_points
                    if result.pred_html is not None:
                        table_res['table_res']['html'] = result.pred_html

                # 更新进度条
                pbar.update(len(results))
