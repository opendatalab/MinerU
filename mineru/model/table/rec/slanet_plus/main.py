import os
import argparse
import copy
import importlib
import time
import html
from dataclasses import asdict, dataclass
from enum import Enum
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

root_dir = Path(__file__).resolve().parent


class ModelType(Enum):
    PPSTRUCTURE_EN = "ppstructure_en"
    PPSTRUCTURE_ZH = "ppstructure_zh"
    SLANETPLUS = "slanet_plus"
    UNITABLE = "unitable"


ROOT_URL = "https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/master/"
KEY_TO_MODEL_URL = {
    ModelType.PPSTRUCTURE_EN.value: f"{ROOT_URL}/en_ppstructure_mobile_v2_SLANet.onnx",
    ModelType.PPSTRUCTURE_ZH.value: f"{ROOT_URL}/ch_ppstructure_mobile_v2_SLANet.onnx",
    ModelType.SLANETPLUS.value: f"{ROOT_URL}/slanet-plus.onnx",
    ModelType.UNITABLE.value: {
        "encoder": f"{ROOT_URL}/unitable/encoder.pth",
        "decoder": f"{ROOT_URL}/unitable/decoder.pth",
        "vocab": f"{ROOT_URL}/unitable/vocab.json",
    },
}


@dataclass
class RapidTableInput:
    model_type: Optional[str] = ModelType.SLANETPLUS.value
    model_path: Union[str, Path, None, Dict[str, str]] = None
    use_cuda: bool = False
    device: str = "cpu"


@dataclass
class RapidTableOutput:
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None


class RapidTable:
    def __init__(self, config: RapidTableInput):
        self.model_type = config.model_type
        if self.model_type not in KEY_TO_MODEL_URL:
            model_list = ",".join(KEY_TO_MODEL_URL)
            raise ValueError(
                f"{self.model_type} is not supported. The currently supported models are {model_list}."
            )

        config.model_path = config.model_path
        if self.model_type == ModelType.SLANETPLUS.value:
            self.table_structure = TableStructurer(asdict(config))
        else:
            raise ValueError(f"{self.model_type} is not supported.")
        self.table_matcher = TableMatch()

    def predict(
        self,
        img: np.ndarray,
        ocr_result: List[Union[List[List[float]], str, str]] = None,
    ) -> RapidTableOutput:
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
        return RapidTableOutput(pred_html, cell_bboxes, logic_points, elapse)

    def batch_predict(
        self,
        images: List[np.ndarray],
        ocr_results: List[List[Union[List[List[float]], str, str]]],
        batch_size: int = 4,
    ) -> List[RapidTableOutput]:
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
            result = RapidTableOutput(pred_html, cell_bboxes, logic_points, 0)
            output_results.append(result)

        total_elapse = time.perf_counter() - s
        for result in output_results:
            result.elapse = total_elapse / len(output_results)

        return output_results

    def get_boxes_recs(
        self, ocr_result: List[Union[List[List[float]], str, str]], h: int, w: int
    ) -> Tuple[np.ndarray, Tuple[str, str]]:
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


def parse_args(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        default=False,
        help="Wheter to visualize the layout results.",
    )
    parser.add_argument(
        "-img", "--img_path", type=str, required=True, help="Path to image for layout."
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default=ModelType.SLANETPLUS.value,
        choices=list(KEY_TO_MODEL_URL),
    )
    args = parser.parse_args(arg_list)
    return args


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class RapidTableModel(object):
    def __init__(self, ocr_engine):
        slanet_plus_model_path = os.path.join(
            auto_download_and_get_model_root_path(ModelPath.slanet_plus),
            ModelPath.slanet_plus,
        )
        input_args = RapidTableInput(
            model_type="slanet_plus", model_path=slanet_plus_model_path
        )
        self.table_model = RapidTable(input_args)
        self.ocr_engine = ocr_engine

    def predict(self, image, table_cls_score):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # Continue with OCR on potentially rotated image
        ocr_result = self.ocr_engine.ocr(bgr_image)[0]
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
                table_results = self.table_model.predict(np.asarray(image), ocr_result)
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
        for index in tqdm(
            range(0, len(table_res_list), batch_size),
            desc=f"Table Batch Predict, total={len(table_res_list)}, batch_size={batch_size}",
        ):
            batch_imgs = [
                cv2.cvtColor(np.asarray(table_res_list[i]["table_img"]), cv2.COLOR_RGB2BGR)
                for i in range(index, min(index + batch_size, len(table_res_list)))
            ]
            batch_ocrs = [
                table_res_list[i]["ocr_result"]
                for i in range(index, min(index + batch_size, len(table_res_list)))
            ]
            results = self.table_model.batch_predict(
                batch_imgs, batch_ocrs, batch_size=batch_size
            )
            for i, result in enumerate(results):
                if result.pred_html:
                    # 检查html_code是否包含'<table>'和'</table>'
                    if '<table>' in result.pred_html and '</table>' in result.pred_html:
                        # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                        start_index = result.pred_html.find('<table>')
                        end_index = result.pred_html.rfind('</table>') + len('</table>')
                        table_res_list[index + i]['table_res']['html'] = result.pred_html[start_index:end_index]
                    else:
                        logger.warning(
                            'table recognition processing fails, not found expected HTML table end'
                        )
                else:
                    logger.warning(
                        "table recognition processing fails, not get html return"
                    )
