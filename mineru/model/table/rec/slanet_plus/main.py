# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import copy
import importlib
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from rapid_table.utils import DownloadModel, LoadImage, Logger, VisTable

from .matcher import TableMatch
from .table_structure import TableStructurer
from .table_structure_unitable import TableStructureUnitable

logger = Logger(logger_name=__name__).get_log()
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

        config.model_path = self.get_model_path(config.model_type, config.model_path)
        if self.model_type == ModelType.UNITABLE.value:
            self.table_structure = TableStructureUnitable(asdict(config))
        else:
            self.table_structure = TableStructurer(asdict(config))

        self.table_matcher = TableMatch()

        try:
            self.ocr_engine = importlib.import_module("rapidocr").RapidOCR()
        except ModuleNotFoundError:
            self.ocr_engine = None

        self.load_img = LoadImage()

    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path],
        ocr_result: List[Union[List[List[float]], str, str]] = None,
    ) -> RapidTableOutput:
        if self.ocr_engine is None and ocr_result is None:
            raise ValueError(
                "One of two conditions must be met: ocr_result is not empty, or rapidocr is installed."
            )

        img = self.load_img(img_content)

        s = time.perf_counter()
        h, w = img.shape[:2]

        if ocr_result is None:
            ocr_result = self.ocr_engine(img)
            ocr_result = list(
                zip(
                    ocr_result.boxes,
                    ocr_result.txts,
                    ocr_result.scores,
                )
            )
        dt_boxes, rec_res = self.get_boxes_recs(ocr_result, h, w)

        pred_structures, cell_bboxes, _ = self.table_structure(copy.deepcopy(img))

        # 适配slanet-plus模型输出的box缩放还原
        if self.model_type == ModelType.SLANETPLUS.value:
            cell_bboxes = self.adapt_slanet_plus(img, cell_bboxes)

        pred_html = self.table_matcher(pred_structures, cell_bboxes, dt_boxes, rec_res)

        # 过滤掉占位的bbox
        mask = ~np.all(cell_bboxes == 0, axis=1)
        cell_bboxes = cell_bboxes[mask]

        logic_points = self.table_matcher.decode_logic_points(pred_structures)
        elapse = time.perf_counter() - s
        return RapidTableOutput(pred_html, cell_bboxes, logic_points, elapse)

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

    @staticmethod
    def get_model_path(
        model_type: str, model_path: Union[str, Path, None]
    ) -> Union[str, Dict[str, str]]:
        if model_path is not None:
            return model_path

        model_url = KEY_TO_MODEL_URL.get(model_type, None)
        if isinstance(model_url, str):
            model_path = DownloadModel.download(model_url)
            return model_path

        if isinstance(model_url, dict):
            model_paths = {}
            for k, url in model_url.items():
                model_paths[k] = DownloadModel.download(
                    url, save_model_name=f"{model_type}_{Path(url).name}"
                )
            return model_paths

        raise ValueError(f"Model URL: {type(model_url)} is not between str and dict.")


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


def main(arg_list: Optional[List[str]] = None):
    args = parse_args(arg_list)

    try:
        ocr_engine = importlib.import_module("rapidocr").RapidOCR()
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Please install the rapidocr by pip install rapidocr"
        ) from exc

    input_args = RapidTableInput(model_type=args.model_type)
    table_engine = RapidTable(input_args)

    img = cv2.imread(args.img_path)

    rapid_ocr_output = ocr_engine(img)
    ocr_result = list(
        zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores)
    )
    table_results = table_engine(img, ocr_result)
    print(table_results.pred_html)

    viser = VisTable()
    if args.vis:
        img_path = Path(args.img_path)

        save_dir = img_path.resolve().parent
        save_html_path = save_dir / f"{Path(img_path).stem}.html"
        save_drawed_path = save_dir / f"vis_{Path(img_path).name}"
        viser(img_path, table_results, save_html_path, save_drawed_path)


if __name__ == "__main__":
    main()
