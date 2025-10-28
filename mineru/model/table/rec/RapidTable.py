import html
import os
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from loguru import logger
from rapid_table import ModelType, RapidTable, RapidTableInput
from rapid_table.utils import RapidTableOutput
from tqdm import tqdm

from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR
from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class CustomRapidTable(RapidTable):
    def __init__(self, cfg: RapidTableInput):
        import logging
        # 通过环境变量控制日志级别
        logging.disable(logging.INFO)
        super().__init__(cfg)
    def __call__(self, img_contents, ocr_results=None, batch_size=1):
        if not isinstance(img_contents, list):
            img_contents = [img_contents]

        s = time.perf_counter()

        results = RapidTableOutput()

        total_nums = len(img_contents)

        with tqdm(total=total_nums, desc="Table-wireless Predict") as pbar:
            for start_i in range(0, total_nums, batch_size):
                end_i = min(total_nums, start_i + batch_size)

                imgs = self._load_imgs(img_contents[start_i:end_i])

                pred_structures, cell_bboxes = self.table_structure(imgs)
                logic_points = self.table_matcher.decode_logic_points(pred_structures)

                dt_boxes, rec_res = self.get_ocr_results(imgs, start_i, end_i, ocr_results)
                pred_htmls = self.table_matcher(
                    pred_structures, cell_bboxes, dt_boxes, rec_res
                )

                results.pred_htmls.extend(pred_htmls)
                # 更新进度条
                pbar.update(end_i - start_i)

        elapse = time.perf_counter() - s
        results.elapse = elapse / total_nums
        return results


class RapidTableModel():
    def __init__(self, ocr_engine):
        slanet_plus_model_path = os.path.join(
            auto_download_and_get_model_root_path(ModelPath.slanet_plus),
            ModelPath.slanet_plus,
        )
        input_args = RapidTableInput(
            model_type=ModelType.SLANETPLUS,
            model_dir_or_path=slanet_plus_model_path,
            use_ocr=False
        )
        self.table_model = CustomRapidTable(input_args)
        self.ocr_engine = ocr_engine

    def predict(self, image, ocr_result=None):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # Continue with OCR on potentially rotated image

        if not ocr_result:
            raw_ocr_result = self.ocr_engine.ocr(bgr_image)[0]
            # 分离边界框、文本和置信度
            boxes = []
            texts = []
            scores = []
            for item in raw_ocr_result:
                if len(item) == 3:
                    boxes.append(item[0])
                    texts.append(escape_html(item[1]))
                    scores.append(item[2])
                elif len(item) == 2 and isinstance(item[1], tuple):
                    boxes.append(item[0])
                    texts.append(escape_html(item[1][0]))
                    scores.append(item[1][1])
            # 按照 rapid_table 期望的格式构建 ocr_results
            ocr_result = [(boxes, texts, scores)]

        if ocr_result:
            try:
                table_results = self.table_model(img_contents=np.asarray(image), ocr_results=ocr_result)
                html_code = table_results.pred_htmls
                table_cell_bboxes = table_results.cell_bboxes
                logic_points = table_results.logic_points
                elapse = table_results.elapse
                return html_code, table_cell_bboxes, logic_points, elapse
            except Exception as e:
                logger.exception(e)

        return None, None, None, None

    def batch_predict(self, table_res_list: List[dict], batch_size: int = 4):
        not_none_table_res_list = []
        for table_res in table_res_list:
            if table_res.get("ocr_result", None):
                not_none_table_res_list.append(table_res)

        if not_none_table_res_list:
            img_contents = [table_res["table_img"] for table_res in not_none_table_res_list]
            ocr_results = []
            # ocr_results需要按照rapid_table期望的格式构建
            for table_res in not_none_table_res_list:
                raw_ocr_result = table_res["ocr_result"]
                boxes = []
                texts = []
                scores = []
                for item in raw_ocr_result:
                    if len(item) == 3:
                        boxes.append(item[0])
                        texts.append(escape_html(item[1]))
                        scores.append(item[2])
                    elif len(item) == 2 and isinstance(item[1], tuple):
                        boxes.append(item[0])
                        texts.append(escape_html(item[1][0]))
                        scores.append(item[1][1])
                ocr_results.append((boxes, texts, scores))
            table_results = self.table_model(img_contents=img_contents, ocr_results=ocr_results, batch_size=batch_size)

            for i, result in enumerate(table_results.pred_htmls):
                if result:
                    not_none_table_res_list[i]['table_res']['html'] = result

if __name__ == '__main__':
    ocr_engine= PytorchPaddleOCR(
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            enable_merge_det_boxes=False,
    )
    table_model = RapidTableModel(ocr_engine)
    img_path = Path(r"D:\project\20240729ocrtest\pythonProject\images\601c939cc6dabaf07af763e2f935f54896d0251f37cc47beb7fc6b069353455d.jpg")
    image = cv2.imread(str(img_path))
    html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(image)
    print(html_code)

