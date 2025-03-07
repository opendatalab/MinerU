import time

import cv2
import torch
from loguru import logger

from magic_pdf.config.constants import MODEL_NAME
from magic_pdf.model.pdf_extract_kit import CustomPEKModel
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img, get_res_list_from_layout_res)
from magic_pdf.model.sub_modules.ocr.paddleocr.ocr_utils import (
    get_adjusted_mfdetrec_res, get_ocr_result_list)

YOLO_LAYOUT_BASE_BATCH_SIZE = 1
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(self, model: CustomPEKModel, batch_ratio: int):
        self.model = model
        self.batch_ratio = batch_ratio

    def __call__(self, images: list) -> list:
        images_layout_res = []
        layout_start_time = time.time()
        if self.model.layout_model_name == MODEL_NAME.LAYOUTLMv3:
            # layoutlmv3
            for image in images:
                layout_res = self.model.layout_model(image, ignore_catids=[])
                images_layout_res.append(layout_res)
        elif self.model.layout_model_name == MODEL_NAME.DocLayout_YOLO:
            # doclayout_yolo
            layout_images = []
            for image_index, image in enumerate(images):
                layout_images.append(image)

            images_layout_res += self.model.layout_model.batch_predict(
                # layout_images, self.batch_ratio * YOLO_LAYOUT_BASE_BATCH_SIZE
                layout_images, YOLO_LAYOUT_BASE_BATCH_SIZE
            )

        logger.info(
            f'layout time: {round(time.time() - layout_start_time, 2)}, image num: {len(images)}'
        )

        if self.model.apply_formula:
            # 公式检测
            mfd_start_time = time.time()
            images_mfd_res = self.model.mfd_model.batch_predict(
                # images, self.batch_ratio * MFD_BASE_BATCH_SIZE
                images, MFD_BASE_BATCH_SIZE
            )
            logger.info(
                f'mfd time: {round(time.time() - mfd_start_time, 2)}, image num: {len(images)}'
            )

            # 公式识别
            mfr_start_time = time.time()
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res,
                images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            mfr_count = 0
            for image_index in range(len(images)):
                images_layout_res[image_index] += images_formula_list[image_index]
                mfr_count += len(images_formula_list[image_index])
            logger.info(
                f'mfr time: {round(time.time() - mfr_start_time, 2)}, image num: {mfr_count}'
            )

        # 清理显存
        clean_vram(self.model.device, vram_threshold=8)

        ocr_time = 0
        ocr_count = 0
        table_time = 0
        table_count = 0
        # reference: magic_pdf/model/doc_analyze_by_custom_model.py:doc_analyze
        for index in range(len(images)):
            layout_res = images_layout_res[index]
            np_array_img = images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )
            # ocr识别
            ocr_start = time.time()
            # Process each area that requires OCR processing
            for res in ocr_res_list:
                new_image, useful_list = crop_img(
                    res, np_array_img, crop_paste_x=50, crop_paste_y=50
                )
                adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                    single_page_mfdetrec_res, useful_list
                )

                # OCR recognition
                new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                if self.model.apply_ocr:
                    ocr_res = self.model.ocr_model.ocr(
                        new_image, mfd_res=adjusted_mfdetrec_res
                    )[0]
                else:
                    ocr_res = self.model.ocr_model.ocr(
                        new_image, mfd_res=adjusted_mfdetrec_res, rec=False
                    )[0]

                # Integration results
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(ocr_res, useful_list)
                    layout_res.extend(ocr_result_list)
            ocr_time += time.time() - ocr_start
            ocr_count += len(ocr_res_list)

            # 表格识别 table recognition
            if self.model.apply_table:
                table_start = time.time()
                for res in table_res_list:
                    new_image, _ = crop_img(res, np_array_img)
                    single_table_start_time = time.time()
                    html_code = None
                    if self.model.table_model_name == MODEL_NAME.STRUCT_EQTABLE:
                        with torch.no_grad():
                            table_result = self.model.table_model.predict(
                                new_image, 'html'
                            )
                            if len(table_result) > 0:
                                html_code = table_result[0]
                    elif self.model.table_model_name == MODEL_NAME.TABLE_MASTER:
                        html_code = self.model.table_model.img2html(new_image)
                    elif self.model.table_model_name == MODEL_NAME.RAPID_TABLE:
                        html_code, table_cell_bboxes, logic_points, elapse = (
                            self.model.table_model.predict(new_image)
                        )
                    run_time = time.time() - single_table_start_time
                    if run_time > self.model.table_max_time:
                        logger.warning(
                            f'table recognition processing exceeds max time {self.model.table_max_time}s'
                        )
                    # 判断是否返回正常
                    if html_code:
                        expected_ending = html_code.strip().endswith(
                            '</html>'
                        ) or html_code.strip().endswith('</table>')
                        if expected_ending:
                            res['html'] = html_code
                        else:
                            logger.warning(
                                'table recognition processing fails, not found expected HTML table end'
                            )
                    else:
                        logger.warning(
                            'table recognition processing fails, not get html return'
                        )
                table_time += time.time() - table_start
                table_count += len(table_res_list)

        if self.model.apply_ocr:
            logger.info(f'ocr time: {round(ocr_time, 2)}, image num: {ocr_count}')
        else:
            logger.info(f'det time: {round(ocr_time, 2)}, image num: {ocr_count}')
        if self.model.apply_table:
            logger.info(f'table time: {round(table_time, 2)}, image num: {table_count}')

        return images_layout_res
