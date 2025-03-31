import time

import cv2
import torch
from loguru import logger

from magic_pdf.config.constants import MODEL_NAME
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from magic_pdf.model.sub_modules.model_utils import (
    clean_vram, crop_img, get_res_list_from_layout_res)
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.ocr_utils import (
    get_adjusted_mfdetrec_res, get_ocr_result_list)

YOLO_LAYOUT_BASE_BATCH_SIZE = 1
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(self, model_manager, batch_ratio: int, show_log, layout_model, formula_enable, table_enable):
        self.model_manager = model_manager
        self.batch_ratio = batch_ratio
        self.show_log = show_log
        self.layout_model = layout_model
        self.formula_enable = formula_enable
        self.table_enable = table_enable

    def __call__(self, images_with_extra_info: list) -> list:
        if len(images_with_extra_info) == 0:
            return []
    
        images_layout_res = []
        layout_start_time = time.time()
        _, fst_ocr, fst_lang = images_with_extra_info[0]
        self.model = self.model_manager.get_model(fst_ocr, self.show_log, fst_lang, self.layout_model, self.formula_enable, self.table_enable)

        images = [image for image, _, _ in images_with_extra_info]

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

        det_time = 0
        det_count = 0
        table_time = 0
        table_count = 0
        # reference: magic_pdf/model/doc_analyze_by_custom_model.py:doc_analyze
        for index in range(len(images)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            self.model = self.model_manager.get_model(ocr_enable, self.show_log, _lang, self.layout_model, self.formula_enable, self.table_enable)
            layout_res = images_layout_res[index]
            np_array_img = images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )
            # ocr识别
            det_start = time.time()
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

                # if ocr_enable:
                #     ocr_res = self.model.ocr_model.ocr(
                #         new_image, mfd_res=adjusted_mfdetrec_res
                #     )[0]
                # else:
                ocr_res = self.model.ocr_model.ocr(
                    new_image, mfd_res=adjusted_mfdetrec_res, rec=False
                )[0]

                # Integration results
                if ocr_res:
                    ocr_result_list = get_ocr_result_list(ocr_res, useful_list, ocr_enable, new_image, _lang)
                    layout_res.extend(ocr_result_list)
            det_time += time.time() - det_start
            det_count += len(ocr_res_list)

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


        logger.info(f'ocr-det time: {round(det_time, 2)}, image num: {det_count}')
        if self.model.apply_table:
            logger.info(f'table time: {round(table_time, 2)}, image num: {table_count}')

        # Create dictionaries to store items by language
        need_ocr_lists_by_lang = {}  # Dict of lists for each language
        img_crop_lists_by_lang = {}  # Dict of lists for each language

        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if layout_res_item['category_id'] in [15]:
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang']

                        # Initialize lists for this language if not exist
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []

                        # Add to the appropriate language-specific lists
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                        # Remove the fields after adding to lists
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')


        if len(img_crop_lists_by_lang) > 0:

            # Process OCR by language
            rec_time = 0
            rec_start = time.time()
            total_processed = 0

            # Process each language separately
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    # Get OCR results for this language's images
                    atom_model_manager = AtomModelSingleton()
                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name='ocr',
                        ocr_show_log=False,
                        det_db_box_thresh=0.3,
                        lang=lang
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False)[0]

                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    # Process OCR results for this language
                    for index, layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(round(ocr_score, 2))

                    total_processed += len(img_crop_list)

            rec_time += time.time() - rec_start
            logger.info(f'ocr-rec time: {round(rec_time, 2)}, total images processed: {total_processed}')



        return images_layout_res
