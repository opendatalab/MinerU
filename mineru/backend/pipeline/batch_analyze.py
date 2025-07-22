import cv2
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from .model_init import AtomModelSingleton
from ...utils.config_reader import get_formula_enable, get_table_enable
from ...utils.model_utils import crop_img, get_res_list_from_layout_res
from ...utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence

YOLO_LAYOUT_BASE_BATCH_SIZE = 8
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(self, model_manager, batch_ratio: int, formula_enable, table_enable, enable_ocr_det_batch: bool = True):
        self.batch_ratio = batch_ratio
        self.formula_enable = get_formula_enable(formula_enable)
        self.table_enable = get_table_enable(table_enable)
        self.model_manager = model_manager
        self.enable_ocr_det_batch = enable_ocr_det_batch

    def __call__(self, images_with_extra_info: list) -> list:
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = []

        self.model = self.model_manager.get_model(
            lang=None,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
        )
        atom_model_manager = AtomModelSingleton()

        images = [image for image, _, _ in images_with_extra_info]

        # doclayout_yolo
        layout_images = []
        for image_index, image in enumerate(images):
            layout_images.append(image)


        images_layout_res += self.model.layout_model.batch_predict(
            layout_images, YOLO_LAYOUT_BASE_BATCH_SIZE
        )

        if self.formula_enable:
            # 公式检测
            images_mfd_res = self.model.mfd_model.batch_predict(
                images, MFD_BASE_BATCH_SIZE
            )

            # 公式识别
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res,
                images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            mfr_count = 0
            for image_index in range(len(images)):
                images_layout_res[image_index] += images_formula_list[image_index]
                mfr_count += len(images_formula_list[image_index])

        # 清理显存
        # clean_vram(self.model.device, vram_threshold=8)

        ocr_res_list_all_page = []
        table_res_list_all_page = []
        for index in range(len(images)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            pil_img = images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'pil_img':pil_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'layout_res':layout_res,
                                          })

            for table_res in table_res_list:
                table_img, _ = crop_img(table_res, pil_img)
                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':table_img,
                                              })

        # OCR检测处理
        if self.enable_ocr_det_batch:
            # 批处理模式 - 按语言和分辨率分组
            # 收集所有需要OCR检测的裁剪图像
            all_cropped_images_info = []

            for ocr_res_list_dict in ocr_res_list_all_page:
                _lang = ocr_res_list_dict['lang']

                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['pil_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )

                    # BGR转换
                    new_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)

                    all_cropped_images_info.append((
                        new_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang
                    ))

            # 按语言分组
            lang_groups = defaultdict(list)
            for crop_info in all_cropped_images_info:
                lang = crop_info[5]
                lang_groups[lang].append(crop_info)

            # 对每种语言按分辨率分组并批处理
            for lang, lang_crop_list in lang_groups.items():
                if not lang_crop_list:
                    continue

                # logger.info(f"Processing OCR detection for language {lang} with {len(lang_crop_list)} images")

                # 获取OCR模型
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name='ocr',
                    det_db_box_thresh=0.3,
                    lang=lang
                )

                # 按分辨率分组并同时完成padding
                resolution_groups = defaultdict(list)
                for crop_info in lang_crop_list:
                    cropped_img = crop_info[0]
                    h, w = cropped_img.shape[:2]
                    # 使用更大的分组容差，减少分组数量
                    # 将尺寸标准化到32的倍数
                    normalized_h = ((h + 32) // 32) * 32  # 向上取整到32的倍数
                    normalized_w = ((w + 32) // 32) * 32
                    group_key = (normalized_h, normalized_w)
                    resolution_groups[group_key].append(crop_info)

                # 对每个分辨率组进行批处理
                for group_key, group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):

                    # 计算目标尺寸（组内最大尺寸，向上取整到32的倍数）
                    max_h = max(crop_info[0].shape[0] for crop_info in group_crops)
                    max_w = max(crop_info[0].shape[1] for crop_info in group_crops)
                    target_h = ((max_h + 32 - 1) // 32) * 32
                    target_w = ((max_w + 32 - 1) // 32) * 32

                    # 对所有图像进行padding到统一尺寸
                    batch_images = []
                    for crop_info in group_crops:
                        img = crop_info[0]
                        h, w = img.shape[:2]
                        # 创建目标尺寸的白色背景
                        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                        # 将原图像粘贴到左上角
                        padded_img[:h, :w] = img
                        batch_images.append(padded_img)

                    # 批处理检测
                    det_batch_size = min(len(batch_images), self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)  # 增加批处理大小
                    # logger.debug(f"OCR-det batch: {det_batch_size} images, target size: {target_h}x{target_w}")
                    batch_results = ocr_model.text_detector.batch_predict(batch_images, det_batch_size)

                    # 处理批处理结果
                    for i, (crop_info, (dt_boxes, elapse)) in enumerate(zip(group_crops, batch_results)):
                        new_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang = crop_info

                        if dt_boxes is not None and len(dt_boxes) > 0:
                            # 直接应用原始OCR流程中的关键处理步骤
                            from mineru.utils.ocr_utils import (
                                merge_det_boxes, update_det_boxes, sorted_boxes
                            )

                            # 1. 排序检测框
                            if len(dt_boxes) > 0:
                                dt_boxes_sorted = sorted_boxes(dt_boxes)
                            else:
                                dt_boxes_sorted = []

                            # 2. 合并相邻检测框
                            if dt_boxes_sorted:
                                dt_boxes_merged = merge_det_boxes(dt_boxes_sorted)
                            else:
                                dt_boxes_merged = []

                            # 3. 根据公式位置更新检测框（关键步骤！）
                            if dt_boxes_merged and adjusted_mfdetrec_res:
                                dt_boxes_final = update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                            else:
                                dt_boxes_final = dt_boxes_merged

                            # 构造OCR结果格式
                            ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]

                            if ocr_res:
                                ocr_result_list = get_ocr_result_list(
                                    ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], new_image, _lang
                                )

                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)
        else:
            # 原始单张处理模式
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
                # Process each area that requires OCR processing
                _lang = ocr_res_list_dict['lang']
                # Get OCR results for this language's images
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name='ocr',
                    ocr_show_log=False,
                    det_db_box_thresh=0.3,
                    lang=_lang
                )
                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['pil_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )
                    # OCR-det
                    new_image = cv2.cvtColor(np.asarray(new_image), cv2.COLOR_RGB2BGR)
                    ocr_res = ocr_model.ocr(
                        new_image, mfd_res=adjusted_mfdetrec_res, rec=False
                    )[0]

                    # Integration results
                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_res_list_dict['ocr_enable'],new_image, _lang
                        )

                        ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        # 表格识别 table recognition
        if self.table_enable:
            for table_res_dict in tqdm(table_res_list_all_page, desc="Table Predict"):
                _lang = table_res_dict['lang']
                table_model = atom_model_manager.get_atom_model(
                    atom_model_name='table',
                    lang=_lang,
                )
                html_code, table_cell_bboxes, logic_points, elapse = table_model.predict(table_res_dict['table_img'])
                # 判断是否返回正常
                if html_code:
                    expected_ending = html_code.strip().endswith('</html>') or html_code.strip().endswith('</table>')
                    if expected_ending:
                        table_res_dict['table_res']['html'] = html_code
                    else:
                        logger.warning(
                            'table recognition processing fails, not found expected HTML table end'
                        )
                else:
                    logger.warning(
                        'table recognition processing fails, not get html return'
                    )

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
            total_processed = 0

            # Process each language separately
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    # Get OCR results for this language's images

                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name='ocr',
                        det_db_box_thresh=0.3,
                        lang=lang
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    # Process OCR results for this language
                    for index, layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(f"{ocr_score:.3f}")
                        if ocr_score < OcrConfidence.min_confidence:
                            layout_res_item['category_id'] = 16
                        else:
                            layout_res_bbox = [layout_res_item['poly'][0], layout_res_item['poly'][1],
                                               layout_res_item['poly'][4], layout_res_item['poly'][5]]
                            layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                            layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                            if ocr_text in ['（204号', '（20', '（2', '（2号', '（20号'] and ocr_score < 0.8 and layout_res_width < layout_res_height:
                                layout_res_item['category_id'] = 16

                    total_processed += len(img_crop_list)

        return images_layout_res
