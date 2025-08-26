import html

import cv2
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...utils.config_reader import get_formula_enable, get_table_enable
from ...utils.model_utils import crop_img, get_res_list_from_layout_res
from ...utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_rotate_crop_image
from ...utils.pdf_image_tools import get_crop_img
from ...utils.ocr_utils import merge_det_boxes, update_det_boxes, sorted_boxes
from ...utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence
from ...utils.pdf_image_tools import get_crop_np_img

YOLO_LAYOUT_BASE_BATCH_SIZE = 1
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 16
ORI_TAB_CLS_BATCH_SIZE = 16


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

        np_images = [np.asarray(image) for image, _, _ in images_with_extra_info]

        # doclayout_yolo

        images_layout_res += self.model.layout_model.batch_predict(
            np_images, YOLO_LAYOUT_BASE_BATCH_SIZE
        )

        if self.formula_enable:
            # 公式检测
            images_mfd_res = self.model.mfd_model.batch_predict(
                np_images, MFD_BASE_BATCH_SIZE
            )

            # 公式识别
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res,
                np_images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            mfr_count = 0
            for image_index in range(len(np_images)):
                images_layout_res[image_index] += images_formula_list[image_index]
                mfr_count += len(images_formula_list[image_index])

        # 清理显存
        # clean_vram(self.model.device, vram_threshold=8)

        ocr_res_list_all_page = []
        table_res_list_all_page = []
        for index in range(len(np_images)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_img = np_images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_img':np_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'layout_res':layout_res,
                                          })

            for table_res in table_res_list:
                # table_img, _ = crop_img(table_res, pil_img)
                # bbox = (241, 208, 1475, 2019)
                scale = 10/3
                crop_xmin, crop_ymin = int(table_res['poly'][0]), int(table_res['poly'][1])
                crop_xmax, crop_ymax = int(table_res['poly'][4]), int(table_res['poly'][5])
                bbox = (int(crop_xmin/scale), int(crop_ymin/scale), int(crop_xmax/scale), int(crop_ymax/scale))
                table_img = get_crop_np_img(bbox, np_img, scale=scale)

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
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )

                    # BGR转换
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                    all_cropped_images_info.append((
                        bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang
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
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.3,
                    lang=lang
                )

                # 按分辨率分组并同时完成padding
                # RESOLUTION_GROUP_STRIDE = 32
                RESOLUTION_GROUP_STRIDE = 64  # 定义分辨率分组的步进值

                resolution_groups = defaultdict(list)
                for crop_info in lang_crop_list:
                    cropped_img = crop_info[0]
                    h, w = cropped_img.shape[:2]
                    # 使用更大的分组容差，减少分组数量
                    # 将尺寸标准化到32的倍数
                    normalized_h = ((h + RESOLUTION_GROUP_STRIDE) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE  # 向上取整到32的倍数
                    normalized_w = ((w + RESOLUTION_GROUP_STRIDE) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    group_key = (normalized_h, normalized_w)
                    resolution_groups[group_key].append(crop_info)

                # 对每个分辨率组进行批处理
                for group_key, group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):

                    # 计算目标尺寸（组内最大尺寸，向上取整到32的倍数）
                    max_h = max(crop_info[0].shape[0] for crop_info in group_crops)
                    max_w = max(crop_info[0].shape[1] for crop_info in group_crops)
                    target_h = ((max_h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    target_w = ((max_w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE

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
                        bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang = crop_info

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
                                    ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang
                                )

                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)
        else:
            # 原始单张处理模式
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
                # Process each area that requires OCR processing
                _lang = ocr_res_list_dict['lang']
                # Get OCR results for this language's images
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    ocr_show_log=False,
                    det_db_box_thresh=0.3,
                    lang=_lang
                )
                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )
                    # OCR-det
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    ocr_res = ocr_model.ocr(
                        bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False
                    )[0]

                    # Integration results
                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_res_list_dict['ocr_enable'],bgr_image, _lang
                        )

                        ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        # 表格识别 table recognition
        if self.table_enable:
            # 图片旋转批量处理

            img_orientation_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.ImgOrientationCls,
            )
            try:
                img_orientation_cls_model.batch_predict(table_res_list_all_page, atom_model_manager, AtomicModel.OCR, self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)
            except Exception as e:
                logger.warning(
                    f"Image orientation classification failed: {e}, using original image"
                )
            # 表格分类
            table_cls_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.TableCls,
            )
            try:
                table_cls_model.batch_predict(table_res_list_all_page)
            except Exception as e:
                logger.warning(
                    f"Table classification failed: {e}, using default model"
                )
            rec_img_lang_group = defaultdict(list)
            # OCR det 过程，顺序执行
            for index, table_res_dict in enumerate(
                tqdm(table_res_list_all_page, desc="Table OCR det")
            ):
                _lang = table_res_dict["lang"]
                ocr_engine = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.5,
                    det_db_unclip_ratio=1.6,
                    lang=_lang,
                    enable_merge_det_boxes=False,
                )
                bgr_image = cv2.cvtColor(
                    np.asarray(table_res_dict["table_img"]), cv2.COLOR_RGB2BGR
                )
                ocr_result = ocr_engine.ocr(bgr_image, det=True, rec=False)[0]
                # 构造需要 OCR 识别的图片字典，包括cropped_img, dt_box, table_id，并按照语言进行分组
                for dt_box in ocr_result:
                    rec_img_lang_group[_lang].append(
                        {
                            "cropped_img": get_rotate_crop_image(
                                bgr_image, np.asarray(dt_box, dtype=np.float32)
                            ),
                            "dt_box": np.asarray(dt_box, dtype=np.float32),
                            "table_id": index,
                        }
                    )
            # OCR rec，按照语言分批处理
            for _lang, rec_img_list in rec_img_lang_group.items():
                ocr_engine = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.5,
                    det_db_unclip_ratio=1.6,
                    lang=_lang,
                    enable_merge_det_boxes=False,
                )
                cropped_img_list = [item["cropped_img"] for item in rec_img_list]
                ocr_res_list = ocr_engine.ocr(
                    cropped_img_list, det=False, rec=True, tqdm_enable=True
                )[0]
                # 按照 table_id 将识别结果进行回填
                for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
                    if table_res_list_all_page[img_dict["table_id"]].get("ocr_result"):
                        table_res_list_all_page[img_dict["table_id"]]["ocr_result"].append(
                            [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                        )
                    else:
                        table_res_list_all_page[img_dict["table_id"]]["ocr_result"] = [
                            [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                        ]

            # 先对所有表格使用无线表格模型，然后对分类为有线的表格使用有线表格模型
            wireless_table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.WirelessTable,
            )

            wireless_table_model.batch_predict(table_res_list_all_page)
            for table_res_dict in tqdm(
                table_res_list_all_page, desc="Wired Table Predict"
            ):
                if table_res_dict["table_res"]["cls_label"] == AtomicModel.WiredTable:
                    wired_table_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.WiredTable,
                        lang=table_res_dict["lang"],
                    )
                    if table_res_dict["table_res"].get("html") is None:
                        logger.warning("Table Wireless Predict Error.")
                    html_code = wired_table_model.predict(
                        table_res_dict["table_img"],
                        table_res_dict["table_res"]["cls_score"],
                        table_res_dict["table_res"]["html"],
                    )
                    # 检查html_code是否包含'<table>'和'</table>'
                    if "<table>" in html_code and "</table>" in html_code:
                        # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                        start_index = html_code.find("<table>")
                        end_index = html_code.rfind("</table>") + len("</table>")
                        table_res_dict["table_res"]["html"] = html_code[
                            start_index:end_index
                        ]
                    else:
                        logger.warning(
                            "wired table recognition processing fails, not found expected HTML table end"
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
                        atom_model_name=AtomicModel.OCR,
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
