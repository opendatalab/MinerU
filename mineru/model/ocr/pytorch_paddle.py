# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

from mineru.model.ocr.seal_crop import CropByPolys, SortPolyBoxes
from mineru.utils.config_reader import get_device
from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.utils.ocr_utils import (
    check_img,
    preprocess_image,
    sorted_boxes,
    merge_det_boxes,
    update_det_boxes,
    get_rotate_crop_image_for_text_rec,
)
from mineru.model.utils.tools.infer.predict_system import TextSystem
from mineru.model.utils.tools.infer import pytorchocr_utility as utility
import argparse


latin_lang = [
        "af",
        "az",
        "bs",
        "cs",
        "cy",
        "da",
        "de",
        "es",
        "et",
        "fr",
        "ga",
        "hr",
        "hu",
        "id",
        "is",
        "it",
        "ku",
        "la",
        "lt",
        "lv",
        "mi",
        "ms",
        "mt",
        "nl",
        "no",
        "oc",
        "pi",
        "pl",
        "pt",
        "ro",
        "rs_latin",
        "sk",
        "sl",
        "sq",
        "sv",
        "sw",
        "tl",
        "tr",
        "uz",
        "vi",
        "french",
        "german",
        "fi",
        "eu",
        "gl",
        "lb",
        "rm",
        "ca",
        "qu",
]
arabic_lang = ["ar", "fa", "ug", "ur", "ps", "ku", "sd", "bal"]
cyrillic_lang = [
        "ru",
        "rs_cyrillic",
        "be",
        "bg",
        "uk",
        "mn",
        "abq",
        "ady",
        "kbd",
        "ava",
        "dar",
        "inh",
        "che",
        "lbe",
        "lez",
        "tab",
        "kk",
        "ky",
        "tg",
        "mk",
        "tt",
        "cv",
        "ba",
        "mhr",
        "mo",
        "udm",
        "kv",
        "os",
        "bua",
        "xal",
        "tyv",
        "sah",
        "kaa",
]
east_slavic_lang = ["ru", "be", "uk"]
devanagari_lang = [
        "hi",
        "mr",
        "ne",
        "bh",
        "mai",
        "ang",
        "bho",
        "mah",
        "sck",
        "new",
        "gom",
        "sa",
        "bgc",
]


def get_model_params(lang, config):
    if lang in config['lang']:
        params = config['lang'][lang]
        det = params.get('det')
        rec = params.get('rec')
        dict_file = params.get('dict')
        return det, rec, dict_file
    else:
        raise Exception (f'Language {lang} not supported')


root_dir = os.path.join(Path(__file__).resolve().parent.parent, 'utils')
DEFAULT_SEAL_DEBUG_DIR = os.path.join(
    Path(__file__).resolve().parents[3],
    'output_images',
    'seal_ocr_debug',
)


class PytorchPaddleOCR(TextSystem):
    def __init__(self, *args, **kwargs):
        parser = utility.init_args()
        args = parser.parse_args(args)

        self.lang = kwargs.get('lang', 'ch')
        self.is_seal = self.lang in ['seal', 'seal_lite']
        self.enable_merge_det_boxes = kwargs.get("enable_merge_det_boxes", True)

        device = get_device()
        if device == 'cpu':
            if self.lang in ['ch', 'ch_server', 'japan', 'chinese_cht']:
                # logger.warning("The current device in use is CPU. To ensure the speed of parsing, the language is automatically switched to ch_lite.")
                self.lang = 'ch_lite'
            elif self.lang in ['seal']:
                self.lang = 'seal_lite'

        if self.lang in latin_lang:
            self.lang = 'latin'
        elif self.lang in east_slavic_lang:
            self.lang = 'east_slavic'
        elif self.lang in arabic_lang:
            self.lang = 'arabic'
        elif self.lang in cyrillic_lang:
            self.lang = 'cyrillic'
        elif self.lang in devanagari_lang:
            self.lang = 'devanagari'
        else:
            pass

        models_config_path = os.path.join(root_dir, 'pytorchocr', 'utils', 'resources', 'models_config.yml')
        with open(models_config_path) as file:
            config = yaml.safe_load(file)
            det, rec, dict_file = get_model_params(self.lang, config)
        ocr_models_dir = ModelPath.pytorch_paddle

        det_model_path = f"{ocr_models_dir}/{det}"
        det_model_path = os.path.join(auto_download_and_get_model_root_path(det_model_path), det_model_path)
        rec_model_path = f"{ocr_models_dir}/{rec}"
        rec_model_path = os.path.join(auto_download_and_get_model_root_path(rec_model_path), rec_model_path)
        kwargs['det_model_path'] = det_model_path
        kwargs['rec_model_path'] = rec_model_path
        kwargs['rec_char_dict_path'] = os.path.join(root_dir, 'pytorchocr', 'utils', 'resources', 'dict', dict_file)
        kwargs['rec_batch_num'] = 6
        if self.is_seal:
            kwargs['det_limit_side_len'] = 736
            kwargs['det_limit_type'] = 'min'
            kwargs['det_max_side_limit'] = 4000
            kwargs['det_db_thresh'] = 0.2
            kwargs['det_db_box_thresh'] = 0.6
            kwargs['det_db_unclip_ratio'] = 0.5
            kwargs['det_box_type'] = 'poly'
            kwargs['use_dilation'] = False
            kwargs['enable_merge_det_boxes'] = False
            kwargs['drop_score'] = 0
            self.enable_merge_det_boxes = False

        kwargs['device'] = device

        default_args = vars(args)
        default_args.update(kwargs)
        args = argparse.Namespace(**default_args)

        super().__init__(args)
        if self.is_seal:
            self._seal_sort_boxes = SortPolyBoxes()
            self._seal_crop_by_polys = CropByPolys(det_box_type='poly')
            self._seal_debug_counter = 0
            self._seal_debug_dir = self._resolve_seal_debug_dir()

    def _resolve_seal_debug_dir(self):
        if not self.is_seal:
            return None

        debug_dir = os.getenv("MINERU_SEAL_OCR_DEBUG_DIR")
        if debug_dir:
            return debug_dir

        debug_enable = os.getenv("MINERU_SEAL_OCR_DEBUG", "").lower()
        if debug_enable in {"1", "true", "yes", "on"}:
            return DEFAULT_SEAL_DEBUG_DIR

        return None

    def _dump_seal_debug_artifacts(self, input_image, dt_boxes, img_crop_list, rec_res=None):
        if not self._seal_debug_dir:
            return

        sample_dir = os.path.join(
            self._seal_debug_dir,
            f"sample_{self._seal_debug_counter:04d}",
        )
        self._seal_debug_counter += 1
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "input.png"), input_image)

        det_vis = input_image.copy()
        for index, box in enumerate(dt_boxes or []):
            points = np.asarray(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(det_vis, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            anchor = tuple(np.asarray(box[0], dtype=np.int32).tolist())
            cv2.putText(
                det_vis,
                str(index),
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(sample_dir, "det_vis.png"), det_vis)

        records = []
        for index, crop_img in enumerate(img_crop_list or []):
            crop_name = f"crop_{index:02d}.png"
            cv2.imwrite(os.path.join(sample_dir, crop_name), crop_img)
            record = {
                "index": index,
                "crop_path": crop_name,
            }
            if rec_res is not None and index < len(rec_res):
                text, score = rec_res[index]
                record["text"] = text
                record["score"] = float(score)
            records.append(record)

        with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def ocr(self,
            img,
            det=True,
            rec=True,
            mfd_res=None,
            tqdm_enable=False,
            tqdm_desc="OCR-rec Predict",
            ):
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        img = check_img(img)
        imgs = [img]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if det and rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, rec_res = self.__call__(img, mfd_res=mfd_res)
                    if not dt_boxes and not rec_res:
                        ocr_res.append(None)
                        continue
                    tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
                    ocr_res.append(tmp_res)
                return ocr_res
            elif det and not rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, elapse = self.text_detector(img)
                    # logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
                    if dt_boxes is None:
                        ocr_res.append(None)
                        continue
                    if self.is_seal:
                        dt_boxes = self._seal_sort_boxes(dt_boxes)
                        img_crop_list = self._seal_crop_by_polys(img, dt_boxes)
                        self._dump_seal_debug_artifacts(img, dt_boxes, img_crop_list)
                    else:
                        dt_boxes = sorted_boxes(dt_boxes)
                        # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
                        if self.enable_merge_det_boxes:
                            dt_boxes = merge_det_boxes(dt_boxes)
                        if mfd_res:
                            dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                    tmp_res = [box.tolist() for box in dt_boxes]
                    ocr_res.append(tmp_res)
                return ocr_res
            elif not det and rec:
                ocr_res = []
                for img in imgs:
                    if not isinstance(img, list):
                        img = preprocess_image(img)
                        img = [img]
                    rec_res, elapse = self.text_recognizer(img, tqdm_enable=tqdm_enable, tqdm_desc=tqdm_desc)
                    # logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
                    ocr_res.append(rec_res)
                return ocr_res

    def __call__(self, img, mfd_res=None):

        if img is None:
            logger.debug("no valid image provided")
            return None, None

        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            return None, None
        else:
            pass
            # logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
        if self.is_seal:
            dt_boxes = self._seal_sort_boxes(dt_boxes)
            img_crop_list = self._seal_crop_by_polys(ori_im, dt_boxes)
        else:
            img_crop_list = []

            dt_boxes = sorted_boxes(dt_boxes)

            # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
            if self.enable_merge_det_boxes:
                dt_boxes = merge_det_boxes(dt_boxes)

            if mfd_res:
                dt_boxes = update_det_boxes(dt_boxes, mfd_res)

            # Standard text OCR rotates tall crops before recognition.
            # Seal OCR keeps its dedicated poly-crop path above.
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = get_rotate_crop_image_for_text_rec(ori_im, tmp_box)
                img_crop_list.append(img_crop)

        rec_res, elapse = self.text_recognizer(img_crop_list)
        # logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
        if self.is_seal:
            self._dump_seal_debug_artifacts(ori_im, dt_boxes, img_crop_list, rec_res)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res

if __name__ == '__main__':
    pytorch_paddle_ocr = PytorchPaddleOCR()
    img = cv2.imread("/Users/myhloli/Downloads/screenshot-20250326-194348.png")
    dt_boxes, rec_res = pytorch_paddle_ocr(img)
    ocr_res = []
    if not dt_boxes and not rec_res:
        ocr_res.append(None)
    else:
        tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        ocr_res.append(tmp_res)
    print(ocr_res)
