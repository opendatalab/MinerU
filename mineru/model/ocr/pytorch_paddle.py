# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
import warnings
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from loguru import logger

from ..registry import MINERU_4_MODELS_TORCH
from ..runtime.device import get_device
from .._internal.pytorchocr.infer import pytorchocr_utility as utility
from .._internal.pytorchocr.infer.predict_system import TextSystem
from .geometry import merge_det_boxes, sorted_boxes, update_det_boxes
from .image import check_img, get_rotate_crop_image_for_text_rec, preprocess_image
from .language import normalize_ocr_model_lang
from .seal_crop import CropByPolys, SortPolyBoxes, filter_valid_seal_crops
from .resources import PPOCRV6_DICT_PATH
import argparse
from typing import Any


def get_model_params(lang, config):
    if lang in config['lang']:
        params = config['lang'][lang]
        det = params.get('det')
        rec = params.get('rec')
        dict_file = params.get('dict')
        return det, rec, dict_file
    else:
        raise Exception (f'Language {lang} not supported')


PYTORCHOCR_RESOURCE_DIR = Path(__file__).resolve().parents[1] / "_internal" / "pytorchocr" / "utils" / "resources"
DEFAULT_SEAL_DEBUG_DIR = os.path.join(
    Path(__file__).resolve().parents[3],
    'output_images',
    'seal_ocr_debug',
)


class PytorchPaddleOCR(TextSystem):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """初始化所选设备的 Torch OCR，并使用跨后端共享字符表。"""
        parser = utility.init_args()
        args = parser.parse_args(args)

        requested_lang = kwargs.get('lang', 'ch')
        self.lang = requested_lang
        self.is_seal = requested_lang in ['seal', 'seal_lite']
        self.enable_merge_det_boxes = kwargs.get("enable_merge_det_boxes", True)

        device = kwargs.get("device") or get_device()
        models_config_path = PYTORCHOCR_RESOURCE_DIR / "models_config.yml"
        with open(models_config_path, encoding='utf-8') as file:
            config = yaml.safe_load(file)
            self.lang = normalize_ocr_model_lang(
                requested_lang,
                device=device,
                supported_langs=config['lang'],
            )
            det, rec, _dict_file = get_model_params(self.lang, config)
        det_model_path = str(MINERU_4_MODELS_TORCH.pytorch_paddle.path(det).ensure())
        rec_model_path = str(MINERU_4_MODELS_TORCH.pytorch_paddle.path(rec).ensure())
        kwargs['det_model_path'] = det_model_path
        kwargs['rec_model_path'] = rec_model_path
        kwargs['rec_char_dict_path'] = str(PPOCRV6_DICT_PATH)
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
        kwargs['lang'] = self.lang

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

    @staticmethod
    def _save_debug_image(path: Path, image: np.ndarray) -> None:
        """独立物化 BGR 诊断图后用 Pillow 保存，调试过程不加载视觉运行时。"""
        pixels = image[:, :, ::-1] if image.ndim == 3 else image
        with Image.fromarray(pixels) as canvas:
            canvas.save(path)

    def _dump_seal_debug_artifacts(self, input_image, dt_boxes, img_crop_list, rec_res=None):

        if not self._seal_debug_dir:
            return

        sample_dir = os.path.join(
            self._seal_debug_dir,
            f"sample_{self._seal_debug_counter:04d}",
        )
        self._seal_debug_counter += 1
        os.makedirs(sample_dir, exist_ok=True)

        self._save_debug_image(Path(sample_dir) / "input.png", input_image)
        with Image.fromarray(input_image[:, :, ::-1]) as canvas:
            draw = ImageDraw.Draw(canvas)
            for index, box in enumerate(dt_boxes or []):
                points = [tuple(map(int, xy)) for xy in np.asarray(box).reshape(-1, 2)]
                draw.line(points + points[:1], fill=(255, 0, 0), width=2)
                draw.text(points[0], str(index), fill=(0, 0, 255))
            canvas.save(Path(sample_dir) / "det_vis.png")

        records = []
        for index, crop_img in enumerate(img_crop_list or []):
            crop_name = f"crop_{index:02d}.png"
            self._save_debug_image(Path(sample_dir) / crop_name, crop_img)
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

    def ocr(
        self,
        img: np.ndarray | list[np.ndarray] | str | bytes,
        det: bool = True,
        rec: bool = True,
        mfd_res: list[dict[str, Any]] | None = None,
        tqdm_enable: bool = False,
        tqdm_desc: str = "OCR-rec Predict",
        tqdm_progress_bar: Any = None,
    ) -> list[list[Any] | None] | None:
        """按模式执行 OCR，印章裁剪同步过滤无效轮廓并保持调试数据对应。"""
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det:
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
                        dt_boxes, img_crop_list = filter_valid_seal_crops(dt_boxes, img_crop_list)
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
                    rec_res, elapse = self.text_recognizer(
                        img,
                        tqdm_enable=tqdm_enable,
                        tqdm_desc=tqdm_desc,
                        tqdm_progress_bar=tqdm_progress_bar,
                    )
                    # logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
                    ocr_res.append(rec_res)
                return ocr_res

    def __call__(
        self, img: np.ndarray | None, mfd_res: list[dict[str, Any]] | None = None
    ) -> tuple[list[np.ndarray] | None, list[tuple[str, float]] | None]:
        """执行检测与识别，跳过无效裁片时同步移除检测框，避免文字错配。"""
        if img is None:
            logger.debug("no valid image provided")
            return None, None

        ori_im = img
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
            dt_boxes, img_crop_list = filter_valid_seal_crops(dt_boxes, img_crop_list)
            if not img_crop_list:
                return [], []
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
            crop_boxes = []
            for bno in range(len(dt_boxes)):
                tmp_box = copy.deepcopy(dt_boxes[bno])
                img_crop = get_rotate_crop_image_for_text_rec(ori_im, tmp_box)
                if img_crop is None:
                    continue
                img_crop_list.append(img_crop)
                crop_boxes.append(dt_boxes[bno])
            dt_boxes = crop_boxes
            if not img_crop_list:
                return [], []

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
    img = np.asarray(Image.open("/Users/myhloli/Downloads/screenshot-20250326-194348.png").convert("RGB"))[:, :, ::-1].copy()
    dt_boxes, rec_res = pytorch_paddle_ocr(img)
    ocr_res = []
    if not dt_boxes and not rec_res:
        ocr_res.append(None)
    else:
        tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
        ocr_res.append(tmp_res)
    print(ocr_res)
