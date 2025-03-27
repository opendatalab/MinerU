# Copyright (c) Opendatalab. All rights reserved.
import copy

import cv2
import numpy as np
from loguru import logger

from magic_pdf.libs.config_reader import get_device
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.ocr_utils import check_img, preprocess_image, sorted_boxes, \
    merge_det_boxes, update_det_boxes, get_rotate_crop_image
from magic_pdf.model.sub_modules.ocr.paddleocr2pytorch.tools.infer.predict_system import TextSystem
import tools.infer.pytorchocr_utility as utility
import argparse


class PytorchPaddleOCR(TextSystem):
    def __init__(self, *args, **kwargs):
        parser = utility.init_args()
        args = parser.parse_args(args)

        self.lang = kwargs.get('lang', 'ch')

        # kwargs['cls_model_path'] = "/Users/myhloli/Downloads/ch_ptocr_mobile_v2.0_cls_infer.pth"

        if self.lang == 'ch':
            kwargs['det_model_path'] = "/Users/myhloli/Downloads/ch_ptocr_v4_det_infer.pth"
            kwargs['rec_model_path'] = "/Users/myhloli/Downloads/ch_ptocr_v4_rec_infer.pth"
            kwargs['det_yaml_path'] = "/Users/myhloli/Downloads/PaddleOCR2Pytorch-main/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_student.yml"
            kwargs['rec_yaml_path'] = "/Users/myhloli/Downloads/PaddleOCR2Pytorch-main/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec.yml"
            kwargs['rec_image_shape'] = '3,48,320'

        kwargs['device'] = get_device()

        default_args = vars(args)
        default_args.update(kwargs)
        args = argparse.Namespace(**default_args)

        super().__init__(args)

    def ocr(self,
            img,
            det=True,
            rec=True,
            mfd_res=None,
            ):
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        img = check_img(img)
        imgs = [img]

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
                if dt_boxes is None:
                    ocr_res.append(None)
                    continue
                dt_boxes = sorted_boxes(dt_boxes)
                # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
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
                rec_res, elapse = self.text_recognizer(img)
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
            logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
        dt_boxes = merge_det_boxes(dt_boxes)

        if mfd_res:
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))

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


