import copy
import platform
import time
import cv2
import numpy as np
import torch

from paddleocr import PaddleOCR
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import alpha_to_color, binarize_img
from tools.infer.predict_system import sorted_boxes
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop

from magic_pdf.model.sub_modules.ocr.paddleocr.ocr_utils import update_det_boxes, merge_det_boxes, check_img, \
    ONNXModelSingleton

logger = get_logger()


class ModifiedPaddleOCR(PaddleOCR):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.lang = kwargs.get('lang', 'ch')
        # 在cpu架构为arm且不支持cuda时调用onnx、
        if not torch.cuda.is_available() and platform.machine() in ['arm64', 'aarch64']:
            self.use_onnx = True
            onnx_model_manager = ONNXModelSingleton()
            self.additional_ocr = onnx_model_manager.get_onnx_model(**kwargs)
        else:
            self.use_onnx = False

    def ocr(self,
            img,
            det=True,
            rec=True,
            cls=True,
            bin=False,
            inv=False,
            alpha_color=(255, 255, 255),
            mfd_res=None,
            ):
        """
        OCR with PaddleOCR
        args：
            img: img for OCR, support ndarray, img_path and list or ndarray
            det: use text detection or not. If False, only rec will be exec. Default is True
            rec: use text recognition or not. If False, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If True, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
            bin: binarize image to black and white. Default is False.
            inv: invert image colors. Default is False.
            alpha_color: set RGB color Tuple for transparent parts replacement. Default is pure white.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        if cls == True and self.use_angle_cls == False:
            pass
            # logger.warning(
            #     'Since the angle classifier is not initialized, it will not be used during the forward process'
            # )

        img = check_img(img)
        # for infer pdf file
        if isinstance(img, list):
            if self.page_num > len(img) or self.page_num == 0:
                self.page_num = len(img)
            imgs = img[:self.page_num]
        else:
            imgs = [img]

        def preprocess_image(_image):
            _image = alpha_to_color(_image, alpha_color)
            if inv:
                _image = cv2.bitwise_not(_image)
            if bin:
                _image = binarize_img(_image)
            return _image

        if det and rec:
            ocr_res = []
            for img in imgs:
                img = preprocess_image(img)
                dt_boxes, rec_res, _ = self.__call__(img, cls, mfd_res=mfd_res)
                if not dt_boxes and not rec_res:
                    ocr_res.append(None)
                    continue
                tmp_res = [[box.tolist(), res]
                           for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for img in imgs:
                img = preprocess_image(img)
                if self.lang in ['ch'] and self.use_onnx:
                    dt_boxes, elapse = self.additional_ocr.text_detector(img)
                else:
                    dt_boxes, elapse = self.text_detector(img)
                if dt_boxes is None:
                    ocr_res.append(None)
                    continue
                dt_boxes = sorted_boxes(dt_boxes)
                # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
                dt_boxes = merge_det_boxes(dt_boxes)
                if mfd_res:
                    bef = time.time()
                    dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                    aft = time.time()
                    logger.debug("split text box by formula, new dt_boxes num : {}, elapsed : {}".format(
                        len(dt_boxes), aft - bef))
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for img in imgs:
                if not isinstance(img, list):
                    img = preprocess_image(img)
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                if self.lang in ['ch'] and self.use_onnx:
                    rec_res, elapse = self.additional_ocr.text_recognizer(img)
                else:
                    rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res

    def __call__(self, img, cls=True, mfd_res=None):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        if self.lang in ['ch'] and self.use_onnx:
            dt_boxes, elapse = self.additional_ocr.text_detector(img)
        else:
            dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict
        else:
            logger.debug("dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
        dt_boxes = merge_det_boxes(dt_boxes)

        if mfd_res:
            bef = time.time()
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)
            aft = time.time()
            logger.debug("split text box by formula, new dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), aft - bef))

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapsed : {}".format(
                len(img_crop_list), elapse))
        if self.lang in ['ch'] and self.use_onnx:
            rec_res, elapse = self.additional_ocr.text_recognizer(img_crop_list)
        else:
            rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict