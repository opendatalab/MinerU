import copy
import time


import cv2
import numpy as np
from paddleocr import PaddleOCR
from paddleocr.paddleocr import check_img, logger
from paddleocr.ppocr.utils.utility import alpha_to_color, binarize_img
from paddleocr.tools.infer.predict_system import sorted_boxes
from paddleocr.tools.infer.utility import slice_generator, merge_fragmented, get_rotate_crop_image, \
    get_minarea_rect_crop

from magic_pdf.model.sub_modules.ocr.paddleocr.ocr_utils import update_det_boxes


class ModifiedPaddleOCR(PaddleOCR):

    def ocr(
        self,
        img,
        det=True,
        rec=True,
        cls=True,
        bin=False,
        inv=False,
        alpha_color=(255, 255, 255),
        slice={},
        mfd_res=None,
    ):
        """
        OCR with PaddleOCR

        Args:
            img: Image for OCR. It can be an ndarray, img_path, or a list of ndarrays.
            det: Use text detection or not. If False, only text recognition will be executed. Default is True.
            rec: Use text recognition or not. If False, only text detection will be executed. Default is True.
            cls: Use angle classifier or not. Default is True. If True, the text with a rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance.
            bin: Binarize image to black and white. Default is False.
            inv: Invert image colors. Default is False.
            alpha_color: Set RGB color Tuple for transparent parts replacement. Default is pure white.
            slice: Use sliding window inference for large images. Both det and rec must be True. Requires int values for slice["horizontal_stride"], slice["vertical_stride"], slice["merge_x_thres"], slice["merge_y_thres"] (See doc/doc_en/slice_en.md). Default is {}.

        Returns:
            If both det and rec are True, returns a list of OCR results for each image. Each OCR result is a list of bounding boxes and recognized text for each detected text region.
            If det is True and rec is False, returns a list of detected bounding boxes for each image.
            If det is False and rec is True, returns a list of recognized text for each image.
            If both det and rec are False, returns a list of angle classification results for each image.

        Raises:
            AssertionError: If the input image is not of type ndarray, list, str, or bytes.
            SystemExit: If det is True and the input is a list of images.

        Note:
            - If the angle classifier is not initialized (use_angle_cls=False), it will not be used during the forward process.
            - For PDF files, if the input is a list of images and the page_num is specified, only the first page_num images will be processed.
            - The preprocess_image function is used to preprocess the input image by applying alpha color replacement, inversion, and binarization if specified.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error("When input a list of images, det must be false")
            exit(0)
        if cls == True and self.use_angle_cls == False:
            logger.warning(
                "Since the angle classifier is not initialized, it will not be used during the forward process"
            )

        img, flag_gif, flag_pdf = check_img(img, alpha_color)
        # for infer pdf file
        if isinstance(img, list) and flag_pdf:
            if self.page_num > len(img) or self.page_num == 0:
                imgs = img
            else:
                imgs = img[: self.page_num]
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
                dt_boxes, rec_res, _ = self.__call__(img, cls, slice, mfd_res=mfd_res)
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
                if dt_boxes.size == 0:
                    ocr_res.append(None)
                    continue
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
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res

    def __call__(self, img, cls=True, slice={}, mfd_res=None):
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        if slice:
            slice_gen = slice_generator(
                img,
                horizontal_stride=slice["horizontal_stride"],
                vertical_stride=slice["vertical_stride"],
            )
            elapsed = []
            dt_slice_boxes = []
            for slice_crop, v_start, h_start in slice_gen:
                dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                if dt_boxes.size:
                    dt_boxes[:, :, 0] += h_start
                    dt_boxes[:, :, 1] += v_start
                    dt_slice_boxes.append(dt_boxes)
                    elapsed.append(elapse)
            dt_boxes = np.concatenate(dt_slice_boxes)

            dt_boxes = merge_fragmented(
                boxes=dt_boxes,
                x_threshold=slice["merge_x_thres"],
                y_threshold=slice["merge_y_thres"],
            )
            elapse = sum(elapsed)
        else:
            dt_boxes, elapse = self.text_detector(img)

        time_dict["det"] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict
        else:
            logger.debug(
                "dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse)
            )
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

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
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
            time_dict["cls"] = elapse
            logger.debug(
                "cls num  : {}, elapsed : {}".format(len(img_crop_list), elapse)
            )
        if len(img_crop_list) > 1000:
            logger.debug(
                f"rec crops num: {len(img_crop_list)}, time and memory cost may be large."
            )

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict["all"] = end - start
        return filter_boxes, filter_rec_res, time_dict
