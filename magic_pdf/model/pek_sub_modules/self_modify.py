import time
import copy
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

from paddleocr import PaddleOCR
from paddleocr.ppocr.utils.logging import get_logger
from paddleocr.ppocr.utils.utility import check_and_read, alpha_to_color, binarize_img
from paddleocr.tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


logger = get_logger()

def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, 'rb') as f:
                img_str = f.read()
                img = img_decode(img_str)
            if img is None:
                try:
                    buf = BytesIO()
                    image = BytesIO(img_str)
                    im = Image.open(image)
                    rgb = im.convert('RGB')
                    rgb.save(buf, 'jpeg')
                    buf.seek(0)
                    image_bytes = buf.read()
                    data_base64 = str(base64.b64encode(image_bytes),
                                      encoding="utf-8")
                    image_decode = base64.b64decode(data_base64)
                    img_array = np.frombuffer(image_decode, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                except:
                    logger.error("error in loading image:{}".format(image_file))
                    return None
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def formula_in_text(mf_bbox, text_bbox):
    x1, y1, x2, y2 = mf_bbox
    x3, y3 = text_bbox[0]
    x4, y4 = text_bbox[2]
    left_box, right_box = None, None
    same_line = abs((y1+y2)/2 - (y3+y4)/2) / abs(y4-y3) < 0.2
    if not same_line:
        return False, left_box, right_box
    else:
        drop_origin = False
        left_x = x1 - 1
        right_x = x2 + 1
        if x3 < x1 and x2 < x4:
            drop_origin = True
            left_box = np.array([text_bbox[0], [left_x, text_bbox[1][1]], [left_x, text_bbox[2][1]], text_bbox[3]]).astype('float32')
            right_box = np.array([[right_x, text_bbox[0][1]], text_bbox[1], text_bbox[2], [right_x, text_bbox[3][1]]]).astype('float32')
        if x3 < x1 and x1 <= x4 <= x2:
            drop_origin = True
            left_box = np.array([text_bbox[0], [left_x, text_bbox[1][1]], [left_x, text_bbox[2][1]], text_bbox[3]]).astype('float32')
        if x1 <= x3 <= x2 and x2 < x4:
            drop_origin = True
            right_box = np.array([[right_x, text_bbox[0][1]], text_bbox[1], text_bbox[2], [right_x, text_bbox[3][1]]]).astype('float32')
        if x1 <= x3 < x4 <= x2:
            drop_origin = True
        return drop_origin, left_box, right_box

    
def update_det_boxes(dt_boxes, mfdetrec_res):
    new_dt_boxes = dt_boxes
    for mf_box in mfdetrec_res:
        flag, left_box, right_box = False, None, None
        for idx, text_box in enumerate(new_dt_boxes):
            ret, left_box, right_box = formula_in_text(mf_box['bbox'], text_box)
            if ret:
                new_dt_boxes.pop(idx)
                if left_box is not None:
                    new_dt_boxes.append(left_box)
                if right_box is not None:
                    new_dt_boxes.append(right_box)
                break
            
    return new_dt_boxes

class ModifiedPaddleOCR(PaddleOCR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vietocr_cfg = Cfg.load_config_from_name('vgg_transformer')
        self.vietocr_cfg['device'] = 'cpu'  # Force CPU usage
        self.vietocr_detector = Predictor(self.vietocr_cfg)
    
    def ocr(self, img, det=True, rec=True, cls=True, bin=False, inv=False, mfd_res=None, alpha_color=(255, 255, 255)):
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
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                # print("SHAPE ", img.shape)
                dt_boxes, rec_res, _ = self.__call__(img, cls, mfd_res=mfd_res)
                if not dt_boxes and not rec_res:
                    ocr_res.append(None)
                    continue
                tmp_res = []
                for box, res in zip(dt_boxes, rec_res):
                    if box is not None:
                        x_min = int(box[0][0])
                        y_min = int(box[0][1])
                        x_max = int(box[2][0])
                        y_max = int(box[2][1])

                        text_img = img[y_min:y_max, x_min:x_max]

                        text = self.vietocr_detector.predict(Image.fromarray(text_img))

                        tmp_res.append([box.tolist(), (text, res[1])])
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                img = preprocess_image(img)
                dt_boxes, elapse = self.text_detector(img)
                if not dt_boxes:
                    ocr_res.append(None)
                    continue
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
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
        
    def __call__(self, img, cls=True, mfd_res=None):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
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
        if mfd_res:
            bef = time.time()
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)
            aft = time.time()
            logger.debug("split text box by formula, new dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), aft-bef))

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