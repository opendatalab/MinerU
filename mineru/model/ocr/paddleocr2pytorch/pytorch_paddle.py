# Copyright (c) Opendatalab. All rights reserved.
import copy
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger

from mineru.utils.config_reader import get_device
from mineru.utils.enum_class import ModelPath
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from mineru.model.ocr.surya_ocr.text_predictor import SuryaTextPredictor
from ....utils.ocr_utils import check_img, preprocess_image, sorted_boxes, merge_det_boxes, update_det_boxes, get_rotate_crop_image
from .tools.infer.predict_system import TextSystem
from .tools.infer import pytorchocr_utility as utility
import argparse


latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',  # noqa: E126
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'french', 'german'
]
arabic_lang = ['ar', 'fa', 'ug', 'ur']
cyrillic_lang = [
        'rs_cyrillic', 'bg', 'mn', 'abq', 'ady', 'kbd', 'ava',  # noqa: E126
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
]
east_slavic_lang = ["ru", "be", "uk"]
devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',  # noqa: E126
        'sa', 'bgc'
]
advanced_lang = ['vi', 'th']

def get_model_params(lang, config):
    if lang in config['lang']:
        params = config['lang'][lang]
        det = params.get('det')
        rec = params.get('rec')
        dict_file = params.get('dict')
        return det, rec, dict_file
    else:
        raise Exception (f'Language {lang} not supported')


root_dir = Path(__file__).resolve().parent


class PytorchPaddleOCR(TextSystem):
    def __init__(self, *args, **kwargs):
        parser = utility.init_args()
        args = parser.parse_args(args)

        self.lang = kwargs.get('lang', 'ch')

        device = get_device()
        if device == 'cpu' and self.lang in ['ch', 'ch_server', 'japan', 'chinese_cht']:
            # logger.warning("The current device in use is CPU. To ensure the speed of parsing, the language is automatically switched to ch_lite.")
            self.lang = 'ch_lite'

        if self.lang in latin_lang:
            self.lang = 'latin'
        elif self.lang in arabic_lang:
            self.lang = 'arabic'
        elif self.lang in cyrillic_lang:
            self.lang = 'cyrillic'
        elif self.lang in devanagari_lang:
            self.lang = 'devanagari'
        elif self.lang in east_slavic_lang:
            self.lang = 'east_slavic'
        elif self.lang in advanced_lang:
            self.lang = 'advanced'
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
        kwargs['rec_batch_num'] = 16

        kwargs['device'] = device

        default_args = vars(args)
        default_args.update(kwargs)
        args = argparse.Namespace(**default_args)

        # Initialize TextSystem first
        super().__init__(args)
        
        # Initialize SuryaTextPredictor for Vietnamese language
        if self.lang == 'advanced':
            self._surya_text_recognizer = SuryaTextPredictor()

    def ocr(self,
            img,
            det=True,
            rec=True,
            mfd_res=None,
            tqdm_enable=False,
            ):
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        img = check_img(img)
        imgs = [img]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if det and rec: # table and formula recogition is enabled
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
                    if self.lang == 'advanced':
                        # Use context manager for consistent resource management
                        with self._surya_text_recognizer as surya_predictor:
                            rec_res, elapse = surya_predictor(img)
                        # Explicit cleanup to ensure resources are released
                        self._surya_text_recognizer.cleanup()
                    else:
                        rec_res, elapse = self.text_recognizer(img, tqdm_enable=tqdm_enable)
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
        if self.lang == 'advanced':
            # Use context manager to prevent semaphore leaks when processing multiple images
            with self._surya_text_recognizer as surya_predictor:
                advanced_rec_res, vi_elapse = surya_predictor(img_crop_list)
                for i, advanced_rec_result in enumerate(advanced_rec_res):
                    if self._is_valid_advanced_recognition(advanced_rec_result):
                        rec_res[i] = advanced_rec_result
                        # logger.debug(f"Advanced OCR selected: {advanced_rec_result[0]} (confidence: {advanced_rec_result[1]:.3f})")
            # Explicit cleanup to ensure resources are released
            self._surya_text_recognizer.cleanup()
        # logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res
    
    def _is_valid_advanced_recognition(self, recognition_result):
        """
        Validate advanced OCR recognition results with comprehensive filtering.
        
        Args:
            recognition_result: Tuple of (text, confidence_score)
            
        Returns:
            bool: True if the recognition result meets quality criteria
        """
        if not recognition_result or len(recognition_result) != 2:
            return False
            
        text, confidence = recognition_result
        
        # Basic validation: confidence threshold and non-empty text
        if confidence <= 0.5 or not text:
            return False
            
        # Skip math expressions
        if '<math>' in text:
            return False
            
        # Clean text for analysis
        cleaned_text = text.strip()
        
        # Minimum meaningful length check (only consider non-empty text)
        if len(cleaned_text) < 1:
            return False
            
        # Vietnamese check for meaningful content
        if not self._contains_meaningful_content(cleaned_text):
            return False
            
        return True
        
    def _contains_meaningful_content(self, text):
        """
        Check if text contains meaningful content.
        
        Args:
            text: Text to analyze
            
        Returns:
            bool: True if text contains meaningful content
        """
        # Remove spaces and check remaining content
        content_chars = [char for char in text if char != ' ']
        
        if len(content_chars) == 0:
            return False
            
        # Define problematic characters that should not be considered meaningful
        problematic_chars = set('.٠۰٩٨٧٦٥٤٣٢١')
        
        # Check if it's mostly punctuation, symbols, or problematic characters
        meaningful_chars = sum(1 for char in content_chars 
                             if (char.isalnum() and char not in problematic_chars) or 
                                char in 'àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ')
        
        if len(content_chars) > 0:
            meaningful_ratio = meaningful_chars / len(content_chars)
            return meaningful_ratio >= 0.5  # Increased threshold
            
        return False

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


