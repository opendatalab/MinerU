# Copyright (c) Opendatalab. All rights reserved.
import sys
from collections import defaultdict

import numpy as np
import time
import torch
from tqdm import tqdm
from ...pytorchocr.base_ocr_v20 import BaseOCRV20
from . import pytorchocr_utility as utility
from ...pytorchocr.data import create_operators, transform
from ...pytorchocr.postprocess import build_post_process


class TextDetector(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.device = args.device
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
                'max_side_limit': args.det_max_side_limit,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            postprocess_params["box_type"] = args.det_box_type
        elif self.det_algorithm == "DB++":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            postprocess_params["box_type"] = args.det_box_type
            pre_process_list[1] = {
                'NormalizeImage': {
                    'std': [1.0, 1.0, 1.0],
                    'mean':
                        [0.48109378172549, 0.45752457890196, 0.40787054090196],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': args.det_limit_side_len
                }
            }
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = args.det_sast_nms_thresh
            self.det_sast_polygon = args.det_sast_polygon
            if self.det_sast_polygon:
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3
        elif self.det_algorithm == "PSE":
            postprocess_params['name'] = 'PSEPostProcess'
            postprocess_params["thresh"] = args.det_pse_thresh
            postprocess_params["box_thresh"] = args.det_pse_box_thresh
            postprocess_params["min_area"] = args.det_pse_min_area
            postprocess_params["box_type"] = args.det_pse_box_type
            postprocess_params["scale"] = args.det_pse_scale
            self.det_pse_box_type = args.det_pse_box_type
        elif self.det_algorithm == "FCE":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'rescale_img': [1080, 736]
                }
            }
            postprocess_params['name'] = 'FCEPostProcess'
            postprocess_params["scales"] = args.scales
            postprocess_params["alpha"] = args.alpha
            postprocess_params["beta"] = args.beta
            postprocess_params["fourier_degree"] = args.fourier_degree
            postprocess_params["box_type"] = args.det_fce_box_type
        else:
            print("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        self.weights_path = args.det_model_path
        self.yaml_path = args.det_yaml_path
        network_config = utility.get_arch_config(self.weights_path)
        super(TextDetector, self).__init__(network_config, **kwargs)
        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
        self._apply_inference_precision(self.device)
        for module in self.net.modules():
            if hasattr(module, 'rep'):
                module.rep()

    def _preprocess_det_image(self, img):
        """执行 OCR-det 单图预处理，并保留后处理需要的原始尺寸信息。"""
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        if data is None:
            return None

        img_processed, shape_list = data
        if img_processed is None:
            return None

        return np.ascontiguousarray(img_processed), shape_list, img.shape

    def _should_only_clip_det_res(self):
        if self.det_algorithm == "SAST" and getattr(self, "det_sast_polygon", False):
            return True
        if self.det_algorithm in ["DB", "DB++", "PSE", "FCE"]:
            return getattr(self.postprocess_op, "box_type", "quad") == "poly"
        return False

    def _filter_det_res(self, dt_boxes, image_shape):
        if self._should_only_clip_det_res():
            return self.filter_tag_det_res_only_clip(dt_boxes, image_shape)
        return self.filter_tag_det_res(dt_boxes, image_shape)

    def _build_det_preds(self, outputs):
        """将 OCR-det 模型输出统一转换为后处理需要的 float32 numpy 结构。"""
        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs['f_geo'].float().cpu().numpy()
            preds['f_score'] = outputs['f_score'].float().cpu().numpy()
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs['f_border'].float().cpu().numpy()
            preds['f_score'] = outputs['f_score'].float().cpu().numpy()
            preds['f_tco'] = outputs['f_tco'].float().cpu().numpy()
            preds['f_tvo'] = outputs['f_tvo'].float().cpu().numpy()
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            preds['maps'] = outputs['maps'].float().cpu().numpy()
        elif self.det_algorithm == 'FCE':
            for i, (_k, output) in enumerate(outputs.items()):
                preds['level_{}'.format(i)] = output.float().cpu().numpy()
        else:
            raise NotImplementedError
        return preds

    def _postprocess_det_batch(self, preds, batch_shapes, ori_shapes):
        """对完整 batch 执行一次 OCR-det 后处理，再逐张裁剪过滤检测框。"""
        post_results = self.postprocess_op(preds, batch_shapes)
        batch_results = []
        for post_result, ori_shape in zip(post_results, ori_shapes):
            dt_boxes = post_result['points']
            dt_boxes = self._filter_det_res(dt_boxes, ori_shape)
            batch_results.append(dt_boxes)
        return batch_results

    def _batch_process_preprocessed(self, batch_items):
        """对已经完成预处理且形状一致的图片执行批量推理。"""
        starttime = time.time()
        if not batch_items:
            return [], 0

        batch_data = [item[1] for item in batch_items]
        batch_shapes = [item[2] for item in batch_items]
        ori_shapes = [item[3] for item in batch_items]

        try:
            batch_tensor = np.ascontiguousarray(np.stack(batch_data, axis=0))
            batch_shapes = np.stack(batch_shapes, axis=0)
        except Exception:
            batch_results = []
            for _index, img_processed, shape_list, ori_shape in batch_items:
                single_tensor = np.expand_dims(np.ascontiguousarray(img_processed), axis=0)
                single_shape = np.expand_dims(shape_list, axis=0)
                with torch.inference_mode():
                    inp = torch.from_numpy(single_tensor)
                    inp = inp.to(self.device)
                    inp = self._to_inference_dtype(inp)
                    outputs = self.net(inp)
                preds = self._build_det_preds(outputs)
                dt_boxes = self._postprocess_det_batch(preds, single_shape, [ori_shape])[0]
                batch_results.append((dt_boxes, 0))
            return batch_results, time.time() - starttime

        with torch.inference_mode():
            inp = torch.from_numpy(batch_tensor)
            inp = inp.to(self.device)
            inp = self._to_inference_dtype(inp)
            outputs = self.net(inp)

        preds = self._build_det_preds(outputs)
        dt_boxes_batch = self._postprocess_det_batch(preds, batch_shapes, ori_shapes)
        total_elapse = time.time() - starttime
        batch_elapse = total_elapse / len(batch_items)
        batch_results = [(dt_boxes, batch_elapse) for dt_boxes in dt_boxes_batch]
        return batch_results, total_elapse

    def _batch_process_same_size(self, img_list):
        """
            对相同尺寸的图像进行批处理

            Args:
                img_list: 相同尺寸的图像列表

            Returns:
                batch_results: 批处理结果列表
                total_elapse: 总耗时
            """
        starttime = time.time()
        batch_items = []
        for index, img in enumerate(img_list):
            preprocessed = self._preprocess_det_image(img)
            if preprocessed is None:
                return [(None, 0) for _ in img_list], 0
            img_processed, shape_list, ori_shape = preprocessed
            batch_items.append((index, img_processed, shape_list, ori_shape))

        batch_results, _elapsed = self._batch_process_preprocessed(batch_items)
        return batch_results, time.time() - starttime

    def batch_predict(
        self,
        img_list,
        max_batch_size=8,
        tqdm_enable=False,
        tqdm_desc="OCR-det Predict",
        tqdm_progress_bar=None,
    ):
        """
        批处理预测方法，支持多张图像同时检测

        Args:
            img_list: 图像列表
            max_batch_size: 最大批处理大小
            tqdm_enable: 是否显示内部 OCR-det 进度条
            tqdm_desc: 内部 OCR-det 进度条描述
            tqdm_progress_bar: 外部复用进度条，传入时不在本方法内关闭

        Returns:
            batch_results: 批处理结果列表，每个元素为(dt_boxes, elapse)
        """
        if not img_list:
            return []

        progress_bar = tqdm_progress_bar
        should_close_progress = False
        if progress_bar is None:
            progress_bar = tqdm(total=len(img_list), desc=tqdm_desc, disable=not tqdm_enable)
            should_close_progress = True

        max_batch_size = max(1, int(max_batch_size))
        batch_results = [(None, 0)] * len(img_list)
        grouped_items = defaultdict(list)

        try:
            for index, img in enumerate(img_list):
                preprocessed = self._preprocess_det_image(img)
                if preprocessed is None:
                    progress_bar.update(1)
                    continue
                img_processed, shape_list, ori_shape = preprocessed
                grouped_items[img_processed.shape].append(
                    (index, img_processed, shape_list, ori_shape)
                )

            for group_items in grouped_items.values():
                for i in range(0, len(group_items), max_batch_size):
                    batch_items = group_items[i:i + max_batch_size]
                    group_results, _batch_elapse = self._batch_process_preprocessed(batch_items)
                    for batch_item, batch_result in zip(batch_items, group_results):
                        original_index = batch_item[0]
                        batch_results[original_index] = batch_result
                    progress_bar.update(len(batch_items))
        finally:
            if should_close_progress:
                progress_bar.close()

        return batch_results

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        # Polygon detectors may emit a variable number of points per box,
        # so this path must preserve a ragged outer container.
        return dt_boxes_new

    def __call__(self, img):
        preprocessed = self._preprocess_det_image(img)
        if preprocessed is None:
            return None, 0
        img_processed, shape_list, ori_shape = preprocessed
        batch_results, _elapsed = self._batch_process_preprocessed(
            [(0, img_processed, shape_list, ori_shape)]
        )
        return batch_results[0]
