# Copyright (c) Opendatalab. All rights reserved.
"""ONNX 后端的 PP-DocLayoutV2 推理封装。

与 ``PPDocLayoutV2LayoutModel``（transformers/torch 后端）公开接口完全一致，
内部用 onnxruntime 推理 PaddlePaddle 官方发布的 ``PP-DocLayoutV2_onnx`` 模型。

后处理逻辑（去重、合并公式框、重标 header/footer、阅读顺序排序等）直接复用
基类 ``PPDocLayoutV2PostProcessor``，保证两个后端的输出结构一致。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

from ..utils.onnxruntime_provider import ort_session
from .pp_doclayout_v2_base import PP_DOCLAYOUT_V2_LABELS, PPDocLayoutV2PostProcessor

__all__ = ["PPDocLayoutV2LayoutModelONNX"]

# PaddlePaddle 官方 PP-DocLayoutV2_onnx 的输入尺寸（与 inference.yml 一致）。
_INPUT_SIZE = (800, 800)


class PPDocLayoutV2LayoutModelONNX(PPDocLayoutV2PostProcessor):
    """PP-DocLayoutV2 的 ONNX 推理封装。

    与 ``PPDocLayoutV2LayoutModel`` 的 ``predict`` / ``batch_predict`` / ``visualize``
    接口完全一致，返回的 box dict 结构也一致（cls_id / label / score / bbox / index）。

    差异：
    - 构造函数接收 onnx 文件路径（单个 ``.onnx``），而非 HF 模型目录
    - 不依赖 transformers / torch，仅用 onnxruntime + numpy + cv2
    - 预处理固定为 800x800 BICUBIC + /255（与 PaddlePaddle 官方 inference.yml 一致）
    - 阅读顺序直接来自 ONNX 输出的第 6/7 列（lexsort），无需 reading_order head
    """

    def __init__(
        self,
        weight: str,
        device: Optional[str] = None,
        imgsz: Tuple[int, int] = _INPUT_SIZE,
        conf: float = 0.45,
        use_paddlex_filter_boxes: bool = True,
        intra_op_num_threads: int = 0,
    ) -> None:
        self.device = device
        self.conf = conf
        self.use_paddlex_filter_boxes = use_paddlex_filter_boxes
        self.model_path = str(weight)
        self.imgsz = imgsz
        self.rescale_factor = 1.0 / 255.0
        self.session = ort_session(self.model_path, device, intra_op_num_threads)
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_name = self.session.get_outputs()[0].name
        logger.debug(
            "PPDocLayoutV2LayoutModelONNX loaded: {} (inputs={}, outputs={})",
            Path(self.model_path).name,
            self._input_names,
            [o.name for o in self.session.get_outputs()],
        )

    def _preprocess_single_image(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Resize 到 800x800 (BICUBIC) + /255，返回 CHW float32 ndarray。"""
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            target_size = pil_image.size[1], pil_image.size[0]
            arr = np.array(pil_image, dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            target_size = arr.shape[0], arr.shape[1]
        else:
            raise TypeError(f"Unsupported image type for PP-DocLayoutV2 ONNX: {type(image)}")

        resized = cv2.resize(arr, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR)
        norm = resized.astype(np.float32) * self.rescale_factor
        chw = norm.transpose(2, 0, 1)
        return chw, target_size

    def _run_session(self, pixel_values: np.ndarray, target_sizes: Sequence[Tuple[int, int]]) -> List[Dict[str, np.ndarray]]:
        """跑一次 ONNX 推理，解析输出为内部 prediction dict。

        PaddlePaddle 官方 PP-DocLayoutV2_onnx 输出 [N, 8]：
        [class_id, score, x1, y1, x2, y2, order_key_a, order_key_b]
        其中 order_key_a/b 是阅读顺序 key，用 lexsort((-col7, col6)) 排序。
        坐标已由模型内部按 scale_factor 还原到原图尺度。
        """
        feed: Dict[str, np.ndarray] = {}
        if "image" in self._input_names:
            feed["image"] = pixel_values
            if "scale_factor" in self._input_names:
                feed["scale_factor"] = np.array(
                    [[self.imgsz[0] / ts[0], self.imgsz[1] / ts[1]] for ts in target_sizes],
                    dtype=np.float32,
                )
            if "im_shape" in self._input_names:
                feed["im_shape"] = np.array([[self.imgsz[0], self.imgsz[1]]] * len(target_sizes), dtype=np.float32)
        else:
            feed[self._input_names[0]] = pixel_values

        preds = self.session.run(None, feed)[0]
        if preds.ndim == 3:
            preds = preds[0]

        batch_predictions: List[Dict[str, np.ndarray]] = []
        for sample_idx, target_size in enumerate(target_sizes):
            sample = preds if preds.ndim == 2 else preds[sample_idx]
            scores = sample[:, 1]
            keep = scores >= self.conf
            sample = sample[keep]

            if sample.shape[1] >= 8:
                order_idx = np.lexsort((-sample[:, 7], sample[:, 6]))
                sample = sample[order_idx]

            batch_predictions.append(
                {
                    "scores": sample[:, 1],
                    "labels": sample[:, 0].astype(np.int64),
                    "boxes": sample[:, 2:6],
                }
            )
        return batch_predictions

    def _parse_prediction(self, result: Dict[str, np.ndarray], image_size: Tuple[int, int]) -> List[Dict]:
        """把 ndarray prediction 转成 box dict 列表（结构与 torch 版一致）。"""
        layout_res: List[Dict] = []
        for index, (score, label_id, box) in enumerate(
            zip(result["scores"], result["labels"], result["boxes"]),
            start=1,
        ):
            bbox = self._clip_bbox(box.tolist(), image_size=image_size)
            if bbox is None:
                continue
            cls_id = int(label_id)
            layout_res.append(
                {
                    "cls_id": cls_id,
                    "label": (PP_DOCLAYOUT_V2_LABELS[cls_id] if 0 <= cls_id < len(PP_DOCLAYOUT_V2_LABELS) else str(cls_id)),
                    "score": round(float(score), 4),
                    "bbox": bbox,
                    "index": index,
                }
            )
        return layout_res

    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        use_paddlex_filter_boxes: Optional[bool] = None,
    ) -> List[Dict]:
        return self.batch_predict(
            [image],
            batch_size=1,
            use_paddlex_filter_boxes=use_paddlex_filter_boxes,
        )[0]

    def batch_predict(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 1,
        use_paddlex_filter_boxes: Optional[bool] = None,
    ) -> List[List[Dict]]:
        if not images:
            return []

        use_filter = self.use_paddlex_filter_boxes if use_paddlex_filter_boxes is None else use_paddlex_filter_boxes
        results: List[List[Dict]] = []
        with tqdm(total=len(images), desc="Layout Predict") as pbar:
            for start in range(0, len(images), batch_size):
                batch_images = images[start : start + batch_size]
                pixel_values_list: List[np.ndarray] = []
                target_sizes: List[Tuple[int, int]] = []
                for image in batch_images:
                    chw, target_size = self._preprocess_single_image(image)
                    pixel_values_list.append(chw)
                    target_sizes.append(target_size)

                batch_tensor = np.stack(pixel_values_list, axis=0).astype(np.float32)
                predictions = self._run_session(batch_tensor, target_sizes)
                for prediction, image_size in zip(predictions, target_sizes):
                    layout_res = self._parse_prediction(prediction, image_size)
                    if use_filter:
                        layout_res = self._apply_paddlex_filter_boxes(layout_res, drop_inline_formula=False)
                    layout_res = self._apply_layout_post_process(layout_res, image_size=image_size)
                    results.append(layout_res)
                pbar.update(len(batch_images))
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PP-DocLayoutV2 ONNX local inference smoke test")
    parser.add_argument("image", nargs="?", help="Path to an input image.")
    parser.add_argument("--model", default=None, help="Path to inference.onnx.")
    parser.add_argument("--device", default=None, help="cpu / cuda / mps.")
    parser.add_argument("--output", default=None, help="Save visualization to this path.")
    args = parser.parse_args()

    if args.model is None:
        from ...utils.model_registry import PP_DOCLAYOUT_V2_ONNX

        args.model = str(PP_DOCLAYOUT_V2_ONNX.onnx.ensure())

    if args.device is None:
        from ...utils.config_reader import get_device

        args.device = get_device()

    model = PPDocLayoutV2LayoutModelONNX(weight=args.model, device=args.device)
    print(f"model loaded on {model.device}")

    if args.image:
        with Image.open(args.image) as img:
            results = model.predict(img)
            print(f"detected {len(results)} regions")
            for r in results:
                print(f"  [{r['index']}] {r['label']:<20} score={r['score']:.2f} bbox={r['bbox']}")
            if args.output:
                vis = model.visualize(img, results)
                vis.save(args.output)
                print(f"visualization saved to {args.output}")
