# Copyright (c) Opendatalab. All rights reserved.
"""复用 Transformers 的官方 PP-DocLayoutV2，保留 MinerU 的预处理和版面后处理。"""

import argparse
import json
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from PIL import Image
from torch import nn
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import PPDocLayoutV2Config, PPDocLayoutV2ForObjectDetection
from transformers.models.pp_doclayout_v2.configuration_pp_doclayout_v2 import PPDocLayoutV2ReadingOrderConfig
from transformers.models.pp_doclayout_v2.modeling_pp_doclayout_v2 import (
    PPDocLayoutV2ForObjectDetectionOutput,
    PPDocLayoutV2ReadingOrder,
)

from .pp_doclayout_v2_base import (
    DEFAULT_CLASS_ORDER,
    DEFAULT_CLASS_THRESHOLDS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RESCALE_FACTOR,
    PP_DOCLAYOUT_V2_LABEL_TO_ID,
    PP_DOCLAYOUT_V2_LABELS,
    PPDocLayoutV2PostProcessor,
    label_to_color,
    load_preprocess_config,
)


class _CpuSinePositionEmbedding(nn.Module):
    """在 CPU 计算官方实现所需的 float64 频率，再把位置编码送回模型设备。"""

    def __init__(self, embedding: nn.Module) -> None:
        """组合无参数的位置编码模块，保持预训练权重名称和数学实现不变。"""
        super().__init__()
        self.embedding = embedding

    def forward(
        self,
        width: int,
        height: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """避开 MPS 和部分 XPU 不支持的 float64 运算，复用官方 CPU 位置编码缓存。"""
        return self.embedding(width=width, height=height, device=torch.device("cpu"), dtype=dtype).to(device=device)


class PPDocLayoutV2LayoutModel(PPDocLayoutV2PostProcessor):
    def __init__(
        self,
        weight: str,
        device: Optional[str] = "cuda",
        imgsz: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        use_paddlex_filter_boxes: bool = True,
    ) -> None:
        """加载模型及其阅读顺序配置，最终输出另按共享准入阈值筛选。"""
        self.device = device or "cpu"
        self.use_paddlex_filter_boxes = use_paddlex_filter_boxes
        self.model_dir = weight
        self.preprocess_config = load_preprocess_config(self.model_dir)
        size = self.preprocess_config.get("size", {})
        self.imgsz = (
            int(size.get("width", imgsz[0])),
            int(size.get("height", imgsz[1])),
        )
        self.rescale_factor = float(self.preprocess_config.get("rescale_factor", DEFAULT_RESCALE_FACTOR))
        self.config = PPDocLayoutV2Config.from_pretrained(self.model_dir)
        # 仅提高行内公式阈值，让模型阅读顺序与最终检测筛选使用同一门槛。
        inline_formula_id = PP_DOCLAYOUT_V2_LABEL_TO_ID["inline_formula"]
        self.config.class_thresholds[inline_formula_id] = DEFAULT_CLASS_THRESHOLDS[inline_formula_id]
        # 保留原有 FP32 推理精度，直接加载到目标设备以减少 CPU 权重中转。
        self.model = PPDocLayoutV2ForObjectDetection.from_pretrained(
            self.model_dir,
            config=self.config,
            device_map={"": self.device},
            dtype=torch.float32,
            attn_implementation="eager" if torch.device(self.device).type == "mps" else None,
        )
        if torch.device(self.device).type in {"mps", "xpu"}:
            # 官方正弦编码内部使用 float64，Arc 等 XPU 不支持；统一在 CPU 缓存这个无参数模块的结果。
            for layer in self.model.model.encoder.aifi:
                layer.position_embedding = _CpuSinePositionEmbedding(layer.position_embedding)
        # 同步加载器生成的非持久化 buffer；已经位于目标设备的参数不会再次复制。
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _get_order_seqs(order_logits: torch.Tensor) -> torch.Tensor:
        order_scores = torch.sigmoid(order_logits)
        batch_size, sequence_length, _ = order_scores.shape
        order_votes = order_scores.triu(diagonal=1).sum(dim=1) + (1.0 - order_scores.transpose(1, 2)).tril(diagonal=-1).sum(
            dim=1
        )
        order_pointers = torch.argsort(order_votes, dim=1)
        order_seq = torch.empty_like(order_pointers)
        ranks = torch.arange(sequence_length, device=order_pointers.device, dtype=order_pointers.dtype).expand(batch_size, -1)
        order_seq.scatter_(1, order_pointers, ranks)
        return order_seq

    def _preprocess_single_image(self, image: Union[np.ndarray, Image.Image]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise TypeError(f"Unsupported image type for PP-DocLayoutV2: {type(image)}")

        pil_image = pil_image.convert("RGB")
        target_size = pil_image.size[1], pil_image.size[0]
        pixel_values = tvF.pil_to_tensor(pil_image)
        pixel_values = tvF.resize(
            pixel_values,
            size=[self.imgsz[1], self.imgsz[0]],
            interpolation=InterpolationMode.BICUBIC,
            antialias=False,
        )
        pixel_values = pixel_values.to(dtype=torch.float32) * self.rescale_factor
        return pixel_values, target_size

    def _post_process_object_detection(
        self,
        outputs: PPDocLayoutV2ForObjectDetectionOutput,
        target_sizes: Sequence[Tuple[int, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        """按各标签的输出准入阈值独立筛选，沿用模型给出的阅读顺序。"""
        boxes = outputs.pred_boxes
        logits = outputs.logits
        order_logits = outputs.order_logits
        order_seqs = self._get_order_seqs(order_logits)

        box_centers, box_dims = torch.split(boxes, 2, dim=-1)
        boxes = torch.cat([box_centers - 0.5 * box_dims, box_centers + 0.5 * box_dims], dim=-1)

        img_height, img_width = torch.as_tensor(target_sizes, device=boxes.device).unbind(1)
        scale_factor = torch.stack([img_width, img_height, img_width, img_height], dim=1).to(boxes.device)
        boxes = boxes * scale_factor[:, None, :]

        num_top_queries = logits.shape[1]
        num_classes = logits.shape[2]
        class_thresholds = torch.as_tensor(DEFAULT_CLASS_THRESHOLDS, dtype=torch.float32, device=logits.device)
        scores = torch.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), num_top_queries, dim=-1)
        labels = index % num_classes
        index = index // num_classes
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
        order_seqs = order_seqs.gather(dim=1, index=index)

        results = []
        for score, label, box, order_seq in zip(scores, labels, boxes, order_seqs):
            keep = score >= class_thresholds[label]
            order_seq = order_seq[keep]
            order_seq, indices = torch.sort(order_seq)
            results.append(
                {
                    "scores": score[keep][indices],
                    "labels": label[keep][indices],
                    "boxes": box[keep][indices],
                    "order_seq": order_seq,
                }
            )
        return results

    def _parse_prediction(self, result: Dict[str, torch.Tensor], image_size: Tuple[int, int]) -> List[Dict]:
        layout_res = []
        for index, (score, label_id, box, _order_seq) in enumerate(
            zip(
                result["scores"],
                result["labels"],
                result["boxes"],
                result["order_seq"],
            ),
            start=1,
        ):
            bbox = self._clip_bbox(box.tolist(), image_size)
            if bbox is None:
                continue

            cls_id = int(label_id.item())
            layout_res.append(
                {
                    "cls_id": cls_id,
                    "label": self._label_id_to_label_name(cls_id),
                    "score": round(float(score.item()), 4),
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
        if len(images) == 0:
            return []

        use_paddlex_filter_boxes = (
            self.use_paddlex_filter_boxes if use_paddlex_filter_boxes is None else use_paddlex_filter_boxes
        )
        results: List[List[Dict]] = []
        with torch.no_grad():
            with tqdm(total=len(images), desc="Layout Predict") as pbar:
                for start in range(0, len(images), batch_size):
                    batch_images = images[start : start + batch_size]
                    pixel_values_list = []
                    target_sizes = []
                    for image in batch_images:
                        pixel_values, target_size = self._preprocess_single_image(image)
                        pixel_values_list.append(pixel_values)
                        target_sizes.append(target_size)

                    batch_tensor = torch.stack(pixel_values_list, dim=0).to(self.device)
                    outputs = self.model(pixel_values=batch_tensor)
                    predictions = self._post_process_object_detection(outputs, target_sizes)
                    for prediction, image_size in zip(predictions, target_sizes):
                        layout_res = self._parse_prediction(prediction, image_size)
                        if use_paddlex_filter_boxes:
                            layout_res = self._apply_paddlex_filter_boxes(layout_res, drop_inline_formula=False)
                        layout_res = self._apply_layout_post_process(layout_res, image_size=image_size)
                        results.append(layout_res)
                    pbar.update(len(batch_images))
        return results


__all__ = [
    "DEFAULT_CLASS_ORDER",
    "DEFAULT_CLASS_THRESHOLDS",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_RESCALE_FACTOR",
    "PP_DOCLAYOUT_V2_LABELS",
    "PP_DOCLAYOUT_V2_LABEL_TO_ID",
    "PPDocLayoutV2Config",
    "PPDocLayoutV2ForObjectDetection",
    "PPDocLayoutV2LayoutModel",
    "PPDocLayoutV2ReadingOrder",
    "PPDocLayoutV2ReadingOrderConfig",
    "label_to_color",
    "load_preprocess_config",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PP-DocLayoutV2 local inference smoke test")
    parser.add_argument("image", nargs="?", help="Path to an input image. If omitted, only model loading is tested.")
    parser.add_argument("--model", default=None, help="Model name or path.")
    parser.add_argument("--device", default=None, help="Runtime device, e.g. cpu/mps/cuda.")
    parser.add_argument("--output", default=None, help="Optional path to save the visualization image.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the visualization window.")
    args = parser.parse_args()

    if args.device is None:
        from ..runtime.device import get_device

        args.device = get_device()

    if args.model is None:
        from ..registry import MINERU_4_MODELS_TORCH

        args.model = str(MINERU_4_MODELS_TORCH.pp_doclayout_v2.ensure())

    args.image = "/Users/myhloli/pdf/png/index.png"

    model = PPDocLayoutV2LayoutModel(
        weight=args.model,
        device=args.device,
    )
    print(f"model loaded on {model.device}")

    if args.image:
        with Image.open(args.image) as img:
            results = model.predict(img)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            vis_img = model.visualize(img, results)
            if args.output:
                vis_img.save(args.output)
            if not args.no_show:
                vis_img.show()
