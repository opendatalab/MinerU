"""验证 det 的 FP32 约束独立于共享 OCR 精度和设备路由。"""

from argparse import Namespace
from typing import Any, Self
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from mineru.model._internal.pytorchocr import base_ocr_v20
from mineru.model._internal.pytorchocr.infer import pytorchocr_utility
from mineru.model._internal.pytorchocr.infer.predict_det import TextDetector
from mineru.model._internal.pytorchocr.infer.predict_system import TextSystem


class _PrecisionNet(torch.nn.Module):
    """使用真实参数和大数值 BN 缓冲区，仅模拟设备迁移以覆盖本机不存在的硬件。"""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        """构造会在 FP16 转换时溢出的 BN 统计值。"""
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(3)
        self.norm.running_var.fill_(1_000_000)
        self.requested_device: str | None = None
        self.input_dtypes: list[torch.dtype] = []

    def to(self, device: str | None = None, *, dtype: torch.dtype | None = None) -> Self:
        """保留真实 dtype 转换，只记录目标设备而不调用加速卡运行时。"""
        if device is not None:
            self.requested_device = str(device)
        return super().to(dtype=dtype) if dtype is not None else self

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """记录真实批处理输入精度，并返回可供 DB 后处理检测的概率图。"""
        self.input_dtypes.append(inputs.dtype)
        return {"maps": self.norm(inputs).mean(dim=1, keepdim=True).sigmoid() + 0.25}


@pytest.fixture
def ocr_args(monkeypatch: pytest.MonkeyPatch) -> Namespace:
    """替换模型文件加载，保留 det/rec/cls 的真实初始化和精度设置。"""
    monkeypatch.setattr(base_ocr_v20, "BaseModel", _PrecisionNet)
    monkeypatch.setattr(pytorchocr_utility, "get_arch_config", Mock(return_value={}))
    monkeypatch.setattr(base_ocr_v20.BaseOCRV20, "load_pytorch_weights", Mock())
    monkeypatch.setattr(base_ocr_v20.BaseOCRV20, "read_pytorch_weights", Mock(return_value={}))
    monkeypatch.setattr(base_ocr_v20.BaseOCRV20, "get_out_channels", Mock(return_value=1))
    monkeypatch.setattr(base_ocr_v20.BaseOCRV20, "load_state_dict", Mock())
    args = pytorchocr_utility.init_args().parse_args([])
    args.use_angle_cls = True
    args.det_inference_precision = "fp16"
    return args


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0", "npu:0", "xpu:0"])
@pytest.mark.parametrize("lang", ["ch", "seal", "seal_lite"])
@pytest.mark.parametrize("precision", ["auto", "fp16", "fp32"])
def test_detector_fp32_keeps_rec_cls_precision_and_device(
    monkeypatch: pytest.MonkeyPatch, ocr_args: Namespace, device: str, lang: str, precision: str
) -> None:
    """所有设备的 det 均保留有限 FP32 权重与输入，且不改变共享参数及 rec/cls 策略。"""
    monkeypatch.setattr(base_ocr_v20, "OCR_INFERENCE_PRECISION", precision)
    ocr_args.device, ocr_args.lang = device, lang
    system = TextSystem(ocr_args)
    detector = system.text_detector
    expected_device = "cpu" if device == "mps" and lang == "ch" else device
    assert detector.device == detector.net.requested_device == expected_device
    assert ocr_args.device == device
    assert detector.ocr_inference_dtype == torch.float32
    for tensor in (*detector.net.parameters(), *detector.net.buffers()):
        if tensor.is_floating_point():
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()
    assert detector._to_inference_dtype(torch.ones(1, dtype=torch.float16)).dtype == torch.float32
    indices = torch.ones(1, dtype=torch.int64)
    assert detector._to_inference_dtype(indices) is indices
    expected_rec_dtype = torch.float32 if device == "cpu" or precision == "fp32" else torch.float16
    for model in (system.text_recognizer, system.text_classifier):
        assert model.device == model.net.requested_device == device
        assert model.ocr_inference_dtype == expected_rec_dtype
        assert next(model.net.parameters()).dtype == expected_rec_dtype
        assert model._to_inference_dtype(torch.ones(1)).dtype == expected_rec_dtype


def test_detector_single_and_mixed_size_batches_use_fp32(monkeypatch: pytest.MonkeyPatch, ocr_args: Namespace) -> None:
    """单图及不同形状分桶的真实前后处理均使用 FP32，且批处理框与单图一致。"""
    monkeypatch.setattr(base_ocr_v20, "OCR_INFERENCE_PRECISION", "fp16")
    detector = TextDetector(ocr_args)
    images = [np.full((height, width, 3), 255, dtype=np.uint8) for height, width in [(32, 64), (64, 64), (32, 64)]]
    singles = [detector(image)[0] for image in images]
    batches = detector.batch_predict(images, max_batch_size=2)
    assert all(len(boxes) > 0 for boxes in singles)
    for single, (batch, _elapsed) in zip(singles, batches):
        np.testing.assert_array_equal(single, batch)
    assert detector.net.input_dtypes == [torch.float32] * 5
