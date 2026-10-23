"""印章零尺寸裁片及检测框、识别结果和调试产物对应关系回归。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from mineru.model.ocr import seal_crop
from mineru.model.ocr.seal_crop import CropByPolys, SortPolyBoxes

_REFERENCE = json.loads((Path(__file__).parents[1] / "fixtures/seal_crop_degenerate.json").read_text())


@pytest.fixture
def image_backend() -> Iterator[None]:
    """共享原生图像内核检查无效轮廓和有效裁片的像素契约。"""
    from docvortex._compute_backend import get_native

    get_native.cache_clear()
    get_native()
    try:
        yield
    finally:
        get_native.cache_clear()


@pytest.fixture
def seal_image() -> np.ndarray:
    """构造与真实失败印章尺寸一致的确定性图像。"""
    return np.random.default_rng(_REFERENCE["image_seed"]).integers(0, 256, _REFERENCE["image_shape"], dtype=np.uint8)


@pytest.mark.parametrize("mode", ["poly", "quad"])
@pytest.mark.parametrize(
    "points",
    [
        _REFERENCE["degenerate_polygon"],
        [[20, 10], [20, 10], [20, 30], [20, 30]],
        [[10, 20], [30, 20], [30, 20], [10, 20]],
        [[10, 20]] * 4,
    ],
    ids=["real-repeated-points", "zero-width", "zero-height", "single-point"],
)
def test_invalid_seal_crop_keeps_original_position(
    image_backend: None, seal_image: np.ndarray, mode: str, points: list[list[int]]
) -> None:
    """真实重复点、线段和单点均保留无效裁片标记，且不影响前后有效轮廓。"""
    valid = np.asarray(_REFERENCE["valid_cases"][0]["points"])
    invalid = np.asarray(points)
    if mode == "quad":
        invalid = invalid[:4]
    crops = CropByPolys(mode)(seal_image, [valid, invalid, valid + 70])
    assert len(crops) == 3
    assert crops[0] is not None and crops[0].size > 0
    assert crops[1] is None
    assert crops[2] is not None and crops[2].size > 0


def test_zero_target_size_is_rejected_before_matrix_creation(
    image_backend: None, seal_image: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实四点裁片高度取整为零时，不再调用透视矩阵或共享采样内核。"""
    matrix = Mock(side_effect=AssertionError("zero-size target must not reach perspective_matrix"))
    warp = Mock(side_effect=AssertionError("zero-size target must not reach warp_image"))
    monkeypatch.setattr(seal_crop, "perspective_matrix", matrix)
    monkeypatch.setattr(seal_crop, "warp_image", warp)
    assert CropByPolys().get_rotate_crop_image(seal_image, np.asarray(_REFERENCE["degenerate_quad"], dtype=np.float32)) is None
    matrix.assert_not_called()
    warp.assert_not_called()


@pytest.mark.parametrize("mode", ["poly", "quad"])
@pytest.mark.parametrize("case", _REFERENCE["valid_cases"], ids=["aligned", "inclined"])
def test_valid_seal_crop_keeps_frozen_pixels(
    image_backend: None, seal_image: np.ndarray, mode: str, case: dict[str, Any]
) -> None:
    """有效裁片保持修改前冻结的尺寸、全部像素和独立存储。"""
    original = seal_image.copy()
    crop = CropByPolys(mode)(seal_image, [np.asarray(case["points"])])[0]
    assert crop is not None
    assert list(crop.shape) == case["shape"]
    assert hashlib.sha256(crop.tobytes()).hexdigest() == case["sha256"]
    np.testing.assert_array_equal(seal_image, original)
    assert not np.shares_memory(crop, seal_image)


def test_valid_crop_does_not_hide_unrelated_warp_errors(seal_image: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    """尺寸有效时共享内核的其他错误继续抛出，避免宽泛回退掩盖真实故障。"""
    monkeypatch.setattr(seal_crop, "warp_image", Mock(side_effect=ValueError("unexpected warp error")))
    with pytest.raises(ValueError, match="unexpected warp error"):
        CropByPolys().get_rotate_crop_image(seal_image, np.asarray(_REFERENCE["valid_cases"][0]["points"], dtype=np.float32))


def _build_seal_model(engine: str, boxes: list[np.ndarray], recognizer: Mock, debug_dir: Path) -> Any:
    """只替换检测和识别推理，保留两个引擎的真实裁剪、过滤与结果封装。"""
    if engine == "torch":
        pytest.importorskip("torch")
        from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR

        model = PytorchPaddleOCR.__new__(PytorchPaddleOCR)
        model._seal_debug_counter = 0
        model._seal_debug_dir = str(debug_dir)
    else:
        from mineru.model.ocr.pp_ocr_v6_onnx import PPOCRv6ONNX

        model = PPOCRv6ONNX.__new__(PPOCRv6ONNX)
    model.is_seal = True
    model.enable_merge_det_boxes = False
    model.drop_score = 0
    model.text_detector = Mock(return_value=(boxes, 0.0))
    model.text_recognizer = recognizer
    model._seal_sort_boxes = SortPolyBoxes()
    model._seal_crop_by_polys = CropByPolys("poly")
    return model


@pytest.mark.parametrize("engine", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
def test_seal_recognition_keeps_boxes_and_debug_outputs_aligned(seal_image: np.ndarray, engine: str, tmp_path: Path) -> None:
    """跳过中间真实无效轮廓后，后一个识别文字和调试裁片仍对应最后的检测框。"""
    first = np.asarray(_REFERENCE["valid_cases"][0]["points"])
    last = first + 70
    invalid = np.asarray(_REFERENCE["degenerate_polygon"])
    texts = [("第一段", 0.8), ("第二段", 0.7)]
    recognizer = Mock(return_value=(texts, 0.0))
    model = _build_seal_model(engine, [first, invalid, last], recognizer, tmp_path)
    assert model.ocr(seal_image) == [[[first.tolist(), texts[0]], [last.tolist(), texts[1]]]]
    crops = recognizer.call_args.args[0]
    assert len(crops) == 2 and all(crop.size > 0 for crop in crops)
    if engine == "torch":
        sample_dir = tmp_path / "sample_0000"
        records = json.loads((sample_dir / "meta.json").read_text())
        assert [(record["index"], record["text"]) for record in records] == [(0, "第一段"), (1, "第二段")]
        for record, crop in zip(records, crops, strict=True):
            with Image.open(sample_dir / record["crop_path"]) as saved:
                np.testing.assert_array_equal(np.asarray(saved), crop[:, :, ::-1])


@pytest.mark.parametrize("engine", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
def test_all_invalid_seal_crops_skip_recognition(seal_image: np.ndarray, engine: str, tmp_path: Path) -> None:
    """全部轮廓无效时外层 OCR 返回既有空结果，并且不执行识别模型。"""
    invalid = np.asarray(_REFERENCE["degenerate_polygon"])
    recognizer = Mock(side_effect=AssertionError("recognizer must not receive invalid crops"))
    model = _build_seal_model(engine, [invalid, invalid.copy()], recognizer, tmp_path)
    assert model.ocr(seal_image) == [None]
    recognizer.assert_not_called()


@pytest.mark.torch
@pytest.mark.parametrize("all_invalid", [False, True])
def test_torch_detection_only_keeps_debug_crops_aligned(seal_image: np.ndarray, tmp_path: Path, all_invalid: bool) -> None:
    """Torch 仅检测入口实际执行裁剪时，同步过滤返回框及调试图，空裁片也能安全输出。"""
    invalid = np.asarray(_REFERENCE["degenerate_polygon"])
    valid = np.asarray(_REFERENCE["valid_cases"][0]["points"])
    boxes = [invalid] if all_invalid else [valid, invalid]
    recognizer = Mock(side_effect=AssertionError("detection-only must not call recognizer"))
    model = _build_seal_model("torch", boxes, recognizer, tmp_path)
    assert model.ocr(seal_image, rec=False) == ([[]] if all_invalid else [[valid.tolist()]])
    recognizer.assert_not_called()
    records = json.loads((tmp_path / "sample_0000/meta.json").read_text())
    assert len(records) == (0 if all_invalid else 1)
