"""正文 OCR 在公式遮罩拆框之后的零尺寸裁图回归。"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from mineru.model.ocr import image as ocr_image
from mineru.model.ocr.geometry import update_det_boxes
from mineru.model.ocr.image import get_rotate_crop_image, get_rotate_crop_image_for_text_rec
from mineru.model.ocr.results import get_ocr_result_list


@pytest.mark.parametrize(
    "points",
    (
        pytest.param([[4, 4], [4, 4], [4, 12], [4, 12]], id="zero-width"),
        pytest.param([[4, 4], [16, 4], [16, 4], [4, 4]], id="zero-height"),
        pytest.param([[4, 4], [16, 6], [16, 6.5], [4, 4.5]], id="inclined-subpixel-height"),
        pytest.param([[4, 4], [4.4, 4], [4.4, 12], [4, 12]], id="subpixel-width"),
        pytest.param([[4, 4], [16, 4], [16, 4.4], [4, 4.4]], id="subpixel-height"),
        pytest.param([[4.8, 4], [5.2, 4], [5.2, 12], [4.8, 12]], id="subpixel-width-crosses-integer"),
        pytest.param([[4, 4.8], [16, 4.8], [16, 5.2], [4, 5.2]], id="subpixel-height-crosses-integer"),
    ),
)
def test_zero_pixel_text_crop_is_discarded(points: list[list[float]], monkeypatch: pytest.MonkeyPatch) -> None:
    """两个裁图入口均拒绝退化框，跨整数边界也不能绕过检查进入采样或 OCR。"""
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    matrix = Mock(side_effect=AssertionError("invalid crop must not create a perspective matrix"))
    warp = Mock(side_effect=AssertionError("invalid crop must not reach warp_image"))
    monkeypatch.setattr(ocr_image, "perspective_matrix", matrix)
    monkeypatch.setattr(ocr_image, "warp_image", warp)
    quad = np.asarray(points, dtype=np.float32)
    assert get_rotate_crop_image(image, quad) is None
    assert get_rotate_crop_image_for_text_rec(image, quad) is None
    assert get_ocr_result_list([points], [0, 0, 0, 0, 24, 24, 24, 24], True, image) == []
    matrix.assert_not_called()
    warp.assert_not_called()


@pytest.mark.parametrize(
    ("bbox", "rotate_for_rec"),
    (
        pytest.param((4, 4, 5, 16), True, id="one-pixel-width"),
        pytest.param((4, 4, 16, 5), False, id="one-pixel-height"),
        pytest.param((4, 4, 16, 12), False, id="horizontal"),
        pytest.param((4, 4, 8, 16), True, id="vertical"),
    ),
)
def test_valid_text_crops_keep_pixels_and_source_ownership(bbox: tuple[int, int, int, int], rotate_for_rec: bool) -> None:
    """有效单像素及横竖裁图保留原始像素和转正规则，支持只读输入且不共享源图存储。"""
    image = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    original = image.copy()
    image.flags.writeable = False
    x0, y0, x1, y1 = bbox
    points = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    original_points = points.copy()
    points.flags.writeable = False
    expected = original[y0:y1, x0:x1]

    crop = get_rotate_crop_image(image, points)
    rec_crop = get_rotate_crop_image_for_text_rec(image, points)
    assert crop is not None and rec_crop is not None
    np.testing.assert_array_equal(crop, expected)
    np.testing.assert_array_equal(rec_crop, np.rot90(expected) if rotate_for_rec else expected)
    assert not np.shares_memory(crop, image) and not np.shares_memory(rec_crop, image)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(points, original_points)


def test_valid_text_crop_does_not_hide_unrelated_warp_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """有效倾斜框的其他采样错误继续抛出，不能被退化裁图处理吞掉。"""
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    points = np.asarray([[4, 4], [16, 6], [16, 14], [4, 12]], dtype=np.float32)
    monkeypatch.setattr(ocr_image, "warp_image", Mock(side_effect=ValueError("unexpected warp error")))
    with pytest.raises(ValueError, match="unexpected warp error"):
        get_rotate_crop_image_for_text_rec(image, points)


@pytest.mark.parametrize("recognized", (False, True))
@pytest.mark.parametrize("ocr_enable", (False, True))
def test_ocr_result_filter_keeps_thresholds_and_read_only_input(recognized: bool, ocr_enable: bool) -> None:
    """合并过滤后仍保留宽度和置信度边界、结果字段及只读源图的裁片像素。"""
    image = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    original = image.copy()
    image.flags.writeable = False
    narrow = [[4, 4], [6, 4], [6, 8], [4, 8]]
    valid = [[10, 4], [13, 4], [13, 8], [10, 8]]
    ocr_res = (
        [[narrow, ("narrow", 0.9)], [valid, ("valid", 0.876)], [valid, ("low-score", 0.49)]] if recognized else [narrow, valid]
    )
    output = get_ocr_result_list(ocr_res, [0, 0, 0, 0, 24, 24, 24, 24], ocr_enable, image)
    assert len(output) == 1
    item = output[0].copy()
    crop = item.pop("np_img", None)
    assert item == {
        "label": "ocr_text",
        "bbox": (10, 4, 13, 8),
        "score": 0.88 if recognized and not ocr_enable else 1.0,
        "text": "valid" if recognized else "",
    }
    if ocr_enable and not recognized:
        assert crop is not None and not np.shares_memory(crop, image)
        np.testing.assert_array_equal(crop, original[4:8, 10:13])
    else:
        assert crop is None
    np.testing.assert_array_equal(image, original)


def test_formula_split_zero_width_fragment_preserves_valid_text() -> None:
    """公式遮罩留下闭区间端点时丢弃零宽残片，合法正文的坐标和裁图保持原样。"""
    image = np.arange(24 * 24 * 3, dtype=np.uint8).reshape(24, 24, 3)
    boxes = update_det_boxes(
        [[[4, 4], [20, 4], [20, 12], [4, 12]]],
        [{"bbox": [5, 4, 10, 12]}],
    )
    assert boxes[0][0][0] == boxes[0][2][0] == 4
    output = get_ocr_result_list(boxes, [0, 0, 0, 0, 24, 24, 24, 24], True, image)
    assert len(output) == 1 and output[0]["bbox"] == (11, 4, 20, 12)
    assert output[0]["text"] == "" and output[0]["score"] == 1.0
    np.testing.assert_array_equal(output[0]["np_img"], image[4:12, 11:20])


def _build_text_ocr_model(engine: str, boxes: np.ndarray, texts: list[tuple[str, float]]) -> Any:
    """仅替换检测和识别模型，保留两个后端的真实排序、裁图、过滤和结果组装。"""
    if engine == "torch":
        pytest.importorskip("torch")
        from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR

        model = object.__new__(PytorchPaddleOCR)
    else:
        from mineru.model.ocr.pp_ocr_v6_onnx import PPOCRv6ONNX

        model = object.__new__(PPOCRv6ONNX)
    model.is_seal = False
    model.enable_merge_det_boxes = False
    model.drop_score = 0.5
    model.text_detector = Mock(return_value=(boxes, 0.0))
    model.text_recognizer = Mock(return_value=(texts, 0.0))
    return model


@pytest.mark.parametrize("engine", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
@pytest.mark.parametrize("mask_right", (10, 20))
def test_formula_split_ocr_keeps_text_alignment(engine: str, mask_right: int) -> None:
    """两个后端同步过滤公式拆框后的零宽裁片，全部被过滤时不调用识别器。"""
    boxes = np.asarray([[[4, 4], [20, 4], [20, 12], [4, 12]]], dtype=np.float32)
    model = _build_text_ocr_model(engine, boxes, [("valid", 0.99)])
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    result = model.ocr(image, mfd_res=[{"bbox": [5, 4, mask_right, 12]}])
    if mask_right == 20:
        assert result == [None]
        model.text_recognizer.assert_not_called()
    else:
        assert result == [[[[[11, 4], [20, 4], [20, 12], [11, 12]], ("valid", 0.99)]]]
        assert len(model.text_recognizer.call_args.args[0]) == 1


@pytest.mark.parametrize("engine", [pytest.param("torch", marks=pytest.mark.torch), "onnx"])
@pytest.mark.parametrize("all_invalid", (False, True))
def test_text_ocr_keeps_alignment_after_subpixel_crop_filter(engine: str, all_invalid: bool) -> None:
    """跳过中间亚像素裁片后文字仍对应前后检测框，全无效输入保持空结果并跳过识别。"""
    first = np.asarray([[4, 4], [12, 4], [12, 12], [4, 12]], dtype=np.float32)
    invalid = np.asarray([[13, 4], [13.4, 4], [13.4, 12], [13, 12]], dtype=np.float32)
    last = first + [16, 0]
    boxes = np.asarray([invalid] if all_invalid else [first, invalid, last])
    texts = [] if all_invalid else [("first", 0.99), ("last", 0.98)]
    model = _build_text_ocr_model(engine, boxes, texts)
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    result = model.ocr(image)
    if all_invalid:
        assert result == [None]
        model.text_recognizer.assert_not_called()
    else:
        assert result == [[[first.tolist(), texts[0]], [last.tolist(), texts[1]]]]
        crops = model.text_recognizer.call_args.args[0]
        assert len(crops) == 2
        np.testing.assert_array_equal(crops[0], image[4:12, 4:12])
        np.testing.assert_array_equal(crops[1], image[4:12, 20:28])
