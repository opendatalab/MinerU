"""正文 OCR 在公式遮罩拆框之后的零尺寸裁图回归。"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from mineru.model.ocr.geometry import update_det_boxes
from mineru.model.ocr.image import get_rotate_crop_image_for_text_rec
from mineru.model.ocr.results import get_ocr_result_list


@pytest.mark.parametrize(
    "points",
    (
        [[4, 4], [4, 4], [4, 12], [4, 12]],
        [[4, 4], [16, 4], [16, 4], [4, 4]],
        [[4, 4], [16, 6], [16, 6.5], [4, 4.5]],
    ),
)
def test_zero_pixel_text_crop_is_discarded(points: list[list[float]]) -> None:
    """零宽、零高及倾斜亚像素高度不能进入透视采样或后续 OCR rec。"""
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    assert get_rotate_crop_image_for_text_rec(image, np.asarray(points, dtype=np.float32)) is None
    assert get_ocr_result_list([points], [0, 0, 0, 0, 24, 24, 24, 24], True, image) == []


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


@pytest.mark.parametrize("mask_right", (10, 20))
def test_torch_ocr_discards_invalid_crops_without_text_misalignment(mask_right: int) -> None:
    """Torch 整体 OCR 同步过滤零宽框和裁图，全部被过滤时不调用识别器。"""
    from mineru.model.ocr.pytorch_paddle import PytorchPaddleOCR

    model = object.__new__(PytorchPaddleOCR)
    model.is_seal = False
    model.enable_merge_det_boxes = False
    model.drop_score = 0.5
    model.text_detector = Mock(return_value=(np.asarray([[[4, 4], [20, 4], [20, 12], [4, 12]]]), 0.0))
    model.text_recognizer = Mock(return_value=([("valid", 0.99)], 0.0))
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    boxes, texts = model(image, mfd_res=[{"bbox": [5, 4, mask_right, 12]}])
    if mask_right == 20:
        assert boxes == texts == []
        model.text_recognizer.assert_not_called()
    else:
        assert len(boxes) == 1 and texts == [("valid", 0.99)]
        np.testing.assert_array_equal(boxes[0], [[11, 4], [20, 4], [20, 12], [11, 12]])
        assert len(model.text_recognizer.call_args.args[0]) == 1
