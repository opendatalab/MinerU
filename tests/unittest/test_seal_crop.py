import numpy as np
import pytest

from mineru.model.ocr.seal_crop import CropByPolys


@pytest.mark.parametrize("det_box_type", ["quad", "poly"])
def test_crop_by_polys_skips_zero_area_boxes_and_keeps_results_aligned(det_box_type: str) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    degenerate_box = np.full((4, 2), 4, dtype=np.int32)
    first_box = np.array([[2, 2], [10, 2], [10, 8], [2, 8]], dtype=np.int32)
    last_box = np.array([[3, 10], [12, 10], [12, 17], [3, 17]], dtype=np.int32)

    boxes, crops = CropByPolys(det_box_type=det_box_type)(image, [first_box, degenerate_box, last_box])

    assert len(boxes) == len(crops) == 2
    np.testing.assert_array_equal(boxes[0], first_box)
    np.testing.assert_array_equal(boxes[1], last_box)
    assert all(crop.size > 0 for crop in crops)


@pytest.mark.parametrize("det_box_type", ["quad", "poly"])
def test_crop_by_polys_returns_empty_results_when_all_boxes_are_zero_area(det_box_type: str) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    degenerate_box = np.full((4, 2), 4, dtype=np.int32)

    boxes, crops = CropByPolys(det_box_type=det_box_type)(image, [degenerate_box])

    assert boxes == []
    assert crops == []
