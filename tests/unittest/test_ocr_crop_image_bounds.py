"""Out-of-image text boxes must use border sampling instead of NumPy wrapping."""

from __future__ import annotations

import numpy as np
import pytest
from docvortex.image import perspective_matrix, warp_image

from mineru.model.ocr.image import get_rotate_crop_image


@pytest.mark.parametrize(
    "bbox",
    [(-2, 1, 11, 6), (1, -2, 8, 9), (5, 1, 13, 6), (1, 5, 8, 11)],
)
def test_axis_aligned_outside_box_uses_requested_extent_and_border_pixels(bbox: tuple[int, int, int, int]) -> None:
    """Each image edge can be crossed without cropping the opposite side or shortening the text."""
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    original = image.copy()
    x0, y0, x1, y1 = bbox
    points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    width, height = x1 - x0, y1 - y0
    target = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    expected = warp_image(image, perspective_matrix(points, target), (width, height), border="replicate", interpolation="cubic")
    if height / width >= 1.5:
        expected = np.rot90(expected)
    actual = get_rotate_crop_image(image, points)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(actual, image)


def test_negative_start_does_not_select_opposite_edge() -> None:
    """A red right edge cannot appear at the start of a crop extending left of the source."""
    image = np.zeros((6, 10, 3), dtype=np.uint8)
    image[:, :8] = [10, 20, 30]
    image[:, 8:] = [200, 210, 220]
    points = np.float32([[-2, 0], [11, 0], [11, 6], [-2, 6]])
    actual = get_rotate_crop_image(image, points)
    assert actual.shape == (6, 13, 3)
    np.testing.assert_array_equal(actual[:, :3], np.broadcast_to([10, 20, 30], (6, 3, 3)))


def test_in_bounds_box_retains_direct_exact_pixel_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary in-image fast path remains an exact copy without a perspective warp."""
    from mineru.model.ocr import image as module

    def unexpected_warp(*args: object, **kwargs: object) -> None:
        pytest.fail("In-bounds axis-aligned crops must keep their direct-copy path")

    monkeypatch.setattr(module, "warp_image", unexpected_warp)
    image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    points = np.float32([[1, 1], [8, 1], [8, 6], [1, 6]])
    np.testing.assert_array_equal(get_rotate_crop_image(image, points), image[1:6, 1:8])
