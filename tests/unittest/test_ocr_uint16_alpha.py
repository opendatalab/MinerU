"""Preserve 16-bit PNG colors when preparing the 8-bit OCR input."""

from __future__ import annotations

import struct
import zlib

import numpy as np

from mineru.model.ocr.image import alpha_to_color, check_img, preprocess_image


def _png16_bgra(image: np.ndarray) -> bytes:
    """Encode a minimal unfiltered 16-bit RGBA PNG without an optional imaging library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    height, width, _ = image.shape
    rgba = image[:, :, [2, 1, 0, 3]].astype(">u2")
    scanlines = b"".join(b"\0" + row.tobytes() for row in rgba)
    header = struct.pack(">IIBBBBB", width, height, 16, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")


def test_native_uint16_png_composites_correct_colors_on_white() -> None:
    """Real native PNG decoding feeds transparent, half-transparent and opaque pixels to OCR."""
    source = np.array([[[0, 0, 65535, 32768], [10000, 20000, 30000, 0], [25700, 12850, 5140, 65535]]], dtype=np.uint16)
    decoded = check_img(_png16_bgra(source))
    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, source)
    output = preprocess_image(decoded)
    np.testing.assert_array_equal(output, np.array([[[127, 127, 255], [255, 255, 255], [100, 50, 20]]], dtype=np.uint8))


def test_uint16_nonwhite_background_and_read_only_input() -> None:
    """Color arguments retain their RGB convention and input storage remains unchanged."""
    source = np.array([[[0, 0, 65535, 0], [65535, 0, 0, 32768], [0, 0, 0, 65535]]], dtype=np.uint16)
    original = source.copy()
    source.flags.writeable = False
    result = alpha_to_color(source, (10, 20, 30))
    np.testing.assert_array_equal(result, np.array([[[30, 20, 10], [142, 9, 4], [0, 0, 0]]], dtype=np.uint8))
    np.testing.assert_array_equal(source, original)
    assert not np.shares_memory(result, source)


def test_uint8_alpha_keeps_existing_rounding() -> None:
    """Existing eight-bit OCR input retains its truncation and BGR channel order."""
    source = np.array([[[12, 34, 56, 0], [12, 34, 56, 128], [12, 34, 56, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        alpha_to_color(source), np.array([[[255, 255, 255], [133, 144, 155], [12, 34, 56]]], dtype=np.uint8)
    )


def test_image_without_alpha_is_returned_unchanged() -> None:
    """This change only affects the alpha compositing path."""
    source = np.zeros((4, 5, 3), dtype=np.uint16)
    assert alpha_to_color(source) is source
