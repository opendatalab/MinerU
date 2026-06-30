import math

from mineru.utils.span_pre_proc import _is_supported_rotation


def test_pdf_italic_skew_is_supported_for_text_backfill():
    assert _is_supported_rotation(math.radians(19.1))


def test_diagonal_rotation_is_not_supported_for_text_backfill():
    assert not _is_supported_rotation(math.radians(45))
