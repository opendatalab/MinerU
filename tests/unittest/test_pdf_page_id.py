from __future__ import annotations

import pytest

from mineru.utils.pdf_page_id import parse_page_range


def test_parse_page_range_empty_means_all_pages() -> None:
    assert parse_page_range("", 5) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("raw", ["6", "6~8", "0", "-20~-10"])
def test_parse_page_range_explicit_out_of_range_does_not_fallback_to_all_pages(raw: str) -> None:
    assert parse_page_range(raw, 5) == []


def test_parse_page_range_explicit_partial_out_of_range_keeps_valid_intersection() -> None:
    assert parse_page_range("3~8", 5) == [2, 3, 4]


def test_parse_page_range_expands_deduplicates_and_sorts_available_pages() -> None:
    assert parse_page_range("1~10,3,4~5", 5) == [0, 1, 2, 3, 4]


def test_parse_page_range_negative_one_is_last_page() -> None:
    assert parse_page_range("-2~-1", 5) == [3, 4]
