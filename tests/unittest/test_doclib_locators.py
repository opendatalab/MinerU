import pytest

from mineru.doclib.locators import block_char_ref, block_ref, locator_for_block, page_ref, parse_content_cursor


def test_doclib_locator_helpers_use_public_one_based_numbers() -> None:
    assert locator_for_block(1, 3) == "page:1/block:3"
    assert page_ref("ab12cd3", "high", 1) == "doc:ab12cd3/tier:high/page:1"
    assert block_ref("ab12cd3", "high", 1, 3) == "doc:ab12cd3/tier:high/page:1/block:3"
    assert block_char_ref("ab12cd3", "high", 1, 3, 0) == "doc:ab12cd3/tier:high/page:1/block:3/char:0"


def test_parse_content_cursor_supports_page_block_and_char_refs() -> None:
    page = parse_content_cursor("doc:ab12cd3/tier:high/page:5")
    block = parse_content_cursor("doc:ab12cd3/tier:high/page:5/block:18")
    char = parse_content_cursor("doc:ab12cd3/tier:high/page:5/block:18/char:32784")

    assert page.page_no == 5
    assert page.block_no is None
    assert block.block_no == 18
    assert block.char_offset is None
    assert char.char_offset == 32784


def test_content_cursor_rejects_zero_based_public_numbers() -> None:
    with pytest.raises(ValueError):
        locator_for_block(0, 1)
    with pytest.raises(ValueError):
        parse_content_cursor("doc:ab12cd3/tier:high/page:0")
    with pytest.raises(ValueError):
        parse_content_cursor("doc:ab12cd3/tier:high/page:1/block:0")
