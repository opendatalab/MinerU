# Copyright (c) Opendatalab. All rights reserved.

import pytest

from mineru.backend.utils.markdown_utils import escape_text_block_markdown_prefix


@pytest.mark.parametrize(
    "content, expected",
    [
        # Block quote: CommonMark does not require a space after ">".
        ("> 5 mV was measured at the gate.", "\\> 5 mV was measured at the gate."),
        (">5 mV was measured at the gate.", "\\>5 mV was measured at the gate."),
        ("  > up to three spaces of indent still count", "  \\> up to three spaces of indent still count"),
        # Ordered list: the backslash goes before the delimiter, since "\1." is
        # not a Markdown escape but "1\." is.
        ("1. Smith, J. et al. Nature 2020.", "1\\. Smith, J. et al. Nature 2020."),
        ("2) Second numbered reference.", "2\\) Second numbered reference."),
        ("1986. A landmark year for the field.", "1986\\. A landmark year for the field."),
        # Already covered before this change.
        ("## Section title", "\\## Section title"),
        ("- bullet item", "\\- bullet item"),
        ("+ bullet item", "\\+ bullet item"),
    ],
)
def test_leading_block_marker_is_escaped(content, expected):
    assert escape_text_block_markdown_prefix(content) == expected


@pytest.mark.parametrize(
    "content",
    [
        "",
        "Plain sentence with no marker.",
        # No space after the delimiter, so this is not an ordered list.
        "1.5 mm is the tolerance.",
        # CommonMark caps an ordered list marker at nine digits.
        "1234567890. ten digits is not a list marker",
        # The marker has to be at the start of the block.
        "100 > 50 in this comparison",
        # Four or more leading spaces is an indented code block, not our concern.
        "     > deeply indented",
    ],
)
def test_text_without_a_block_marker_is_untouched(content):
    assert escape_text_block_markdown_prefix(content) == content
