# Copyright (c) Opendatalab. All rights reserved.

from mineru.backend.utils.markdown_utils import escape_conservative_markdown_text


def test_escape_html_like_placeholders_in_plain_text() -> None:
    content = "-a <address>, then -p <port>"

    assert escape_conservative_markdown_text(content) == (
        r"-a \<address>, then -p \<port>"
    )


def test_preserve_existing_angle_bracket_escape() -> None:
    content = r"-a \<address>"

    assert escape_conservative_markdown_text(content) == content


def test_escape_angle_bracket_after_even_backslashes() -> None:
    content = r"-a \\<address>"

    assert escape_conservative_markdown_text(content) == r"-a \\\<address>"


def test_preserve_existing_conservative_markdown_escaping() -> None:
    content = "*value* _name_ `code` ~old~ $x$"

    assert escape_conservative_markdown_text(content) == (
        r"\*value\* \_name\_ \`code\` \~old\~ \$x\$"
    )
