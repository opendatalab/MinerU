from __future__ import annotations

from rich.table import Table
from rich.text import Text

from mineru.cli import output


MARKUP_LIKE_TEXT = "MinerU home [/Users/jinzhenj/.mineru] [bold] [/invalid]"


def test_stdout_console_renders_markup_like_strings_literally() -> None:
    with output.console.capture() as capture:
        output.console.print(MARKUP_LIKE_TEXT)

    assert MARKUP_LIKE_TEXT in capture.get()


def test_stderr_notice_renders_markup_like_strings_literally() -> None:
    with output.stderr_console.capture() as capture:
        output.print_notice(MARKUP_LIKE_TEXT)

    assert MARKUP_LIKE_TEXT in capture.get()


def test_rich_table_renders_markup_like_title_and_cell_literally() -> None:
    table = Table(title=MARKUP_LIKE_TEXT)
    table.add_column("Value")
    table.add_row(MARKUP_LIKE_TEXT)

    with output.console.capture() as capture:
        output.print_rich(table)

    rendered = capture.get()
    assert "[/Users/jinzhenj/.mineru]" in rendered
    assert "[bold]" in rendered
    assert "[/invalid]" in rendered


def test_explicit_rich_text_remains_renderable() -> None:
    text = Text("styled", style="green")

    with output.console.capture() as capture:
        output.print_rich(text)

    assert "styled" in capture.get()
