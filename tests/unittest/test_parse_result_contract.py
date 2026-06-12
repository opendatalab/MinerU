from dataclasses import fields

from mineru.parser.base import ParseResult


def test_parse_result_does_not_expose_backend_version_or_file_name() -> None:
    field_names = {field.name for field in fields(ParseResult)}

    assert "_backend" not in field_names
    assert "_version_name" not in field_names
    assert "_file_name" not in field_names
