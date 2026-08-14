import pytest

from mineru.backend.utils.formula_number import (
    build_tagged_formula_content,
    normalize_formula_tag_content,
)


@pytest.mark.parametrize(
    ("raw_tag", "expected"),
    [
        ("(3)", "3"),
        ("（３）", "3"),
        ("......(3)", "3"),
        ("......（３）", "3"),
        ("A.3", "A.3"),
    ],
)
def test_normalize_formula_tag_content(raw_tag: str, expected: str) -> None:
    assert normalize_formula_tag_content(raw_tag) == expected


def test_build_tagged_formula_content_drops_dot_leader() -> None:
    assert build_tagged_formula_content("E=x", {"content": "......(3)"}) == (r"E=x\tag{3}")
