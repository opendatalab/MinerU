import pytest

from mineru.utils.text_utils import resolve_text_line_boundary


@pytest.mark.parametrize(
    ("previous_content", "next_content", "block_language", "expected"),
    [
        pytest.param(
            "See https://example.test/current-research",
            "/image-processing/",
            "en",
            "See https://example.test/current-research/image-processing/",
            id="url-path-continuation",
        ),
        pytest.param(
            "Download from ftp://example.test/archive",
            "?format=zip",
            "en",
            "Download from ftp://example.test/archive?format=zip",
            id="url-query-continuation",
        ),
        pytest.param(
            "Visit www.example.test/docs",
            "#install",
            "en",
            "Visit www.example.test/docs#install",
            id="url-fragment-continuation",
        ),
        pytest.param(
            "Use https://example.test/search?key=one",
            "&page=2",
            "en",
            "Use https://example.test/search?key=one&page=2",
            id="url-query-parameter-continuation",
        ),
        pytest.param(
            "Use https://example.test/search?key",
            "=value",
            "en",
            "Use https://example.test/search?key=value",
            id="url-query-value-continuation",
        ),
        pytest.param("input", "/ output", "en", "input / output", id="ordinary-slash-text"),
        pytest.param(
            "See https://example.test/complete",
            "Figure 2 follows.",
            "en",
            "See https://example.test/complete Figure 2 follows.",
            id="url-followed-by-prose",
        ),
        pytest.param("inter-", "national", "en", "international", id="western-hyphen"),
        pytest.param("first line", "second line", "en", "first line second line", id="western-space"),
        pytest.param("中文", "继续", "zh", "中文继续", id="cjk-direct-join"),
    ],
)
def test_resolve_text_line_boundary_keeps_conservative_joining_rules(
    previous_content: str,
    next_content: str,
    block_language: str,
    expected: str,
) -> None:
    """验证 URL、普通西文、断词和 CJK 的物理行边界规则互不干扰。"""
    processed_previous, separator = resolve_text_line_boundary(
        previous_content,
        block_language=block_language,
        next_content=next_content,
    )

    assert f"{processed_previous}{separator}{next_content}" == expected
