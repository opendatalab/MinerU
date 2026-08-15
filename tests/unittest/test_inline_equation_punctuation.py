import pytest

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import _render_span
from mineru.utils.enum_class import ContentType


def _inline_equation(content: str) -> tuple[str, str] | None:
    return _render_span({
        'type': ContentType.INLINE_EQUATION,
        'content': content,
    })


@pytest.mark.parametrize(
    ('content', 'expected'),
    [
        (r'P(G\mid ¬U),', r'$P(G\mid ¬U)$,'),
        ('f(x).', '$f(x)$.'),
        ('f(x)，', '$f(x)$，'),
        ('f(x)。', '$f(x)$。'),
    ],
)
def test_render_inline_equation_moves_sentence_punctuation_outside_math(
    content: str,
    expected: str,
) -> None:
    assert _inline_equation(content) == (ContentType.INLINE_EQUATION, expected)


def test_render_inline_equation_keeps_internal_punctuation() -> None:
    assert _inline_equation('f(x,y)') == (
        ContentType.INLINE_EQUATION,
        '$f(x,y)$',
    )
    assert _inline_equation('n!') == (ContentType.INLINE_EQUATION, '$n!$')


def test_render_inline_equation_does_not_create_empty_math() -> None:
    assert _inline_equation(',') == (ContentType.INLINE_EQUATION, '$,$')
    assert _inline_equation('') is None
