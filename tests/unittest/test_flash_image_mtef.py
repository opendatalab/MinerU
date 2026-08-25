from __future__ import annotations

import pytest

from mineru.model.flash.office.legacy.errors import LegacyOfficeResourceLimitError
from mineru.model.flash.office import image_equation as image_equation_module
from mineru.model.flash.office.image_equation import (
    OfficeImageEquationDecoder,
    decode_image_embedded_equation,
)

from _image_mtef_test_utils import (
    apps_mfcc_comment,
    apps_mfcc_comments,
    baseline_wmf_comment,
    build_baseline_only_gif,
    build_gif_with_extensions,
    build_gif_with_mtef,
    build_wmf,
    gif_mtef_extension,
    pre6_wmf_comment,
)
from _mtef_test_utils import formula_corpus
from _mtef_v5_test_utils import v5_equation, v5_formula_corpus, v5_text


@pytest.mark.parametrize(
    ("mtef", "expected"),
    [
        (formula_corpus()[1][1], formula_corpus()[1][2]),
        (v5_formula_corpus()[1][1], v5_formula_corpus()[1][2]),
    ],
    ids=["v3", "v5"],
)
def test_pre6_wmf_comment_decodes_mtef_versions(
    mtef: bytes,
    expected: str,
) -> None:
    """验证带/不带 placeable header 的 pre-6 WMF 可恢复 v3/v5。"""

    comment = pre6_wmf_comment(mtef)

    assert decode_image_embedded_equation(
        build_wmf([comment]),
        part_name="image.wmf",
    ) == expected
    assert decode_image_embedded_equation(
        build_wmf([comment], placeable=True),
        content_type="image/x-wmf",
    ) == expected


@pytest.mark.parametrize(
    "signature",
    [
        "Design Science, Inc./MTEF",
        "Wiris/MTEF/MathType7",
        "Acme/MTEF",
        "Design Science, Inc.",
        "Wiris",
    ],
)
def test_apps_mfcc_single_and_multi_chunk_signatures(signature: str) -> None:
    """验证规范及历史 signature 的单/多 chunk AppsMFCC。"""

    _name, mtef, expected = v5_formula_corpus()[2]

    single = build_wmf(
        apps_mfcc_comments(
            mtef,
            chunk_size=len(mtef),
            signature=signature,
        )
    )
    multiple = build_wmf(
        apps_mfcc_comments(
            mtef,
            chunk_size=7,
            signature=signature,
        )
    )

    assert decode_image_embedded_equation(single) == expected
    assert decode_image_embedded_equation(multiple) == expected


def test_apps_mfcc_reassembles_mtef_larger_than_32k() -> None:
    """验证 AppsMFCC 可跨 WMF 单 comment 上限重组大型 MTEF。"""

    mtef = v5_equation(v5_text("x" * 7000))
    assert len(mtef) > 0x7FFE
    image = build_wmf(
        apps_mfcc_comments(
            mtef,
            chunk_size=30_000,
        )
    )

    assert decode_image_embedded_equation(image) == "x" * 7000


@pytest.mark.parametrize(
    ("mtef", "expected"),
    [
        (formula_corpus()[0][1], formula_corpus()[0][2]),
        (v5_formula_corpus()[0][1], v5_formula_corpus()[0][2]),
    ],
    ids=["v3", "v5"],
)
def test_gif_mathtype_001_decodes_across_subblocks(
    mtef: bytes,
    expected: str,
) -> None:
    """验证 GIF MathType/001 跨 sub-block 恢复 v3/v5。"""

    image = build_gif_with_mtef(
        mtef,
        chunk_size=5,
        include_baseline=True,
    )

    assert decode_image_embedded_equation(image) == expected


def test_wmf_gif_decode_full_v3_v5_formula_corpora() -> None:
    """验证两种图片载体复用全部既有 v3/v5 公式语料。"""

    for _name, mtef, expected in formula_corpus():
        assert decode_image_embedded_equation(
            build_wmf([pre6_wmf_comment(mtef)])
        ) == expected
        assert decode_image_embedded_equation(
            build_gif_with_mtef(mtef, chunk_size=5)
        ) == expected
    for _name, mtef, expected in v5_formula_corpus():
        assert decode_image_embedded_equation(
            build_wmf(apps_mfcc_comments(mtef, chunk_size=7))
        ) == expected
        assert decode_image_embedded_equation(
            build_gif_with_mtef(mtef, chunk_size=5)
        ) == expected


def test_baseline_comments_and_ordinary_images_are_ignored() -> None:
    """验证 WMF baseline、GIF/002 和普通图片不被误判为公式。"""

    assert decode_image_embedded_equation(
        build_wmf([baseline_wmf_comment(12)])
    ) is None
    assert decode_image_embedded_equation(build_baseline_only_gif()) is None
    assert decode_image_embedded_equation(b"\x89PNG\r\n\x1a\n") is None


def test_conflicting_wmf_and_gif_candidates_fail_closed() -> None:
    """验证同一图片中互相冲突的公式 candidates 整体回退。"""

    v3 = formula_corpus()[0][1]
    v5 = v5_formula_corpus()[1][1]
    wmf = build_wmf(
        [
            pre6_wmf_comment(v3),
            apps_mfcc_comment(v5, total_length=len(v5)),
        ]
    )
    gif = build_gif_with_extensions(
        [
            gif_mtef_extension(v3),
            gif_mtef_extension(v5),
        ]
    )

    assert decode_image_embedded_equation(wmf) is None
    assert decode_image_embedded_equation(gif) is None


@pytest.mark.parametrize(
    "image",
    [
        build_wmf([pre6_wmf_comment(bytes([4, 1, 0, 3, 5, 0]))]),
        build_wmf(
            [
                apps_mfcc_comment(
                    b"truncated",
                    total_length=100,
                )
            ]
        ),
        build_gif_with_mtef(bytes([4, 1, 0, 3, 5, 0])),
        build_gif_with_mtef(v5_formula_corpus()[0][1])[:-1],
    ],
)
def test_unsupported_or_truncated_image_comments_fail_closed(
    image: bytes,
) -> None:
    """验证 v4、缺 chunk 和截断 GIF 不输出部分公式。"""

    assert decode_image_embedded_equation(image) is None


def test_reordered_apps_chunks_and_strict_image_prefixes_fail_closed() -> None:
    """验证 AppsMFCC 乱序及任意 WMF/GIF 截断不会输出部分公式。"""

    mtef = v5_formula_corpus()[1][1]
    comments = apps_mfcc_comments(mtef, chunk_size=7)
    reordered = build_wmf(list(reversed(comments)))
    valid_wmf = build_wmf(comments)
    valid_gif = build_gif_with_mtef(mtef, chunk_size=5)

    assert decode_image_embedded_equation(reordered) is None
    assert all(
        decode_image_embedded_equation(valid_wmf[:end]) is None
        for end in range(len(valid_wmf))
    )
    assert all(
        decode_image_embedded_equation(valid_gif[:end]) is None
        for end in range(len(valid_gif))
    )


def test_image_equation_decoder_cache_and_total_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证相同图片缓存不重复计费，唯一图片受累计预算限制。"""

    first_mtef = v5_formula_corpus()[0][1]
    second_mtef = v5_formula_corpus()[1][1]
    first = build_gif_with_mtef(first_mtef)
    second = build_gif_with_mtef(second_mtef)
    monkeypatch.setattr(
        image_equation_module,
        "MAX_ASSET_TOTAL_BYTES",
        len(first) + 1,
    )
    decoder = OfficeImageEquationDecoder()

    assert decoder.decode(first) == v5_formula_corpus()[0][2]
    assert decoder.decode(first) == v5_formula_corpus()[0][2]
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_asset_total_bytes"):
        decoder.decode(second)


def test_image_equation_record_limit_raises_stable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 WMF/GIF record 超限抛稳定 resource-limit 错误。"""

    monkeypatch.setattr(image_equation_module, "MAX_PICTURE_RECORDS", 1)
    image = build_gif_with_mtef(v5_formula_corpus()[0][1], chunk_size=1)

    with pytest.raises(LegacyOfficeResourceLimitError, match="max_picture_records"):
        decode_image_embedded_equation(image)
