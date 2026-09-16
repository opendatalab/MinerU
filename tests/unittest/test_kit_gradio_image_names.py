"""原始页索引、所属父块和表内图片序号的语义命名回归。"""

from __future__ import annotations

import base64
import json
import zipfile
from pathlib import Path

import pytest
from bs4 import BeautifulSoup
from PIL import Image, ImageStat

from mineru.kit.gradio.artifacts import (
    _build_image_context,
    _close_image_context,
    _materialize_middle_json,
    _write_materialized_asset,
    persist_parse_result,
    render_download,
)
from mineru.parser.base import ParseResult
from mineru.types import (
    AlgorithmBodyBlock,
    BlockType,
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
    CodeBodyBlock,
    EquationBlock,
    TableBlock,
    TableBodyBlock,
    TextSpan,
)
from test_kit_gradio import _colored_pdf_bytes, _middle_json
from test_kit_gradio_html import _image_bytes


def _image_uri(color: str) -> str:
    """生成有效 PNG data URI，供不同语义位置的命名测试复用。"""
    return "data:image/png;base64," + base64.b64encode(_image_bytes(color)).decode()


@pytest.mark.parametrize("page_indices,page_range", [((1, 2), "2-3"), ((1, 3), "2,4")])
def test_crops_use_original_pages_and_parent_type(tmp_path: Path, page_indices: tuple[int, ...], page_range: str) -> None:
    """两页使用相同块索引时，裁后页号不得覆盖原始页号对应的图片。"""
    source = tmp_path / "colors.pdf"
    source.write_bytes(_colored_pdf_bytes([(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 1)]))
    middle = _middle_json(page_indices=page_indices)
    for page in middle.pages:
        text, image = page.blocks
        body = image.content[0].model_copy(update={"index": 1})
        page.blocks = [text.model_copy(update={"index": 0}), image.model_copy(update={"index": 1, "content": [body]})]
    original = middle.model_dump_json()
    artifacts = persist_parse_result(
        ParseResult(middle_json=middle), source, output_root=tmp_path / "output", page_range=page_range
    )
    assert middle.model_dump_json() == original
    saved = ParseResult.from_json(artifacts.middle_json_path.read_text())
    expected = {f"page_{page_idx}_image_1.jpg" for page_idx in page_indices}
    assert {path.name for path in (artifacts.root / "images").iterdir()} == expected
    for page, channel in zip(saved.pages, (1, 2)):
        image_path = page.blocks[1].content[0].image_path
        assert image_path == f"images/page_{page.page_idx}_image_1.jpg"
        with Image.open(artifacts.root / image_path) as image:
            mean = ImageStat.Stat(image.convert("RGB")).mean
        assert mean[channel] > 240 and sum(value for index, value in enumerate(mean) if index != channel) < 20

    # 单独读取裁图缓存也不应产生任何临时图片文件。
    crop_root = tmp_path / "crop-only"
    context = _build_image_context(saved.middle_json, artifacts.origin_pdf_path, crop_root, page_indices=page_indices)
    try:
        block = saved.pages[0].blocks[1].content[0]
        first = context.crop_for_block(block, middle_page_idx=page_indices[0])
        assert first and context.crop_for_block(block, middle_page_idx=page_indices[0]) is first
        assert not crop_root.exists()
    finally:
        _close_image_context(context)


def test_parent_names_table_images_and_repeated_materialization(tmp_path: Path) -> None:
    """截图按父块命名，表内多图按位置编号，相同图片内容不合并不同语义名称。"""
    source = tmp_path / "sample.docx"
    source.write_bytes(b"source")
    middle = _middle_json(file_suffix="docx")
    middle.pages[0].blocks[1].content[0].image_base64 = _image_uri("red")
    middle.pages[0].blocks[1].content[0].content = f'<img src="{_image_uri("blue")}">'
    middle.pages[0].blocks.extend(
        [
            TableBlock(
                type=BlockType.TABLE,
                index=2,
                content=[
                    TableBodyBlock(
                        type=BlockType.TABLE_BODY,
                        index=2,
                        image_base64=_image_uri("red"),
                        content=(
                            f'<table><tr><td><img src="{_image_uri("blue")}"></td>'
                            f'<td><img src="{_image_uri("blue")}"></td></tr></table>'
                        ),
                    ),
                ],
            ),
            ChartBlock(
                type=BlockType.CHART,
                index=3,
                content=[
                    ChartBodyBlock(type=BlockType.CHART_BODY, index=3, image_base64=_image_uri("red"), content=""),
                ],
            ),
            EquationBlock(type=BlockType.EQUATION, index=4, image_base64=_image_uri("red"), content="x^2"),
        ]
    )
    artifacts = persist_parse_result(ParseResult(middle_json=middle), source, output_root=tmp_path / "output", page_range="")
    expected = {
        "page_0_image_1.png",
        "page_0_image_1_1.png",
        "page_0_table_2.png",
        "page_0_table_image_2_1.png",
        "page_0_table_image_2_2.png",
        "page_0_chart_3.png",
        "page_0_equation_4.png",
    }
    before = {path.name: path.read_bytes() for path in (artifacts.root / "images").iterdir()}
    assert set(before) == expected
    saved = ParseResult.from_json(artifacts.middle_json_path.read_text())
    assert saved.pages[0].blocks[2].content[0].image_path == "images/page_0_table_2.png"
    table = BeautifulSoup(saved.pages[0].blocks[2].content[0].content, "html.parser")
    assert [image["src"] for image in table.find_all("img")] == [
        "images/page_0_table_image_2_1.png",
        "images/page_0_table_image_2_2.png",
    ]
    assert before["page_0_image_1.png"] == before["page_0_table_2.png"]
    context = _build_image_context(saved.middle_json, None, artifacts.root)
    repeated = _materialize_middle_json(saved.middle_json, context)
    assert repeated == saved.middle_json
    assert {path.name: path.read_bytes() for path in (artifacts.root / "images").iterdir()} == before
    for format_name in ("markdown", "json", "latex"):
        with zipfile.ZipFile(render_download(artifacts.as_state(), format_name)) as archive:
            images = {Path(name).name: archive.read(name) for name in archive.namelist() if name.startswith("images/")}
            assert images == before
    structured = json.loads(artifacts.structured_content_path.read_text())
    assert structured["pages"][0]["blocks"][2]["image_source"] == "images/page_0_table_2.png"


def test_identical_images_on_different_pages_keep_their_names(tmp_path: Path) -> None:
    """像素相同的跨页图片仍使用各自的原始页索引，保留定位意义。"""
    middle = _middle_json(file_suffix="docx", page_indices=(2, 5))
    for page in middle.pages:
        page.blocks[1].content[0].image_base64 = _image_uri("red")
    context = _build_image_context(middle, None, tmp_path)
    result = _materialize_middle_json(middle, context)
    assert [page.blocks[1].content[0].image_path for page in result.pages] == [
        "images/page_2_image_21.png",
        "images/page_5_image_51.png",
    ]
    assert len(list((tmp_path / "images").iterdir())) == 2


def test_conflicting_contents_never_overwrite_or_take_inline_ordinals(tmp_path: Path) -> None:
    """同名不同内容分配独立冲突后缀，并避免占用正文内嵌图片的正式序号。"""
    owner = _middle_json(file_suffix="docx").pages[0].blocks[1]
    first = _write_materialized_asset(tmp_path, _image_bytes("red"), "png", page_idx=1, owner=owner)
    second = _write_materialized_asset(tmp_path, _image_bytes("blue"), "png", page_idx=1, owner=owner)
    assert first == "images/page_1_image_1.png"
    assert second == "images/page_1_image_1_duplicate_1.png"
    assert _write_materialized_asset(tmp_path, _image_bytes("blue"), "png", page_idx=1, owner=owner) == second
    inline = _write_materialized_asset(tmp_path, _image_bytes("blue"), "png", page_idx=1, owner=owner, ordinal=1)
    assert inline == "images/page_1_image_1_1.png"
    assert (tmp_path / first).read_bytes() == _image_bytes("red")
    assert len(list((tmp_path / "images").iterdir())) == 3


def test_code_and_algorithm_img_literals_are_not_materialized(tmp_path: Path) -> None:
    """代码与算法的 img 字面量不得读取资源、改变正文或生成截图。"""
    source = tmp_path / "code.docx"
    source.write_bytes(b"source")
    middle = _middle_json(with_image=False, file_suffix="docx")
    literal = '<img src="missing.png">'
    middle.pages[0].blocks.extend(
        [
            CodeBlock(
                type=BlockType.CODE,
                index=1,
                sub_type="code",
                guess_lang="html",
                content=[
                    CodeBodyBlock(type=BlockType.CODE_BODY, index=1, bbox=(0, 0, 1, 1), content=literal),
                ],
            ),
            CodeBlock(
                type=BlockType.CODE,
                index=2,
                sub_type="algorithm",
                content=[
                    AlgorithmBodyBlock(
                        type=BlockType.ALGORITHM_BODY,
                        index=2,
                        bbox=(0, 0, 1, 1),
                        content=[TextSpan(type="text", content=literal)],
                    ),
                ],
            ),
        ]
    )
    original = middle.model_dump_json()
    artifacts = persist_parse_result(ParseResult(middle_json=middle), source, output_root=tmp_path / "output", page_range="")
    saved = ParseResult.from_json(artifacts.middle_json_path.read_text())
    assert saved.middle_json == middle and middle.model_dump_json() == original
    assert not (artifacts.root / "images").exists()
    assert saved.pages[0].blocks[1].content[0].content == literal
    assert saved.pages[0].blocks[2].content[0].content[0].content == literal
