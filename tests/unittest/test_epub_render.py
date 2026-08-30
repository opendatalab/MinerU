from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock
from zipfile import ZIP_STORED, ZipFile

from lxml import etree
from PIL import Image
import pytest

from mineru.backend.analyze import doc_analyze
from mineru.model.flash.epub import EpubPackage
from mineru.render import render_epub
from mineru.render._internal.epub import assets as epub_assets
from mineru.types import (
    AlgorithmBodyBlock,
    ChartBlock,
    ChartBodyBlock,
    CodeBlock,
    CodeBodyBlock,
    DocTitleBlock,
    EquationBlock,
    ImageBlock,
    ImageBodyBlock,
    IndexBlock,
    MiddleJson,
    PageAuxTextBlock,
    PageBlock,
    PageFootnoteBlock,
    PageInfo,
    ParagraphTitleBlock,
    TableBlock,
    TableBodyBlock,
    TextBlock,
)
from mineru.utils.image_payload import (
    MAX_DECODED_RASTER_DIMENSION,
    MAX_DECODED_RASTER_PIXELS,
    validate_decoded_raster_size,
)
from mineru.utils import image_payload as image_payload_utils

from _epub_test_utils import build_epub_fixture
from _span_test_utils import equation, hyperlink, inline

_FIXED_TIME = datetime(2026, 1, 2, 3, 4, 6, tzinfo=timezone.utc)
_PNG_URI = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
_PNG_BYTES = base64.b64decode(_PNG_URI.split(",", 1)[1])
_NS = {
    "container": "urn:oasis:names:tc:opendocument:xmlns:container",
    "dc": "http://purl.org/dc/elements/1.1/",
    "epub": "http://www.idpf.org/2007/ops",
    "math": "http://www.w3.org/1998/Math/MathML",
    "opf": "http://www.idpf.org/2007/opf",
    "xhtml": "http://www.w3.org/1999/xhtml",
}


def _middle(*pages: PageInfo) -> MiddleJson:
    """构造无需固定版式 bbox 的测试 MiddleJson。"""
    return MiddleJson(
        pages=list(pages),
        is_full_document=True,
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _page(page_idx: int, *blocks: PageBlock) -> PageInfo:
    """按调用方顺序构造一页测试内容。"""
    return PageInfo(page_idx=page_idx, blocks=list(blocks))


def _archive(payload: bytes) -> ZipFile:
    """从内存字节打开 EPUB ZIP，供测试读取成员。"""
    return ZipFile(BytesIO(payload))


def _xml(archive: ZipFile, name: str) -> etree._Element:
    """使用禁用网络和实体的 parser 读取 EPUB XML 成员。"""
    parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True, recover=False)
    return etree.fromstring(archive.read(name), parser=parser)


def _text(element: etree._Element) -> str:
    """合并 XML 节点全部可见文本，便于断言。"""
    return "".join(element.itertext())


def test_epub_package_is_single_spine_epub33_with_stable_metadata_and_mathml() -> None:
    """验证 OCF、必需元数据、单正文 spine、导航和确定性输出。"""
    middle = _middle(
        _page(
            0,
            DocTitleBlock(type="doc_title", index=0, level=1, anchor="chapter-one", content=inline("Book Title")),
            TextBlock(
                type="text",
                index=1,
                content=[*inline("Inline "), equation("x^2"), *inline(" and text")],
            ),
            EquationBlock(type="equation", index=2, content=r"\frac{a}{b}"),
        )
    )

    first = render_epub(
        middle,
        authors=("Alice", "Bob"),
        language="en-US",
        modified_at=_FIXED_TIME,
    )
    second = render_epub(
        middle,
        authors=("Alice", "Bob"),
        language="en-US",
        modified_at=_FIXED_TIME,
    )
    assert first == second
    later = render_epub(
        middle,
        authors=("Alice", "Bob"),
        language="en-US",
        modified_at=datetime(2027, 2, 3, tzinfo=timezone.utc),
    )

    with _archive(first) as archive:
        members = archive.infolist()
        assert members[0].filename == "mimetype"
        assert members[0].compress_type == ZIP_STORED
        assert members[0].extra == b""
        assert archive.read("mimetype") == b"application/epub+zip"
        assert [item.filename for item in members if item.filename.endswith(".xhtml")] == [
            "EPUB/nav.xhtml",
            "EPUB/text/content.xhtml",
        ]

        container = _xml(archive, "META-INF/container.xml")
        assert (
            container.xpath("string(container:rootfiles/container:rootfile/@full-path)", namespaces=_NS) == "EPUB/package.opf"
        )

        package = _xml(archive, "EPUB/package.opf")
        assert package.xpath("string(opf:metadata/dc:title)", namespaces=_NS) == "Book Title"
        assert package.xpath("string(opf:metadata/dc:language)", namespaces=_NS) == "en-US"
        assert package.xpath("opf:metadata/dc:creator/text()", namespaces=_NS) == ["Alice", "Bob"]
        assert package.xpath("string(opf:metadata/opf:meta[@property='dcterms:modified'])", namespaces=_NS) == (
            "2026-01-02T03:04:06Z"
        )
        assert package.xpath("string(opf:metadata/dc:identifier)", namespaces=_NS).startswith("urn:uuid:")
        assert package.xpath("count(opf:spine/opf:itemref)", namespaces=_NS) == 1.0
        assert package.xpath("string(opf:manifest/opf:item[@id='content']/@properties)", namespaces=_NS) == "mathml"

        navigation = _xml(archive, "EPUB/nav.xhtml")
        assert navigation.xpath("count(xhtml:body/xhtml:nav[@epub:type='toc'])", namespaces=_NS) == 1.0
        assert navigation.xpath("string(//xhtml:nav[@epub:type='toc']//xhtml:a/@href)", namespaces=_NS) == (
            "text/content.xhtml#chapter-one"
        )
        content = _xml(archive, "EPUB/text/content.xhtml")
        assert content.xpath("count(//math:math)", namespaces=_NS) == 2.0
        assert not content.xpath("//xhtml:script", namespaces=_NS)
    with _archive(later) as archive:
        later_package = _xml(archive, "EPUB/package.opf")
        later_identifier = later_package.xpath("string(opf:metadata/dc:identifier)", namespaces=_NS)
    with _archive(first) as archive:
        first_package = _xml(archive, "EPUB/package.opf")
        first_identifier = first_package.xpath("string(opf:metadata/dc:identifier)", namespaces=_NS)
    assert later_identifier == first_identifier


def test_epub_table_preserves_standard_inline_style_tags() -> None:
    """验证表格中的标准文字样式标签安全复制为 XHTML。"""
    table = TableBlock(
        type="table",
        index=0,
        content=[
            TableBodyBlock(
                type="table_body",
                index=0,
                content=(
                    "<table><tr><td><strong>bold</strong><em>italic</em><u>under</u>"
                    "<s>strike</s><sup>sup</sup><sub>sub</sub></td></tr></table>"
                ),
            )
        ],
    )

    with _archive(render_epub(_middle(_page(0, table)), modified_at=_FIXED_TIME)) as archive:
        content = _xml(archive, "EPUB/text/content.xhtml")

    assert [
        content.xpath(f"string(//xhtml:td/xhtml:{tag})", namespaces=_NS) for tag in ("strong", "em", "u", "s", "sup", "sub")
    ] == [
        "bold",
        "italic",
        "under",
        "strike",
        "sup",
        "sub",
    ]


def test_epub_uses_default_planner_without_source_page_boundaries() -> None:
    """验证固定默认 EPUB 连续阅读、隐藏辅助块且不保留源页边界。"""
    middle = _middle(
        _page(
            0,
            PageAuxTextBlock(type="header", index=0, content=inline("HEADER")),
            TextBlock(type="text", index=1, content=inline("inter-")),
        ),
        _page(5),
        _page(
            9,
            TextBlock(type="text", index=0, content=inline("national"), continues_prev=True),
            PageAuxTextBlock(type="footer", index=1, content=inline("FOOTER")),
        ),
    )
    original = deepcopy(middle)

    payload = render_epub(middle, modified_at=_FIXED_TIME)

    with _archive(payload) as archive:
        content = _xml(archive, "EPUB/text/content.xhtml")
        paragraphs = content.xpath("//xhtml:p[contains(@class, 'mineru-text')]", namespaces=_NS)
        assert [_text(item) for item in paragraphs] == ["international"]
        assert content.xpath("string(//xhtml:article/@class)", namespaces=_NS) == "mineru-document"
        assert not content.xpath("//xhtml:section[contains(@class, 'mineru-page')]", namespaces=_NS)
        assert not content.xpath("//xhtml:hr[contains(@class, 'mineru-page-break')]", namespaces=_NS)
        assert "HEADER" not in _text(content) and "FOOTER" not in _text(content)
    with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
        render_epub(middle, mode="default")  # type: ignore[call-arg]
    assert middle == original


def test_epub_navigation_prefers_valid_index_then_falls_back_to_heading_hierarchy() -> None:
    """验证源目录优先、无效项过滤和标题层级回退。"""
    source_index = IndexBlock(
        type="index",
        index=0,
        content=[
            ParagraphTitleBlock(type="paragraph_title", level=2, anchor="a", content=inline("Source A\t1")),
            IndexBlock(
                type="index",
                content=[
                    ParagraphTitleBlock(type="paragraph_title", level=3, anchor="b", content=inline("Source B\t2")),
                    ParagraphTitleBlock(type="paragraph_title", level=3, anchor="missing", content=inline("Missing")),
                ],
            ),
        ],
    )
    indexed = _middle(
        _page(
            0,
            source_index,
            ParagraphTitleBlock(type="paragraph_title", index=1, level=2, anchor="a", content=inline("Heading A")),
            ParagraphTitleBlock(type="paragraph_title", index=2, level=3, anchor="b", content=inline("Heading B")),
        )
    )
    with _archive(render_epub(indexed, modified_at=_FIXED_TIME)) as archive:
        navigation = _xml(archive, "EPUB/nav.xhtml")
        toc = navigation.xpath("//xhtml:nav[@epub:type='toc']", namespaces=_NS)[0]
        assert toc.xpath(".//xhtml:a/text()", namespaces=_NS) == ["Source A", "Source B"]
        assert toc.xpath(".//xhtml:a/@href", namespaces=_NS) == [
            "text/content.xhtml#a",
            "text/content.xhtml#b",
        ]
        assert toc.xpath("count(./xhtml:ol/xhtml:li/xhtml:ol/xhtml:li)", namespaces=_NS) == 1.0

    fallback = _middle(
        _page(
            0,
            ParagraphTitleBlock(type="paragraph_title", index=0, level=2, anchor="same", content=inline("First")),
            ParagraphTitleBlock(type="paragraph_title", index=1, level=4, anchor="child", content=inline("Child")),
            ParagraphTitleBlock(type="paragraph_title", index=2, level=2, anchor="same", content=inline("Second")),
        )
    )
    with _archive(render_epub(fallback, modified_at=_FIXED_TIME)) as archive:
        navigation = _xml(archive, "EPUB/nav.xhtml")
        toc = navigation.xpath("//xhtml:nav[@epub:type='toc']", namespaces=_NS)[0]
        assert toc.xpath(".//xhtml:a/text()", namespaces=_NS) == ["First", "Child", "Second"]
        content = _xml(archive, "EPUB/text/content.xhtml")
        assert content.xpath("//xhtml:h2/@id | //xhtml:h4/@id", namespaces=_NS) == ["same", "child", "same-2"]


def test_epub_assets_are_embedded_deduplicated_and_missing_sources_degrade_to_text() -> None:
    """验证 resolver/base64 去重、富 HTML 图片重写及缺图文字降级。"""
    requested: list[str] = []

    def resolve_asset(path: str) -> bytes:
        """记录 sidecar 请求，并仅为已知图片返回有效 PNG。"""
        requested.append(path)
        if path != "images/shared.png":
            raise FileNotFoundError(path)
        return _PNG_BYTES

    middle = _middle(
        _page(
            0,
            ImageBlock.model_validate(
                {
                    "type": "image",
                    "index": 0,
                    "content": [
                        {"type": "image_caption", "content": inline("Primary caption")},
                        {
                            "type": "image_body",
                            "index": 0,
                            "content": "Primary OCR",
                            "image_path": "images/shared.png",
                            "image_base64": _PNG_URI,
                        },
                    ],
                }
            ),
            TableBlock(
                type="table",
                index=1,
                content=[
                    TableBodyBlock(
                        type="table_body",
                        index=1,
                        content=(
                            '<table><tr><td><img src="images/shared.png" alt="Cell image"/></td></tr></table>'
                            "<ul>Loose text<div>Nested text</div><li>List item</li></ul><!--hidden comment-->"
                            '<script>alert(1)</script><a href="missing.xhtml">Relative link text</a>'
                        ),
                    )
                ],
            ),
            ImageBlock(
                type="image",
                index=2,
                content=[
                    ImageBodyBlock(
                        type="image_body",
                        index=2,
                        content="Remote OCR",
                        image_url="https://example.com/remote.png",
                    )
                ],
            ),
            ImageBlock(
                type="image",
                index=3,
                content=[
                    ImageBodyBlock(
                        type="image_body",
                        index=3,
                        content="Broken OCR",
                        image_path="images/missing.png",
                        image_base64="data:image/png;base64,broken",
                    )
                ],
            ),
        )
    )

    payload = render_epub(middle, asset_resolver=resolve_asset, modified_at=_FIXED_TIME)
    with _archive(payload) as archive:
        assets = [name for name in archive.namelist() if name.startswith("EPUB/assets/")]
        assert len(assets) == 1
        content = _xml(archive, "EPUB/text/content.xhtml")
        sources = content.xpath("//xhtml:img/@src", namespaces=_NS)
        assert len(sources) == 2 and len(set(sources)) == 1
        assert all(source.startswith("../assets/") for source in sources)
        visible = _text(content)
        assert all(value in visible for value in ("Primary OCR", "Primary caption", "Remote OCR", "Broken OCR"))
        assert "Relative link text" in visible
        assert "Loose textNested textList item" in visible
        assert "hidden comment" not in visible
        assert not content.xpath("//xhtml:script", namespaces=_NS)
        assert not content.xpath("//xhtml:a[@href='missing.xhtml']", namespaces=_NS)
        assert "https://example.com/remote.png" not in archive.read("EPUB/text/content.xhtml").decode()
    assert requested == ["images/shared.png", "images/missing.png"]


@pytest.mark.parametrize(
    ("width", "height", "is_valid"),
    [
        (4_000, 4_000, True),
        (8_192, 1, True),
        (4_001, 4_000, False),
        (8_193, 1, False),
        (0, 1, False),
        (-1, 1, False),
    ],
)
def test_decoded_raster_size_uses_shared_dimension_and_pixel_limits(width: int, height: int, is_valid: bool) -> None:
    """验证共享 raster 解码预算固定为单边 8192 与总像素 1600 万。"""
    assert MAX_DECODED_RASTER_DIMENSION == 8_192
    assert MAX_DECODED_RASTER_PIXELS == 16_000_000
    if is_valid:
        validate_decoded_raster_size(width, height)
        return
    with pytest.raises(ValueError, match="Decoded raster image exceeds limits"):
        validate_decoded_raster_size(width, height)


def test_image_data_uri_enforces_encoded_and_decoded_byte_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 raster data URI 会在 Base64 前后分别执行固定字节预算。"""
    decode = MagicMock(return_value=b"\x89PNG\r\n\x1a\nx")
    monkeypatch.setattr(image_payload_utils, "MAX_RASTER_IMAGE_BYTES", 8)
    monkeypatch.setattr(image_payload_utils.base64, "b64decode", decode)

    with pytest.raises(ValueError, match="Raster image payload exceeds its byte limit"):
        image_payload_utils.parse_image_data_uri_strict("data:image/png;base64,AAAAAAAAAAAAAAAA")
    decode.assert_not_called()

    with pytest.raises(ValueError, match="Raster image payload exceeds its byte limit"):
        image_payload_utils.parse_image_data_uri_strict("data:image/png;base64,AAAA")
    decode.assert_called_once_with("AAAA", validate=True)


def test_epub_oversized_data_uri_is_omitted_before_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 EPUB 在缓存键哈希前拒绝超过编码预算的 data URI。"""
    digest = MagicMock(side_effect=AssertionError("oversized data URI must not be hashed"))
    monkeypatch.setattr(epub_assets, "MAX_IMAGE_DATA_URI_BYTES", len(_PNG_URI) - 1)
    monkeypatch.setattr(epub_assets.hashlib, "sha256", digest)
    registry = epub_assets.EpubAssetRegistry(None)

    href = registry.resolve_block(ImageBodyBlock(type="image_body", index=0, content="OCR", image_base64=_PNG_URI))

    assert href is None
    assert registry.assets == ()
    digest.assert_not_called()


def test_epub_oversized_image_is_omitted_before_pixel_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 EPUB 超限 raster 在 load 前省略且保留图片识别文字。"""
    image = MagicMock()
    image.__enter__.return_value = image
    image.format = "PNG"
    image.size = (4_001, 4_000)
    image.load.side_effect = AssertionError("oversized image must not be decoded")
    monkeypatch.setattr(epub_assets.Image, "open", lambda _source: image)
    middle = _middle(
        _page(
            0,
            ImageBlock(
                type="image",
                index=0,
                content=[
                    ImageBodyBlock(
                        type="image_body",
                        index=0,
                        content="Oversized OCR",
                        image_path="images/oversized.png",
                    )
                ],
            ),
        )
    )

    payload = render_epub(middle, asset_resolver=lambda _path: b"oversized", modified_at=_FIXED_TIME)

    with _archive(payload) as archive:
        assert not [name for name in archive.namelist() if name.startswith("EPUB/assets/")]
        assert "Oversized OCR" in _text(_xml(archive, "EPUB/text/content.xhtml"))
    image.load.assert_not_called()


def test_epub_mathml_failure_uses_visible_latex_without_false_manifest_property() -> None:
    """验证无效 LaTeX 回退为可见文本且不会误声明 mathml。"""
    middle = _middle(_page(0, EquationBlock(type="equation", index=0, content="{")))
    with _archive(render_epub(middle, modified_at=_FIXED_TIME)) as archive:
        package = _xml(archive, "EPUB/package.opf")
        assert package.xpath("string(opf:manifest/opf:item[@id='content']/@properties)", namespaces=_NS) == ""
        content = _xml(archive, "EPUB/text/content.xhtml")
        assert not content.xpath("//math:math", namespaces=_NS)
        fallback = content.xpath("//xhtml:code[contains(@class, 'mineru-latex-fallback')]", namespaces=_NS)
        assert len(fallback) == 1 and _text(fallback[0]) == "{"


def test_epub_static_chart_code_and_algorithm_cover_remaining_visual_bodies() -> None:
    """验证 chart、空代码和带行内公式算法均有无脚本静态表示。"""
    middle = _middle(
        _page(
            0,
            ChartBlock(
                type="chart",
                index=0,
                sub_type="bar",
                content=[
                    ChartBodyBlock(
                        type="chart_body",
                        index=0,
                        content="| A | B |\n| --- | --- |\n| 1 | 2 |",
                    )
                ],
            ),
            CodeBlock(
                type="code",
                index=1,
                sub_type="code",
                guess_lang="python",
                content=[CodeBodyBlock(type="code_body", index=1, content="")],
            ),
            CodeBlock(
                type="code",
                index=2,
                sub_type="algorithm",
                content=[
                    AlgorithmBodyBlock(
                        type="algorithm_body",
                        index=2,
                        content=[*inline("Step ", styles=["bold"]), equation("x+1")],
                    )
                ],
            ),
        )
    )
    with _archive(render_epub(middle, modified_at=_FIXED_TIME)) as archive:
        content = _xml(archive, "EPUB/text/content.xhtml")
        assert "| A | B |" in _text(content)
        code = content.xpath("//xhtml:pre[contains(@class, 'mineru-code')]/xhtml:code", namespaces=_NS)
        assert len(code) == 1 and code[0].get("class") == "language-python"
        assert content.xpath("//xhtml:div[contains(@class, 'mineru-algorithm')]//xhtml:strong", namespaces=_NS)
        assert content.xpath("//xhtml:div[contains(@class, 'mineru-algorithm')]//math:math", namespaces=_NS)
        assert not content.xpath("//xhtml:script", namespaces=_NS)


def test_epub_resolver_converts_webp_to_manifested_png() -> None:
    """验证 reader 兼容性较弱的 WebP sidecar 会在内存中转为 PNG。"""
    source = BytesIO()
    Image.new("RGBA", (2, 2), (10, 20, 30, 128)).save(source, format="WEBP", lossless=True)
    middle = _middle(
        _page(
            0,
            ImageBlock(
                type="image",
                index=0,
                content=[
                    ImageBodyBlock(
                        type="image_body",
                        index=0,
                        content="WebP image",
                        image_path="images/source.webp",
                    )
                ],
            ),
        )
    )
    payload = render_epub(
        middle,
        modified_at=_FIXED_TIME,
        asset_resolver=lambda _: source.getvalue(),
    )
    with _archive(payload) as archive:
        asset_names = [name for name in archive.namelist() if name.startswith("EPUB/assets/")]
        assert len(asset_names) == 1 and asset_names[0].endswith(".png")
        assert archive.read(asset_names[0]).startswith(b"\x89PNG\r\n\x1a\n")
        package = _xml(archive, "EPUB/package.opf")
        assert package.xpath("string(opf:manifest/opf:item[starts-with(@id, 'asset-')]/@media-type)", namespaces=_NS) == (
            "image/png"
        )


def test_real_epub_middlejson_renders_and_roundtrips_through_existing_flash_parser() -> None:
    """验证丰富 EPUB fixture 的所有核心结构可生成并由现有 parser 回读。"""
    source, _ = doc_analyze(build_epub_fixture(), file_suffix="epub")
    payload = render_epub(source, modified_at=_FIXED_TIME)

    package = EpubPackage(payload)
    try:
        assert len(package.spine) == 1
        assert package.metadata.title == "Chapter One"
    finally:
        package.close()

    with _archive(payload) as archive:
        content = _xml(archive, "EPUB/text/content.xhtml")
        assert content.xpath("//xhtml:table", namespaces=_NS)
        assert content.xpath("//math:math", namespaces=_NS)
        assert content.xpath("//xhtml:aside[@epub:type='footnote']", namespaces=_NS)
        assert content.xpath("//xhtml:pre", namespaces=_NS)
        assert content.xpath("//xhtml:ol | //xhtml:ul", namespaces=_NS)
        assert content.xpath("//xhtml:img", namespaces=_NS)

    roundtrip, _ = doc_analyze(payload, file_suffix="epub")
    assert len(roundtrip.pages) == 1
    visible = " ".join(str(block.content) for page in roundtrip.pages for block in page.blocks if hasattr(block, "content"))
    assert "Chapter One" in visible and "Section Two" in visible and "Footnote body" in visible


def test_epub_fragment_links_gain_noteref_semantics_and_external_links_remain() -> None:
    """验证脚注 fragment 被标为 noteref，安全外链保持可点击。"""
    middle = _middle(
        _page(
            0,
            TextBlock(
                type="text",
                index=0,
                content=[
                    *inline("See "),
                    hyperlink("#note", "[1]"),
                    *inline(" and "),
                    hyperlink("https://example.com/a b", "external"),
                ],
            ),
            PageFootnoteBlock(
                type="page_footnote",
                index=1,
                anchor="note",
                content=inline("Footnote body"),
            ),
        )
    )
    with _archive(render_epub(middle, modified_at=_FIXED_TIME)) as archive:
        content = _xml(archive, "EPUB/text/content.xhtml")
        note_link = content.xpath("//xhtml:a[@href='#note']", namespaces=_NS)[0]
        assert note_link.get(f"{{{_NS['epub']}}}type") == "noteref"
        assert note_link.get("role") == "doc-noteref"
        assert content.xpath("string(//xhtml:a[normalize-space(.)='external']/@href)", namespaces=_NS) == (
            "https://example.com/a%20b"
        )
