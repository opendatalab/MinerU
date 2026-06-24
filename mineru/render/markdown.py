from ..types import Block, BlockType, ContentType
from .merge import merge_para_text
from .merge_visual import _build_media_path, merge_visual_para_text
from .markdown_table import to_markdown_table


def blocks_to_markdown(
    para_blocks: list[Block],
    img_bucket_path: str = "",
    table_as_image: bool = False,
    formula_as_image: bool = False,
    no_rich_content: bool = False,
    prefer_markdown_table: bool = False,
) -> list[str]:
    para_texts: list[str] = []
    for para_block in para_blocks:
        para_text = _block_to_markdown(
            para_block=para_block,
            img_bucket_path=img_bucket_path,
            table_as_image=table_as_image,
            formula_as_image=formula_as_image,
            no_rich_content=no_rich_content,
            prefer_markdown_table=prefer_markdown_table,
        )
        if para_text := (para_text or "").strip():
            para_texts.append(para_text)
    return para_texts


def _block_to_markdown(
    para_block: Block,
    img_bucket_path: str = "",
    table_as_image: bool = False,
    formula_as_image: bool = False,
    no_rich_content: bool = False,
    prefer_markdown_table: bool = False,
) -> str | None:
    para_type = para_block.type
    if para_type == BlockType.TEXT:
        return merge_para_text(para_block)
    if para_type == BlockType.TITLE:
        title_level = _get_title_level(para_block)
        return f"{'#' * title_level} {merge_para_text(para_block)}"
    if para_type == BlockType.INDEX:
        return merge_para_text(para_block)
    if para_type == BlockType.ABSTRACT:
        return merge_para_text(para_block)
    if para_type == BlockType.REF_TEXT:
        return merge_para_text(para_block)
    if para_type == BlockType.LIST:
        return merge_para_text(para_block)
    if para_type == BlockType.PHONETIC:
        return merge_para_text(para_block)
    if para_type == BlockType.CODE:
        return merge_visual_para_text(para_block)

    if para_type == BlockType.INTERLINE_EQUATION:
        if not formula_as_image:
            if (para_text := merge_para_text(para_block)).strip():
                return para_text
        return _build_image_link(para_block, ContentType.INTERLINE_EQUATION, img_bucket_path)

    if no_rich_content:
        return None
    if para_type == BlockType.IMAGE:
        return merge_visual_para_text(para_block, img_bucket_path)
    if para_type == BlockType.TABLE:
        if not table_as_image:
            if prefer_markdown_table:
                return _render_table_block_as_markdown_table(para_block, img_bucket_path)
            return merge_visual_para_text(para_block, img_bucket_path)
        return _build_image_link(para_block, ContentType.TABLE, img_bucket_path)
    if para_type == BlockType.CHART:
        return merge_visual_para_text(para_block, img_bucket_path)
    return None


def _get_title_level(block: Block) -> int:
    title_level = block.level
    if title_level is None:
        title_level = 1
    if title_level > 6:
        title_level = 6
    elif title_level < 1:
        title_level = 0
    return title_level


def _build_image_link(block: Block, target_span_type: str, img_bucket_path: str) -> str | None:
    for line in block.lines:
        for span in line.spans:
            if span.type != target_span_type or not span.image_path:
                continue
            if media_path := _build_media_path(img_bucket_path, span.image_path):
                return f"![]({media_path})"
    return None


def _render_table_block_as_markdown_table(para_block: Block, img_bucket_path: str) -> str:
    rendered = merge_visual_para_text(para_block, img_bucket_path)
    for span in para_block.all_spans():
        if span.type != ContentType.TABLE or not span.html:
            continue
        candidate = to_markdown_table(span.html)
        if candidate != span.html.strip():
            return candidate
    return rendered
