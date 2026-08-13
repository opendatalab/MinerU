# Copyright (c) Opendatalab. All rights reserved.
"""PDF 分析各领域阶段共享的阈值、类型集合与映射。"""

from __future__ import annotations

import re

from mineru.types import RAW_ALGORITHM, RAW_CAPTION, RAW_FOOTNOTE, RAW_PHONETIC, BlockType

LAYOUT_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 8
LAYOUT_TITLE_SPLIT_OVERLAP_THRESHOLD = 0.8
IMAGE_BLOCK_CONTAINMENT_THRESHOLD = 0.8
IMAGE_BLOCK_LAYOUT_COVERAGE_THRESHOLD = 0.9
IMAGE_BLOCK_LAYOUT_MIN_VISUAL_COUNT = 2
BATCH_RATIO = 2
TABLE_TEXT_LINE_OVERLAP_THRESHOLD = 0.5
TABLE_TEXT_ORIENTATION_MIN_VALID_LINES = 3
TABLE_TEXT_ORIENTATION_MIN_DOMINANCE_RATIO = 0.6
TABLE_TEXT_ORIENTATION_ANGLES = frozenset({0, 90, 180, 270})
_VLM_UNCLASSIFIED_TITLE_TYPE = "title"

TITLE_BLOCK_TYPES = {
    _VLM_UNCLASSIFIED_TITLE_TYPE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}
CODE_CONTENT_BLOCK_TYPES = {BlockType.CODE, RAW_ALGORITHM}
LINE_METADATA_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    RAW_CAPTION,
    RAW_FOOTNOTE,
}
NATURAL_LANGUAGE_CONTENT_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.ASIDE_TEXT,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.REF_TEXT,
    BlockType.LIST,
    BlockType.INDEX,
    RAW_CAPTION,
    RAW_FOOTNOTE,
}
VLM_VISUAL_ANNOTATION_TYPE_MAP = {
    BlockType.TABLE_CAPTION: RAW_CAPTION,
    BlockType.IMAGE_CAPTION: RAW_CAPTION,
    BlockType.CODE_CAPTION: RAW_CAPTION,
    BlockType.TABLE_FOOTNOTE: RAW_FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE: RAW_FOOTNOTE,
}
VLM_MODEL_LIST_FIELDS = frozenset({"type", "bbox", "content", "angle", "score", "sub_type", "cell_merge"})
MODEL_JSON_VISUAL_BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.CHART,
    BlockType.TABLE,
    BlockType.EQUATION,
}
LOCAL_LAYOUT_IMAGE_BLOCK_BODY_TYPES = {BlockType.IMAGE, BlockType.CHART}
LOCAL_LAYOUT_IMAGE_BLOCK_AREA_TYPES = {
    *LOCAL_LAYOUT_IMAGE_BLOCK_BODY_TYPES,
    BlockType.IMAGE_CAPTION,
    RAW_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    RAW_FOOTNOTE,
}
_INLINE_FORMULA_PATTERN = re.compile(r"\\\((.*?)\\\)")

VLM_LAYOUT_LABEL_MAP = {
    "abstract": BlockType.TEXT,
    "algorithm": BlockType.CODE,
    "aside_text": BlockType.ASIDE_TEXT,
    "chart": BlockType.CHART,
    "content": BlockType.INDEX,
    "display_formula": BlockType.EQUATION,
    "doc_title": BlockType.DOC_TITLE,
    "figure_title": RAW_CAPTION,
    "footer": BlockType.FOOTER,
    "footer_image": BlockType.FOOTER,
    "footnote": BlockType.PAGE_FOOTNOTE,
    "formula_number": BlockType.FORMULA_NUMBER,
    "header": BlockType.HEADER,
    "header_image": BlockType.HEADER,
    "image": BlockType.IMAGE,
    "number": BlockType.PAGE_NUMBER,
    "paragraph_title": BlockType.PARAGRAPH_TITLE,
    "reference_content": BlockType.REF_TEXT,
    "seal": BlockType.IMAGE,
    "table": BlockType.TABLE,
    "text": BlockType.TEXT,
    "vertical_text": BlockType.TEXT,
    "vision_footnote": RAW_FOOTNOTE,
}
PIPELINE_DET_TYPE = {
    BlockType.TEXT,
    BlockType.CODE,
    BlockType.ASIDE_TEXT,
    BlockType.INDEX,
    BlockType.DOC_TITLE,
    RAW_CAPTION,
    BlockType.FOOTER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.HEADER,
    BlockType.PAGE_NUMBER,
    BlockType.PARAGRAPH_TITLE,
    BlockType.REF_TEXT,
    RAW_FOOTNOTE,
}
NOT_EXTRACT_TYPES = {
    BlockType.TEXT,
    _VLM_UNCLASSIFIED_TITLE_TYPE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.REF_TEXT,
    BlockType.TABLE_CAPTION,
    BlockType.IMAGE_CAPTION,
    RAW_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE,
    RAW_FOOTNOTE,
    BlockType.CODE_CAPTION,
    RAW_PHONETIC,
}
VLM_TXT_DET_TYPE = NOT_EXTRACT_TYPES
VLM_OCR_DET_TYPE = {
    BlockType.TEXT,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
    RAW_CAPTION,
    RAW_FOOTNOTE,
}
_LOW_TXT_VISUAL_RUN_ANGLES = (0.0, 90.0, 180.0, 270.0)
