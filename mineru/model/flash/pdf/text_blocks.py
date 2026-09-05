# Copyright (c) Opendatalab. All rights reserved.
"""保留既有入口的 Flash PDF 门面，内部实现按职责显式组织。"""

from .text_assembly.annotations import (
    _merge_image_caption_text_blocks as _merge_image_caption_text_blocks,
    _caption_image_group_bboxes as _caption_image_group_bboxes,
    _caption_seed_matches_image as _caption_seed_matches_image,
    _caption_body_has_structural_gap as _caption_body_has_structural_gap,
    _caption_tail_matches_seed as _caption_tail_matches_seed,
    _merge_multiline_title_blocks as _merge_multiline_title_blocks,
    _merge_fragmented_header_blocks as _merge_fragmented_header_blocks,
    _merge_front_matter_column_blocks as _merge_front_matter_column_blocks,
    _merge_repeated_compact_title_continuations as _merge_repeated_compact_title_continuations,
)
from .text_assembly.assembly import _build_text_blocks as _build_text_blocks
from .text_assembly.common import (
    _REFERENCE_ENTRY_RE as _REFERENCE_ENTRY_RE,
    _FIGURE_CAPTION_MARKER_RE as _FIGURE_CAPTION_MARKER_RE,
    _INLINE_MATH_RECOVERY_MARKER as _INLINE_MATH_RECOVERY_MARKER,
    _PARAGRAPH_FORMULA_CONTEXT_MARKER as _PARAGRAPH_FORMULA_CONTEXT_MARKER,
    _FRONT_MATTER_FIELD_RE as _FRONT_MATTER_FIELD_RE,
    _LIST_ITEM_RE as _LIST_ITEM_RE,
    _BULLET_ITEM_RE as _BULLET_ITEM_RE,
    _EMAIL_METADATA_RE as _EMAIL_METADATA_RE,
    _ABSTRACT_METADATA_RE as _ABSTRACT_METADATA_RE,
    _LABELLED_METADATA_RE as _LABELLED_METADATA_RE,
    _URL_LINE_RE as _URL_LINE_RE,
    _SHORT_SAME_BASELINE_PREFIX_RE as _SHORT_SAME_BASELINE_PREFIX_RE,
    _merge_internal_text_block_group as _merge_internal_text_block_group,
    _component_declared_lane_interval as _component_declared_lane_interval,
    _component_lane_interval as _component_lane_interval,
    _component_reference_width as _component_reference_width,
    _compatible_component_lane_width as _compatible_component_lane_width,
    _components_share_lane_role as _components_share_lane_role,
    _block_starts_with_short_wide_rows as _block_starts_with_short_wide_rows,
    _find_short_opener_pairs as _find_short_opener_pairs,
    _nearest_following_text_component as _nearest_following_text_component,
    _has_parallel_text_component as _has_parallel_text_component,
    _nearest_tapered_tail_component as _nearest_tapered_tail_component,
    _component_connection_skips_block as _component_connection_skips_block,
    _text_component_sort_key as _text_component_sort_key,
    _merge_text_line_content as _merge_text_line_content,
)
from .text_assembly.footnotes import (
    _build_grouped_page_footnote_blocks as _build_grouped_page_footnote_blocks,
    _split_page_footnote_entries as _split_page_footnote_entries,
    _find_page_footnote_marker_rows as _find_page_footnote_marker_rows,
    _find_geometric_page_footnote_marker_rows as _find_geometric_page_footnote_marker_rows,
    _split_marked_page_footnote_entries as _split_marked_page_footnote_entries,
    _split_unmarked_page_footnote_entries as _split_unmarked_page_footnote_entries,
    _tight_page_footnote_bboxes as _tight_page_footnote_bboxes,
)
from .text_assembly.merging import (
    _merge_short_same_baseline_prefix_blocks as _merge_short_same_baseline_prefix_blocks,
    _blocks_share_boundary_visual_row as _blocks_share_boundary_visual_row,
    _merge_overlapping_same_line_text_blocks as _merge_overlapping_same_line_text_blocks,
    _merge_inline_math_fragment_text_blocks as _merge_inline_math_fragment_text_blocks,
    _component_local_union_bbox as _component_local_union_bbox,
    _merge_paragraph_formula_context_blocks as _merge_paragraph_formula_context_blocks,
    _merge_residual_narrow_math_text_blocks as _merge_residual_narrow_math_text_blocks,
    _merge_hostless_inline_math_fragment_blocks as _merge_hostless_inline_math_fragment_blocks,
    _merge_inline_math_recovery_group as _merge_inline_math_recovery_group,
    _merge_inline_math_paragraph_continuations as _merge_inline_math_paragraph_continuations,
    _merge_spatial_text_components as _merge_spatial_text_components,
    _merge_list_intro_text_components as _merge_list_intro_text_components,
    _merge_unterminated_text_components as _merge_unterminated_text_components,
)
from .text_assembly.rows import (
    _local_tight_output_line_bboxes as _local_tight_output_line_bboxes,
    _starts_structural_reference_entry as _starts_structural_reference_entry,
    _build_hanging_indent_group_map as _build_hanging_indent_group_map,
    _infer_local_text_lane_map as _infer_local_text_lane_map,
    _structured_text_break_sources as _structured_text_break_sources,
    _isolated_indented_paragraph_break_sources as _isolated_indented_paragraph_break_sources,
    _centered_visual_reset_break_sources as _centered_visual_reset_break_sources,
    _leading_typography_reset_break_sources as _leading_typography_reset_break_sources,
    _formula_style_text_row_break_sources as _formula_style_text_row_break_sources,
    _front_matter_keyword_break_sources as _front_matter_keyword_break_sources,
    _component_starts_with_emphasized_row as _component_starts_with_emphasized_row,
    _explicit_text_break_sources as _explicit_text_break_sources,
)

__all__ = []
