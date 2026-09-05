# Copyright (c) Opendatalab. All rights reserved.
"""保留既有入口的 Flash PDF 门面，内部实现按职责显式组织。"""

from .title_analysis.body_profile import (
    _infer_document_body_profile as _infer_document_body_profile,
    _document_font_is_regular as _document_font_is_regular,
    _infer_lane_body_profile as _infer_lane_body_profile,
    _line_uses_document_regular_font as _line_uses_document_regular_font,
)
from .title_analysis.common import (
    _NUMBERED_SECTION_TITLE_RE as _NUMBERED_SECTION_TITLE_RE,
    _SECTION_NUMBER_ONLY_RE as _SECTION_NUMBER_ONLY_RE,
    _SECTION_TITLE_TERMINAL_RE as _SECTION_TITLE_TERMINAL_RE,
    _UNNUMBERED_SECTION_HEADING_RE as _UNNUMBERED_SECTION_HEADING_RE,
    _build_physical_title_gap_map as _build_physical_title_gap_map,
    _line_inside_visual_container as _line_inside_visual_container,
    _line_near_visual_container as _line_near_visual_container,
)
from .title_analysis.document_profile import _infer_document_title_profile as _infer_document_title_profile
from .title_analysis.lane_titles import (
    _classify_paragraph_titles_in_lane as _classify_paragraph_titles_in_lane,
    _protect_front_matter_title_types as _protect_front_matter_title_types,
    _visual_row_has_body_style_sibling as _visual_row_has_body_style_sibling,
    _is_near_full_mixed_inline_row as _is_near_full_mixed_inline_row,
    _continues_local_body_row as _continues_local_body_row,
    _is_continuous_field_row as _is_continuous_field_row,
    _is_full_width_inline_heading as _is_full_width_inline_heading,
    _has_following_body_row as _has_following_body_row,
    _has_following_compact_text_section as _has_following_compact_text_section,
    _unify_visual_row_title_types as _unify_visual_row_title_types,
    _infer_front_matter_boundary as _infer_front_matter_boundary,
    _normalized_title_gap as _normalized_title_gap,
    _expand_paragraph_title_neighbors as _expand_paragraph_title_neighbors,
)
from .title_analysis.page_titles import (
    _classify_page_titles as _classify_page_titles,
    _demote_non_structural_anomaly_titles as _demote_non_structural_anomaly_titles,
    _find_repeated_grid_title_suppressions as _find_repeated_grid_title_suppressions,
    _find_container_visual_row_title_suppressions as _find_container_visual_row_title_suppressions,
    _classify_document_title as _classify_document_title,
    _expand_document_title_across_lanes as _expand_document_title_across_lanes,
    _classify_additional_document_title_bands as _classify_additional_document_title_bands,
    _classify_cross_lane_centered_section_titles as _classify_cross_lane_centered_section_titles,
    _demote_cross_lane_body_continuation_titles as _demote_cross_lane_body_continuation_titles,
    _classify_cross_lane_emphasized_section_titles as _classify_cross_lane_emphasized_section_titles,
    _is_wide_leading_title_continuation as _is_wide_leading_title_continuation,
    _expand_cross_lane_paragraph_title_neighbors as _expand_cross_lane_paragraph_title_neighbors,
    _demote_hanging_multiline_text_titles as _demote_hanging_multiline_text_titles,
    _demote_visual_container_caption_titles as _demote_visual_container_caption_titles,
    _demote_sentence_tail_titles as _demote_sentence_tail_titles,
    _document_title_fonts_compatible as _document_title_fonts_compatible,
    _document_title_uses_page_fallback as _document_title_uses_page_fallback,
)
from .title_analysis.prototype import (
    _title_profile_seed_matches_cluster as _title_profile_seed_matches_cluster,
    _title_profile_alignment as _title_profile_alignment,
    _matching_document_title_prototype as _matching_document_title_prototype,
    _line_conflicts_document_title_profile as _line_conflicts_document_title_profile,
    _title_font_families_compatible as _title_font_families_compatible,
)
from .title_analysis.structural import (
    _normalized_section_title_text as _normalized_section_title_text,
    _is_plausible_section_number as _is_plausible_section_number,
    _section_title_has_body_followers as _section_title_has_body_followers,
    _classify_explicit_section_titles as _classify_explicit_section_titles,
    _classify_document_structural_titles as _classify_document_structural_titles,
    _promote_noninitial_document_title_band as _promote_noninitial_document_title_band,
    _canonical_title_style_key as _canonical_title_style_key,
    _classify_document_structural_title_candidates as _classify_document_structural_title_candidates,
    _collect_legacy_paragraph_title_sources as _collect_legacy_paragraph_title_sources,
    _classify_inline_typography_reset_titles as _classify_inline_typography_reset_titles,
    _classify_body_height_section_titles as _classify_body_height_section_titles,
    _body_height_section_followers as _body_height_section_followers,
)

__all__ = []
