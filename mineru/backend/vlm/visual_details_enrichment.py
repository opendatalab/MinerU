# Copyright (c) Opendatalab. All rights reserved.
import base64
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from mineru.utils.enum_class import BlockType, ContentType


DEFAULT_DETAILS_VLM_TIMEOUT = 120
DEFAULT_DETAILS_VLM_MAX_CONCURRENCY = 1
DEFAULT_DETAILS_VLM_LANGUAGE = "auto"
NEARBY_CONTEXT_CHAR_LIMIT = 800
LIST_CONTEXT_CHAR_LIMIT = 800


_TEXT_CONTEXT_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.REF_TEXT,
}

_EXCLUDED_CONTEXT_BLOCK_TYPES = {
    BlockType.IMAGE,
    BlockType.IMAGE_BODY,
    BlockType.CHART,
    BlockType.CHART_BODY,
    BlockType.TABLE,
    BlockType.TABLE_BODY,
    BlockType.CODE,
    BlockType.CODE_BODY,
    BlockType.ALGORITHM,
    BlockType.INTERLINE_EQUATION,
    BlockType.EQUATION,
    BlockType.CAPTION,
    BlockType.IMAGE_CAPTION,
    BlockType.IMAGE_FOOTNOTE,
    BlockType.CHART_CAPTION,
    BlockType.CHART_FOOTNOTE,
    BlockType.TABLE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.CODE_CAPTION,
    BlockType.CODE_FOOTNOTE,
    BlockType.FOOTNOTE,
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.ASIDE_TEXT,
    BlockType.PAGE_FOOTNOTE,
}

_VISUAL_BLOCK_TYPES = {
    BlockType.IMAGE: {
        "body": BlockType.IMAGE_BODY,
        "span": ContentType.IMAGE,
        "caption": BlockType.IMAGE_CAPTION,
        "footnote": BlockType.IMAGE_FOOTNOTE,
    },
    BlockType.CHART: {
        "body": BlockType.CHART_BODY,
        "span": ContentType.CHART,
        "caption": BlockType.CHART_CAPTION,
        "footnote": BlockType.CHART_FOOTNOTE,
    },
}


@dataclass(frozen=True)
class DetailsVlmConfig:
    enabled: bool = False
    url: str | None = None
    model: str | None = None
    api_key: str = ""
    timeout: int = DEFAULT_DETAILS_VLM_TIMEOUT
    max_concurrency: int = DEFAULT_DETAILS_VLM_MAX_CONCURRENCY
    language: str = DEFAULT_DETAILS_VLM_LANGUAGE

    def is_complete(self) -> bool:
        return bool(self.enabled and self.url and self.model)


def build_details_vlm_config(
    *,
    details_image_analysis: bool = False,
    details_vlm_url: str | None = None,
    details_vlm_model: str | None = None,
    details_vlm_api_key: str = "",
    details_vlm_timeout: int = DEFAULT_DETAILS_VLM_TIMEOUT,
    details_vlm_max_concurrency: int = DEFAULT_DETAILS_VLM_MAX_CONCURRENCY,
    details_vlm_language: str = DEFAULT_DETAILS_VLM_LANGUAGE,
) -> DetailsVlmConfig:
    return DetailsVlmConfig(
        enabled=details_image_analysis,
        url=(details_vlm_url or "").strip() or None,
        model=(details_vlm_model or "").strip() or None,
        api_key=details_vlm_api_key or "",
        timeout=max(1, int(details_vlm_timeout or DEFAULT_DETAILS_VLM_TIMEOUT)),
        max_concurrency=max(1, int(details_vlm_max_concurrency or DEFAULT_DETAILS_VLM_MAX_CONCURRENCY)),
        language=(details_vlm_language or DEFAULT_DETAILS_VLM_LANGUAGE).strip() or DEFAULT_DETAILS_VLM_LANGUAGE,
    )


def validate_details_vlm_config(config: DetailsVlmConfig) -> None:
    if not config.enabled:
        return
    missing = []
    if not config.url:
        missing.append("details_vlm_url")
    if not config.model:
        missing.append("details_vlm_model")
    if missing:
        raise ValueError(
            "details image analysis requires: " + ", ".join(missing)
        )


def enrich_visual_details(
    pdf_info: list[dict[str, Any]],
    local_image_dir: str,
    config: DetailsVlmConfig,
    *,
    document_name: str = "",
) -> int:
    if not config.enabled:
        return 0
    if not config.is_complete():
        logger.warning(
            "Skipping details image analysis for {} because VLM URL/model is missing.",
            document_name or "document",
        )
        return 0

    targets = list(_iter_referenced_visual_detail_targets(pdf_info, local_image_dir))
    if not targets:
        return 0

    total_targets = len(targets)
    logger.info(
        "Details image analysis started for {}: {} visual detail(s), concurrency={}, model={}",
        document_name or "document",
        total_targets,
        config.max_concurrency,
        config.model,
    )

    enriched_count = 0
    completed_count = 0
    with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
        future_to_target = {
            executor.submit(
                _request_visual_description_with_progress,
                target,
                config,
                document_name or "document",
                index,
                total_targets,
            ): (index, target)
            for index, target in enumerate(targets, start=1)
        }
        for future in as_completed(future_to_target):
            index, target = future_to_target[future]
            completed_count += 1
            try:
                enriched_content, elapsed, request_error = future.result()
            except Exception as exc:
                logger.warning(
                    "Details image analysis request {}/{} failed for {} page={} image={}: {}",
                    index,
                    total_targets,
                    document_name or "document",
                    target["page_idx"],
                    target["image_path"],
                    exc,
                )
                continue
            if request_error is not None:
                logger.warning(
                    "Details image analysis progress for {}: {}/{} complete, {} enriched; failed request #{} page={} image={} after {:.1f}s: {}",
                    document_name or "document",
                    completed_count,
                    total_targets,
                    enriched_count,
                    index,
                    target["page_idx"],
                    target["image_path"],
                    elapsed,
                    request_error,
                )
                continue
            if not enriched_content:
                logger.warning(
                    "Details image analysis progress for {}: {}/{} complete, {} enriched; empty response from request #{} page={} image={} after {:.1f}s",
                    document_name or "document",
                    completed_count,
                    total_targets,
                    enriched_count,
                    index,
                    target["page_idx"],
                    target["image_path"],
                    elapsed,
                )
                continue
            target["span"]["content"] = _append_didactic_interpretation(
                target["original_content"],
                enriched_content,
            )
            target["span"]["details_source"] = "external_vlm"
            target["span"]["details_model"] = config.model
            enriched_count += 1
            logger.info(
                "Details image analysis progress for {}: {}/{} complete, {} enriched; request #{} page={} image={} took {:.1f}s",
                document_name or "document",
                completed_count,
                total_targets,
                enriched_count,
                index,
                target["page_idx"],
                target["image_path"],
                elapsed,
            )

    logger.info(
        "Details image analysis finished for {}: {}/{} enriched.",
        document_name or "document",
        enriched_count,
        total_targets,
    )
    return enriched_count


def _request_visual_description_with_progress(
    target: dict[str, Any],
    config: DetailsVlmConfig,
    document_name: str,
    index: int,
    total_targets: int,
) -> tuple[str, float, Exception | None]:
    logger.info(
        "Details image analysis request {}/{} started for {} page={} image={}",
        index,
        total_targets,
        document_name,
        target["page_idx"],
        target["image_path"],
    )
    started_at = time.monotonic()
    try:
        content = _request_visual_description(target, config)
        return content, time.monotonic() - started_at, None
    except Exception as exc:
        return "", time.monotonic() - started_at, exc


def _iter_referenced_visual_detail_targets(pdf_info, local_image_dir):
    for page_info in pdf_info:
        page_idx = page_info.get("page_idx", 0)
        para_blocks = _get_blocks_in_index_order(page_info.get("para_blocks", []))
        for block_index, para_block in enumerate(para_blocks):
            para_type = para_block.get("type")
            visual_config = _VISUAL_BLOCK_TYPES.get(para_type)
            if visual_config is None:
                continue
            nearby_context = _collect_nearby_context(para_blocks, block_index)
            visual_context = _collect_visual_context(para_block, visual_config)
            yield from _iter_body_span_targets(
                para_block,
                visual_config["body"],
                visual_config["span"],
                local_image_dir,
                page_idx,
                visual_context,
                nearby_context,
            )


def _get_blocks_in_index_order(blocks):
    return [
        block
        for _, block in sorted(
            enumerate(blocks or []),
            key=lambda item: (item[1].get("index", float("inf")), item[0]),
        )
    ]


def _iter_body_span_targets(
    para_block,
    body_type,
    span_type,
    local_image_dir,
    page_idx,
    visual_context,
    nearby_context,
):
    for block in para_block.get("blocks", []):
        if block.get("type") != body_type:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("type") != span_type:
                    continue
                image_path = span.get("image_path", "")
                content = span.get("content", "")
                if not image_path or not isinstance(content, str) or not content.strip():
                    continue
                absolute_image_path = os.path.join(local_image_dir, image_path)
                if not os.path.exists(absolute_image_path):
                    logger.warning(
                        "Skipping details image analysis because image file is missing: {}",
                        absolute_image_path,
                    )
                    continue
                yield {
                    "span": span,
                    "span_type": span_type,
                    "image_path": image_path,
                    "absolute_image_path": absolute_image_path,
                    "page_idx": page_idx,
                    "original_content": content,
                    "sub_type": para_block.get("sub_type", ""),
                    **visual_context,
                    **nearby_context,
                }


def _collect_visual_context(para_block, visual_config):
    captions = []
    footnotes = []
    for block in para_block.get("blocks", []):
        block_type = block.get("type")
        if block_type == visual_config["caption"]:
            captions.append(_collapse_whitespace(_extract_text_from_block(block)))
        elif block_type == visual_config["footnote"]:
            footnotes.append(_collapse_whitespace(_extract_text_from_block(block)))

    return {
        "caption": "\n".join(text for text in captions if text),
        "footnote": "\n".join(text for text in footnotes if text),
    }


def _collect_nearby_context(para_blocks, visual_block_index):
    preceding_context = ""
    following_context = ""

    for block in reversed(para_blocks[:visual_block_index]):
        preceding_context = _extract_nearby_context_text(block)
        if preceding_context:
            preceding_context = _truncate_context(preceding_context, keep_end=True)
            break

    for block in para_blocks[visual_block_index + 1:]:
        following_context = _extract_nearby_context_text(block)
        if following_context:
            following_context = _truncate_context(following_context, keep_end=False)
            break

    return {
        "preceding_context": preceding_context,
        "following_context": following_context,
    }


def _extract_nearby_context_text(block):
    block_type = block.get("type")
    if block_type in _EXCLUDED_CONTEXT_BLOCK_TYPES:
        return ""
    if block_type in _TEXT_CONTEXT_BLOCK_TYPES:
        return _collapse_whitespace(_extract_text_from_block(block))
    if block_type == BlockType.LIST:
        text = _collapse_whitespace(_extract_text_from_block(block))
        if len(text) <= LIST_CONTEXT_CHAR_LIMIT:
            return text
    return ""


def _extract_text_from_block(block):
    text_parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.TEXT:
                content = span.get("content", "")
                if isinstance(content, str) and content.strip():
                    text_parts.append(content)

    for child_block in block.get("blocks", []):
        child_text = _extract_text_from_block(child_block)
        if child_text:
            text_parts.append(child_text)

    return " ".join(text_parts)


def _collapse_whitespace(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _truncate_context(text, *, keep_end):
    text = _collapse_whitespace(text)
    if len(text) <= NEARBY_CONTEXT_CHAR_LIMIT:
        return text
    if keep_end:
        return text[-NEARBY_CONTEXT_CHAR_LIMIT:].lstrip()
    return text[:NEARBY_CONTEXT_CHAR_LIMIT].rstrip()


def _request_visual_description(target: dict[str, Any], config: DetailsVlmConfig) -> str:
    image_path = target["absolute_image_path"]
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    image_data_uri = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_visual_details_prompt(target, config.language)},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ],
        "temperature": 0.2,
        "stream": False,
    }
    endpoint = f"{config.url.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=config.timeout) as client:
        response = client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    return _normalize_visual_details_content(_extract_chat_completion_content(data))


def _build_visual_details_prompt(target: dict[str, Any], language: str) -> str:
    visual_kind = "chart" if target["span_type"] == ContentType.CHART else "image"
    if language and language.lower() != "auto":
        language_instruction = f"Write the answer in this language: {language}."
    else:
        language_instruction = "Use the document's apparent language when inferable; otherwise write in English."

    original_content = target.get("original_content", "").strip()
    caption = target.get("caption", "").strip()
    footnote = target.get("footnote", "").strip()
    preceding_context = target.get("preceding_context", "").strip()
    following_context = target.get("following_context", "").strip()
    return f"""Analyze this referenced {visual_kind} from a document and write only the didactic interpretation to append to an existing Markdown <details> section.

Goal: add concise, RAG-friendly teaching context without duplicating the visual data already extracted by MinerU.

Instructions:
- {language_instruction}
- Do not include <details>, <summary>, HTML wrappers, or a top-level title.
- Start exactly with a "### Didactic interpretation" heading.
- Explain what the visual means, why it matters in the document context, and how a reader should reason about it.
- Do not create a "### Visual data" section.
- Do not repeat axes, labels, categories, values, legends, annotations, or raw data already present in the MinerU description unless a brief reference is needed to explain meaning.
- Do not invent exact values that are not visible.
- Use the caption and nearby context to explain relevance, but do not summarize unrelated surrounding text.

Existing MinerU description, if useful:
{original_content}

Caption, if present:
{caption or "(none)"}

Footnote, if present:
{footnote or "(none)"}

Nearby text before the visual, if present:
{preceding_context or "(none)"}

Nearby text after the visual, if present:
{following_context or "(none)"}

Didactic interpretation to append:"""


def _normalize_visual_details_content(content: str) -> str:
    content = (content or "").strip()
    if not content:
        return ""

    content = re.sub(
        r"^#{1,6}\s+Visual data\b.*?(?=^#{1,6}\s+|\Z)",
        "",
        content,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    ).strip()
    if not content:
        return ""

    has_didactic_heading = re.search(
        r"^#{1,6}\s+Didactic interpretation\b",
        content,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if has_didactic_heading:
        return content

    content = re.sub(
        r"^#{1,6}\s+(Interpretation|Meaning|Explanation)\b",
        "### Didactic interpretation",
        content,
        count=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if re.search(r"^#{1,6}\s+Didactic interpretation\b", content, flags=re.IGNORECASE | re.MULTILINE):
        return content
    return f"### Didactic interpretation\n{content}"


def _append_didactic_interpretation(original_content: str, didactic_content: str) -> str:
    original_content = (original_content or "").strip()
    didactic_content = _normalize_visual_details_content(didactic_content)
    if not didactic_content:
        return original_content
    if not original_content:
        return didactic_content
    return f"{original_content}\n\n{didactic_content}"


def _extract_chat_completion_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "") or ""))
            else:
                parts.append(str(item))
        content = "\n".join(part for part in parts if part.strip())
    if not isinstance(content, str):
        return ""
    content = _strip_response_wrappers(_strip_thinking_content(content)).strip()
    return content


def _strip_thinking_content(content: str) -> str:
    if "</think>" in content:
        return content.split("</think>", 1)[1]
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)


def _strip_response_wrappers(content: str) -> str:
    content = content.strip()
    fence_match = re.fullmatch(r"```[^\n]*\n?(.*?)\s*```", content, flags=re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()

    details_match = re.fullmatch(r"<details\b[^>]*>\s*(.*?)\s*</details>", content, flags=re.DOTALL | re.IGNORECASE)
    if details_match:
        content = details_match.group(1).strip()
        content = re.sub(
            r"^<summary\b[^>]*>.*?</summary>\s*",
            "",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    return content
