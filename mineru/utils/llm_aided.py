# Copyright (c) Opendatalab. All rights reserved.
from concurrent.futures import ThreadPoolExecutor

import json_repair
from loguru import logger
from openai import OpenAI

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text
from mineru.utils.enum_class import BlockType


TITLE_BLOCK_TYPES = {
    BlockType.TITLE,
    BlockType.DOC_TITLE,
    BlockType.PARAGRAPH_TITLE,
}
MAX_TITLE_GROUP_WORKERS = 4


def _get_title_line_avg_height(block):
    if "line_avg_height" in block:
        return block["line_avg_height"]

    title_block_line_height_list = []
    for line in block.get("lines", []):
        bbox = line["bbox"]
        title_block_line_height_list.append(int(bbox[3] - bbox[1]))

    if len(title_block_line_height_list) > 0:
        return sum(title_block_line_height_list) / len(title_block_line_height_list)

    return int(block["bbox"][3] - block["bbox"][1])


def _collect_title_block_refs(page_info_list):
    title_block_refs = []
    title_types = set()

    for page_info in page_info_list:
        for block in page_info.get("para_blocks", []):
            block_type = block.get("type")
            if block_type in TITLE_BLOCK_TYPES:
                title_block_refs.append((page_info, block))
                title_types.add(block_type)

    return title_block_refs, title_types


def _build_title_dict(title_block_refs):
    title_dict = {}

    for i, (page_info, block) in enumerate(title_block_refs):
        title_dict[str(i)] = [
            merge_para_with_text(block),
            _get_title_line_avg_height(block),
            int(page_info["page_idx"]) + 1,
        ]

    return title_dict


def _build_title_optimize_prompt(title_dict):
    return f"""输入的内容是一篇文档中所有标题组成的字典，请根据以下指南优化标题的结果，使结果符合正常文档的层次结构：

1. 字典中每个value均为一个list，包含以下元素：
    - 标题文本
    - 文本行高是标题所在块的平均行高
    - 标题所在的页码

2. 保留原始内容：
    - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
    - 请务必保证输出的字典中元素的数量和输入的数量一致

3. 保持字典内key-value的对应关系不变

4. 优化层次结构：
    - 根据标题内容的语义为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题从前至后的层级必须是连续的，不能跳过层级
    - 标题层级最多为4级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息

5. 合理性检查与微调：
    - 在完成初步分级后，仔细检查分级结果的合理性
    - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
    - 确保最终的分级结果符合文档的实际结构和逻辑

IMPORTANT:
请直接返回优化过的由标题层级组成的字典，格式为{{标题id:标题层级}}，如下：
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
不需要对字典格式化，不需要返回任何其他信息。

Input title list:
{title_dict}

Corrected title list:
"""


def _build_relative_title_optimize_prompt(title_dict):
    return f"""输入内容是某一篇文档中除文章标题外的全部章节/段落标题组成的字典。

请注意：
- 文章标题不在本次输入中，已经由系统单独识别并设置为1级标题

1. 字典中每个value均为一个list，包含以下元素：
    - 标题文本
    - 文本行高是标题所在块的平均行高
    - 标题所在的页码

2. 保留原始内容：
    - 输入的字典中所有元素都是有效的，不能删除字典中的任何元素
    - 请务必保证输出的字典中元素的数量和输入的数量一致

3. 保持字典内key-value的对应关系不变

4. 优化层次结构：
    - 根据标题内容的语义为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题从前至后的层级必须是连续的，不能跳过层级
    - 标题层级最多为4级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息

5. 合理性检查与微调：
    - 在完成初步分级后，仔细检查分级结果的合理性
    - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
    - 确保最终的分级结果符合文档的实际结构和逻辑

IMPORTANT:
请直接返回优化后的标题层级字典，格式为{{标题id:标题层级}}，如下：
{{
  0:1,
  1:2,
  2:2,
  3:3
}}
不要返回 Markdown，不要返回代码块，不要返回任何解释文字。

Input title list:
{title_dict}

Corrected title list:
"""


def _request_title_levels(title_aided_config, title_dict, prompt_builder=None):
    if len(title_dict) == 0:
        return {}

    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )

    retry_count = 0
    max_retries = 3
    expected_keys = set(range(len(title_dict)))
    if prompt_builder is None:
        prompt_builder = _build_title_optimize_prompt
    title_optimize_prompt = prompt_builder(title_dict)

    logger.debug(f"Requesting LLM for title optimization with prompt: {title_optimize_prompt}")

    api_params = {
        "model": title_aided_config["model"],
        "messages": [{"role": "user", "content": title_optimize_prompt}],
        "temperature": 0.7,
        "stream": True,
    }
    if "enable_thinking" in title_aided_config:
        api_params["extra_body"] = {
            "enable_thinking": title_aided_config["enable_thinking"]
        }

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(**api_params)
            content_pieces = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_pieces.append(chunk.choices[0].delta.content)

            content = "".join(content_pieces).strip()
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[idx:].strip()

            logger.debug(f"Raw LLM output for title levels: {content}")
            dict_completion = json_repair.loads(content)
            dict_completion = {int(k): int(v) for k, v in dict_completion.items()}

            if set(dict_completion.keys()) == expected_keys:
                return dict_completion

            logger.warning(
                "The keys in the optimized title result do not match the input titles."
            )
        except Exception as e:
            logger.exception(e)

        retry_count += 1

    logger.error("Failed to decode dict after maximum retries.")
    return None


def _apply_levels_to_blocks(title_block_refs, levels_by_index):
    if levels_by_index is None:
        return

    for i, (_, block) in enumerate(title_block_refs):
        block["level"] = int(levels_by_index[i])


def _normalize_title_types(title_block_refs):
    for _, block in title_block_refs:
        if block.get("type") in [BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE]:
            block["type"] = BlockType.TITLE


def _get_title_block_identity(block):
    block_index = block.get("index")
    if block_index is not None:
        return ("index", block_index)

    return (
        "bbox_text",
        tuple(block.get("bbox", [])),
        merge_para_with_text(block),
    )


def _sync_para_titles_to_preproc(page_info_list):
    for page_info in page_info_list:
        para_title_map = {}
        for block in page_info.get("para_blocks", []):
            if block.get("type") in TITLE_BLOCK_TYPES:
                para_title_map[_get_title_block_identity(block)] = block

        if len(para_title_map) == 0:
            continue

        for block in page_info.get("preproc_blocks", []):
            if block.get("type") not in TITLE_BLOCK_TYPES:
                continue

            para_block = para_title_map.get(_get_title_block_identity(block))
            if para_block is None:
                continue

            block["type"] = para_block.get("type", block.get("type"))
            if "level" in para_block:
                block["level"] = para_block["level"]


def _run_single_pass_title_leveling(title_block_refs, title_aided_config):
    title_dict = _build_title_dict(title_block_refs)
    levels_by_index = _request_title_levels(title_aided_config, title_dict)
    _apply_levels_to_blocks(title_block_refs, levels_by_index)


def _split_paragraph_title_groups(title_block_refs):
    groups = []
    current_group = []

    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            if current_group:
                groups.append(current_group)
                current_group = []
        elif block.get("type") == BlockType.PARAGRAPH_TITLE:
            current_group.append(title_ref)

    if current_group:
        groups.append(current_group)

    return groups


def _offset_paragraph_title_levels(levels_by_index):
    if not levels_by_index:
        return levels_by_index

    return {
        index: 2 if level == 1 else level
        for index, level in levels_by_index.items()
    }


def _request_paragraph_group_levels(title_block_refs, title_aided_config):
    title_dict = _build_title_dict(title_block_refs)
    levels_by_index = _request_title_levels(
        title_aided_config,
        title_dict,
        prompt_builder=_build_relative_title_optimize_prompt,
    )
    return _offset_paragraph_title_levels(levels_by_index)


def _run_grouped_title_leveling(title_block_refs, title_aided_config):
    doc_title_refs = []
    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            block["level"] = 1
            doc_title_refs.append(title_ref)

    paragraph_title_groups = _split_paragraph_title_groups(title_block_refs)
    group_levels = []

    if len(paragraph_title_groups) > 1:
        max_workers = min(len(paragraph_title_groups), MAX_TITLE_GROUP_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _request_paragraph_group_levels,
                    title_group,
                    title_aided_config,
                )
                for title_group in paragraph_title_groups
            ]
            group_levels = [future.result() for future in futures]
    else:
        group_levels = [
            _request_paragraph_group_levels(title_group, title_aided_config)
            for title_group in paragraph_title_groups
        ]

    for title_group, levels_by_index in zip(paragraph_title_groups, group_levels):
        _apply_levels_to_blocks(title_group, levels_by_index)

    _normalize_title_types(doc_title_refs)
    for title_group in paragraph_title_groups:
        _normalize_title_types(title_group)


def llm_aided_title(page_info_list, title_aided_config):
    title_block_refs, title_types = _collect_title_block_refs(page_info_list)
    if len(title_block_refs) == 0:
        logger.info("No titles detected, skipping LLM-aided title optimization.")
        return

    has_doc_title = BlockType.DOC_TITLE in title_types
    has_paragraph_title = BlockType.PARAGRAPH_TITLE in title_types
    has_generic_title = BlockType.TITLE in title_types

    if has_doc_title and has_paragraph_title and not has_generic_title:
        _run_grouped_title_leveling(title_block_refs, title_aided_config)
        _sync_para_titles_to_preproc(page_info_list)
        return

    doc_title_refs = []
    title_refs_for_llm = []
    for title_ref in title_block_refs:
        _, block = title_ref
        if block.get("type") == BlockType.DOC_TITLE:
            block["level"] = 1
            doc_title_refs.append(title_ref)
        else:
            title_refs_for_llm.append(title_ref)

    if len(title_refs_for_llm) > 0:
        _run_single_pass_title_leveling(title_refs_for_llm, title_aided_config)

    _normalize_title_types(doc_title_refs)
    _normalize_title_types(title_refs_for_llm)
    _sync_para_titles_to_preproc(page_info_list)
