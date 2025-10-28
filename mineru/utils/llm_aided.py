# Copyright (c) Opendatalab. All rights reserved.
from loguru import logger
from openai import OpenAI
import json_repair

from mineru.backend.pipeline.pipeline_middle_json_mkcontent import merge_para_with_text


def llm_aided_title(page_info_list, title_aided_config):
    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )
    title_dict = {}
    origin_title_list = []
    i = 0
    for page_info in page_info_list:
        blocks = page_info["para_blocks"]
        for block in blocks:
            if block["type"] == "title":
                origin_title_list.append(block)
                title_text = merge_para_with_text(block)

                if 'line_avg_height' in block:
                    line_avg_height = block['line_avg_height']
                else:
                    title_block_line_height_list = []
                    for line in block['lines']:
                        bbox = line['bbox']
                        title_block_line_height_list.append(int(bbox[3] - bbox[1]))
                    if len(title_block_line_height_list) > 0:
                        line_avg_height = sum(title_block_line_height_list) / len(title_block_line_height_list)
                    else:
                        line_avg_height = int(block['bbox'][3] - block['bbox'][1])

                title_dict[f"{i}"] = [title_text, line_avg_height, int(page_info['page_idx']) + 1]
                i += 1
    # logger.info(f"Title list: {title_dict}")

    title_optimize_prompt = f"""输入的内容是一篇文档中所有标题组成的字典，请根据以下指南优化标题的结果，使结果符合正常文档的层次结构：

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
    #5.
    #- 字典中可能包含被误当成标题的正文，你可以通过将其层级标记为 0 来排除它们

    retry_count = 0
    max_retries = 3
    dict_completion = None

    # Build API call parameters
    api_params = {
        "model": title_aided_config["model"],
        "messages": [{'role': 'user', 'content': title_optimize_prompt}],
        "temperature": 0.7,
        "stream": True,
    }

    # Only add extra_body when explicitly specified in config
    if "enable_thinking" in title_aided_config:
        api_params["extra_body"] = {"enable_thinking": title_aided_config["enable_thinking"]}

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(**api_params)
            content_pieces = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content_pieces.append(chunk.choices[0].delta.content)
            content = "".join(content_pieces).strip()
            # logger.info(f"Title completion: {content}")
            if "</think>" in content:
                idx = content.index("</think>") + len("</think>")
                content = content[idx:].strip()
            dict_completion = json_repair.loads(content)
            dict_completion = {int(k): int(v) for k, v in dict_completion.items()}

            # logger.info(f"len(dict_completion): {len(dict_completion)}, len(title_dict): {len(title_dict)}")
            if len(dict_completion) == len(title_dict):
                for i, origin_title_block in enumerate(origin_title_list):
                    origin_title_block["level"] = int(dict_completion[i])
                break
            else:
                logger.warning(
                    "The number of titles in the optimized result is not equal to the number of titles in the input.")
                retry_count += 1
        except Exception as e:
            logger.exception(e)
            retry_count += 1

    if dict_completion is None:
        logger.error("Failed to decode dict after maximum retries.")
