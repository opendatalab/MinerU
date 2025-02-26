# Copyright (c) Opendatalab. All rights reserved.
import json
from loguru import logger
from magic_pdf.dict2md.ocr_mkcontent import merge_para_with_text
from openai import OpenAI
import ast


#@todo: 有的公式以"\"结尾，这样会导致尾部拼接的"$"被转义，也需要修复
formula_optimize_prompt = """请根据以下指南修正LaTeX公式的错误，确保公式能够渲染且符合原始内容：

1. 修正渲染或编译错误：
    - Some syntax errors such as mismatched/missing/extra tokens. Your task is to fix these syntax errors and make sure corrected results conform to latex math syntax principles.
    - 包含KaTeX不支持的关键词等原因导致的无法编译或渲染的错误

2. 保留原始信息：
   - 保留原始公式中的所有重要信息
   - 不要添加任何原始公式中没有的新信息

IMPORTANT:请仅返回修正后的公式，不要包含任何介绍、解释或元数据。

LaTeX recognition result:
$FORMULA

Your corrected result:
"""

text_optimize_prompt = f"""请根据以下指南修正OCR引起的错误，确保文本连贯并符合原始内容：

1. 修正OCR引起的拼写错误和错误：
   - 修正常见的OCR错误（例如，'rn' 被误读为 'm'）
   - 使用上下文和常识进行修正
   - 只修正明显的错误，不要不必要的修改内容
   - 不要添加额外的句号或其他不必要的标点符号

2. 保持原始结构：
   - 保留所有标题和子标题

3. 保留原始内容：
   - 保留原始文本中的所有重要信息
   - 不要添加任何原始文本中没有的新信息
   - 保留段落之间的换行符

4. 保持连贯性：
   - 确保内容与前文顺畅连接
   - 适当处理在句子中间开始或结束的文本
   
5. 修正行内公式：
   - 去除行内公式前后多余的空格
   - 修正公式中的OCR错误
   - 确保公式能够通过KaTeX渲染
   
6. 修正全角字符
    - 修正全角标点符号为半角标点符号
    - 修正全角字母为半角字母
    - 修正全角数字为半角数字

IMPORTANT:请仅返回修正后的文本，保留所有原始格式，包括换行符。不要包含任何介绍、解释或元数据。

Previous context:

Current chunk to process:

Corrected text:
"""

def llm_aided_formula(pdf_info_dict, formula_aided_config):
    pass

def llm_aided_text(pdf_info_dict, text_aided_config):
    pass

def llm_aided_title(pdf_info_dict, title_aided_config):
    client = OpenAI(
        api_key=title_aided_config["api_key"],
        base_url=title_aided_config["base_url"],
    )
    title_dict = {}
    origin_title_list = []
    i = 0
    for page_num, page in pdf_info_dict.items():
        blocks = page["para_blocks"]
        for block in blocks:
            if block["type"] == "title":
                origin_title_list.append(block)
                title_text = merge_para_with_text(block)
                page_line_height_list = []
                for line in block['lines']:
                    bbox = line['bbox']
                    page_line_height_list.append(int(bbox[3] - bbox[1]))
                if len(page_line_height_list) > 0:
                    line_avg_height = sum(page_line_height_list) / len(page_line_height_list)
                else:
                    line_avg_height = int(block['bbox'][3] - block['bbox'][1])
                title_dict[f"{i}"] = [title_text, line_avg_height, int(page_num[5:])+1]
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
    - 为每个标题元素添加适当的层次结构
    - 行高较大的标题一般是更高级别的标题
    - 标题从前至后的层级必须是连续的，不能跳过层级
    - 标题层级最多为4级，不要添加过多的层级
    - 优化后的标题只保留代表该标题的层级的整数，不要保留其他信息
    
5. 合理性检查与微调：
    - 在完成初步分级后，仔细检查分级结果的合理性
    - 根据上下文关系和逻辑顺序，对不合理的分级进行微调
    - 确保最终的分级结果符合文档的实际结构和逻辑
    - 字典中可能包含被误当成标题的正文，你可以通过将其层级标记为 0 来排除它们
    
IMPORTANT: 
请直接返回优化过的由标题层级组成的字典，格式为{{标题id:标题层级}}，如下：
{{0:1,1:2,2:2,3:3}}
不需要对字典格式化，不需要返回任何其他信息。

Input title list:
{title_dict}

Corrected title list:
"""

    retry_count = 0
    max_retries = 3
    dict_completion = None

    while retry_count < max_retries:
        try:
            completion = client.chat.completions.create(
                model=title_aided_config["model"],
                messages=[
                    {'role': 'user', 'content': title_optimize_prompt}],
                temperature=0.7,
            )
            # logger.info(f"Title completion: {completion.choices[0].message.content}")
            dict_completion = ast.literal_eval(completion.choices[0].message.content)
            # logger.info(f"len(dict_completion): {len(dict_completion)}, len(title_dict): {len(title_dict)}")

            if len(dict_completion) == len(title_dict):
                for i, origin_title_block in enumerate(origin_title_list):
                    origin_title_block["level"] = int(dict_completion[i])
                break
            else:
                logger.warning("The number of titles in the optimized result is not equal to the number of titles in the input.")
                retry_count += 1
        except Exception as e:
            logger.exception(e)
            retry_count += 1

    if dict_completion is None:
        logger.error("Failed to decode dict after maximum retries.")
