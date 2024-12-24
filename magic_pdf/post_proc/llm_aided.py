# Copyright (c) Opendatalab. All rights reserved.

formula_correction_prompt = """请根据以下指南修正LaTeX公式的错误，确保公式能够渲染且符合原始内容：

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

text_correction_prompt = f"""请根据以下指南修正OCR引起的错误，确保文本连贯并符合原始内容：

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