---
name: ocr-mineru
description: 使用 MinerU 对 PDF 或图片进行 OCR 解析，提取文本、表格、公式和结构化内容
triggers:
  - ocr
  - parse pdf
  - extract text
  - 识别图片文字
  - 解析 PDF
  - 提取 PDF 内容
  - OCR 识别
  - mineru
examples:
  - prompt: 用 MinerU 解析这个 PDF 文件
    response: 好的，我将使用 MinerU 解析该 PDF，并返回 markdown 和结构化内容。
  - prompt: OCR 识别这张图片里的文字
    response: 我将使用 MinerU 对该图片进行 OCR 识别。
  - prompt: 提取 PDF 中的所有表格
    response: 我将解析该 PDF 并提取其中的表格内容。
---

# ocr-mineru

当用户需要对 PDF、图片等文件进行 OCR 或结构化解析时，调用本 skill。

## 适用场景

- 从扫描版 PDF 或图片中提取文字
- 将 PDF 转换为 Markdown
- 提取文档中的表格、公式、图片
- 获取文档的结构化内容（content list）

## 调用方式

使用 Python 工具执行：

```python
from mineru.skill import parse_file_sync, ParseOptions

result = parse_file_sync(
    input_path="<用户提供的文件路径>",
    options=ParseOptions(
        backend="hybrid-engine",   # 可选：pipeline, vlm-engine, hybrid-engine 等
        parse_method="auto",       # 可选：auto, txt, ocr
        language="ch",             # OCR 语言提示
        formula_enable=True,       # 是否识别公式
        table_enable=True,         # 是否识别表格
        image_analysis=True,       # 是否分析图片/图表
        effort="medium",           # hybrid 模式精度：medium / high
    ),
)

# 输出结果
print(result.markdown)              # Markdown 全文
print(result.content_list_v2)       # 结构化内容列表
print(result.get_text())            # 纯文本
print(result.get_tables())          # 表格列表
print(result.images)                # 图片 {文件名: base64 data URL}
```

## 输出处理

- 如果用户要求保存到指定路径：
  - 保存 markdown：`result.save_markdown("/path/to/output.md")`
  - 保存结构化内容：`result.save_content_list("/path/to/output.json")`
  - 保存所有产物：`result.save_all("/path/to/output_dir")`
- 如果用户只需要摘要：`result.get_text()`
- 如果用户需要表格：`result.get_tables()`
- 如果用户需要图片：`result.get_images()` 或 `result.images`

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_path` | str | 必填 | PDF 或图片文件路径 |
| `backend` | str | `hybrid-engine` | 解析后端 |
| `parse_method` | str | `auto` | 解析方法：auto / txt / ocr |
| `language` | str | `ch` | OCR 语言 |
| `formula_enable` | bool | `True` | 启用公式识别 |
| `table_enable` | bool | `True` | 启用表格识别 |
| `image_analysis` | bool | `True` | 启用图片/图表分析 |
| `effort` | str | `medium` | hybrid 模式精度 |
| `output_dir` | str | `./mineru_skill_output` | 临时输出目录 |

## 注意事项

- 首次使用可能需要下载模型权重文件，请确保网络可访问 HuggingFace 或 ModelScope。
- 大文档解析可能耗时较长，默认超时时间为 600 秒。
- 若出现显存不足，可尝试切换为 `pipeline` 后端在 CPU 上运行。
