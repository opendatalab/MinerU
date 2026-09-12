# 输出格式与结果协议

4.0 使用统一文档模型。区分“渲染器能生成什么”与“当前 CLI/API 暴露什么”，不要将渲染层的能力直接当作某个产品入口的参数。

## 入口支持范围

| 入口 | 输出 |
| --- | --- |
| `mineru parse` | Markdown；`--json` 为含状态、内容、定位和继续阅读信息的命令响应 |
| `mineru read` | Markdown 或 `--format image`；只读取已缓存内容 |
| `mineru-kit parse` | `markdown`、`middle_json`、`zip` |
| 自部署 V1 API | `markdown`、`middle_json`、`structured_content`、`zip`；以服务能力为准 |
| `ParseResult` | `markdown()`、`structured_content()`、`to_dict()`、`to_json()`、`save(writer)` |
| `mineru.render.render()` | 下列九种目标格式 |

`mineru parse --json` 的响应不是 MiddleJson，不能直接传给 `ParseResult.from_dict()`。远端服务的产物范围由该服务声明。

## 九种渲染目标

| `RenderFormat` | 格式 | Python 返回类型 |
| --- | --- | --- |
| `MARKDOWN` | Markdown | `str` |
| `HTML` | HTML | `str` |
| `LATEX` | LaTeX | `str` |
| `DOCX` | Word | `bytes` |
| `EPUB` | EPUB | `bytes` |
| `PDF` | 语义重排版 PDF | `bytes` |
| `STRUCTURED_CONTENT` | 通用结构化内容 | `dict` |
| `CONTENT_LIST` | Content List V1 | `list[dict]` |
| `CONTENT_LIST_V2` | 按页组织的 Content List V2 | `list[list[dict]]` |

PDF 渲染是语义重排版，不是原 PDF 版式的逐像素复刻，也不是布局调试 PDF。

```python
from pathlib import Path
from mineru.parser import parse
from mineru.render import render, RenderFormat

result = parse("report.docx", tier="flash")
html = render(result.middle_json, RenderFormat.HTML)
Path("report.html").write_text(html, encoding="utf-8")
```

## 中间 JSON

`ModelJson` 保存分析结果 `pages` 和 `page_index_map`；`MiddleJson` 保存后处理后的有序页面和语义块。`schema_id` 区分 `docvortex.model` 与 `docvortex.middle`，`schema_version` 标识协议版本。不要只根据版本数字判断文档种类。

`metadata` 包含文件类型、生产者和文档属性；`extensions["mineru"]` 记录实际执行的 `tier` 与最终 `parse_mode`。版本来自 `metadata.producer.version`，不重复放在产品扩展中。页面包含 `page_idx` 与 `blocks`，`page_idx` 从 0 开始，区别于 CLI 中从 1 开始的 PDF 页码。

以下示例由当前公开类型序列化生成，`4.0.0` 是正式版文档的示意生产者版本；可选字段可被省略，实际输出为准：

```json
{
  "metadata": {
    "file_suffix": "html",
    "producer": {
      "name": "mineru",
      "version": "4.0.0"
    }
  },
  "extensions": {
    "mineru": {
      "tier": "flash",
      "parse_mode": "txt"
    }
  },
  "pages": [
    {
      "page_idx": 0,
      "blocks": [
        {
          "type": "text",
          "index": 0,
          "content": [
            {
              "type": "text",
              "content": "Hello MinerU"
            }
          ]
        }
      ]
    }
  ],
  "is_full_document": true,
  "schema": "docvortex.middle",
  "schema_version": "2.0"
}
```

可以通过 `ParseResult.from_json(result.to_json())` 往返读取当前结果。旧版 `_backend`、`pdf_info`、`_version_name` 不属于此协议；历史数据兼容见[迁移指南](migration_4.md)，不要手工改一个版本号就当作格式已迁移。

## Structured Content

`structured_content()` 返回面向消费端的内容结构，而非中间协议的另一个名称。它保留 `metadata` 和 `extensions`，将自然语言 span 转为更易消费的文本；不要为它补造 `schema_id` 或 `schema_version`。

```json
{
  "pages": [
    {
      "page_idx": 0,
      "blocks": [
        {
          "type": "text",
          "content": "Hello MinerU"
        }
      ]
    }
  ],
  "metadata": {
    "file_suffix": "html",
    "producer": {
      "name": "mineru",
      "version": "4.0.0"
    }
  },
  "extensions": {
    "mineru": {
      "tier": "flash",
      "parse_mode": "txt"
    }
  },
  "is_full_document": true
}
```

## 文件保存、ZIP 与素材

`ParseResult.save(writer)` 写出 `markdown.md`、`middle_json.json`、`structured_content.json`；有原始模型结果时还写出 `model_output.json`。`mineru-kit parse --format zip` 打包这一组结果。

```python
from mineru.parser.writer import FileBasedDataWriter

result.save(FileBasedDataWriter("output"))
```

素材可能以内嵌数据或图片路径表示，取决于来源与输出入口。PDF 的 `ParseResult.to_dict()` 会省略块中的 `image_base64`；只保存中间 JSON 不等于保存了所有外部素材。消费 API 结果时按产物引用下载并保留对应素材，不能依赖已关闭的 PDF 对象或原文件继续渲染。

WebUI 显示的布局 PDF 是调试产物，可用时用于预览检测结果；不可用时使用原始/裁页 PDF 预览。它与上述 `RenderFormat.PDF` 是不同用途。
