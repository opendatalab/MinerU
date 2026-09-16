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
| `PDF` | 原始块布局或语义重排 PDF | `bytes` |
| `STRUCTURED_CONTENT` | 通用结构化内容 | `dict` |
| `CONTENT_LIST` | Content List V1 | `list[dict]` |
| `CONTENT_LIST_V2` | 按页组织的 Content List V2 | `list[list[dict]]` |

PDF 默认采用 `PdfLayout.AUTO`：PDF 来源且几何完整时按原始块布局导出；旧结果缺少几何时整份回退重排并记录诊断。`ORIGINAL` 严格要求 PDF 来源、页面尺寸和必要 bbox；`REFLOW` 显式使用语义重排。OFD、Office 等来源默认仍重排。块内文字可选择和复制，表格、图表允许使用区域图；不承诺逐像素复刻。

```python
from mineru.render import PdfLayout, PdfRenderOptions, RenderFormat, render, render_pdf

pdf_bytes = render_pdf(result.middle_json, layout=PdfLayout.ORIGINAL)
pdf_bytes = render(result.middle_json, RenderFormat.PDF,
                   options=PdfRenderOptions(layout=PdfLayout.REFLOW))
```

固定布局中的正文、图注、代码、表格和图片继续独立适配各自原框；`continues_prev` 不移动原框之间的文字。原页数与空白页保留。原布局与重排 PDF 中，含中日韩文字的自然语言段落、图注及表格单元格使用 CJK 字符断行，避免将空格间的长串中文整体移到下一行；纯英文和代码、算法字面块保留既有规则，不向内容插入排版字符或连字符。

标题参考实际适配后的正文字号：优先同页同栏后续正文，其次同栏最近正文，再回退到导出正文的字号中位数；没有正文时采用 10.5 pt。章节标题按 `type + level` 取参考字号中位数 +2 pt，向上取整至 0.1 pt；附近正文较大时只上调该标题。主标题比最大章节标题目标再大 2 pt，没有章节标题时采用正文 +4 pt，并保留局部层次。取消按 90% 容纳率压低整组字号和旧样式字号上限。

原框放得下则保持位置，否则标题可利用上下及同栏右侧空白，保持左边缘；优先原顶边，必要时向上移动。栏宽由对应正文确定，不可靠时保持原宽，原有跨栏标题保留跨度。其他原框视为占用，与相邻内容保留 2 pt 间距且不越页；相邻标题按间隙中线分配空间。仍放不下时只缩小该标题，低于 6 pt 沿用完整块缩放。上下标和公式的真实外伸范围计入标题测量，目录引用及缺子坐标的近似组合不参与。

`pdf_title_layout_expanded` 记录扩展；`pdf_layout_font_exception` 记录局部正文较大或空间不足的调整原因、参考/目标/最终字号、原框和绘制框。原始几何重叠时不扩大占用并报告 `pdf_title_geometry_conflict`；原始间距过紧、没有安全扩展区域时报告 `pdf_title_clearance_unavailable`。字号和绘制区域仅存在于渲染上下文，不修改 MiddleJson 或素材，不新增接口参数。

使用 `docvortex>=0.4.7,<1`（MinerU 4.0 当前声明的最低依赖）。DocVortex 在生产原生 TXT PDF 模型输出时将行间公式的 `content` 清空一次，MinerU Flash TXT 直接使用该输出；Flash OCR 原本不填充行间公式内容，MinerU 不再重复清空。两条 Flash 路径保留 bbox、方向、图片及检测到的编号区域，PDF、Markdown、HTML、DOCX、EPUB、LaTeX 沿用图片回退。行内公式、非 Flash tier 的公式文本不变。旧缓存不会自动改写，重新解析才获得新几何和空内容公式。

```python
from pathlib import Path
from mineru.parser import parse
from mineru.render import render, RenderFormat

result = parse("report.docx", tier="flash")
html = render(result.middle_json, RenderFormat.HTML)
Path("report.html").write_text(html, encoding="utf-8")
```

## 中间 JSON

`ModelJson` 保存分析结果 `pages` 和 `page_index_map`；`MiddleJson` 保存后处理后的有序页面和语义块。JSON 消费端通过序列化后的 `schema` 与 `schema_version` 两个键识别数据（`docvortex.model` 或 `docvortex.middle`，协议版本 `2.0`）。协议身份在 Python 类型中的属性名为 `schema_id`，但序列化键名是 `schema`；读写 JSON 时一律使用序列化键。不要只根据版本数字判断文档种类。

`metadata` 包含文件类型、生产者和文档属性；`extensions["mineru"]` 记录实际执行的 `tier` 与最终 `parse_mode`。版本来自 `metadata.producer.version`，不重复放在产品扩展中。页面包含 `page_idx` 与 `blocks`，`page_idx` 从 0 开始，区别于 CLI 中从 1 开始的 PDF 页码。

所有 PDF tier 的同步/异步分析同时写入 `extensions.docvortex_layout`：`version=1`，`pages` 包含源 `page_idx`、`width_pt`、`height_pt` 及需要时的 `image_rotations`。尺寸方向与 bbox 一致；选页、空白页、多窗口和分页缓存汇总按源页号保留几何。ModelJson → MiddleJson → 序列化结果不丢失扩展，主协议仍为 2.0。导出只需要 MiddleJson 和图片素材，无需重新打开源 PDF。

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

`structured_content()` 返回面向消费端的内容结构，而非中间协议的另一个名称。它保留 `metadata` 和 `extensions`，将自然语言 span 转为更易消费的文本；它不携带协议身份，不要为它补造 `schema` 或 `schema_version` 字段。

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

`ParseResult.save(writer)` 先在文档副本上物化图片，再写出 `markdown.md`、`middle_json.json`、`structured_content.json` 和 `images/` 素材；有原始模型结果时还写出 `model_output.json`。自部署 V1 API ZIP 与 `mineru-kit parse --format zip` 共用这一保存入口。包内三种消费格式引用同一组素材，图片字节、源页号、块索引及旋转元数据保持不变。

```python
from mineru.parser.writer import FileBasedDataWriter

result.save(FileBasedDataWriter("output"))
```

PDF 的 `ParseResult.to_dict()` / `to_json()` 仍省略块中的 `image_base64`，因此独立结构 JSON 不等于携带素材的结果包。`save(writer)` 不重新裁图或旋转图片；已有路径却缺少字节时在写出前失败，不隐式读取当前目录或网络。API 客户端设置 `include_images=True` 后，从 ZIP 恢复直接图片和视觉 HTML 内嵌图片；Gradio 复用这些素材，PDF 导出只需 MiddleJson 与图片文件，无需源 PDF 或 ModelJson。`include_images=False` 保持结构读取行为。历史错误结果包需要重新生成。

WebUI 显示的布局 PDF 是调试产物，可用时用于预览检测结果；不可用时使用原始/裁页 PDF 预览。它与上述 `RenderFormat.PDF` 是不同用途。
