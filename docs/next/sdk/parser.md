# 本地解析 SDK

核对入口：`mineru/parser/__init__.py`、`mineru/parser/mineru_parser.py`、`mineru/backend/analyze.py`。

## 公开入口

`mineru.parser` 导出 `parse`、`parse_async`、`MinerUParser`、`MinerUApiParser`、`DocumentParser`、
`ParseResult`、`MIDDLE_JSON_SCHEMA_VERSION`、`ApiTaskStatus` 和 `ApiJobStatus`。
`ApiJobStatus` 只是 `ApiTaskStatus` 的 Python 类型别名，不恢复 Jobs HTTP 接口。

`parse()` 和 `parse_async()` 接受同一组参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `path` | 必填 | 本地 `str` 或 `Path` |
| `tier` | `standard` | `flash/basic/standard/advanced`；不接受 `None` 或 `auto` |
| `ocr_mode` | `auto` | `auto/txt/ocr` |
| `image_analysis` | `True` | 图片分析开关 |
| `page_range` | 空字符串 | PDF 页范围；空字符串表示全部 |
| `source_context` | `None` | 内部调用方保留 HTML 来源上下文 |
| `vlm_config` | `None` | 使用全局 VLM 配置；显式对象完整覆盖并复制 |

`MinerUParser` 构造参数为 `tier/parse_mode/image_analysis/vlm_config`，其中 `parse_mode` 对应便捷函数的 `ocr_mode`。
实例的 `parse()` / `parse_async()` 接收 `path/page_range/source_context`。

```python
from mineru.parser import MinerUParser
from mineru.parser.writer import FileBasedDataWriter

with MinerUParser(tier="flash", parse_mode="txt") as parser:
    result = parser.parse("report.pdf", page_range="1-3")
    result.save(FileBasedDataWriter("output"))
```

## 路由与执行

本地路径读取后按文件类型进入 `doc_analyze()` / `aio_doc_analyze()`。
图片先转为 PDF；PDF 支持 [页范围](../page-ranges.md)，非 PDF 输入的非空页范围报错。
原生 Office、HTML/MHTML、CSV/TSV、EPUB、OFD 由 DocVortex 解析并记录实际 Flash/txt 元数据。
直接 Python parser 的有效 tier 参数不改变这些原生路径；CLI 单文件对显式质量档位另有校验，见 [档位边界](tiers-errors.md)。

Flash PDF 的 `auto` 会分类，`txt` 直接走原生解析，`ocr` 走 Flash OCR；不会在显式 txt 后追加 OCR 回退。
Basic/Standard/Advanced 的 effort 分别为 medium/high/xhigh。

`DocumentParser` 的默认异步实现在线程中调用同步方法，默认 batch 按输入顺序调用，异步 batch 使用 gather。
`MinerUParser` 覆盖异步方法并调用实际异步分析门面，不能将所有引擎概括为线程池实现。
`close()` 与上下文管理器属于 parser 生命周期；结果不依赖已关闭的源文档。

SDK 不使用 Doclib 数据库或缓存。导入公开包不启动服务或加载推理模型；实际执行按需加载。
远程 VLM 配置会把模型输入发往指定推理服务，它与使用 [MinerUApiParser](api-parser.md) 上传整份源文件的 Tasks 接口不同。
