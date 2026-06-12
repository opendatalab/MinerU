# Tool SDK: `mineru.parser`

状态: Draft
读者: SDK 开发者、`mineru-kit` 开发者、核心开发者
范围: 无状态解析工具层的公开入口、parser 类和目标契约
底稿: `../../../NEXT-SDK.md`

## 定位

`mineru.parser` 是无状态解析 SDK。它接受本地文件路径，按文件类型和 backend 选择具体 parser，返回 `ParseResult`。

适用场景:

- 用户在 Python 代码中直接解析单个文件。
- `mineru-kit parse` 执行一次性解析。
- parse-server worker 调用 parser 执行实际解析。
- doclib worker 使用 `flash` 做快速索引或调用本地轻量解析。

非目标:

- 不管理 doclib 缓存。
- 不做 watch、搜索、配置持久化。
- 不隐式启动 doclib server。
- 不在未显式配置远端时上传文件。

## 当前公开导出

`mineru.parser` 当前导出:

| 名称 | 类型 | 说明 |
|------|------|------|
| `parse` | function | 根据文件后缀和 backend 分派到具体 parser。 |
| `DocumentParser` | abstract class | 所有 parser 的统一接口。 |
| `ParseResult` | dataclass | 解析结果对象。 |
| `PdfFlashParser` | class | CPU-only PDF/image 快速解析。 |
| `PdfPipelineParser` | class | pipeline backend。 |
| `PdfVlmParser` | class | VLM backend。 |
| `PdfHybridParser` | class | hybrid backend。 |
| `DocxParser` | class | DOCX parser。 |
| `PptxParser` | class | PPTX parser。 |
| `XlsxParser` | class | XLSX parser。 |
| `HtmlParser` | class | HTML parser。 |
| `MinerUApiParser` | class | API-backed parser，详见 [API-backed Parser](api-parser.md)。 |

## `parse()` 目标契约

目标公开签名:

```python
from pathlib import Path
from mineru.parser import ParseResult

def parse(
    path: str | Path,
    *,
    tier: str | None = None,
    backend: str | None = None,
    language: str = "ch",
    ocr_mode: str | None = None,
    disable_table: bool = False,
    disable_formula: bool = False,
    disable_image_analysis: bool = False,
    server_url: str | None = None,
    page_range: str = "",
) -> ParseResult: ...

async def parse_async(...) -> ParseResult: ...
```

设计规则:

- `tier` 是面向用户的质量档位。
- `backend` 是高级参数；显式传入时覆盖 `tier`。
- `language`、`ocr_mode`、`disable_table`、`disable_formula`、`disable_image_analysis` 与 `mineru-kit api-server` 启动参数保持一致。
- `page_range` 使用与 CLI/API 一致的页码表达。
- `server_url` 只用于需要委托 VLM/remote backend 的情况，不能触发隐式远端上传。
- 返回值始终是 `ParseResult`。

`method`、`lang`、`table_enable`、`formula_enable`、`image_analysis` 仅作为兼容参数保留；新增实现和文档应使用 `ocr_mode`、`language` 和 `disable_*`。

`mineru-kit api-server` 内部应复用 `parse_async()`，避免在 server 层重复维护 parser dispatch 和 tier/backend 兼容规则。

## `DocumentParser`

所有 parser 都应实现:

```python
class DocumentParser:
    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult: ...
    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult: ...
    def parse_batch(self, paths: list[str | Path], *, page_range: str = "") -> list[ParseResult]: ...
    async def parse_batch_async(self, paths: list[str | Path], *, page_range: str = "") -> list[ParseResult]: ...
    def close(self) -> None: ...
```

约束:

- `parse()` 必须检查输入路径是否存在。
- `parse_async()` 默认可以通过线程池调用同步实现。
- `parse_batch()` 默认按顺序解析；能批处理的 parser 可以覆盖。
- parser 可以作为 context manager 使用，退出时调用 `close()`。

## Parser 类

| Parser | 输入 | 主要 backend | 说明 |
|--------|------|--------------|------|
| `PdfFlashParser` | PDF/image | flash | CPU-only，快速但质量最低，主要用于发现和索引。 |
| `PdfPipelineParser` | PDF/image | pipeline | 本地 pipeline 解析。 |
| `PdfVlmParser` | PDF/image | VLM | VLM 解析，可通过 server URL 委托。 |
| `PdfHybridParser` | PDF/image | hybrid | pipeline + VLM 混合解析。 |
| `DocxParser` | DOCX | office | Office 文档解析。 |
| `PptxParser` | PPTX | office | Office 文档解析。 |
| `XlsxParser` | XLSX | office | Office 文档解析。 |
| `HtmlParser` | HTML/HTM | html | HTML 转结构化 blocks。 |

## Tier 到 backend

目标映射:

| Tier | 默认 backend | 说明 |
|------|--------------|------|
| `flash` | `flash` | 快速 CPU-only。 |
| `standard` | `pipeline` | 消费级硬件可用的模型/规则组合。 |
| `pro` | hybrid 默认高质量 backend | 最高质量。 |

`tier=None` 表示使用默认选择策略，选择可用的最高非 `flash` tier。只有用户显式请求 `tier="flash"` 或 `backend="flash"` 时，才返回 flash 结果。

## 重依赖边界

公开 import 不应触发 torch、transformers、模型权重或解析 server 启动。重依赖应在 parser 构造或执行时惰性加载。

目标:

- `from mineru.parser import parse, ParseResult` 应足够轻。
- `PdfPipelineParser` 只在执行 pipeline 时加载 pipeline backend。
- `PdfVlmParser` / `PdfHybridParser` 只在执行 VLM/hybrid 时加载对应 backend。
- Office parser 可以在构造时绑定 analyze function，但不应在模块 import 时加载重依赖。

## 示例

```python
from mineru.parser import parse

result = parse("report.pdf", tier="standard", page_range="1~5")
print(result.markdown())
```

高级用法:

```python
from mineru.parser import PdfPipelineParser

with PdfPipelineParser(method="txt", lang="ch") as parser:
    result = parser.parse("report.pdf", page_range="1~10")
    images = result.images()
```

## 未决问题

`server_url` 归属、`parse_batch()` 进度回调等未决项集中维护在 [开放问题清单](../open-questions.md)。`parse()` 支持 `tier` 已作为目标契约写入正文。
