# ADR-0014: MinerU Read Command

状态: Accepted
日期: 2026-06-16
相关文档: 0012-doclib-block-locator.md, 0013-doc-content-progressive-reading.md, ../cli/mineru-parse.md, ../workflows.md

## 背景

`mineru parse` 从文件 path 出发，负责入库、解析、等待任务完成，并读取默认内容。Agent 后续继续阅读时，不应一直依赖本地绝对路径或重新传递 path，而应进入稳定 locator 世界。

ADR-0011 定义了 `docs.short_id`，ADR-0012 定义了 page/block locator，ADR-0013 定义了 parse content 的渐进式阅读协议。基于这些能力，需要新增一个 locator-first 的顶级命令:

```bash
mineru read <locator>
```

`read` 的目标是读取 doclib 中已有解析结果。它不负责 discover、scan、ingest，也不默认创建 parse task。

## 决策

新增 `mineru read` 顶级命令:

```bash
mineru read <locator> [--format markdown|image] [--limit 30000] [--context N] [--output PATH] [--json] [--no-marker]
```

`parse` 与 `read` 的职责边界:

```text
parse(path) = ensure document is parsed, then read default content
read(locator) = read existing parsed content by stable locator
```

`parse` 当前只支持 `--format markdown`。

`read` P0 支持:

```text
ContentFormat = Literal["markdown", "image"]
```

`--format` 表示 content format；`--json` 表示 CLI response envelope 使用 JSON。二者不是同一层语义。

## Locator 输入

`read` 支持以下 locator 粒度:

```text
doc:{short_id}
doc:{short_id}/tier:{tier}
doc:{short_id}/tier:{tier}/page:{page_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}/char:{offset}
```

规则:

- `page_no` 和 `block_no` 使用 1-based 编号。
- `char:{offset}` 使用 block 渲染文本内的 0-based 字符 offset。
- `doc:{short_id}` 会解析为当前可用的默认高质量 tier；不会静默把 `flash` 当作高质量读取结果。
- `doc:{short_id}/tier:{tier}` 明确读取指定 tier 的已有结果。

## Markdown 读取语义

> visual block 的 Markdown 引用与 image 可用性规则已由 [ADR-0027](0027-doclib-visual-block-locators.md) 补充；`image_path` 不再作为 doclib 用户接口暴露。

`--format markdown` 是默认格式。

| Locator | 默认读取范围 |
|---------|--------------|
| `doc:{short_id}` | 默认 tier 的默认阅读范围。 |
| `doc:{short_id}/tier:{tier}` | 该 tier 的默认阅读范围。 |
| `.../page:{page_no}` | 该页。 |
| `.../block:{block_no}` | 该 block。 |
| `.../char:{offset}` | 从该 block 的 offset 起读取。 |

`--limit` 是软字符限制，沿用 ADR-0013 的规则，尽量在页、block、段落、句子、换行或空白边界停止。

### `--context`

`--context N` 的含义随 locator 粒度变化:

| Locator | `--context` 行为 |
|---------|------------------|
| page locator | 读取前后各 N 页。 |
| block locator | 读取前后各 N 个 block。 |
| char locator | 按 block locator 处理上下文，char 只影响起点。 |
| doc locator | 返回 `context_not_applicable`。 |
| doc/tier locator | 返回 `context_not_applicable`。 |

## Image 读取语义

`--format image` 是 P0 能力，但范围受文档类型和 locator 粒度约束。

不支持多页 image 输出。

| 文档类型 | doc locator | doc/tier locator | page locator | block locator |
|----------|:-----------:|:----------------:|:------------:|:-------------:|
| PDF / image | 不支持 | 不支持 | 支持 | 有有效 bbox 时支持；否则可回退到 sidecar |
| Office / HTML | 不支持 | 不支持 | 不支持 | visual block 有可访问 image sidecar 时支持 |

PDF / image 的 image 输出规则:

- page image 从源文件重新渲染得到。
- block image 从源页面渲染后按 bbox 裁剪得到。
- block image 不按 block type 判断；只要 bbox 存在且非 empty，就可以输出 image。
- PDF 直接构造 `PDFDocument`；image 使用 `PDFDocument.from_image()` 重建解析时使用的单页 PDF。两者复用 `render_page()` / `crop_image()`。

Office / HTML 的 image 输出规则:

- Office / HTML 不支持 doc、doc/tier、page locator 输出 image。
- image、table、chart、formula 等 visual block 仅在 doclib parsed 目录中存在对应 image sidecar 时支持 image 输出。
- `image_path` 只用于 doclib 内部安全解析 sidecar，不向 Markdown 读取者暴露。

默认情况下，image 输出打印 asset path。传入 `--output` 时，CLI 先根据输出路径后缀选择 `image_format`，由 server 生成匹配编码的临时 asset，再由 CLI 在 client 侧 copy 到指定路径。传入 `--json` 时输出完整 `DocContentResponse`。

image 输出只接受 `.png`、`.jpg`、`.jpeg`、`.webp` 输出路径。无后缀、其它后缀以及 `--output -` 都在 CLI 参数校验阶段返回 `image_output_extension_unsupported`。CLI 不负责图片转码。

## 响应模型扩展

`read` 与 `parse` 的读取结果都继续使用 `DocContentResponse`。不能为了 locator-first 删除 parse read 已有字段。

在 ADR-0013 的基础上做最小扩展:

```python
ContentFormat = Literal["markdown", "image"]


class ContentRequestScope(DoclibModel):
    page_range: str | None = None
    after: str | None = None
    limit: int = 30000
    locator: str | None = None
    context: int = 0


class ContentNextRequest(DoclibModel):
    page_range: str | None = None
    after: str | None = None
    locator: str | None = None


class ContentAsset(DoclibModel):
    path: str
    mime_type: str
    size_bytes: int | None = None
    width: int | None = None
    height: int | None = None


class DocContentResponse(DoclibModel):
    sha256: str
    short_id: str
    tier: Tier
    format: ContentFormat
    content: str
    request_scope: ContentRequestScope
    content_ranges: list[ContentRange]
    truncated: bool = False
    next_request: ContentNextRequest | None = None
    asset: ContentAsset | None = None
```

字段规则:

- parse read 保留现有字段语义。
- parse read 的 `request_scope.locator = None`，`request_scope.context = 0`。
- locator read 的 `request_scope.locator` 填入规范化 locator。
- locator read 的 `request_scope.context` 填入实际生效的 context。
- parse read 的 `next_request` 只写 `page_range` / `after`，不写 `locator`。
- read 的 `next_request` 只写 `locator`，不写 `page_range` / `after`。
- image 输出时 `content=""`，asset 信息写入 `asset`，不把图片路径塞进 `content`。

parse read 返回示例:

```json
{
  "sha256": "...",
  "short_id": "ab12cd3",
  "tier": "basic",
  "format": "markdown",
  "content": "...",
  "request_scope": {
    "page_range": "1~10",
    "after": null,
    "limit": 30000,
    "locator": null,
    "context": 0
  },
  "content_ranges": [
    {
      "page_range": "1~10",
      "start": "doc:ab12cd3/tier:basic/page:1",
      "end": "doc:ab12cd3/tier:basic/page:10/block:18"
    }
  ],
  "truncated": false,
  "next_request": {
    "page_range": "11~20",
    "after": null,
    "locator": null
  },
  "asset": null
}
```

locator read 返回示例:

```json
{
  "sha256": "...",
  "short_id": "ab12cd3",
  "tier": "basic",
  "format": "markdown",
  "content": "...",
  "request_scope": {
    "page_range": "4",
    "after": null,
    "limit": 30000,
    "locator": "doc:ab12cd3/tier:basic/page:4/block:12",
    "context": 2
  },
  "content_ranges": [
    {
      "page_range": "4",
      "start": "doc:ab12cd3/tier:basic/page:4/block:10",
      "end": "doc:ab12cd3/tier:basic/page:4/block:14"
    }
  ],
  "truncated": false,
  "next_request": null,
  "asset": null
}
```

image read 返回示例:

```json
{
  "sha256": "...",
  "short_id": "ab12cd3",
  "tier": "basic",
  "format": "image",
  "content": "",
  "request_scope": {
    "page_range": "4",
    "after": null,
    "limit": 30000,
    "locator": "doc:ab12cd3/tier:basic/page:4/block:12",
    "context": 0
  },
  "content_ranges": [
    {
      "page_range": "4",
      "start": "doc:ab12cd3/tier:basic/page:4/block:12",
      "end": "doc:ab12cd3/tier:basic/page:4/block:12"
    }
  ],
  "truncated": false,
  "next_request": null,
  "asset": {
    "path": "/Users/me/.mineru/exports/ab12cd3-page-4-block-12.png",
    "mime_type": "image/png",
    "size_bytes": 12345,
    "width": 800,
    "height": 320
  }
}
```

## Continuation 与 marker

parse read 保持 ADR-0013 现有行为:

- `next_request.page_range` / `next_request.after` 用于下一次 `mineru parse`。
- CLI continuation marker 渲染为 `mineru parse <path> --pages ... --after ...`。

read 使用 locator continuation:

- `next_request.locator` 用于下一次 `mineru read`。
- CLI continuation marker 渲染为:

```markdown
<!-- Next: mineru read doc:ab12cd3/tier:basic/page:5 -->
```

P0 暂不实现 block locator marker，即不在每个 block 前默认输出:

```markdown
<!-- mineru:locator ... -->
```

`--no-marker` 关闭 continuation marker。

## 实现方案

`parse` 和 `read` 不合并 CLI 参数，也不共享入口 request model。二者共享内部读取计划和执行器。

```text
parse_cmd
  -> ensure_parse
  -> ParseReadRequest
  -> build_read_plan_from_parse()
  -> execute_read_plan()
  -> output_doc_content_response()

read_cmd
  -> LocatorReadRequest
  -> build_read_plan_from_locator()
  -> execute_read_plan()
  -> output_doc_content_response()
```

入口 request:

```python
class ParseReadRequest(DoclibModel):
    sha256: str
    tier: Tier
    page_range: str | None = None
    after: str | None = None
    limit: int = 30000
    format: Literal["markdown"] = "markdown"
    no_marker: bool = False


class LocatorReadRequest(DoclibModel):
    locator: str
    context: int = 0
    limit: int = 30000
    format: Literal["markdown", "image"] = "markdown"
    no_marker: bool = False
```

内部计划:

```python
class ReadPlan(DoclibModel):
    sha256: str
    short_id: str
    tier: Tier
    page_range: str | None = None
    after: str | None = None
    locator: str | None = None
    context: int = 0
    limit: int = 30000
    format: Literal["markdown", "image"] = "markdown"
    no_marker: bool = False
```

转换规则:

- `build_read_plan_from_parse()` 只处理 `sha256/tier/page_range/after/limit`。
- `build_read_plan_from_parse()` 生成的 plan 中 `locator=None`、`context=0`。
- `build_read_plan_from_locator()` 解析 locator，查出 `sha256/short_id/tier`，并根据 locator 粒度生成 `page_range`、`after`、`locator` 和 `context`。
- doc locator 和 doc/tier locator 使用默认阅读范围。
- page locator 使用单页或 context 扩展后的页范围。
- block locator 使用所在页，并由执行器选择目标 block 和上下文 block。
- char locator 使用所在页，并从 char offset 起读取。

执行器:

```python
def execute_read_plan(plan: ReadPlan) -> DocContentResponse: ...
```

执行器负责:

- load done JSON batches。
- filter pages。
- select page/block/char/context。
- render markdown。
- render page/block image。
- 有有效 bbox 的 block 使用 PDF/image 源页面渲染和裁剪；其他 visual block 使用 doclib parsed 目录中可访问的 sidecar。
- apply soft limit。
- build `content_ranges`。
- build `next_request`。
- create temporary image assets when `format="image"`。

CLI 输出 helper:

```python
def output_doc_content_response(
    response: DocContentResponse,
    *,
    json_mode: bool,
    output: str | None,
    source_path: str | None,
    read_mode: bool,
) -> None: ...
```

规则:

- `json_mode=True` 时输出完整 `DocContentResponse`。
- markdown 非 JSON 时输出 `content`，并按入口输出 continuation marker。
- image 非 JSON 时输出 `asset.path`。
- `output` 由 CLI 在 client 侧处理: markdown 写入 `content`，image copy `asset.path` 到目标路径。
- parse marker 使用 `page_range/after`。
- read marker 使用 `locator`。

## HTTP / SDK

P0 新增 locator-first 接口:

```python
def read_content(
    self,
    locator: str,
    *,
    context: int = 0,
    limit: int = 30000,
    format: Literal["markdown", "image"] = "markdown",
    image_format: Literal["jpeg", "png", "webp"] = "jpeg",
    no_marker: bool = False,
) -> DocContentResponse: ...
```

HTTP 路由:

```http
GET /api/v1/content?locator=doc:ab12cd3/tier:basic/page:4&format=markdown&limit=30000
GET /api/v1/content?locator=doc:ab12cd3/tier:basic/page:4/block:12&format=image&image_format=png
```

P0 不为 locator-first content API 提供 POST export。`GET /api/v1/content` 可以在 server 内部生成临时 image asset，并接受 `image_format=jpeg|png|webp` 控制 asset 编码，但不接受 client 指定的 output path。CLI `read --output` 是 client 侧文件写入/copy 行为，不是 doclib HTTP API 行为。

实现时在 Interface 中定义同步和异步方法，再由 client/server 使用相同签名实现。

## 错误码

| code | 场景 |
|------|------|
| `invalid_locator` | locator 格式非法。 |
| `doc_not_found` | short_id 找不到。 |
| `tier_not_cached` | 指定 tier 没有缓存解析结果。 |
| `page_not_cached` | 指定页没有缓存解析结果。 |
| `block_not_found` | block 不存在。 |
| `char_offset_out_of_range` | char offset 超出 block 内容长度。 |
| `context_not_applicable` | doc 或 doc/tier locator 使用了 context。 |
| `format_not_supported` | 当前 locator 或文档类型不支持该 format。 |
| `multi_page_image_not_supported` | 请求范围覆盖多页 image。 |
| `bbox_not_available` | block 没有有效 bbox。 |
| `asset_not_available` | Office image block 没有可读取 asset。 |

## P0 范围

必须实现:

1. `mineru read <locator>`。
2. doc / doc+tier / page / block / char locator。
3. `--format markdown`。
4. `--format image`，按本文 image 边界实现。
5. `--json`。
6. `--limit`。
7. `--context`，按 locator 粒度规则实现。
8. read continuation marker。
9. parse/read 共享 `ReadPlan` 与 `execute_read_plan()`。
10. `DocContentResponse` 增加 `asset`，`ContentRequestScope` 增加 `locator/context`，`ContentNextRequest` 增加 `locator`。

暂不实现:

1. 多页 image 输出。
2. `--format structured_content`。
3. `--format text`。
4. `--format csv` / `--format html`。
5. block locator marker。
6. search result locator。
7. doc/tier locator 的 `--format image` 多页导出包。

## 后果

正面:

- Agent 可以从 search、content_ranges 或 continuation 中拿到 locator 后，用统一命令继续读取。
- `parse` 和 `read` 共享实际读取执行器，避免两套 render/limit/marker/image 逻辑。
- parse read 已有响应结构不被破坏。
- image 读取成为 locator 的通用能力，而不是 image block 的特殊命令。

代价:

- `DocContentResponse` 会同时承载 parse read 和 locator read 的字段。
- `ContentNextRequest` 同时有 `page_range/after` 和 `locator`，但通过入口规则避免混用。
- image 输出需要补 PDF page render/crop 和 Office asset 读取两条实现路径。
