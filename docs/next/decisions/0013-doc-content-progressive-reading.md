# ADR-0013: Doc Content Progressive Reading

状态: Accepted
日期: 2026-06-15
相关文档: 0012-doclib-block-locator.md, ../cli/mineru-parse.md, ../workflows.md, ../middle-json/agent-gaps.md

## 背景

`mineru parse` 和 doclib content API 面向 Agent 主动阅读文档。长文档不能默认一次性输出全文，需要支持:

- 默认读取有限内容。
- 在内容被截断时继续读取当前请求范围内的剩余内容。
- 在当前请求范围已完整返回时，提示 Agent 下一段建议读取范围。
- 对分页文档、非分页文档和极端长 block 使用一致但不过度复杂的接口。
- 避免 Server 在机器 cursor 中暴露本地绝对路径或 filename。

ADR-0011 定义了 `docs.short_id`，ADR-0012 定义了 block locator。基于这些能力，content continuation 不再需要依赖自然语言 marker，也不需要让 CLI 反解析 Markdown。Server 返回结构化的请求范围、实际输出范围和下一次建议请求参数。

## 决策

content 接口使用以下模型:

```python
class ContentRequestScope(DoclibModel):
    page_range: str | None = None
    after: str | None = None
    limit: int = 30000


class ContentRange(DoclibModel):
    page_range: str | None = None
    start: str
    end: str


class ContentNextRequest(DoclibModel):
    page_range: str | None = None
    after: str | None = None


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
```

字段含义:

| 字段 | 说明 |
|------|------|
| `request_scope` | Server 规范化后的本次读取范围，不是原始 query 参数。 |
| `content_ranges` | 本次响应实际输出内容覆盖的范围。 |
| `truncated` | 本次 `request_scope` 内是否还有内容未返回。 |
| `next_request` | Server 建议下一次请求使用的参数。可以用于继续当前范围，也可以用于读取下一段内容。 |

## 方法签名

同步 Interface:

```python
def get_doc_content(
    self,
    sha256: str,
    *,
    tier: Tier,
    page_range: str | None = None,
    after: str | None = None,
    limit: int = 30000,
    format: ContentFormat = "markdown",
    no_marker: bool = False,
) -> DocContentResponse: ...
```

异步 Interface 使用相同参数和返回类型。

HTTP API 使用 GET query，不使用 request body:

```http
GET /api/v1/docs/{sha256}/content?tier=medium&page_range=1~10&limit=30000
GET /api/v1/docs/{sha256}/content?tier=medium&page_range=7~10&after=doc:ab12cd3/tier:medium/page:7/block:4&limit=30000
GET /api/v1/docs/{sha256}/content?tier=medium&after=doc:ab12cd3/tier:medium/page:1/block:42&limit=30000
```

## 默认值

- content 接口的默认 `limit` 是 `30000`。
- CLI、HTTP API、SDK 使用同一个默认 `limit`。
- export 接口没有 `limit`，也不设计 `limit` 参数。
- 分页文档默认 `page_range=1~10`。
- 分页文档如果显式传 `page_range=all`，Server 规范化为 `1~{page_count}`。
- 非分页文档可以不传 `page_range`，也可以只传 `after`。

`limit` 是 soft limit，单位是渲染后的字符预算。Server 应尽量在自然边界停止，但协议不承诺精确等于 `limit`。

## Cursor 格式

content cursor 复用 ADR-0012 的 block reference，并允许在必要时细化到 page、block 或 char:

```text
doc:{short_id}/tier:{tier}/page:{page_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}/char:{offset}
```

规则:

- `page_no`、`block_no` 均为 1-based。
- `char:{offset}` 是 block 渲染文本内的 0-based 字符 offset。
- `char:{offset}` 是准确继续点，不是近似值。
- Server 可以为了可读性把截断点推进到段落、句子、换行或空白边界；一旦确定截断点，返回的 `char:{offset}` 必须等于实际结束位置。
- 下一次传入 `after` 时，从该 cursor 之后继续，不能重复，也不能遗漏。

## `request_scope`

`request_scope` 是 Server 最终采用的读取范围:

```json
{
  "page_range": "1~10",
  "after": null,
  "limit": 30000
}
```

规则:

- `request_scope.page_range` 使用展开和归一化后的页码范围。
- `request_scope.after` 使用请求中实际生效的 cursor。
- 如果分页文档只传 `after`，Server 根据 `after` 所在页生成默认 `page_range`，即从该页开始连续 10 页。
- 如果分页文档同时传 `page_range` 和 `after`，`after` 只在 `page_range` 定义的范围内继续，不自动扩大 scope。
- 如果 `after` 不属于 `page_range` 范围，应返回 `invalid_request_error`。

示例:

```http
GET /api/v1/docs/{sha}/content?tier=medium&after=doc:ab12cd3/tier:medium/page:21
```

若文档至少有 30 页，规范化为:

```json
{
  "request_scope": {
    "page_range": "21~30",
    "after": "doc:ab12cd3/tier:medium/page:21",
    "limit": 30000
  }
}
```

## `content_ranges`

`content_ranges` 表示本次实际输出覆盖范围:

- `content_ranges` 按输出顺序排列。
- 每个 `ContentRange` 内部连续。
- 多个 `ContentRange` 之间可以不连续。
- 分页文档常规情况下使用 page 级 cursor。
- 非分页文档常规情况下使用 block 级 cursor。
- 只有在单页或单 block 过长时，才细化到 block 或 char。

默认 `page_range=1~10` 且未截断:

```json
{
  "content_ranges": [
    {
      "page_range": "1~10",
      "start": "doc:ab12cd3/tier:medium/page:1",
      "end": "doc:ab12cd3/tier:medium/page:10"
    }
  ]
}
```

显式非连续页码且未截断:

```json
{
  "request_scope": {
    "page_range": "1~5,20~25",
    "after": null,
    "limit": 30000
  },
  "content_ranges": [
    {
      "page_range": "1~5",
      "start": "doc:ab12cd3/tier:medium/page:1",
      "end": "doc:ab12cd3/tier:medium/page:5"
    },
    {
      "page_range": "20~25",
      "start": "doc:ab12cd3/tier:medium/page:20",
      "end": "doc:ab12cd3/tier:medium/page:25"
    }
  ]
}
```

## `truncated` 与 `next_request`

`truncated` 和 `next_request` 不是一一对应关系。

规则:

- `truncated=true` 表示本次 `request_scope` 内仍有内容未返回。
- `truncated=false` 表示本次 `request_scope` 已完整返回。
- `next_request != null` 表示 Server 建议下一次请求。
- `truncated=true` 时，`next_request` 必须存在。
- `truncated=false` 时，如果文档仍有下一段建议读取内容，`next_request` 也可以存在。
- `truncated=false` 且已无后续建议内容时，`next_request=null`。

因此:

```text
truncated=true  => next_request != null
truncated=false => next_request may be null or non-null
```

## 分页文档的 `next_request`

分页文档包括 PDF 和任何有明确页序的文档。分页文档中，只要 `next_request != null`，`next_request.page_range` 必须存在。

规则:

- 多页类文档不会出现只有 `after`、没有 `page_range` 的 `next_request`。
- 页边缘截断时，`next_request` 只需要 `page_range`。
- 页内 block/char 截断时，`next_request` 同时包含 `page_range` 和 `after`。
- `next_request.page_range` 是下一次请求的 scope。
- `next_request.after` 表示在这个 scope 内从哪里继续。

### 未截断，建议下一段

请求:

```http
GET /api/v1/docs/{sha}/content?tier=medium&page_range=1~10&limit=30000
```

响应:

```json
{
  "request_scope": {
    "page_range": "1~10",
    "after": null,
    "limit": 30000
  },
  "content_ranges": [
    {
      "page_range": "1~10",
      "start": "doc:ab12cd3/tier:medium/page:1",
      "end": "doc:ab12cd3/tier:medium/page:10"
    }
  ],
  "truncated": false,
  "next_request": {
    "page_range": "11~20"
  }
}
```

`truncated=false`，但仍有 `next_request`，表示当前 scope 已读完，建议继续读下一段。

如果后续不足 10 页，则读到最后一页:

```json
{
  "next_request": {
    "page_range": "36~38"
  }
}
```

如果已经到最后一页:

```json
{
  "truncated": false,
  "next_request": null
}
```

### 页边缘截断

请求:

```http
GET /api/v1/docs/{sha}/content?tier=medium&page_range=1~10&limit=30000
```

如果 Server 只输出到第 6 页，并在页边缘停止:

```json
{
  "content_ranges": [
    {
      "page_range": "1~6",
      "start": "doc:ab12cd3/tier:medium/page:1",
      "end": "doc:ab12cd3/tier:medium/page:6"
    }
  ],
  "truncated": true,
  "next_request": {
    "page_range": "7~10"
  }
}
```

这里不需要 `after`。去掉默认尾页采样后，分页文档的大多数 continuation 都可以简化成新的连续 `page_range` 请求。

### 页内 block 截断

分页文档通常在页边缘截断。但如果单页过长，允许页内 block 截断。

请求:

```http
GET /api/v1/docs/{sha}/content?tier=medium&page_range=7~10&limit=30000
```

响应:

```json
{
  "content_ranges": [
    {
      "page_range": "7",
      "start": "doc:ab12cd3/tier:medium/page:7",
      "end": "doc:ab12cd3/tier:medium/page:7/block:4"
    }
  ],
  "truncated": true,
  "next_request": {
    "page_range": "7~10",
    "after": "doc:ab12cd3/tier:medium/page:7/block:4"
  }
}
```

如果第 7 页内某个 block 本身过长:

```json
{
  "content_ranges": [
    {
      "page_range": "7",
      "start": "doc:ab12cd3/tier:medium/page:7/block:4/char:0",
      "end": "doc:ab12cd3/tier:medium/page:7/block:4/char:32784"
    }
  ],
  "truncated": true,
  "next_request": {
    "page_range": "7~10",
    "after": "doc:ab12cd3/tier:medium/page:7/block:4/char:32784"
  }
}
```

## 非连续 `page_range`

显式 `page_range` 和默认 `page_range` 一样生成 `next_request`。

如果请求是非连续页码，且没有发生截断，下一段建议从当前请求的最后一页的下一页开始，忽略当前 `page_range` 中间略过的页。

示例:

```http
GET /api/v1/docs/{sha}/content?tier=medium&page_range=1~5,20~25
```

若文档还有第 26 页之后的内容:

```json
{
  "request_scope": {
    "page_range": "1~5,20~25",
    "after": null,
    "limit": 30000
  },
  "truncated": false,
  "next_request": {
    "page_range": "26~35"
  }
}
```

`6~19` 不会被自动补读。用户显式跳过的页段不属于 Server 的下一段建议。

## 非分页文档

非分页文档通常只有一个逻辑 page，包含多个 block。它的 continuation 依赖 block locator。

规则:

- 非分页文档可以只传 `after`。
- 非分页文档的 `next_request` 可以只有 `after`。
- 常规截断边界是 block。
- 极端长单 block 才使用 char offset。

普通 block 截断:

```json
{
  "request_scope": {
    "page_range": null,
    "after": null,
    "limit": 30000
  },
  "content_ranges": [
    {
      "page_range": null,
      "start": "doc:ab12cd3/tier:medium/page:1/block:1",
      "end": "doc:ab12cd3/tier:medium/page:1/block:42"
    }
  ],
  "truncated": true,
  "next_request": {
    "after": "doc:ab12cd3/tier:medium/page:1/block:42"
  }
}
```

长单 block 截断:

```json
{
  "content_ranges": [
    {
      "page_range": null,
      "start": "doc:ab12cd3/tier:medium/page:1/block:1/char:0",
      "end": "doc:ab12cd3/tier:medium/page:1/block:1/char:32784"
    }
  ],
  "truncated": true,
  "next_request": {
    "after": "doc:ab12cd3/tier:medium/page:1/block:1/char:32784"
  }
}
```

## CLI Marker

CLI marker 是展示层，不是协议源头。Server 返回 `next_request`，CLI 根据 `next_request` 生成用户可复制的命令提示。

分页文档:

```html
<!-- Next: mineru parse report.pdf --pages 11~20 -->
```

分页文档页内 continuation:

```html
<!-- Next: mineru parse report.pdf --pages 7~10 --after doc:ab12cd3/tier:medium/page:7/block:4 -->
```

非分页文档:

```html
<!-- Next: mineru parse long.docx --after doc:ab12cd3/tier:medium/page:1/block:42 -->
```

机器可读协议以 `DocContentResponse.next_request` 为准。CLI marker 不应被 SDK 或 Server 反解析。

## 替代方案

### 方案 A: 只返回 `truncated`，让客户端用 `content_ranges[-1].end` 推断

拒绝。它只能处理当前请求范围内的 continuation，不能表达“当前范围读完后建议下一段读什么”。

### 方案 B: 保留 `continuation` 对象

拒绝。`continuation.after` 与 `content_ranges[-1].end` 大多数情况下重复，`reason` 也可从 cursor 粒度和 `next_request` 形态推断。最终保留 `next_request`，直接表达下一次请求参数。

### 方案 C: 用 server-side read session 记录已读页

拒绝。它需要状态存储、过期、清理和并发语义。P0 采用无状态请求。默认页改为线性 `1~10` 后，分页文档可以用简单的下一段 `page_range` 推荐避免尾页重复。

### 方案 D: 分页文档 continuation 总是用 `after`

拒绝。分页文档最自然的继续方式是页码范围。常规页边缘截断应简化为 `next_request.page_range`，只有页内 block/char 截断才需要 `after`。

### 方案 E: 继续使用默认 `1~5,-5~-1`

拒绝。头尾采样会导致无状态渐进阅读时容易重复读取尾页，也让 `next_request` 必须携带额外 read state 才能正确推荐后续页。P0 改为默认 `1~10`，使阅读计划线性、可预测。

## 影响

- `DocContentResponse` 需要增加 `short_id`、`format`、`request_scope`、`content_ranges`、`truncated`、`next_request`。
- doclib content 渲染层需要支持 `limit`、`page_range`、`after` 和范围计算。
- render markdown 应支持 soft limit，并尽量在 page/block/自然文本边界停止。
- CLI 应根据 `next_request` 渲染下一条命令提示。
- CLI 不应反解析 Markdown marker 来决定下一次请求。
- export 接口保持不截断，不增加 `limit`。
- 分页文档默认页码范围使用 `1~10`。
- 分页文档只要 `next_request` 非空，就必须包含 `page_range`。

## 后续动作

1. 更新 doclib Interface、Client、Server 类型和签名。
2. 实现 content cursor parser 和 validator。
3. 实现 `request_scope` 规范化:
   - 默认 `page_range=1~10`。
   - 分页文档只传 `after` 时推导连续 10 页。
   - 校验 `after` 属于 `page_range` scope。
4. 实现 `content_ranges` 和 `next_request` 计算。
5. 为 Markdown render 增加 soft limit。
6. 更新 CLI parse 输出 marker，使其从 `next_request` 生成。
7. 增加测试:
   - 默认 `1~10`。
   - 未截断时生成下一段 `page_range`。
   - 页边缘截断时只返回 `next_request.page_range`。
   - 页内 block/char 截断时返回 `page_range + after`。
   - 多页类文档 `next_request` 非空时一定有 `page_range`。
   - 非分页文档可以只有 `after`。
   - 非连续 page_range 的下一段从最后一页之后开始。
   - export 不受 `limit` 影响。
