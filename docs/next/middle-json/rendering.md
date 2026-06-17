# Rendering Contract

状态: Draft
读者: render 开发者、backend 开发者、SDK 开发者
范围: Markdown / Content List / Structured Content 对 Middle JSON 的消费约束和收敛计划
底稿: `../../../NEXT-JSON.md`

## 当前状态

当前已有统一入口:

```python
from mineru.render import render_markdown, render_content_list, render_content_list_v2
```

`ParseResult` 也通过这些入口生成输出:

- `ParseResult.markdown()`
- `ParseResult.content_list()`
- `ParseResult.content_list_v2()`

`content_list_v2` 是当前代码函数名；NEXT 版公开格式名定为 Structured Content，命名决策见 [ADR-0001](../decisions/0001-json-output-formats.md)。

这说明底稿中“三套 union_make 完全割裂”的问题已经部分解决。

## 仍未完全收敛

统一入口内部仍存在 backend dispatch:

| 输出 | 当前状态 |
|------|----------|
| Markdown | 有统一 facade；Office 仍走 office-specific markdown。 |
| Content List v1 | 有统一 facade；Office 仍走 office-specific converter。 |
| Content List v2 | Pipeline / VLM / Office 仍分别调用不同实现。 |

此外，render 当前通过 `PageInfo._backend` 判断 backend。这是临时实现，不应成为长期 schema。

## Render 输入契约

render 函数应接受:

```python
list[PageInfo]
```

或 canonical envelope 经解析后的:

```python
payload["pages"] -> list[PageInfo]
```

render 不应直接依赖:

- 原始模型输出。
- doclib 数据库。
- 文件系统路径，除非渲染图片链接需要 `img_bucket_path`。
- 私有 `_meta` 之外的 backend-specific hidden state。

## Block 消费规则

| Block type | Markdown | Content List |
|------------|----------|--------------|
| `text` | 普通段落。 | text item。 |
| `title` | heading，使用 `level`。 | text item with level。 |
| `list` | 列表。 | list item。 |
| `index` | 目录。 | index/list item。 |
| `interline_equation` | display math。 | equation item。 |
| `image` | image markdown，包含 caption/footnote。 | image item。 |
| `table` | HTML 或 table image。 | table item。 |
| `chart` | chart image/content。 | chart item。 |
| `code` | fenced code block。 | code item。 |
| `header/footer/page_number` | 默认可忽略或作为 discarded。 | 可进入 content list，但需标注类型。 |

## Unknown BBox

render 不应因为 `bbox=EMPTY_BBOX` 失败。Content List 若输出 bbox，应遵循:

- 已知 bbox: 输出 bbox。
- unknown bbox: 省略 bbox 或输出 null。
- 不应把 `(0,0,0,0)` 当成真实 bbox 给 UI 画框。

## `_backend` 迁移

当前:

```python
backend = pages[0]._backend
```

目标:

```python
backend = envelope["_meta"]["backend"]
```

迁移策略:

1. render facade 新增可选 `backend` 参数。
2. 如果传入 envelope，则从 `_meta.backend` 读取。
3. 如果只传 pages，短期继续 fallback 到 `PageInfo._backend`。
4. validator 提醒 `_backend` 是 legacy/internal。
5. backend-specific 逻辑收敛后，render 不再需要 backend 判断。

## Content List v2 收敛

P0:

- 明确 Content List v2 的目标 schema。
- 确保所有 backend 输出都能生成 v2。
- 保留当前 backend dispatch，先补 validator。

P1:

- 抽出通用 block-to-v2 映射。
- 将 pipeline/vlm/office 特殊逻辑变成 type-specific helper，而不是 backend-specific module。

P2:

- 删除 backend-specific v2 converter。
- render 只依赖 block type，不依赖 backend。

## Agent Marker

为了 Agent 引用，Markdown 可以支持可选 locator marker:

```markdown
<!-- mineru:locator doc:ab12cd3/tier:standard/page:3/block:12 -->
正文内容
```

要求:

- 默认不输出，避免影响普通用户。
- SDK/Agent 可显式开启。
- marker 中不得包含本地文件路径或 filename。

## 验收

- `render_markdown()` 可消费 migrated canonical envelope 的 pages。
- `render_content_list()` 不依赖旧 `pdf_info`。
- Office unknown bbox 不导致 content list 坐标错误。
- 所有 render 输出可以带可选 Agent locator。
- 逐步减少 backend-specific dispatch。
