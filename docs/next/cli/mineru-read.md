# mineru read

状态: Draft
读者: Agent skill 作者、CLI 使用者、核心开发者
范围: `mineru read` 的定位、locator 输入、markdown/image 输出、context 和 continuation
非目标: 从文件 path 触发解析；批量目录扫描；解析 backend 专家参数
来源: 由根目录旧 CLI 底稿迁移整理而来，相关 ADR: [0014-mineru-read-command](../decisions/0014-mineru-read-command.md)

## 1. 定位

`mineru read` 是 `mineru` 的 locator-first 读取命令。它面向“文档已经进入 doclib，后续要继续读其中某个 doc、tier、page、block”的场景。

职责边界：

```text
parse(path) = ensure document is parsed, then read default content
read(locator) = read existing parsed content by stable locator
```

`read` 不负责 discover、scan、ingest，也不默认创建 parse task。它读取的是 doclib 中已有的解析结果。

text 文件没有解析结果；当 locator 指向 text doc 时，返回 `parse_not_required`，调用方应直接读取源文件。

## 2. Usage

```bash
mineru read <locator> [flags]
```

当前 P0 支持：

```bash
mineru read <locator> [--format markdown|image] [--limit 30000] [--context N] [--output PATH] [--json] [--no-marker]
```

## 3. Locator 输入

`read` 支持以下粒度的 locator：

```text
doc:{short_id}
doc:{short_id}/tier:{tier}
doc:{short_id}/tier:{tier}/page:{page_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}/char:{offset}
```

规则：

- `short_id` 是 doclib 为文档生成的稳定短 ID。
- `tier` 取值为 `flash`、`medium`、`high`、`xhigh`。
- `page_no` 和 `block_no` 使用 1-based 编号。
- `char:{offset}` 使用 block 渲染文本内的 0-based 字符 offset。

当只给出 `doc:{short_id}` 时，系统不会创建新解析，而是在当前已缓存的非 `flash` 质量 tier 中选择最高质量结果，顺序为 `xhigh` -> `high` -> `medium`。如果不存在非 `flash` 质量 tier 结果，则返回错误，不静默降级到 `flash`。

## 4. 核心参数

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--format`, `-f` | `markdown` / `image` | `markdown` | 输出 content format |
| `--json` | bool | false | 将整个 CLI 响应包装为 JSON |
| `--limit` | int | `30000` | 文本输出软字符上限 |
| `--context` | int | `0` | 按 locator 周围扩展上下文 |
| `--output`, `-o` | path | 不传 | 将返回内容或 image asset 写入本地路径；image 只支持 `.png`、`.jpg`、`.jpeg`、`.webp` |
| `--no-marker` | bool | false | 关闭 continuation marker |

语义区分：

- `--format` 控制 content 本身的格式。
- `--json` 控制 CLI 最外层输出是否为 JSON envelope。

## 5. Markdown 读取

`--format markdown` 是默认模式。

| Locator | 默认读取范围 |
|---------|--------------|
| `doc:{short_id}` | 默认 tier 的默认阅读范围 |
| `doc:{short_id}/tier:{tier}` | 该 tier 的默认阅读范围 |
| `.../page:{page_no}` | 该页 |
| `.../block:{block_no}` | 该 block |
| `.../char:{offset}` | 从该 block 的 offset 起读取 |

`--limit` 是软截断，尽量在页、block、段落、句子、换行或空白边界停止。

### 可视 block 引用

Markdown 继续优先输出 table、chart 和 formula 的结构化内容。需要输出图片时，doclib 使用可读取的 block locator，不暴露 Middle JSON 内部的 `image_path`：

```markdown
![Image block](doc:ab12cd3/tier:high/page:1/block:4)
![Table block]()
```

非空 locator 表示该 block 当前可通过 `mineru read <locator> --format image` 读取；空 locator 表示没有可用图片。`--output` 只写 Markdown 文件，保留这些虚拟引用，不创建 `images/` 目录。

### `--context`

`--context N` 的含义随 locator 粒度变化：

| Locator | `--context` 行为 |
|---------|------------------|
| page locator | 读取前后各 N 页 |
| block locator | 读取前后各 N 个 block |
| char locator | 按 block locator 处理上下文，char 只影响起点 |
| doc locator | 返回 `context_not_applicable` |
| doc/tier locator | 返回 `context_not_applicable` |

## 6. Image 读取

`--format image` 是 P0 能力，但范围受文档类型和 locator 粒度限制。

不支持多页 image。

| 文档类型 | doc locator | doc/tier locator | page locator | block locator |
|----------|:-----------:|:----------------:|:------------:|:-------------:|
| PDF / image | 不支持 | 不支持 | 支持 | 有有效 bbox 时支持；否则可回退到 sidecar |
| Office / HTML | 不支持 | 不支持 | 不支持 | visual block 有可访问 image sidecar 时支持 |

实现语义：

- PDF / image page image：从源文件重新渲染页面。
- 有有效 bbox 的 block image：从源页面渲染后按 bbox 裁剪。
- 无有效 bbox 的 visual block：读取 doclib parsed 目录中的 image sidecar，并按请求格式生成临时 asset。
- Markdown 中出现的非空 visual block locator 即表示该 block 支持 image 读取；源文件或 sidecar 在读取前被外部删除时，读取仍可能失败。
- `read --output` 是 CLI 本地写文件行为，不是 doclib HTTP API 写文件行为。

如果不带 `--output`，image 模式默认输出 server 返回的临时 asset path。

如果带 `--output`，CLI 根据输出路径后缀决定请求的图片编码：

| 后缀 | server `image_format` |
|------|------------------------|
| `.png` | `png` |
| `.jpg`, `.jpeg` | `jpeg` |
| `.webp` | `webp` |

image 输出路径必须带上述后缀。无后缀、其它后缀以及 `--output -` 都返回 `image_output_extension_unsupported`。CLI 不做图片转码；server 按 `image_format` 生成临时 asset，CLI 只把该 asset copy 到用户指定路径。

## 7. JSON 输出

`--json` 时，CLI 输出 `DocContentResponse`。

locator-first 读取时：

- `request_scope.locator` 为规范化 locator。
- `request_scope.context` 为实际生效的 context。
- `next_request` 只写 `locator`。

## 8. Continuation

`read` 的 continuation 使用 locator 模型。

普通 markdown 输出中，如果还有下一段可读内容，CLI 会输出：

```text
<!-- Next: mineru read doc:ab12cd3/tier:medium/page:5 -->
```

`--json` 时，通过 `next_request.locator` 表示下一次建议读取的位置。

## 9. 与 mineru parse 的关系

建议工作流：

1. 第一次从 path 出发，使用 `mineru parse <file>`。
2. 需要继续阅读时，切换到 `mineru read <locator>`。

`parse` 适合“把文件纳入 MinerU 并读默认内容”。`read` 适合“基于已有定位继续读指定位置或导出 page/block 图像”。
