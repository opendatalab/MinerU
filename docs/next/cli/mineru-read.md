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
mineru read <locator> [--after CURSOR] [--format markdown|image] [--limit 30000] [--context N] [--output PATH] [--json] [--no-marker]
```

## 3. Locator 输入

`read` 支持以下粒度的 locator：

```text
doc:{short_id}
doc:{short_id}/tier:{tier}
doc:{short_id}/tier:{tier}/page:{page_selection}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}
doc:{short_id}/tier:{tier}/page:{page_no}/block:{block_no}/char:{offset}
```

规则：

- `short_id` 是 doclib 为文档生成的稳定短 ID。
- `tier` 取值为 `flash`、`basic`、`standard`、`advanced`。
- `page_no` 和 `block_no` 使用 1-based 编号。
- `page_selection` 与解析输入页范围共用语法：数字、`rN`、区间、逗号组合和 `all`。`r1` 为最后一页；倒数页和 `all` 必须有可靠的文档总页数。
- 页选择排序、去重并裁剪越界页；返回的规范定位器和块引用使用实际页码。带 block/char 时必须解析为单页。
- `char:{offset}` 使用 block 渲染文本内的 0-based 字符 offset。

```bash
mineru read doc:ab12cd3/tier:flash/page:r1
mineru read doc:ab12cd3/tier:flash/page:r3-r1
mineru read doc:ab12cd3/tier:flash/page:1,3,r1
mineru read doc:ab12cd3/tier:flash/page:all
```

当只给出 `doc:{short_id}` 时，系统不会创建新解析，而是在当前已缓存的非 `flash` 质量 tier 中选择最高质量结果，顺序为 `advanced` -> `standard` -> `basic`。如果不存在非 `flash` 质量 tier 结果，则返回错误，不静默降级到 `flash`。

## 4. 核心参数

| Flag | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--format`, `-f` | `markdown` / `image` | `markdown` | 输出 content format |
| `--json` | bool | false | 将整个 CLI 响应包装为 JSON |
| `--limit` | int | `30000` | 文本输出软字符上限 |
| `--after` | locator | 不传 | 使用返回的内容游标续读；要求 Markdown 文档定位或页选择，且 context=0 |
| `--context` | int | `0` | 按 locator 周围扩展上下文 |
| `--output`, `-o` | path | 不传 | 仅将 image asset 写入本地路径；Markdown 仅接受 `-` 表示 stdout；image 只支持 `.png`、`.jpg`、`.jpeg`、`.webp` |
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
![Image block](doc:ab12cd3/tier:standard/page:1/block:4)
![Table block]()
```

非空 locator 表示该 block 当前可通过 `mineru read <locator> --format image` 读取；空 locator 表示没有可用图片。Markdown 阅读仅返回 stdout／JSON；保存当前片段请使用重定向或 SDK，不支持 `--output PATH`。返回的文本保留这些虚拟引用。

### `--context`

`--context N` 的含义随 locator 粒度变化：

| Locator | `--context` 行为 |
|---------|------------------|
| page locator | 读取前后各 N 页 |
| block locator | 读取前后各 N 个 block |
| char locator | 按 block locator 处理上下文，char 只影响起点 |
| doc locator | 返回 `context_not_applicable` |
| doc/tier locator | 返回 `context_not_applicable` |
| 多页选择 | 非零 context 返回 `context_not_applicable` |

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

仅 `--format image --output PATH --json` 会保存 server 返回的 asset，并返回 `content=null` 和额外的 `output` 字段（`{"status": "written", "path": "..."}`）。Markdown 携带实际输出路径时，在读取服务或创建目录之前返回 `invalid_request`（`param=output`）；`--output -` 与省略此参数等价。

locator-first 读取时：

- `request_scope.locator` 为规范化 locator。
- `request_scope.context` 为实际生效的 context。
- 页或文档读取在页内截断时，`next_request` 使用文档 `locator` 与 `after`；读完当前页后恢复下一页建议。
- 单块读取保留块/字符定位的续读方式，不自动扩大到同页其他块。
- 范围内截断时同时返回原范围 `locator` 和 `after`。
- 多页选择只读取范围内的已缓存内容，读完后 `next_request=null`；不会创建新解析或继续推荐范围外页面。

## 8. Continuation

`read` 的 continuation 使用 locator 模型。

普通 markdown 输出中，如果还有下一段可读内容，CLI 会输出：

```text
<!-- Next: mineru read doc:ab12cd3/tier:basic/page:5 -->
```

`--json` 时，`next_request` 表示下一次建议读取的请求；续读时应同时传递
其中的 `locator` 和非空 `after`，而不是只使用 `locator`。

页或文档读取因长度限制在页内停止时，续读使用文档定位符和已消费位置：

```text
<!-- Next: mineru read doc:ab12cd3/tier:flash --after doc:ab12cd3/tier:flash/page:1/block:2 -->
```

这会读取第 1 页第 2 块之后的剩余内容；若游标带有 `char`，从对应字符偏移
继续。当前页读完后，仍按单点阅读规则建议下一页。页级 `after` 游标表示该页
已经读完，从下一页开始。它不会重新解析文档，也不会扩大直接单块读取的范围。

以上页或文档续读语义适用于 `context=0`。块或字符定位的 `context>0` 上下文窗口在截断续读时存在重复与遗漏，属于待单独修复的已知限制。

范围读取的续读保留原选择，并以实际页码的块/字符游标推进。例如：

```text
<!-- Next: mineru read doc:ab12cd3/tier:flash/page:1,3 --after doc:ab12cd3/tier:flash/page:1/block:2/char:1000 -->
```

可以直接执行该命令；第 2 页不会因续读被加入。图片读取要求页选择解析为单页，多页选择返回 `multi_page_image_not_supported`。

## 9. 与 mineru parse 的关系

建议工作流：

1. 第一次从 path 出发，使用 `mineru parse <file>`。
2. 需要继续阅读时，切换到 `mineru read <locator>`。

`parse` 适合“把文件纳入 MinerU 并读默认内容”。`read` 适合“基于已有定位继续读指定位置或导出 page/block 图像”。

## 10. 阅读与保存

`read` 返回受字符预算约束的阅读片段。保存当前片段可使用 `mineru read <locator> --no-marker > part.md`，或通过 SDK 保存返回文本。这里不会自动续读、追加或拼接。

需要导出已解析的完整指定页范围时使用 `mineru parse <path> --pages <range> -o result.md`；该入口复用缓存并执行完整内容导出，不受阅读字符预算限制。图片读取与图片路径输出继续保留。
