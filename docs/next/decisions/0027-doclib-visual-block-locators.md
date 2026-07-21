# ADR-0027: Doclib 视觉 Block Locator 图片引用

状态: Accepted
日期: 2026-07-17
相关文档: 0012-doclib-block-locator.md, 0014-mineru-read-command.md, 0015-cli-output-json-composition.md, ../architecture.md, ../cli/mineru-read.md

本 ADR 补充 ADR-0014 的 Markdown 图片引用和 block image 读取语义。ADR-0014 中 locator-first、源页面 bbox 裁图、`DocContentResponse` 与 CLI `--output` 的其他决策保持不变。

## 背景

doclib 的目标是保存可渐进读取的 Middle JSON，而不是为大量 PDF 或 image 长期保存所有裁图。当前 PDF 和 image block 使用可映射回源页面的 bbox；Office 因缺少可靠 bbox，会保存解析得到的图片 sidecar。

PDF Middle JSON 仍可能包含解析阶段生成的 `image_path`。通用 Markdown renderer 会把它渲染为:

```md
![](images/<internal-hash>.jpg)
```

但 doclib 没有对应的 PDF sidecar，导致用户得到悬空链接。即使 sidecar 存在，哈希文件名也是缓存实现细节，不是 doclib 的稳定公共标识。doclib 已经使用 `doc/tier/page/block` locator 标识内容，因此视觉 block 也应使用同一套 locator，而不是暴露内部 `image_path`。

## 决策

### 1. Doclib Markdown 使用虚拟 block 图片引用

doclib 对外生成的 Markdown 使用 block locator 表示可读取的视觉 block:

```md
![Image block](doc:ab12cd3/tier:standard/page:1/block:4)
![Table block](doc:ab12cd3/tier:standard/page:1/block:5)
![Chart block](doc:ab12cd3/tier:standard/page:1/block:6)
![Formula block](doc:ab12cd3/tier:standard/page:1/block:7)
```

locator 为空表示原文存在对应视觉 block，但 doclib 当前没有可读取的图片:

```md
![Image block]()
![Table block]()
![Chart block]()
![Formula block]()
```

非空 locator 是公共能力承诺: 在生成该 Markdown 时，这个 block 具备明确的 image 读取路径。用户或 Agent 可以执行:

```bash
mineru read "doc:ab12cd3/tier:standard/page:1/block:4" --format image --output image.jpg
```

源文件或缓存文件在 Markdown 生成后仍可能被移动、删除或损坏；locator 不对未来文件状态提供永久保证。

### 2. Locator 可用性判定

视觉 block 按以下顺序判定是否输出非空 locator:

1. block 有有效 bbox 时，输出 canonical block locator。当前只有 PDF 和 image block 会产生这种 bbox，且两者都能重新渲染源页面；渲染 Markdown 时不检查源文件是否存在。
2. 否则，递归检查 block 中的视觉 span。如果存在安全的 `image_path`，并且它在当前文档和 tier 的 doclib sidecar 目录中对应实际文件，则输出 canonical block locator。
3. 其他情况输出对应类型的空 locator。

有效 bbox 沿用现有规则: bbox 必须包含四个有效数值，不等于 empty bbox，并且 `x0 < x1`、`y0 < y1`。

sidecar 只能从以下目录解析:

```text
{doclib_data_dir}/parsed/{sha256_prefix}/{sha256}/{tier}/images/{image_path}
```

`image_path` 必须经过既有安全相对路径校验，最终目标必须满足 `is_file()`。不能对原始 `image_path` 直接调用 `Path.exists()`，也不能把 URL、绝对路径或包含上跳片段的路径当作本地 sidecar。

如果一个 block 含有多个候选 sidecar，resolver 按 block/span 文档顺序选择第一个安全且存在的文件。Markdown 渲染和 image 读取必须共享同一个 resolver，保证选择规则一致。

### 3. Block Image 读取

`mineru read <block-locator> --format image` 使用与 Markdown 相同的解析顺序:

1. block 有有效 bbox 时，从当前可访问的 PDF 或 image 源页面即时裁剪。
2. 否则，读取 resolver 选中的 doclib sidecar。
3. 两种方式都不可用时，返回现有结构化错误。

这会扩展 ADR-0014 中“Office 仅 image block 支持 image 输出”的限制。只要 table、chart 或其他视觉 block 有可用 sidecar，它的 block locator 也可以读取图片。HTML 的内嵌图片 sidecar 使用相同规则，不再错误进入 PDF 裁图路径。

运行时错误沿用现有语义:

- PDF 或 image 源文件不可访问: `no_accessible_file`。
- bbox 在实际读取时不可用: `bbox_not_available`。
- sidecar 不存在或不可读取: `asset_not_available`。
- locator 指向不存在的 block: `block_not_found`。

### 4. 结构化内容与图片占位

Doclib renderer 只替换原本会产生图片引用的部分:

- image block 使用 `Image block`。
- 图片型 table 使用 `Table block`。
- 图片型 chart 使用 `Chart block`。
- 图片型公式使用 `Formula block`。
- table、chart 或公式已有 Markdown、HTML、文本或 LaTeX 内容时，继续输出结构化内容，不额外增加图片占位。
- caption、footnote 和其他文本内容保持原有顺序。

通用 Markdown renderer 增加可选的 block-level 图片扩展点:

```python
ImageRenderer = Callable[[Block], str]

def blocks_to_markdown(
    blocks: list[Block],
    ...,
    image_renderer: ImageRenderer | None = None,
) -> list[str]:
    ...
```

未传 `image_renderer` 时，继续使用既有 image path 渲染逻辑，按 `image_path` 和 `img_bucket_path` 生成普通 Markdown 图片引用。doclib 传入绑定当前文档上下文的 closure 或 partial，返回完整的类型化虚拟图片引用。

`Block` 不包含 doclib 的 `data_dir`、`sha256`、`short_id`、`tier` 和 `page_no`。doclib 因此通过 factory 创建 `Callable[[Block], str]`，在 callable 中绑定这些上下文。通用 render 模块保持纯函数，不访问 doclib 数据库或磁盘。

renderer 只在原本需要输出图片的位置调用 `image_renderer`。caption、footnote、结构化 table/chart/formula 内容继续复用现有逻辑。默认实现的输出行为保持不变，避免影响 Parse SDK、Parse API、`mineru-kit` 和其他直接消费 ParseResult/Middle JSON 的调用方。

### 5. `image_path` 是 Doclib 内部实现细节

`image_path` 可以继续存在于 doclib 内部缓存的 Middle JSON 中，用于定位 Office、HTML 或其他解析结果的 sidecar，但不得出现在 doclib 返回的 Markdown 中。

本 ADR 不修改公共 Middle JSON、Parse SDK、Parse API 或 `mineru-kit` 的 `image_path` 契约。它们仍可以将 `image_path` 作为 parse output sidecar 的相对路径。

### 6. 所有 Doclib Markdown 输出保持一致

以下入口使用同一虚拟图片引用语义:

- `mineru read` 的终端输出。
- `mineru read --json` 的 `content`。
- `mineru read --output <file.md>`。
- `mineru parse` 完成后的 doclib 内容输出。
- doclib content 和 export endpoints。

`--output` 只写 Markdown，不创建 `images/` 目录，不复制 Office sidecar，也不按需生成 PDF 图片。输出文件保留 `doc:...` 虚拟引用；需要图片时由用户或 Agent 显式调用 `mineru read --format image`。

## 替代方案

### 继续输出 `images/<image_path>`

未采用。PDF 默认不保存 sidecar，链接会悬空；内部哈希名称也会成为不必要的公共兼容契约。

### 导出 Markdown 时自动生成或复制 `images/`

未采用。它会让普通读取隐式产生大量 PDF 图片，增加磁盘占用，并使 `--output` 同时承担内容读取和资源打包职责。

### Markdown 生成后通过正则替换图片链接

未采用。正则无法可靠映射 block locator，也难以正确处理 HTML 图片、一个 block 内的多个 span、caption 顺序和渐进式字符截断。

### 从 Middle JSON 删除 `image_path`

未采用。`image_path` 对 Office sidecar 和其他 parse output 仍有内部价值，并且删除会扩大到 Parse SDK、Parse API 和 `mineru-kit` 的兼容性变化。

## 影响

- Doclib Markdown 不再是普通 Markdown reader 可直接加载图片的自包含文档；`doc:` 是 MinerU-aware client 和 Agent 使用的虚拟引用。
- 视觉 block 是否具有非空 locator，明确表达其当前 image 读取能力。
- PDF 和 image 继续避免长期保存大量裁图，只在显式 image read 时按 bbox 生成。
- Office 和 HTML 可以通过现有 sidecar 支持 block image read。
- 内部图片命名和存储策略可以演进，不影响 doclib 的公共 Markdown 契约。
- 现有依赖 `images/<hash>` 的 doclib Markdown 消费方需要改为识别 block locator。

## 测试要求

- block 有有效 bbox 时输出非空 locator，并可通过同一 locator 从 PDF 或 image 源页面裁图。
- block 无有效 bbox，但存在 sidecar 时使用 sidecar。
- 无有效 bbox且 sidecar 不存在时输出类型化空 locator。
- Office/HTML 的 image、table、chart sidecar locator 可以通过 `--format image` 读取。
- image、table、chart、formula 使用对应 alt text。
- caption、footnote、结构化 table/chart/formula 内容保持不变。
- Doclib Markdown 中不出现内部 `image_path` 或 `images/<internal-name>`。
- `--output` 只写 Markdown，不创建资源目录。
- URL、绝对路径、上跳路径和目录不能被当作 sidecar。
- sidecar 在 Markdown 生成后被删除时，image read 返回 `asset_not_available`。

## 后续动作

- 在 doclib server 中增加共享的 block image resolver。
- 为通用 Markdown renderer 增加默认兼容的 `image_renderer` 扩展点。
- 让 doclib 通过绑定文档上下文的 `image_renderer` 生成类型化虚拟 block 图片引用。
- 扩展 block image read，使无 bbox block 可以读取已有 sidecar。
- 更新 ADR-0014、CLI、API、SDK 和 workflow 文档中的 doclib 图片引用说明。
- 在 issue #5274 中说明最终行为和兼容边界。
