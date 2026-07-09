# ADR-0022: Doclib 文件类型、Tier 与 Remote 语义

状态: Accepted，已由 ADR-0024 补充
日期: 2026-06-29
相关文档: ../cli/mineru-parse.md, ../tiers.md, ../errors.md, 0024-file-type-tier-normalization.md

## 背景

`mineru parse` 支持多种输入格式，但不同格式背后的解析路径并不相同：

- PDF 和 image 可以进入质量 tier 路径，支持 `flash`、`medium`、`high` 和 `xhigh`。
- Office 文件走专门的 Office parser，实际只支持 `flash`。
- text / html 类输入不走 PDF 质量后端，也只具备 `flash` 语义。

此前 Office 文件显式传入质量 tier 时，会在服务层静默覆盖为 `flash`。CLI 和 JSON 输出只显示最终 tier，无法表达“用户请求被忽略”。这会让自动化脚本、测试和用户判断误以为质量 tier 已被执行。

同时，image 文件需要支持显式 `parse` 并进入 doclib，但 watch / scan 不应把目录中发现的图片自动当作文档纳入库。doclib 单对象主动解析的 `--remote` 也需要明确边界：remote 后端只用于质量解析，不支持 `flash`，也不支持非 PDF / image 输入。

因此需要把文件类型、tier、remote 的组合语义显式化，避免静默降级。

## 决策

### 文件类型集合

doclib 区分两个文件集合：

- `DISCOVERABLE_EXTENSIONS`：watch / scan 自动发现的文档格式。
- `PARSEABLE_EXTENSIONS`：显式 `mineru parse` 可接受的格式。

image 扩展名只进入 `PARSEABLE_EXTENSIONS`，不进入 `DISCOVERABLE_EXTENSIONS`。因此：

- 用户显式执行 `mineru parse image.png` 时，图片会入库、生成 `files` / `docs` 记录，并创建 parse 任务。
- watch / scan 遇到图片文件时忽略，不自动把图片当作文档发现。
- 显式 parse 后，image 与其它文档一样可出现在 show / list / search / find 等 doclib 查询面。

`ALLOWED_EXTENSIONS` 不再保留。调用方应根据语义选择 `DISCOVERABLE_EXTENSIONS` 或 `PARSEABLE_EXTENSIONS`。

### Tier 能力矩阵

质量 tier 只对 PDF 和 image 有意义。

| 文件类型 | `flash` | `medium` | `high` | `xhigh` |
|----------|---------|----------|--------|---------|
| PDF | 支持 | 支持 | 支持 | 支持 |
| image | 支持 | 支持 | 支持 | 支持 |
| Office (`docx` / `pptx` / `xlsx`) | 支持 | 不支持 | 不支持 | 不支持 |
| text / html (`txt` / `md` / `csv` / `rst` / `tex` / `html` / `htm`) | 支持 | 不支持 | 不支持 | 不支持 |

当非 PDF / image 文件在 doclib 单对象主动解析中显式请求质量 tier 时，doclib 不再自动降级为 `flash`，而是返回 `InvalidRequestError`。批量、parsing-rule、API Server 和 mineru-kit 的归一化例外见 [ADR-0024](0024-file-type-tier-normalization.md)。

- `code`: `tier_unsupported_for_file_type`
- `param`: `tier`

未显式传入 tier 的非 PDF / image 文件继续按 `flash` 语义处理。

### Remote 能力矩阵

doclib 单对象主动解析中的 `--remote` 只支持 PDF 和 image，并且不支持 `flash` tier。

| 文件类型 | `--remote` |
|----------|------------|
| PDF | 支持非 `flash` 质量 tier，不支持 `flash` |
| image | 支持非 `flash` 质量 tier，不支持 `flash` |
| Office | 不支持 |
| text / html | 不支持 |

当非 PDF / image 文件在 doclib 单对象主动解析中请求 `--remote` 时，doclib 返回 `InvalidRequestError`：

- `code`: `remote_unsupported_for_file_type`
- `param`: `remote`

当 PDF / image 同时请求 `--remote --tier flash` 时，doclib 返回 `InvalidRequestError`：

- `code`: `tier_unsupported_for_remote`
- `param`: `tier`

当 PDF / image 请求 `--remote` 且未显式传入 tier 时，继续使用 [解析 Tier](../tiers.md) 中定义的默认选择策略。

### 错误优先级

当多个参数同时冲突时，优先返回更贴近显式冲突参数的错误：

1. doclib 单对象主动解析中，`--remote` 与非 PDF / image 文件冲突时，返回 `remote_unsupported_for_file_type`。
2. PDF / image 的 `--remote --tier flash` 冲突时，返回 `tier_unsupported_for_remote`。
3. doclib 单对象主动解析中，非 PDF / image 的质量 tier 冲突时，返回 `tier_unsupported_for_file_type`。

例如 `sample.docx --remote --tier flash` 优先报告 remote 不支持该文件类型，而不是报告 remote 不支持 flash。

## 替代方案

### 方案 A：继续自动降级为 `flash`，在 `tip` 中提示

未采用。

原因：

- 自动化脚本可能只检查退出码和 `status=done`，warning / tip 容易被忽略。
- 用户显式指定质量 tier 表示质量预期，静默或半静默降级会制造错误判断。
- 当前响应模型只有 `tip`，没有稳定的 warnings 列表；把冲突语义塞进提示字段不够可靠。

### 方案 B：为所有格式保留质量 tier 名义兼容

未采用。

原因：

- Office、text、html 并没有对应 PDF 质量后端能力。
- 名义兼容会让 `parse.tier` 和 locator tier 与实际解析路径脱节。
- 后续如果这些格式真的支持质量 tier，应作为新能力显式加入，而不是提前保留虚假语义。

### 方案 C：watch / scan 也自动发现 image

未采用。

原因：

- 用户目录中的图片数量通常远高于文档数量，自动发现会显著增加噪声。
- image 更适合作为显式 parse 的输入，而不是默认文档库扫描目标。
- 显式 parse 已能覆盖需要把图片纳入 doclib 的场景。

## 影响

### 对 CLI / API

- 冲突参数组合从“成功但实际降级”改为 400 类 `InvalidRequestError`。
- JSON 输出、普通 CLI 输出和 `-v` 模式都通过统一错误通道表达冲突，不依赖 warning 文本。
- 新增错误码需要在错误文档中保持可发现。

### 对 doclib 存储

- 显式 parse 的 image 会写入 `files` 和 `docs`，`docs.file_type` 为 `image`。
- `docs.is_image_based` 保留 metadata 中的图片型信息。
- watch / scan 不会因为目录中存在图片而自动创建 doclib 记录。

### 对兼容性

- 这是有意的 doclib 单对象主动解析行为收紧。依赖非 PDF / image 显式请求质量 tier 或 `--remote` 成功退出的脚本需要调整。
- 未显式指定质量 tier 的 Office、text、html 解析仍保持 `flash` 语义。
- PDF / image 的本地 `flash` 和本地 / remote 质量 tier 路径保持可用。

### 对测试

测试应覆盖：

- 显式 parse image 能入库并创建 parse 任务。
- watch / scan 不自动发现 image。
- doclib 单对象主动解析中，非 PDF / image 请求质量 tier 报 `tier_unsupported_for_file_type`。
- doclib 单对象主动解析中，非 PDF / image 请求 `--remote` 报 `remote_unsupported_for_file_type`。
- PDF / image 请求 `--remote --tier flash` 报 `tier_unsupported_for_remote`。

## 后续动作

1. 在 CLI / tier / error 文档中同步文件类型、tier、remote 的能力矩阵。
2. 如果未来 Office 或 text/html 支持质量 tier，应通过新 ADR 或本 ADR 的 superseding 记录更新能力矩阵。
3. 如果需要非错误型 warning 通道，应设计稳定的 `warnings` / `hints` 响应字段，而不是复用 `tip` 承载参数冲突。
