# ADR-0024: 文件类型与 Tier 归一化语义

状态: Accepted
日期: 2026-07-09
相关文档: ../tiers.md, ../cli/mineru-parse.md, ../cli/mineru-kit-parse.md, ../api/parse-jobs.md, 0022-doclib-file-type-tier-remote-semantics.md

## 背景

MinerU doclib 可以接收 PDF、image、Office、text 和 HTML 输入，但这些输入并不具备相同的解析能力。

PDF 和 image 支持质量 tier: `flash`、`medium`、`high`、`xhigh`。Office 和 HTML 当前只有 `flash` 语义，没有对应的质量 tier。text 可入库和搜索，但不需要解析。此前不同入口对“非 PDF/image + 质量 tier”的处理不完全一致:

- 有的入口报错。
- 有的入口静默走 Office / HTML parser，或为 text 返回没有对应缓存的合成成功响应。
- parsing-rule 可能创建 `tier=high` 等 parse row，但实际执行的仍是 Office / flash 语义。

这种不一致会让 parse row、缓存目录、locator 和用户看到的实际解析能力脱节。因此需要定义一套按操作类型区分的归一化规则。

本 ADR 细化并部分修订 ADR-0022。ADR-0022 中 doclib 主动单文件解析的严格错误语义仍保留；本 ADR 补充 batch、parsing-rule、API Server 和 mineru-kit 的例外。

## 决策

### 1. 文件类型分组

MinerU 在 tier 语义上把输入分为三组:

| 分组 | 文件类型 | Tier 能力 |
|------|----------|-----------|
| 支持多 tier 的输入 | PDF、image | `flash`、`medium`、`high`、`xhigh` |
| 仅支持 flash tier 的输入 | Office、HTML | 仅 `flash` 语义 |
| 无需解析的文本输入 | `.txt`、`.md`、`.markdown`、`.csv`、`.rst`、`.tex` | 无 tier；直接读取源文件 |

Office 和 HTML 即使通过高层接口传入了 `medium`、`high` 或 `xhigh`，也不能把结果记录为这些质量 tier，除非未来另有 ADR 明确引入对应能力。text 不创建 parse row 或 Middle JSON，显式 parse 返回 `parse_not_required`。

### 2. 单对象主动操作

单对象主动操作包括:

- `mineru parse <single-file>` 经 doclib 解析。
- `DoclibClient.ensure_parse(ParseRequest(...))`。
- SDK 直接解析单个文件。
- `mineru-kit parse` 展开后只有一个输入文件。

规则:

| 输入类型 | 未指定 tier | 指定 `flash` | 指定 `medium/high/xhigh` |
|----------|-------------|--------------|--------------------------|
| PDF / image | 使用该入口默认质量选择策略 | 使用 `flash` | 使用指定质量 tier |
| Office / HTML | 归一为 `flash` | 使用 `flash` | 报错 |
| text | 不解析；doclib 返回 `parse_not_required` | 不解析；doclib 返回 `parse_not_required` | 不解析；doclib 返回 `parse_not_required` |

因此，单对象主动操作中，用户显式指定质量 tier 表达明确质量预期。对于没有质量 tier 能力的输入，系统不能静默降级。SDK 和 `mineru-kit` 的解析输入校验直接排除 text；doclib 使用更具体的 `parse_not_required` 指导用户读取源文件。

### 3. 批量操作

批量操作包括:

- API Server v1 parse job。
- `mineru-kit parse` 展开后包含多个输入文件，或输入目录。
- parsing-rule 触发的后台解析。

批量操作的目标是处理一组异构文件。为了避免一个 Office 或 HTML 文件阻断整个批次，仅支持 flash tier 的输入允许静默归一为 `flash`。text 文件不进入 parse 批次；需要保留内容时直接读取或建立文本索引。

规则:

| 输入类型 | 未指定 tier | 指定 `flash` | 指定 `medium/high/xhigh` |
|----------|-------------|--------------|--------------------------|
| PDF / image | 使用该入口默认选择策略 | 使用 `flash` | 使用指定质量 tier |
| Office / HTML | 归一为 `flash` | 使用 `flash` | 归一为 `flash` |
| text | 不进入 parse 批次 | 不进入 parse 批次 | 不进入 parse 批次 |

批量操作中，Office/HTML 的实际 parse row、缓存目录、locator 和 metadata 必须记录 `flash`，不能记录用户传入的质量 tier。text 不得生成这些解析产物。

API Server 当前只有 job 级 `tier` 字段，暂不扩展 file-level effective tier 字段。因此 API job 响应中的 job tier 可以表示请求 tier；但内部执行 Office/HTML 文件时，应按 `flash` 语义处理，不产生伪质量缓存。text 不进入 API parse job。

### 4. Parsing-Rule 语义

parsing-rule 是批量/后台规则，不是用户对单个文件的显式 parse 请求。

规则:

- `tier` 和 `remote` 参数只适用于 PDF 和 image。
- Office 和 HTML 命中 parsing-rule 时，忽略 rule 中的 `tier` 和 `remote`，按 `flash` 解析。
- text 命中 parsing-rule 时只完成入库和文本索引，不创建 parse row。
- PDF/image 命中 parsing-rule 且指定 tier 时，使用 rule tier。
- PDF/image 命中 parsing-rule 但未指定 tier 时，按以下顺序选择可用 tier:

```text
high -> xhigh -> medium -> flash
```

这与用户主动阅读默认选择不同。parsing-rule 是后台批量策略，可以接受最后降级到 `flash`，因为 watch 自动发现本身也接受 `flash`。

### 5. Remote 语义

remote 的含义取决于入口:

| 入口 | Office/HTML + remote | text + remote |
|------|----------------------|---------------|
| doclib / `mineru parse` 单文件主动解析 | 报 `remote_unsupported_for_file_type` | 报 `parse_not_required` |
| parsing-rule | 忽略 remote，按 `flash` | 只入库和索引 |
| API Server job | 不适用；API Server 自身就是解析服务 | 不接受 text 作为解析输入 |
| `mineru-kit parse` | 允许，作为低层工具特例 | 不接受 text 作为解析输入 |

doclib 单文件主动解析中，`remote=true` 表示用户显式要求把该文件交给 remote parse-server。因为 remote parse-server 只支持 PDF/image，该组合必须报错，不能静默改成本地 `flash`，否则会改变用户显式选择的执行路径和隐私语义。

`mineru-kit parse` 是低层工具，允许 `--remote` 处理 Office/HTML；当这些输入不能使用质量 tier 时，按批量归一规则处理。text 不作为解析输入。

## 替代方案

### 方案 A: Office/HTML 一律静默忽略 tier

未采用。

原因:

- 单文件主动操作中，用户显式指定质量 tier 通常表示明确质量预期。
- 静默忽略会让脚本和用户误以为质量 tier 已执行。
- doclib 的 parse row、缓存和 locator 应准确表达实际结果能力。

### 方案 B: Office/HTML 一律遇到质量 tier 报错

未采用。

原因:

- 批量任务经常包含混合文件类型。
- parsing-rule 是后台批量策略，不应因为 Office/HTML 携带无效质量 tier 而阻断整个目录处理。
- API Server v1 parse job 当前是多文件接口，也应支持异构输入。

### 方案 C: 为 API Server 增加 file-level effective tier 字段

暂不采用。

原因:

- 当前目标是先统一行为，不扩展 API response schema。
- API job 的 job-level tier 可以继续表达请求 tier。
- 如未来需要精确暴露每个文件的实际 tier，可另行设计 file-level effective tier 字段。

## 影响

### 对 doclib

- doclib 单文件主动解析保持严格语义: Office/HTML 显式质量 tier 报错；text 返回 `parse_not_required`。
- Office/HTML 未指定 tier 或指定 `flash` 时，实际记录为 `flash`。
- parsing-rule 触发 Office/HTML 时，实际 parse row 必须为 `flash`；text 只入库和索引。
- parsing-rule 对 PDF/image 未指定 tier 时，允许 `high -> xhigh -> medium -> flash` 兜底。

### 对 SDK 和 mineru-kit

- SDK 单文件解析应按单对象主动操作语义处理，并排除 text 解析输入。
- `mineru-kit parse` 需要根据展开后的输入数量区分单对象和批量。
- `mineru-kit parse --remote` 对 Office/HTML 是允许的低层工具特例；text 不作为解析输入。

### 对 API Server

- API Server job 按批量操作语义处理。
- Office/HTML 即使 job tier 是质量 tier，执行时也按 `flash` 语义；text 不进入 parse job。
- 暂不新增 file-level effective tier 字段。

### 对缓存和 locator

- 缓存键、parse row、locator 和产物 metadata 必须记录实际 tier。
- Office/HTML 不应产生 `medium`、`high` 或 `xhigh` 缓存；text 不应产生任何解析缓存。

## 后续动作

1. 更新 doclib parsing-rule 入队逻辑，确保 Office/HTML 创建 `tier=flash` 的 parse row，text 不创建 parse row。
2. 更新 parsing-rule 的 PDF/image 默认 tier 选择顺序为 `high -> xhigh -> medium -> flash`。
3. 检查 SDK、mineru-kit 和 API Server，对单对象和批量操作分别应用本 ADR 的归一化规则。
4. 同步更新 CLI、SDK 和 API 文档，避免继续表述为“所有入口统一报错”或“所有入口统一静默降级”。
