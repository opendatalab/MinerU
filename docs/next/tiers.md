# 解析 Tier

状态: Draft
读者: CLI/API/SDK 使用者、Agent skill 作者、核心开发者
范围: `flash`、`basic`、`standard`、`advanced` 解析 tier 的产品语义、默认选择策略、适用场景、质量边界和执行路径
非目标: 固定具体模型名称或 backend 实现；定义 middle_json 字段级 schema
底稿: 用户讨论；旧路线图、旧 CLI 与旧设计底稿迁移内容

## 1. 定位

Tier 是用户可理解的解析档位，不是 backend 名称。它表达的是质量、速度、资源消耗和隐私路径之间的产品承诺。

MinerU 当前定义四个公开 tier。文档中使用面向用户的显示名；代码、CLI 参数、API 字段和 locator 中使用对应字面量。

| 显示名 | 字面量 | 定位 | 一句话说明 |
|--------|--------|------|------------|
| Flash | `flash` | 快速发现与索引 | 随 MinerU 默认安装，最快、质量最低，用于大量文件的低成本预处理 |
| Basic | `basic` | 基础解析 | 需要额外依赖，CPU 可运行，GPU 可加速，质量和资源消耗折中 |
| Standard | `standard` | 标准解析 | 需要 VLM 相关依赖和支持的本地加速器，也是 MinerU 线上 API 提供的档位 |
| Advanced | `advanced` | 高级解析 | 与 Standard 硬件要求相同，但消耗更多算力和时间，困难文档质量更高 |

`auto` 不再作为公开 tier 值。CLI 中不传 `--tier`、HTTP API 中省略 `tier` 或传 JSON `null`、Python SDK 中传 `None`，都表示使用当前入口的默认选择策略。

核心原则：

- `flash` 用于发现、预览和建立搜索索引，不是 PDF/image 默认阅读质量。
- 当 Agent 或用户决定读取 PDF/image 文档时，如果没有指定 tier，默认选择不能解析为 `flash`；有能力发现上下文时优先使用 `standard`。
- 默认选择策略在 PDF/image 质量解析语境下不会等价于 `flash`；Office/HTML 这类仅支持 flash tier 的输入按 [ADR-0024](decisions/0024-file-type-tier-normalization.md) 归一为 `flash` 语义；text 直接读取。
- `standard` 是高质量 tier，也是 `mineru.net/api` 对外提供的解析层级。
- `advanced` 是当前公开 tier 中质量和成本最高的档位。
- 本地运行 `standard` 和 `advanced` 是正式支持能力，不是实验能力；它们依赖 VLM 相关依赖、支持的本地加速器和足够的显存或统一内存。
- `mineru.net/api` 在相当长时间内只提供 `standard`，不暴露低成本远端 `basic`。
- 不使用 `--remote` 时，MinerU 不会把文档发送到 `mineru.net/api`。

## 2. 实体 tier 区别

| 维度 | Flash (`flash`) | Basic (`basic`) | Standard (`standard`) | Advanced (`advanced`) |
|------|---------|------------|--------|--------------|
| 主要目标 | 快速扫大量文件，建立基础索引 | 在普通本地硬件上提供可接受的高质量解析 | 提供高质量解析 | 提供最高成本/最高质量档位 |
| 安装依赖 | 随 MinerU 默认安装 | 需要额外 runtime 依赖 | 需要更多 runtime 依赖和 VLM 依赖 | 同 `standard` |
| 底层原理 | 纯 CPU 实现 | 基于模型 | 基于 VLM 的高质量模型路径 | 基于 VLM 的更高成本模型路径 |
| 质量 | 低 | 中 | 高 | 更高 |
| 速度 | 最快 | 中等 | 较慢，取决于硬件和远端服务 | 最慢，通常比 `standard` 消耗更多算力和时间 |
| 本地硬件要求 | 无特殊硬件要求 | 需要至少 16GB 总内存；CPU 可运行，GPU 可加速 | 需要至少 16GB 总内存，并具备支持的本地加速器和足够的显存或统一内存 | 同 `standard` |
| 是否适合 watch 自动处理 | 是 | 一般不默认使用 | 否 | 否 |
| 是否适合用户主动阅读 | 仅在用户明确选择时 | 是，默认最低阅读质量 | 是，质量优先场景 | 是，专家质量优先场景 |
| 典型场景 | 文件发现、快速预览、搜索索引 | 本地隐私解析、普通文档阅读 | 复杂版面、学术论文、高价值文档 | 极高质量要求、专家批处理或自部署 |

## 3. 默认选择策略

当用户没有指定 tier，或在 HTTP API 中显式传入 JSON `null` 时，MinerU 使用默认选择策略。Python SDK 中的等价表达是 `tier=None`。

默认选择策略分两类场景。本节描述 PDF/image 质量解析的默认选择；Office/HTML 这类仅支持 flash tier 的输入见 [ADR-0024](decisions/0024-file-type-tier-normalization.md)。text 没有 tier，不进入解析。

### 3.1 无能力列表的场景

启动服务或直接解析时，如果当前入口没有可用 tier 列表可参考，默认 tier 是 `standard`。

典型场景:

- `mineru-kit api-server` 未传 `--tier`。
- `mineru-kit parse` local 模式未传 `--tier` / `--backend`。
- Tool SDK 直接本地解析且无法做能力发现。

这类场景的默认值不是 `advanced`。`advanced` 消耗更多算力和时间，需要用户显式指定，或由具备能力发现上下文的调用方在没有 `standard` 时选择。

### 3.2 有能力发现上下文的场景

当调用某个 parse-server，或有等价的可用 tier 上下文时，默认选择策略基于当前可发现能力选择实体 tier。MinerU 有 local parse-server，也有 remote parse-server。无论 local 还是 remote，parse-server 都应支持通过 API 发现当前 server 提供了哪些解析 tier。当前代码支持发现 `basic`、`standard`、`advanced` 等质量 tier。

这类场景的默认选择顺序是:

```text
standard -> basic -> 报错
```

也就是说，`standard` 是默认优先档位；没有 `standard` 时回退到 `basic`。`advanced` 必须由用户显式指定，不参与默认选择。按照当前服务能力契约，提供 `advanced` 的服务一定同时提供 `standard`。

| 可发现能力 | 默认解析结果 |
|------------|-----------------|
| 只支持 `basic` | `basic` |
| 只支持 `standard` | `standard` |
| 只支持 `advanced` | 非法能力组合，不选择默认 tier |
| 同时支持 `basic` 和 `standard` | `standard` |
| 同时支持 `standard` 和 `advanced` | `standard` |
| 同时支持 `basic` 和 `advanced`，不支持 `standard` | `basic` |
| 同时支持 `basic`、`standard` 和 `advanced` | `standard` |
| 只支持 `flash` 或没有可用能力 | 报错 |

PDF/image 默认选择策略不会解析为 `flash`。如果系统找不到 `basic` 或 `standard`，应返回 `quality_tier_unavailable` 或等价结构化错误，而不是自动选择 `advanced` 或用 `flash` 内容替代。错误码见 [错误码体系](errors.md)。

默认选择策略是请求时的选择逻辑，不是长期缓存语义。任务入队、缓存命中、产物目录和结果 metadata 应使用实际选择的实体 tier（例如 `basic`、`standard` 或 `advanced`），避免默认选择与实体 tier 产生重复缓存。

读取已有缓存时不使用这套解析默认顺序。`mineru read doc:{short_id}` 不带 `/tier:{tier}` 时不会创建新解析，应在已缓存的非 `flash` 结果中选择最高质量 tier，顺序为 `advanced` -> `standard` -> `basic`；没有非 `flash` 缓存时返回错误。

## 4. Flash

`flash` 随 MinerU 默认安装，面向没有额外 runtime 依赖、GPU 或加速器硬件的用户。它的质量通常最低，但必须足够快。

`flash` 的核心用途是支撑大量文件的自动发现和低成本预处理。MinerU 的 watch 机制会发现很多文件；在系统刚看到这些文件时，并不知道每个文件的价值，也不适合立即使用高算力方案处理。因此 watch 默认使用 `flash` 快速提取最基础内容，用于建立搜索索引和后续发现。

P0 不做基于启发式的自动提示或自动排队升级。watch 使用 `flash` 后，如果需要在后台自动解析到 `basic`、`standard` 或 `advanced`，必须来自用户显式配置的 parsing-rules；系统不根据文件名、内容特征、搜索命中或“高价值文档”判断自行升级。

`flash` 不应被当成 PDF/image 默认阅读质量。当 Agent 或用户决定真正读取 PDF/image 文档时，需求已经从“发现它”变成“尽可能准确地理解它”，此时未指定 tier 应使用默认选择策略，并解析到可用的非 `flash` 质量 tier。Office/HTML 这类仅支持 flash tier 的输入没有质量 tier 分层，按 `flash` 语义处理。text 直接读取。

只有在用户明确指定 `flash` 时，才返回 `flash` 的解析内容作为最终阅读结果。

## 5. Basic

`basic` 是基础解析 tier，当前使用 `hybrid-engine` 的 `medium` effort。它需要额外 runtime 依赖，是一系列本地模型串联的解析方案。

`basic` 可以在 CPU 上运行，也可以由 GPU 加速。GPU 和显存不是启用 `basic` 的硬门槛，但会影响速度、吞吐和稳定性。因此 `basic` 是一个算力消耗较低、质量中等、仍然基于模型的解析方案。

`basic` 的核心价值是本地隐私解析：当用户不接受把文档发送到远端解析，同时硬件又不是特别高端时，`basic` 应成为本地可用的最佳默认方案。

本地 managed parse server 只有 `basic` 和 `standard` 两个启动能力上限，两者都要求至少 16GB 总内存。Standard 服务同时提供 Standard 和 Advanced 请求能力。低于该基线时，不建议启用本地 managed 质量 tier；应考虑远端解析，或在用户明确接受低质量时显式使用 `flash`。

## 6. Standard

`standard` 是 MinerU 的高质量 tier，也是 `mineru.net/api` 当前提供的解析层级。

`standard` 代表高质量解析能力。它既可以由 `mineru.net/api` 提供，也可以由本地高端硬件运行。

本地 managed `standard` 需要安装 VLM 相关 runtime 依赖，要求至少 16GB 总内存，并满足以下本地加速器条件之一:

- Volta 或更新架构的 NVIDIA GPU，且可供 MinerU 使用的 VRAM 至少 8GB。
- Apple Silicon，且统一内存至少 16GB。
- MinerU 支持的特殊 AI 加速器，例如 `npu`、`gcu`、`musa`、`mlu` 或 `sdaa`。

`standard` 适合复杂版面、学术论文、高价值文档和多数高质量解析场景。它的代价是更高的算力、显存、等待时间或远端调用成本。

## 7. Advanced

`advanced` 是当前公开 tier 中最高的本地/自部署档位。

它适合极高质量要求、专家批处理或自部署服务。`advanced` 与 `standard` 的本地硬件要求相同，复用 `mineru[standard]` extra 和 Standard 模型集；区别是 `advanced` 会消耗更多推理算力，通常需要更长解析时间，并可能带来更高运行成本。准备本地 Advanced 环境时安装 Standard extra、使用 `mineru-kit models download --tier standard`，并以 Standard 启动 parse-server；解析请求再显式选择 `--tier advanced`。Advanced 不需要也不支持独立的服务启动配置。

## 8. 隐私优先与质量优先

MinerU 的 tier 策略同时遵守隐私优先和质量优先。

### 8.1 隐私优先

默认不上传文档。用户没有显式使用 `--remote` 或等价远端配置时，MinerU 不会把文档发送到 `mineru.net/api`。

本地可用能力不足时，应返回可解释错误，而不是静默上传。

### 8.2 质量优先

用户或 Agent 主动读取 PDF/image 文档时，默认应解析到可用的非 `flash` 质量 tier。

如果 `basic`、`standard` 和 `advanced` 都不可用，且远端也不可用，系统应报错，而不是擅自把 `flash` 结果作为最终阅读结果返回。

`flash` 结果可以用于索引、发现、预览和提示用户选择更高 tier；只有用户明确选择 `flash` 时，才可作为最终解析返回。

## 9. 执行路径

| 请求 | 默认执行路径 |
|------|--------------|
| watch 自动发现文件 | 本地 `flash` |
| `mineru-kit api-server` 未传 `--tier` | 以 `standard` 作为服务默认 tier |
| `mineru-kit parse` local 模式未传 `--tier` / `--backend` | PDF/image 直接按 `standard` 解析；Office/HTML 按 `flash` 语义处理；text 不作为解析输入 |
| watch 命中 parsing-rule 且 rule 指定 tier | PDF/image 按 rule 中的 tier、页码范围和 remote 配置执行；Office/HTML 忽略 rule tier 和 remote，按 `flash`；text 只入库和索引 |
| watch 命中 parsing-rule 但 rule 未指定 tier | PDF/image 按 `standard` -> `advanced` -> `basic` -> `flash` 选择；Office/HTML 按 `flash`；text 只入库和索引 |
| 用户主动 parse，未指定 tier；HTTP API 传 JSON `null`；Python SDK 传 `None` | PDF/image 有能力发现上下文时按 `standard` -> `advanced` -> `basic` 选择；Office/HTML 按 [ADR-0024](decisions/0024-file-type-tier-normalization.md) 归一为 `flash`；text 不进入解析 |
| 用户主动指定 `--tier flash` | 本地 `flash` |
| 用户主动指定 `--tier basic` | 本地 basic；不可用时报错 |
| 用户主动指定 `--tier standard` | 本地 standard；显式 remote 时使用 `mineru.net/api` standard |
| 用户主动指定 `--tier advanced` | 本地或自部署 Advanced；不可用时报错 |
| 用户显式 `--remote` | 允许上传到 `mineru.net/api`，使用远端 `standard` |

执行路径需要与 [系统架构](architecture.md) 中的 ParseWorker 路由保持一致。

## 10. 对 CLI/API/SDK 的要求

CLI、API 和 SDK 暴露 tier 时，应遵守同一语义：

- 对用户显示 tier，不显示具体 backend 名称作为主要选择项。
- 公开 tier 取值包括 `flash`、`basic`、`standard`、`advanced`。
- CLI 不传 `--tier`、HTTP API 省略 `tier` 或传 JSON `null`、Python SDK 传 `None`，表示使用当前入口按文件类型定义的默认选择策略。
- 没有能力列表的启动服务或直接解析场景默认使用 `standard`。
- PDF/image 有能力发现上下文时，默认选择策略按 `standard` -> `advanced` -> `basic` 解析为非 `flash` 质量 tier。
- PDF/image 默认阅读质量不能静默降级到 `flash`。
- 当更高 tier 不可用时，应返回结构化错误和修复建议。
- 当用户显式选择 `flash` 时，可以返回 `flash` 内容。
- `--remote` 或等价字段必须是显式选择。

## 11. 未决问题

与 tier 相关的未决问题集中维护在 [开放问题清单](open-questions.md)。
