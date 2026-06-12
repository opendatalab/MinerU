# 解析 Tier

状态: Draft
读者: CLI/API/SDK 使用者、Agent skill 作者、核心开发者
范围: `flash`、`standard`、`pro` 解析 tier 的产品语义、默认选择策略、适用场景、质量边界和执行路径
非目标: 固定具体模型名称或 backend 实现；定义 middle_json 字段级 schema
底稿: 用户讨论；`../../NEXT-ROADMAP.md`；`../../NEXT-CLI.md`；`../../NEXT-DESIGN.md`

## 1. 定位

Tier 是用户可理解的解析档位，不是 backend 名称。它表达的是质量、速度、资源消耗和隐私路径之间的产品承诺。

MinerU 当前定义三个公开 tier：

| Tier | 定位 | 一句话说明 |
|------|------|------------|
| `flash` | 快速发现与索引 | 纯 CPU、最快、质量最低，用于大量文件的低成本预处理 |
| `standard` | 本地主力解析 | 基于模型，面向普通消费级电脑，质量和资源消耗折中 |
| `pro` | 最高质量解析 | MinerU 线上 API 的最高质量层级，也可在高端本地硬件上运行 |

`auto` 不再作为公开 tier 值。CLI 中不传 `--tier`、HTTP API 中省略 `tier` 或传 JSON `null`、Python SDK 中传 `None`，都表示使用默认选择策略。

核心原则：

- `flash` 用于发现、预览和建立搜索索引，不是默认阅读质量。
- 当 Agent 或用户决定读取某个文档时，如果没有指定 tier，默认选择当前可发现能力中最高的非 `flash` tier，最低可接受解析结果是 `standard`。
- 默认选择策略在任何语境下都不会等价于 `flash`。
- `pro` 是当前质量最好的 tier，也是 `mineru.net/api` 对外提供的解析层级。
- 本地运行 `pro` 是正式支持能力，不是实验能力；它依赖高端 GPU / 加速器硬件和对应模型环境。
- `mineru.net/api` 在相当长时间内只提供 `pro`，不暴露低成本远端 `standard`。
- 不使用 `--remote` 时，MinerU 不会把文档发送到 `mineru.net/api`。

## 2. 实体 tier 区别

| 维度 | `flash` | `standard` | `pro` |
|------|---------|------------|-------|
| 主要目标 | 快速扫大量文件，建立基础索引 | 在普通本地硬件上提供可接受的高质量解析 | 提供当前最高质量解析 |
| 底层原理 | 纯 CPU 实现 | 基于模型 | 基于更强模型 |
| 质量 | 低 | 中 | 高 |
| 速度 | 最快 | 中等 | 最慢，取决于硬件和远端服务 |
| 本地硬件要求 | 无 GPU / 加速器也可用 | 消费级电脑可用，如 MacBook 或 Windows 笔记本 | 需要高端 GPU / 加速器，或使用远端 |
| 是否适合 watch 自动处理 | 是 | 一般不默认使用 | 否 |
| 是否适合用户主动阅读 | 仅在用户明确选择时 | 是，默认最低阅读质量 | 是，质量优先场景 |
| 典型场景 | 文件发现、快速预览、搜索索引 | 本地隐私解析、普通文档阅读 | 复杂版面、学术论文、高价值文档 |

## 3. 默认选择策略

当用户没有指定 tier，或在 HTTP API 中显式传入 JSON `null` 时，MinerU 使用默认选择策略。Python SDK 中的等价表达是 `tier=None`。

MinerU 有 local parse-server，也有 remote parse-server。无论 local 还是 remote，parse-server 都应支持通过 API 发现当前 server 提供了哪些解析 tier。通常情况下，server 只支持一种能力（`standard` 或 `pro`），或者同时支持 `standard` 和 `pro`。

默认选择策略的含义是：在当前目标 parse-server 可发现的能力中，选择最高的非 `flash` tier。

| 可发现能力 | 默认解析结果 |
|------------|-----------------|
| 只支持 `standard` | `standard` |
| 只支持 `pro` | `pro` |
| 同时支持 `standard` 和 `pro` | `pro` |
| 只支持 `flash` 或没有可用能力 | 报错 |

默认选择策略永远不会解析为 `flash`。如果系统找不到 `standard` 或 `pro`，应返回 `quality_tier_unavailable` 或等价结构化错误，而不是擅自用 `flash` 内容替代。错误码见 [错误码体系](errors.md)。

默认选择策略是请求时的选择逻辑，不是长期缓存语义。任务入队、缓存命中、产物目录和结果 metadata 应使用实际选择的实体 tier（`standard` 或 `pro`），避免默认选择与实体 tier 产生重复缓存。

## 4. Flash

`flash` 是纯 CPU tier，面向没有 GPU 或加速器硬件的用户。它的质量通常最低，但必须足够快。

`flash` 的核心用途是支撑大量文件的自动发现和低成本预处理。MinerU 的 watch 机制会发现很多文件；在系统刚看到这些文件时，并不知道每个文件的价值，也不适合立即使用高算力方案处理。因此 watch 默认使用 `flash` 快速提取最基础内容，用于建立搜索索引和后续发现。

`flash` 不应被当成默认阅读质量。当 Agent 或用户决定真正读取某个文档时，需求已经从“发现它”变成“尽可能准确地理解它”，此时未指定 tier 应使用默认选择策略，并至少解析到 `standard`。

只有在用户明确指定 `flash` 时，才返回 `flash` 的解析内容作为最终阅读结果。

## 5. Standard

`standard` 是本地主力解析 tier，底层使用模型，面向普通消费级电脑用户。

这类用户可能拥有 GPU 或加速器，但显存和算力有限。它们不一定完全跑不动 `pro`，但运行 `pro` 通常会很慢。因此 `standard` 是一个算力消耗较低、质量中等、仍然基于模型的解析方案。

`standard` 的核心价值是本地隐私解析：当用户不接受把文档发送到远端解析，同时硬件又不是特别高端时，`standard` 应成为本地可用的最佳默认方案。

## 6. Pro

`pro` 是 MinerU 当前最高质量 tier，也是 `mineru.net/api` 提供的解析层级。

`pro` 默认代表最高质量解析能力。它既可以由 `mineru.net/api` 提供，也可以由本地高端硬件运行。拥有高端 NVIDIA 数据中心或工作站级计算卡的用户，应能通过本地 parse-server 正式运行这个 tier。

`pro` 适合复杂版面、学术论文、高价值文档和用户明确追求最高质量的场景。它的代价是更高的算力、显存、等待时间或远端调用成本。

## 7. 隐私优先与质量优先

MinerU 的 tier 策略同时遵守隐私优先和质量优先。

### 7.1 隐私优先

默认不上传文档。用户没有显式使用 `--remote` 或等价远端配置时，MinerU 不会把文档发送到 `mineru.net/api`。

本地可用能力不足时，应返回可解释错误，而不是静默上传。

### 7.2 质量优先

用户或 Agent 主动读取文档时，默认应解析到 `standard` 或 `pro`。

如果 `standard` 和 `pro` 都不可用，且远端也不可用，系统应报错，而不是擅自把 `flash` 结果作为最终阅读结果返回。

`flash` 结果可以用于索引、发现、预览和提示用户选择更高 tier；只有用户明确选择 `flash` 时，才可作为最终解析返回。

## 8. 执行路径

| 请求 | 默认执行路径 |
|------|--------------|
| watch 自动发现文件 | 本地 `flash` |
| 用户主动 parse，未指定 tier；HTTP API 传 JSON `null`；Python SDK 传 `None` | 解析为当前可发现的最高非 `flash` tier |
| 用户主动指定 `--tier flash` | 本地 `flash` |
| 用户主动指定 `--tier standard` | 本地 standard；不可用时报错 |
| 用户主动指定 `--tier pro` | 本地 pro；显式 remote 时使用 `mineru.net/api` pro |
| 用户显式 `--remote` | 允许上传到 `mineru.net/api`，使用远端 `pro` |

执行路径需要与 [系统架构](architecture.md) 中的 ParseWorker 路由保持一致。

## 9. 对 CLI/API/SDK 的要求

CLI、API 和 SDK 暴露 tier 时，应遵守同一语义：

- 对用户显示 tier，不显示具体 backend 名称作为主要选择项。
- 公开 tier 取值只有 `flash`、`standard`、`pro`。
- CLI 不传 `--tier`、HTTP API 省略 `tier` 或传 JSON `null`、Python SDK 传 `None`，表示使用默认选择策略。
- 默认选择策略必须通过 parse-server 能力发现解析为 `standard` 或 `pro`。
- 默认阅读质量不能静默降级到 `flash`。
- 当更高 tier 不可用时，应返回结构化错误和修复建议。
- 当用户显式选择 `flash` 时，可以返回 `flash` 内容。
- `--remote` 或等价字段必须是显式选择。

## 10. 未决问题

与 tier 相关的硬件基线、watch 升级提示和 `mineru-kit` 的 tier/backend 边界，集中维护在 [开放问题清单](open-questions.md)。
