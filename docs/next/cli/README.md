# CLI 文档

状态: Draft
读者: CLI 使用者、Agent skill 作者、核心开发者、SDK/API 设计参与者
范围: `mineru` 与 `mineru-kit` 两款命令行工具的定位、分工和阅读路径
非目标: 重新定义解析 tier；展开 REST API 字段级规格
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 两款工具

MinerU 仓库提供两款命令行工具：

| 工具 | 定位 | 主要读者 | 状态 |
|------|------|----------|------|
| `mineru` | 本地文档解析和管理中心 | 普通用户、Agent、轻量开发者 | Agent 默认入口 |
| `mineru-kit` | 无状态解析工具和服务工具 | 批处理开发者、服务部署者、解析内核开发者 | 专家入口 |

`mineru` 叠加本地数据库、去重缓存、搜索、watch 和远端切换，是默认推荐入口。`mineru-kit` 暴露更完整的解析、批处理、并发和服务部署能力，不维护本地文档库。

## 2. 分工原则

| 维度 | `mineru` | `mineru-kit` |
|------|----------|--------------|
| 产品定位 | 本地文档能力中心 | 纯解析与服务工具 |
| 默认受众 | Agent / 普通用户 | 开发者 / 批处理用户 |
| 状态管理 | 维护 `doclib.db` | 无状态 |
| 去重缓存 | SHA256 内容寻址缓存 | 不负责 |
| 搜索 | 支持 | 不支持 |
| 默认输出 | STDOUT，适合 Agent 渐进式阅读 | 文件输出，适合批处理 |
| 参数暴露 | 精简、稳定、隐藏专家选项 | 完整、显式、可批量控制 |
| Skill 暴露 | 是 | 默认不暴露 |

`mineru-kit parse` 中的专家参数，例如 `--backend`、`--tier`、`--remote-url`、`--api-key`，不应直接暴露到 `mineru parse`。

## 3. 推荐阅读顺序

1. [mineru](mineru.md): 理解用户/Agent 入口的整体定位。
2. [mineru parse](mineru-parse.md): 理解主动解析、默认 tier 选择、隐私、缓存、STDOUT 和 marker。
3. [mineru read](mineru-read.md): 理解 locator-first 读取、page/block 定位、image 输出和 continuation。
4. [mineru server](mineru-server.md): 理解 doclib 生命周期和 parse-server 协作。
5. [mineru library](mineru-library.md): 理解 search、find、list、show、config、watch、scan、invalidate、forget 和 cleanup。
6. [mineru-kit](mineru-kit.md): 理解无状态工具定位。
7. [mineru-kit models](mineru-kit-models.md): 理解模型下载、当前模型配置查看和轻量校验。
8. [mineru-kit parse](mineru-kit-parse.md): 理解批处理解析、目录输入、local/remote 规则和输出命名。
9. [mineru-kit api-server](mineru-kit-api-server.md): 理解本地/自部署 parse-server。
10. [mineru-kit vlm-server](mineru-kit-vlm-server.md): 理解本地 VLM 服务与 api-server backend 的边界。

## 4. 共享约束

- 解析 tier 语义以 [解析 Tier](../tiers.md) 为准。
- 不显式 `--remote` 时，不上传用户文档。
- 用户主动读取 PDF/image 时，不把 `flash` 作为默认最终质量。
- 未指定 tier 使用默认选择策略；PDF/image 默认选择不会解析为 `flash`，Office/text/HTML 归一规则见 [ADR-0024](../decisions/0024-file-type-tier-normalization.md)。
- CLI 错误应使用结构化错误码和可执行修复建议。
- 错误码、`retryable` 和 `user_action` 语义见 [错误码体系](../errors.md)。

## 未决问题

CLI 选项命名和 Agent message contract，集中维护在 [开放问题清单](../open-questions.md)。
