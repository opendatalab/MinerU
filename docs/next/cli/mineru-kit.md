# mineru-kit

状态: Draft
读者: 批处理开发者、解析内核开发者、服务部署者
范围: `mineru-kit` 的定位、子命令和与 `mineru` 的边界
非目标: 本地文档库、搜索、watch、Agent 默认体验
底稿: `../../../NEXT-CLI.md`

## 1. 定位

`mineru-kit` 是无状态解析工具和服务工具。它不维护 `mineru.db`，不做本地文档库搜索，不负责 Agent 渐进式阅读体验。

`mineru-kit` 适合：

- 大规模批处理。
- 解析内核调试。
- 自部署 parse-server。
- 暴露完整解析参数、并发控制和输出策略。

## 2. 子命令

| 子命令 | 作用 | 文档 |
|--------|------|------|
| `mineru-kit parse` | 无状态文件/目录/URL/stdin 批处理解析 | [mineru-kit parse](mineru-kit-parse.md) |
| `mineru-kit api-server` | 启动兼容统一 API 的本地解析服务 | [mineru-kit api-server](mineru-kit-api-server.md) |
| `mineru-kit vlm-server` | 本地 VLM 服务，兼容 OpenAI API | 第一阶段纳入，文档待补充 |

## 3. 与 mineru 的边界

| 能力 | `mineru` | `mineru-kit` |
|------|----------|--------------|
| 本地数据库 | 有 | 无 |
| SHA256 缓存 | 有 | 无 |
| STDOUT 默认阅读 | 有 | 不是默认重点 |
| 目录批处理 | 非首要 | 核心能力 |
| 输出冲突策略 | 简化 | 完整 |
| 并发控制 | 简化 | 完整 |
| backend 专家参数 | 隐藏 | 可暴露 |

`mineru-kit` 可以作为 `mineru` 的底层能力来源，但不是普通用户和 Agent 的默认入口。Agent skill 长期只暴露 `mineru`，不直接暴露 `mineru-kit`。

`mineru-kit parse` 完全不复用 `mineru` 的本地 doclib 缓存。它是纯工具，不感知 doclib、watch、search 或长期数据库状态。

`mineru-kit` 参数暂不划分 `stable` / `experimental` 等稳定性等级。第一阶段保持参数体系简单，后续只有在兼容性压力明确出现时再引入分级。

## 未决问题

`mineru-kit` 的默认 tier 选择策略，集中维护在 [开放问题清单](../open-questions.md)。
