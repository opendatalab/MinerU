# 设计决策记录

状态: Draft
读者: 项目核心开发者、后续维护者
范围: 记录已经做出的关键设计决策、背景、替代方案和影响
非目标: 替代主题设计文档；记录所有临时讨论

## 何时写 ADR

当一个选择满足任一条件时，应进入本目录：

- 会影响 API、CLI、SDK 或 middle_json 的长期兼容性。
- 会影响模块边界、数据模型、进程边界或存储模型。
- 有多个合理替代方案，且未来维护者需要知道为什么选了当前方案。
- 决策一旦落地，回滚成本较高。

## 命名规则

使用递增编号和短标题：

```text
0001-local-doclib-storage.md
0002-api-upload-flow.md
0003-middle-json-coordinate-system.md
```

## 已记录决策

| ADR | 状态 | 主题 |
|-----|------|------|
| [ADR-0001](0001-json-output-formats.md) | Accepted | JSON 输出格式命名 |
| [ADR-0002](0002-force-vs-invalidate.md) | Accepted | Force 与 Invalidate 缓存语义 |
| [ADR-0003](0003-parse-request-wait-batches.md) | Accepted | Parse 请求等待批次语义 |
| [ADR-0004](0004-doclib-http-api-resources.md) | Accepted | Doclib HTTP API 资源模型 |
| [ADR-0005](0005-doclib-interface-client-server-contract.md) | Accepted | Doclib Interface、Client 与 Server 契约一致性 |
| [ADR-0006](0006-doclib-file-change-detection.md) | Proposed | Doclib 文件变化检测与重新入库语义 |
| [ADR-0007](0007-doclib-file-availability-lifecycle.md) | Accepted | Doclib 文件可达性生命周期 |
| [ADR-0008](0008-doclib-forget-path.md) | Accepted | Doclib Forget Path 语义 |
| [ADR-0009](0009-doclib-scan-task.md) | Accepted | Doclib Scan 后台任务 |
| [ADR-0010](0010-doclib-watch-cli.md) | Accepted | Doclib Watch CLI 与 Rescan 边界 |
| [ADR-0011](0011-doclib-doc-short-id.md) | Accepted | Doclib Doc Short ID |
| [ADR-0012](0012-doclib-block-locator.md) | Accepted | Doclib Block Locator |
| [ADR-0013](0013-doc-content-progressive-reading.md) | Accepted | Doc Content Progressive Reading |
| [ADR-0014](0014-mineru-read-command.md) | Accepted | MinerU Read Command |
| [ADR-0015](0015-cli-output-json-composition.md) | Accepted | CLI Output 与 JSON 组合语义 |

## 模板

```md
# ADR-0000: 决策标题

状态: Proposed / Accepted / Deprecated / Superseded
日期: YYYY-MM-DD
相关文档: ../architecture.md

## 背景

描述为什么需要做这个决策。

## 决策

描述最终选择。

## 替代方案

列出主要替代方案，以及没有选择它们的原因。

## 影响

描述对实现、兼容性、测试、文档和用户体验的影响。

## 后续动作

列出需要跟进的具体工作。
```
