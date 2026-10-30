# MinerU 开发文档

本目录描述当前 4.1 系列的代码边界和维护契约。`next` 仅保留为既有路径，不表示所有内容都是未来设计。
安装与集成从 [用户文档](https://opendatalab.github.io/MinerU/zh/) 开始；历史计划不作为当前实现依据。

## 阅读路径

| 目的 | 文档 |
|---|---|
| 理解模块和调用链 | [模块地图](code-map.md)、[系统架构](architecture.md)、[端到端工作流](workflows.md) |
| 修改代码前核对边界 | [不变量](invariants.md)、[开发陷阱](pitfalls.md)、[术语](glossary.md) |
| CLI 与 SDK | [CLI 参考](cli/README.md)、[SDK 分层](sdk/README.md) |
| HTTP 集成 | [API 参考](api/README.md)、[Tasks](https://opendatalab.github.io/MinerU/zh/usage/tasks/)、[file-parse](https://opendatalab.github.io/MinerU/zh/usage/file_parse/) |
| 文档结构与输出 | [Middle JSON](middle-json/README.md)、[输出格式](https://opendatalab.github.io/MinerU/zh/reference/output_files/) |
| 配置、设备与模型 | [配置](config.md)、[档位](tiers.md)、[ONNX 设备](onnx-device.md)、[模型资产](model-assets.md)、[硬件报告](cli/mineru-kit-hardware.md) |
| 页码与元数据 | [页范围](page-ranges.md)、[命名](page-naming.md)、[元数据](metadata.md) |
| 运行时与观测 | [原生异步运行时](../native-async-analysis.md)、[错误边界](errors.md)、[Telemetry](telemetry-minimum.md) |
| 决策与后续工作 | [ADR](decisions/README.md)、[当前待办](backlog.md) |

## 文档状态

`Implemented` 表示已对照对应实现核验；`Draft` 表示仍需核验的部分，不能自动转化为产品承诺。
ADR 使用 `Proposed / Accepted / Deprecated / Superseded`。公共行为以对应版本实现和公开参考为准。

[历史档案](https://github.com/opendatalab/MinerU/tree/master/docs/archive)保存原计划、迁移和实测数据；
[提案区](https://github.com/opendatalab/MinerU/tree/master/docs/proposals)保存未实现设计。
归档材料不进入站点搜索。维护及验证方法见 [仓库文档约定](https://github.com/opendatalab/MinerU/blob/master/docs/README.md)。
