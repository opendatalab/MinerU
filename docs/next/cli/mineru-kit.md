# mineru-kit

状态: Draft
读者: 批处理开发者、解析内核开发者、服务部署者
范围: `mineru-kit` 的定位、子命令和与 `mineru` 的边界
非目标: 本地文档库、搜索、watch、Agent 默认体验
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru-kit` 是无状态解析工具和服务工具。它不维护 `doclib.db`，不做本地文档库搜索，不负责 Agent 渐进式阅读体验。

`mineru-kit` 适合：

- 大规模批处理。
- 解析内核调试。
- 自部署 parse-server。
- 暴露完整解析参数、并发控制和输出策略。

## 2. 子命令

| 子命令 | 作用 | 文档 |
|--------|------|------|
| `mineru-kit models` | 下载、查看和校验本地模型配置 | [mineru-kit models](mineru-kit-models.md) |
| `mineru-kit parse` | 无状态文件/目录批处理解析 | [mineru-kit parse](mineru-kit-parse.md) |
| `mineru-kit api-server` | 启动兼容统一 API 的本地解析服务 | [mineru-kit api-server](mineru-kit-api-server.md) |
| `mineru-kit vlm-server` | 本地 VLM 服务，兼容 OpenAI Chat Completions 协议 | [mineru-kit vlm-server](mineru-kit-vlm-server.md) |
| `mineru-kit router` | 启动路由服务，转发到已有 upstream 或管理本地 worker | 当前继承旧 router 实现 |

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

当前 `mineru-kit parse` 已确定：

- 只支持文件和目录输入，不支持 stdin、路径列表、URL 输入和递归目录。
- `--output` 必填。
- local 模式支持 `tier` 与 `backend`；PDF/image 二者都不传时当前默认使用 `high` 对应 backend，仅支持 flash tier 的输入按 ADR-0024 归一。
- remote 模式支持 `--remote` / `--remote-url` / `--api-key`，允许传 `--tier`，但禁止传 `--backend`；`mineru-kit parse` 允许 remote 处理非 PDF/image 输入。

详细命令契约见 [ADR-0016](../decisions/0016-mineru-kit-parse-command.md)。

当前 `mineru-kit models` 已确定：

- 第一阶段只提供 `download`、`show`、`verify` 三个子命令。
- 继续使用旧 `mineru.json` 配置文件体系。
- `download` 用位置参数显式选择 `pipeline`、`vlm` 或 `all`，不提供默认 bundle。
- 下载完成后默认更新配置文件，不支持 `--no-config` 或自定义配置文件路径。

详细命令契约见 [ADR-0019](../decisions/0019-mineru-kit-models-command.md)。

当前 `mineru-kit router` 已确定：

- 入口参数包括 `--host`、`--port`、`--reload`、`--allow-public-http-client`、`--upstream-url`、`--local-gpus`、`--worker-host`、`--enable-vlm-preload`。
- `--upstream-url` 可重复，用于接入已有 MinerU FastAPI base URL。
- `--local-gpus` 支持 `auto`、`none` 或 GPU CSV，例如 `0,1,2`。
- 当前实现懒加载并转发到旧 router，未知额外参数继续透传到底层实现。
