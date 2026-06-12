# mineru server

状态: Draft
读者: 核心开发者、Agent skill 作者、高级 CLI 用户
范围: `mineru server` 的职责、生命周期、状态和 parse-server 协作
非目标: parse-server API 字段级定义；Web UI 交互设计
底稿: `../../../NEXT-CLI.md`

## 1. 定位

`mineru server` 管理本地 doclib 服务。doclib 是 `mineru` 的本地文档库后台，负责入库、缓存、解析任务、搜索、watch、配置和与 parse-server 的协作。

详细内部设计见 [系统架构](../architecture.md)。

## 2. 子命令

| 子命令 | 作用 |
|--------|------|
| `mineru server start` | 启动 doclib |
| `mineru server stop` | 停止 doclib |
| `mineru server restart` | 重启 doclib |
| `mineru server status` | 查看 doclib、worker 和 parse-server 状态 |

## 3. 生命周期

启动时：

1. 初始化数据目录和 SQLite。
2. 运行 migration 和默认配置种子。
3. 清理 stale lock 和崩溃前未完成任务。
4. 创建 services。
5. managed 模式下拉起 local parse-server。
6. 启动 watch、ingest、parse、health check、device monitor 和 compaction。
7. 通过 UDS 提供本地 HTTP + JSON 协议。

关闭时：

1. 停止后台任务。
2. managed 模式下停止 local parse-server。
3. 关闭数据库资源。
4. 删除 socket 文件。

## 4. 状态输出

`mineru server status` 应返回人类可读摘要，也应支持机器可读格式。

状态至少包括：

- doclib 是否运行。
- UDS 路径。
- ingest / parse 队列长度。
- worker 状态。
- local parse-server 模式和健康状态。
- remote parse-server 健康状态。
- 可发现的 tier 能力。

## 5. 与 parse-server 的关系

local parse-server 是独立进程，由 `mineru-kit api-server` 提供。doclib 可以用三种模式连接它：

| 模式 | 行为 |
|------|------|
| `disabled` | 不启用本地 standard/pro 解析 |
| `managed` | doclib 启停时自动管理 parse-server |
| `self_hosted` | 用户自己启动 parse-server，doclib 只连接 URL |

默认选择策略需要通过 local 或 remote parse-server 的能力发现解析为 `standard` 或 `pro`。

## 6. 崩溃恢复

doclib 启动时应释放 stale ingest 锁，处理崩溃前处于 parsing 状态的任务，并检查 managed parse-server 是否需要重新拉起。

连续拉起失败时，managed parse-server 可以降级为 disabled，并要求用户显式修复。

## 未决问题

`server status` JSON schema、managed parse-server 重启退避、多用户或多项目 UDS 命名空间，集中维护在 [开放问题清单](../open-questions.md)。
