# mineru server

状态: Draft
读者: 核心开发者、Agent skill 作者、高级 CLI 用户
范围: `mineru server` 的职责、生命周期、状态和 parse-server 协作
非目标: parse-server API 字段级定义；Web UI 交互设计
来源: 由根目录旧 CLI 底稿迁移整理而来

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

1. 获取 `$MINERU_HOME/doclib.lock`，确保当前 home 只有一个 doclib owner。
2. 清理 stale endpoint 和 UDS socket。
3. 写入包含当前 PID、`server_id` 和空 transports 的启动阶段 endpoint。
4. 绑定 UDS/TCP listener，并用实际 transports 更新 endpoint。
5. 初始化数据目录和 SQLite。
6. 运行 migration 和默认配置种子。
7. 清理 stale task lock 和崩溃前未完成任务。
8. 创建 services，并按需拉起 managed local parse-server。
9. 启动 watch、ingest、parse、health check、device monitor 和 compaction。

默认情况下，UDS 可用时使用 `$MINERU_HOME/doclib.sock`；UDS 不可用时自动启用 TCP loopback fallback。server 启动成功后会写入 `$MINERU_HOME/doclib.endpoint.json`，供 CLI / SDK 发现实际 endpoint。

如需覆盖，可设置 `doclib.uds.*` / `doclib.tcp.*` 或对应环境变量，例如 `MINERU_DOCLIB_UDS_PATH`、`MINERU_DOCLIB_TCP_PORT`。

关闭时：

1. 停止后台任务。
2. managed 模式下停止 local parse-server。
3. 关闭数据库资源。
4. 在 ownership 保护下清理 socket 文件和 endpoint discovery 文件；CLI stop 在取得已释放的锁后执行兜底清理。
5. 释放 ownership lock；`doclib.lock` 文件是否保留取决于平台锁实现。

`mineru server stop` 会等待 server 不可连接且 `doclib.lock` 已释放后才返回成功。`restart` 只有在 stop 完成后才启动
replacement，避免两个 server 同时操作一个 MinerU home。

`doclib.start.lock` 只序列化 CLI spawn；`doclib.lock` 由 server 进程持有整个生命周期，是判断 home ownership 的唯一权威锁。
直接执行 `python -m mineru.doclib.app` 也必须获取该锁。未取得 ownership 的进程不得清理 endpoint、socket 或数据库状态；
lock 文件是否存在不能用于判断 ownership。

如果 endpoint 不可连接但 `doclib.lock` 仍被占用，status/start/stop 返回 `service_unavailable`，说明当前 home 已由另一个 doclib
server process 持有；如果 endpoint 中存在有效 PID，错误会将其标记为 `reported PID` 供诊断。用户消息不显示 lock 路径，PID
不参与 ownership 判断，CLI 也不会据此自动终止进程。不能将该状态降级为普通的“server 未运行”，也不能自动抢占或清理。

## 4. 状态输出

`mineru server status` 应返回人类可读摘要，也应支持机器可读格式。

顶层状态至少包括：

- `running`
- `pid`
- `server_id`
- `uptime_seconds`
- `mineru_home`
- `socket_path`
- `data_dir`
- `sqlite_path`
- `log_path`
- `version`
- `python_version`
- `tcp.enabled`
- `tcp.host`
- `tcp.port`
- `files_total`
- `docs_total`
- `parse_queue_length`
- `ingest_queue_length`
- `watch_count`
- `active_scan_count`
- `last_scan_at`
- `sqlite_size_bytes`

`workers` 至少包括：

- `watch_running`
- `scan_running`
- `scan_workers`
- `ingest_running`
- `ingest_workers`
- `parse_running`
- `parse_workers`
- `device_monitor_running`
- `compaction_running`
- `health_check_running`

`parse_server.local` 至少包括：

- `healthy`
- `mode`
- `url`
- `port`
- `self_hosted_url`
- `managed_tier`
- `managed_pid`
- `managed_running`
- `starting`
- `started_at`
- `restart_count`
- `max_restart_attempts`
- `last_probe_at`
- `last_success_at`
- `last_failure_at`
- `supported_tiers`

`parse_server.remote` 至少包括：

- `healthy`
- `url`
- `port`
- `last_probe_at`
- `last_success_at`
- `last_failure_at`
- `supported_tiers`

人类可读输出应至少展示：

- Server 进程与 uptime
- `MINERU_HOME`、UDS、TCP、SQLite、log、data 路径
- 文件、文档、队列、watch、scan 摘要
- SQLite 基础信息
- worker 运行状态与 worker 数量
- local / remote parse-server endpoint、模式、探测时间戳和 tier 能力

## 5. 与 parse-server 的关系

local parse-server 是独立进程，由 `mineru-kit api-server` 提供。doclib 可以用三种模式连接它：

| 模式 | 行为 |
|------|------|
| `disabled` | 不启用本地质量解析 |
| `managed` | doclib 启停时自动管理 parse-server |
| `self_hosted` | 用户自己启动 parse-server，doclib 只连接 URL |

PDF/image 默认选择策略需要通过 local 或 remote parse-server 的能力发现解析为可用的非 `flash` tier，选择顺序见 [解析 Tier](../tiers.md)。Office/HTML 的归一规则见 [ADR-0024](../decisions/0024-file-type-tier-normalization.md)；text 直接读取。

## 6. 崩溃恢复

doclib 启动时应释放 stale ingest 锁，处理崩溃前处于 parsing 状态的任务，并检查 managed parse-server 是否需要重新拉起。

连续拉起失败时，managed parse-server 可以降级为 disabled，并要求用户显式修复。

## 未决问题

managed parse-server 重启退避、多用户或多项目 UDS 命名空间，以及 `server status` JSON schema 的稳定发布边界，集中维护在 [开放问题清单](../open-questions.md)。
