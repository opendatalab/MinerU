# 配置体系

状态: Draft
读者: 想了解 doclib / mineru 配置的用户、实现配置能力的核心开发者
范围: 本地配置、远端配置、解析参数、环境变量和优先级规则
非目标: 替代 CLI 参数手册；替代部署文档
底稿: `../../NEXT-CFG.md`

## 1. 定位

配置体系定义 MinerU 如何在不同启动阶段得到最终行为。

MinerU 有两类配置：

- **启动前配置**：doclib server 启动之前就必须知道，通常来自文件形式的配置或内置默认值。
- **运行时配置**：doclib server 启动之后可从 SQLite 读取和修改，由 `config`、`watch_targets`、`exclude_rules`、`parsing_rules` 等表承载。

配置必须服务两个产品原则：

- 隐私优先：任何配置都不能导致静默上传文档。
- 质量优先：主动阅读未指定 tier 时使用默认选择策略，不能静默降级到 `flash`。

## 2. 两阶段配置模型

### 2.1 启动前配置

启动前配置解决“server 还没起来，不能读数据库”的问题。

这类配置包括：

| 分组 | 字段 | 默认值 | 说明 |
|------|------|--------|------|
| UDS | `server.uds.path` | `/tmp/mineru.sock` | CLI / doclib 通信 socket |
| UDS | `server.uds.permission` | `0o600` | socket 权限 |
| HTTP | `server.http.enabled` | `False` | 是否启用 TCP HTTP |
| HTTP | `server.http.host` | `127.0.0.1` | TCP 监听地址 |
| HTTP | `server.http.port` | `15980` | TCP 监听端口 |
| HTTP | `server.http.strict_port` | `False` | 端口占用时是否报错 |
| HTTP | `server.http.backlog` | `128` | socket backlog |
| HTTP | `server.http.timeout` | `600` | keep-alive timeout |
| log | `server.log.path` | `~/MinerU/mineru.log` | 日志路径 |
| log | `server.log.level` | `info` | 日志级别 |
| server | `server.data_dir` | `~/MinerU` | 数据目录 |
| server | `server.ingest_workers` | `2` | ingest worker 数 |
| server | `server.parse_workers` | `2` | parse worker 数 |
| server | `server.compaction_interval_sec` | `3600` | compaction 间隔 |
| sqlite | `sqlite.path` | `~/MinerU/mineru.db` | SQLite DB 路径 |
| sqlite | `sqlite.mmap_size` | `268435456` | mmap size |
| sqlite | `sqlite.cache_size` | `-20000` | SQLite cache size |
| sqlite | `sqlite.wal_autocheckpoint` | `1000` | WAL checkpoint 阈值 |
| sqlite | `sqlite.journal_size_limit` | `33554432` | WAL journal size limit |
| sqlite | `sqlite.temp_store` | `memory` | temp store |
| sqlite | `sqlite.synchronous` | `NORMAL` | synchronous pragma |

这些配置影响 doclib 如何启动，因此不能依赖 `mineru config` 读取。

### 2.2 运行时配置

运行时配置在 doclib server 启动后可读写。

| 来源 | 例子 | 存储 | 说明 |
|------|------|------|------|
| SQLite `config` 表 | `watch_default_tier`、`parse_server.local.mode` | KV | doclib 运行时配置 |
| `watch_targets` 表 | watch 目录、可插拔设备状态 | 表结构 | 文件发现配置 |
| `exclude_rules` 表 | exclude | 表结构 | 路径排除规则 |
| `parsing_rules` 表 | parsing-rules | 表结构 | 路径解析规则 |
| 当前命令显式参数 | `--tier pro`、`--remote` | 不持久化 | 当前请求覆盖 |
| 环境变量 | `MINERU_API_KEY` | 不持久化 | 临时凭证或 CI 覆盖 |
| SDK client 参数 | `base_url`、`api_key` | 调用方决定 | 当前 client 实例 |

## 3. 优先级

运行时行为的优先级从高到低：

1. 当前命令显式参数。
2. 当前进程环境变量。
3. 启动前文件配置或 SQLite 配置。
4. 内置默认值。

启动前文件配置和 SQLite 配置不应定义同一配置项。需要在 doclib server 启动前确定的配置使用文件配置；doclib server 启动后可读写的配置使用 SQLite。设计上不允许同一项配置同时存在于文件和 DB，因此二者之间不应产生优先级冲突。

SDK client 显式参数属于当前调用方传入的请求上下文；当它最终转化为 CLI/API 请求时，按“当前命令显式参数”处理。

如果一个配置会改变隐私边界，例如启用远端上传，必须要求当前请求或规则显式允许，不能只因为全局配置存在 remote URL 或 API Key 就上传文档。

启动前配置只用于必须在 doclib 初始化前确定的字段，比如 UDS 路径、DB 路径、SQLite pragma。它不与 SQLite 配置定义同一 key，也不在 doclib 启动后被 SQLite 覆盖。

## 4. 运行时 KV 配置

运行时 KV 配置由 doclib 初始化默认值，并写入 SQLite `config` 表。

| key | 默认值 | 说明 |
|-----|--------|------|
| `data_dir` | `~/MinerU` | 数据目录，包含 `mineru.db`、日志和解析产物 |
| `watch_default_tier` | `flash` | watch 自动发现文件时使用的默认 tier |
| `scan_interval_sec` | `300` | watch 全量扫描间隔 |
| `ingest_lock_timeout_sec` | `60` | ingest 锁超时 |
| `parse_lock_timeout_sec` | `1800` | parse 锁超时 |
| `device_check_interval_sec` | `5` | 可插拔设备检测间隔 |
| `parse_server.local.mode` | `disabled` | 本地 parse-server 模式 |
| `parse_server.local.managed_tier` | `standard` | managed 模式启动 tier |
| `parse_server.remote.url` | `https://mineru.net/api` | 默认远端 API 地址 |

`watch_default_tier=flash` 只适用于 watch 自动发现和索引，不代表 `mineru parse` 的默认阅读质量。主动读取文档时，未指定 tier 表示使用默认选择策略；Python SDK 中的等价表达是 `tier=None`。

## 5. Parse-server 配置

parse-server 是 standard/pro 解析能力的来源。它可以是本地独立进程，也可以是远端 `mineru.net/api`。

### 5.1 Local parse-server

| key | 默认值 | 说明 |
|-----|--------|------|
| `parse_server.local.mode` | `disabled` | `disabled` / `managed` / `self_hosted` |
| `parse_server.local.managed_tier` | `standard` | managed 模式启动的 tier |
| `parse_server.local.self_hosted_url` | 无 | self-hosted parse-server URL |
| `parse_server.local.self_hosted_api_key` | 无 | self-hosted API Key，可选 |

模式语义：

| mode | 行为 |
|------|------|
| `disabled` | 不使用本地 parse-server；本地 standard/pro 请求返回 `no_engine` 或相关错误 |
| `managed` | doclib 启动和停止时自动管理 parse-server |
| `self_hosted` | 用户自行启动 parse-server，doclib 只负责连接和探活 |

### 5.2 Remote parse-server

| key | 默认值 | 说明 |
|-----|--------|------|
| `parse_server.remote.url` | `https://mineru.net/api` | 默认远端 API 地址 |
| `parse_server.remote.api_key` | 无 | 远端 API Key |

远端配置存在不等于允许上传。只有当前请求显式 `--remote`，或 parsing-rule 显式带 remote 语义时，才可以上传文档。

## 6. API Key 与环境变量

API Key 读取优先级：

1. 环境变量 `MINERU_API_KEY`。
2. SQLite config 中的 `parse_server.remote.api_key` 或 `parse_server.local.self_hosted_api_key`。

环境变量适合无 tty 场景和 CI，但不应改变是否允许 remote 上传的判断。

建议保留的环境变量：

| 变量 | 说明 |
|------|------|
| `MINERU_API_KEY` | 默认远端 API Key |
| `MINERU_BASE_URL` | SDK 或 CLI 的默认远端 URL，是否采用待定 |
| `MINERU_TELEMETRY` | 遥测开关，是否采用待定 |

## 7. CLI 参数与配置关系

`mineru parse` 的显式参数只影响当前请求：

| CLI 参数 | 配置关系 |
|----------|----------|
| `--tier` | 覆盖当前请求 tier；未指定时使用默认选择策略 |
| `--remote` | 显式允许当前请求上传远端；远端 URL 和 API Key 来自 config 或环境变量 |
| `--pages` | 当前请求页码范围 |
| `--force` | 当前请求跳过 done 缓存；复用 active parse 或为未覆盖页创建新 parse；不删除或作废旧缓存 |
| `--wait` / `--no-wait` | 当前请求等待策略 |

`mineru config` 修改持久配置：

```bash
mineru config parse-server local.mode managed
mineru config parse-server local.managed-tier standard
mineru config parse-server remote.api-key sk-...
```

当前已实现的 config 命令分组：

| 命令 | 作用 |
|------|------|
| `mineru config show` | 展示 KV 配置、watch 和 rules |
| `mineru watch add/list/remove/rescan` | 管理 watch 目录 |
| `mineru config exclude add/list/rm` | 管理排除规则 |
| `mineru config parsing-rules add/list/rm` | 管理解析规则 |
| `mineru config parse-server local.mode` | 设置本地 parse-server 模式 |
| `mineru config parse-server local.managed-tier` | 设置 managed tier |
| `mineru config parse-server local.self-hosted-url` | 设置 self-hosted URL |
| `mineru config parse-server local.self-hosted-api-key` | 设置 self-hosted API Key |
| `mineru config parse-server remote.url` | 设置远端 URL |
| `mineru config parse-server remote.api-key` | 设置远端 API Key |

## 8. Watch 与 Rules

Watch 目录、parsing-rules 和 exclude 规则不直接存储在 `config` KV 中，但属于配置体系。

### 8.1 Watch

Watch 用于自动发现文件和建立搜索索引。

约束：

- watch 默认 tier 是 `flash`。
- watch 默认 tier 的配置项是 `watch_default_tier`。
- watch 的目标是发现、预览和索引，不是最终阅读质量。
- watch 目录不允许嵌套。
- removable watch 路径不可达时，应标记 `unreachable`，不永久删除记录。

### 8.2 Parsing-rules

Parsing-rules 通过路径 glob 指定自动解析策略。

```bash
mineru config parsing-rules add "*/论文/*" --tier standard --pages all
mineru config parsing-rules add "*/合同/*" --tier pro --remote
```

约束：

- 规则必须显式指定 remote，才允许上传远端。
- 规则命中的 tier 必须经过能力检查。
- 能力不足时记录结构化错误，不静默降级到 `flash`。
- 允许不指定 tier。执行时必须通过能力发现解析为实体 tier，并只记录实际使用的实体 tier。
- 默认选择策略不能解析为 `flash`。

### 8.3 Exclude

Exclude 使用 glob 模式排除路径。默认 exclude 规则由 DB 初始化写入 `exclude_rules` 表，而不是写死在 config KV 中。

默认规则覆盖：

- `*/Library/*`
- `*/.git/*`
- `*/node_modules/*`
- `*/vendor/*`
- `*/go/pkg/*`
- `*/__pycache__/*`
- `*/.venv/*`
- `*/miniconda3/*`
- `*/.nvm/*`
- `*/.docker/*`
- `*/target/*`
- `*/dist/*`
- `*/build/*`

## 9. SDK Client 配置

SDK 配置应显式、可类型检查，并避免 `**kwargs: Any` 式透传。

建议字段：

| 字段 | 说明 |
|------|------|
| `base_url` | API 地址 |
| `api_key` | API Key |
| `timeout` | 请求超时 |
| `default_tier` | SDK client 自身默认 tier；未设置时使用默认选择策略，不读取 `watch_default_tier` |
| `allow_remote` | 是否允许上传远端，应显式设置 |

SDK client 的配置不应在 import 时读取重依赖或启动服务。

## 10. 隐私与遥测

隐私配置和遥测配置必须分开。

文档隐私：

- 默认 local。
- 远端上传必须由当前请求或规则显式允许。
- API Key 存在不代表允许上传。

遥测：

- 不包含文档内容、文件名和路径。
- doclib server 是 P0 唯一 telemetry 上报主体。
- 首次启动应要求用户选择，默认勾选开启，用户可关闭。
- parser SDK、`mineru-kit parse`、`mineru-kit api-server` 等纯工具无 telemetry 能力。
- telemetry 配置存储在 SQLite 中，不使用启动前文件配置。
- telemetry 聚合数据也存储在 SQLite 中，flush 成功后删除已确认上报的数据。

P0 必须完成 telemetry 设计，明确字段、开关、默认值、隐私边界和上报时机。具体设计见 [Telemetry 设计](telemetry.md)。

## 与其他文档的关系

- CLI 参数见 [CLI 规格](cli.md)。
- CLI 配置命令见 [mineru library](cli/mineru-library.md)。
- 启动前 server 配置和 SQLite `config` 表见 [系统架构](architecture.md)。
- API key、base_url、timeout 等 client 配置见 [SDK 设计](sdk.md)。
- 隐私默认策略见 [产品路线图](roadmap.md)。
- Tier 默认语义见 [解析 Tier](tiers.md)。
- 配置相关错误见 [错误码体系](errors.md)。

## 未决问题

项目级配置、用户级配置路径和正式环境变量等问题，集中维护在 [开放问题清单](open-questions.md)。
