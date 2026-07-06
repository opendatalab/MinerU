# Telemetry 设计

状态: Draft
读者: 产品负责人、核心开发者、doclib server 开发者、数据分析与合规参与者
范围: 下一代 MinerU 的匿名使用统计、稳定性指标和粗粒度环境画像
非目标: 计费、审计、安全风控、精确故障追踪、上传文档内容或用户输入

## 1. 目标

Telemetry 的首要目标是产品使用统计，用于评估 MinerU 的使用规模和产品影响力。

P0 同时兼顾三类问题:

1. 产品使用统计: 有多少安装实例在使用、每天使用多少次、处理了多少文件和页面。
2. 质量与稳定性: 哪些主链路失败、失败发生在哪个阶段、错误码分布如何。
3. 性能与环境画像: 不采集细粒度设备指纹，只用粗粒度字段理解运行环境和处理规模。

Telemetry 数据只用于趋势分析和产品改进，不用于计费、安全判断或强一致审计。

## 2. 采集边界

只有 doclib server 上报 telemetry。

`mineru` CLI 依赖 doclib server，因此可以被间接统计。以下纯工具无 telemetry 能力:

- parser SDK，包括 `mineru.parser.parse()` 和 `parse_async()`
- `mineru-kit parse`
- `mineru-kit api-server`
- 其他不依赖 doclib server 的专家工具

这个边界避免 parser/tool 层在作为库或底层工具使用时包含 telemetry 相关配置、存储或网络行为。

## 3. 默认策略

首次启动 doclib server 时，必须要求用户完成 telemetry 选择。

默认行为:

- 首次选择界面默认勾选开启 telemetry。
- 用户可以关闭。
- 用户选择结果记录为 `consent_state`。
- 用户关闭 telemetry 后，不再记录新的 telemetry metrics，也不再上报。
- 关闭时，DB 中未上报 telemetry 数据应删除，避免后续误发。

用户必须可以通过命令查看和修改状态:

```bash
mineru telemetry status
mineru telemetry enable
mineru telemetry disable
mineru telemetry preview
mineru telemetry flush
```

`preview` 应展示将要上报的 JSON，但不得展示任何 token 或服务端内部凭据。

## 4. 隐私边界

Telemetry 禁止采集:

- 文档内容
- 文件名
- 文件路径
- 原始 URL
- search query
- prompt
- sha256
- job_id
- parse_id
- traceback
- exception message
- hostname
- 用户名
- 邮箱
- 账号 ID
- CPU/GPU 具体型号
- GPU 显存、驱动版本、CUDA 版本

允许采集的数据必须来自白名单字段。不得把请求体、异常对象、配置对象、`kwargs` 或任意字典直接透传到 telemetry。

## 5. 统计口径

P0 使用匿名 `installation_id` 作为主统计口径。

- `installation_id` 在 doclib server 首次启用 telemetry 时生成。
- `installation_id` 存储在 doclib DB 的 telemetry 内部状态中。
- 不依赖账号体系。
- 用户删除配置或重装后，会被视为新的安装实例。

服务端基于 `installation_id` 计算:

- 安装量: 出现过的唯一 `installation_id`
- DAU: 当日有 telemetry 上报的唯一 `installation_id`
- WAU / MAU: 对应周期内有 telemetry 上报的唯一 `installation_id`

如果未来引入账号，账号只能作为可选维度，不替代 `installation_id`。

## 6. 上报模型

P0 采用本地聚合 metrics 上报，不采用每个事件即时上报。

运行时行为:

```text
doclib server runtime
  -> record metric delta
  -> update telemetry aggregate in doclib DB
  -> periodic flush
  -> POST telemetry endpoint
  -> delete flushed rows from doclib DB after success
```

建议聚合窗口:

- 本地 telemetry 配置和 telemetry 聚合数据都存储在 doclib DB 中。
- 本地按 1 小时 period 聚合。
- 上报时可以一次提交多个未 flush 的小时级 period。
- 启动后尝试 flush 一次。
- 运行中每 24 小时尝试 flush 一次。
- 网络失败或服务端 5xx 时保留本地聚合结果，后续重试。
- 服务端 4xx 表示 payload 不合法，对应 batch 应丢弃或进入本地失败记录，不应无限重试。
- flush 成功后，已被服务端确认接收的 telemetry 聚合数据应从 doclib DB 中删除。
- 同一个 period 内 context 变化可以接受；P0 flush 时使用当前 context 上报该 period 的聚合 metrics。

主流程不得等待 telemetry 上报。telemetry 失败不得影响 parse、watch、search 或 doclib API。

## 7. Endpoint

建议使用独立 telemetry endpoint:

```http
POST https://telemetry.mineru.net/v1/metrics
Content-Type: application/json
```

不建议与 `mineru.net/api` 的 parse API 混在一起，因为 telemetry 的认证、限流、存储和合规策略不同。

响应示例:

```json
{
  "object": "telemetry_metrics_result",
  "accepted": 42,
  "rejected": 0,
  "server_time": "2026-06-10T12:00:00Z"
}
```

如果服务端支持部分拒绝，可以返回:

```json
{
  "object": "telemetry_metrics_result",
  "accepted": 40,
  "rejected": 2,
  "errors": [
    {
      "index": 5,
      "code": "invalid_dimension",
      "message": "dimension value is not allowed"
    }
  ],
  "server_time": "2026-06-10T12:00:00Z"
}
```

错误信息只用于解释 schema 问题，不应包含用户上报原文中的敏感字段。

## 8. Payload 结构

一次上报包含一个 batch。环境信息放在 batch-level `context` 中，不重复放入每个 metric。

```json
{
  "batch_id": "tb_01HX...",
  "schema_version": "1",
  "installation_id": "inst_01HX...",
  "period_start": "2026-06-10T14:00:00Z",
  "period_end": "2026-06-10T15:00:00Z",
  "context": {
    "app": "mineru",
    "app_version": "3.0.0",
    "os": "macos",
    "arch": "arm64",
    "python_version": "3.11",
    "install_channel": "pip",
    "cpu_count_bucket": "9_16",
    "gpu_vendor": "apple"
  },
  "metrics": [
    {
      "name": "parse.completed.count",
      "value": 12,
      "dimensions": {
        "source": "cli",
        "caller": "agent",
        "tier": "medium(default)",
        "status": "succeeded"
      }
    },
    {
      "name": "parse.processed_page_count",
      "value": 1240,
      "dimensions": {
        "source": "cli",
        "caller": "agent",
        "tier": "medium(default)"
      }
    }
  ]
}
```

规则:

- `batch_id` 由本地 flush 任务生成，用于服务端短窗口幂等。
- `value` 必须是非负数。
- 字符串数据只能出现在 `context` 或 `dimensions`。
- `context` 描述安装实例、运行环境和版本。
- `metrics[].dimensions` 描述当前 metric 的业务分类。
- `metrics[].dimensions` 必须来自白名单，不能包含任意字符串。
- `context` 表示 flush 时 doclib server 的当前运行环境。同一 period 内 context 变化可以接受，不要求拆分 batch。

## 9. Context 字段

P0 context 字段:

| 字段 | 类型 | 枚举 / 格式 | 说明 |
|------|------|-------------|------|
| `app` | string | `mineru` | 固定应用名 |
| `app_version` | string | semver 或发布版本 | MinerU 版本 |
| `os` | string | `macos` / `windows` / `linux` / `other` / `unknown` | 操作系统 |
| `arch` | string | `x86_64` / `arm64` / `other` / `unknown` | CPU 架构 |
| `python_version` | string | `3.10` / `3.11` / `3.12` / `3.13` / `other` / `unknown` | Python 主次版本 |
| `install_channel` | string | `pip` / `uv` / `docker` / `source` / `unknown` | 安装来源 |
| `cpu_count_bucket` | string | `1_4` / `5_8` / `9_16` / `gt_16` / `unknown` | CPU 核心数分桶 |
| `gpu_vendor` | string | `nvidia` / `apple` / `amd` / `none` / `unknown` | GPU 或常见加速器厂商 |

`gpu_vendor` 规则:

- `nvidia`: 识别到 NVIDIA GPU。
- `apple`: 识别到 Apple GPU。
- `amd`: 识别到 AMD GPU。
- `none`: 明确没有发现 GPU。
- `unknown`: 无法检测、检测失败，或发现了 P0 未分类的其他加速器。

P0 不采集 CPU/GPU family、具体型号、显存、驱动和 CUDA 版本。

## 10. Metric 与 Dimension 规则

Metric 的 `name` 表示数值指标，`value` 表示计数或数值累加。

字符串型信息必须作为 dimension:

```json
{
  "name": "parse.completed.count",
  "value": 1,
  "dimensions": {
    "tier": "medium",
    "status": "succeeded"
  }
}
```

不允许:

```json
{
  "name": "parse.tier",
  "value": "medium"
}
```

Dimension 只允许低基数字段:

| 字段 | 枚举 / 说明 |
|------|-------------|
| `source` | `http_api` / `sdk` / `cli` / `mcp` / `web` / `app` / `watch` / `background` / `unknown` |
| `caller` | `agent` / `user` / `http_client` / `sdk` / `web` / `app` / `system` / `unknown` |
| `tier` | `default` / `flash` / `medium` / `high` / `medium(default)` / `high(default)` / `unknown` |
| `status` | `succeeded` / `failed` / `partial` / `canceled` |
| `input_type` | `pdf` / `docx` / `pptx` / `xlsx` / `html` / `other` / `unknown` |
| `output_format` | `middle_json` / `markdown` / `structured_content` / `other` |
| `stage` | `enqueue` / `cache_lookup` / `parse_server_call` / `parsing` / `merge_pages` / `persist` / `index` / `unknown` |
| `error_code` | 稳定错误码枚举，不允许 exception message |
| `retryable` | `true` / `false` / `unknown` |
| `bucket` | 当前 metric 定义的分桶值 |
| `server` | `local(flash)` / `local(managed)` / `local(self-hosted)` / `remote(official)` / `remote(custom)` / `none` / `unknown` |
| `rule_type` | `exclude` / `parsing` / `unknown` |
| `remote` | `true` / `false` / `unknown` |
| `command` | 当前 metric 定义的命令枚举 |
| `wait_mode` | `sync` / `async` / `unknown` |
| `wait_bucket` | `0` / `1_10s` / `11_60s` / `gt_60s` / `unknown` |
| `page_range_mode` | `all` / `partial` / `default` / `unknown` |
| `config_area` | `watch` / `exclude` / `parsing_rules` / `parse_server` / `telemetry` / `unknown` |
| `cleanup_target` | `orphan_docs` / `deleted_files` / `temp` / `mixed` / `unknown` |
| `dry_run` | `true` / `false` / `unknown` |

`caller` 与 `source` 语义不同:

- `source` 表示请求从哪个调用通道进入 doclib，例如 HTTP API、SDK、CLI、MCP、Web、App、watch 或后台任务。
- `caller` 表示发起行为的主体，例如 Agent、用户、HTTP client、SDK、Web、App 或系统后台任务。

Agent 时代的核心产品目标要求 telemetry 能区分 `agent` 与 `user`。因此，CLI 通道必须尽量识别真实调用主体。

调用通道与主体规则:

| 使用方式 | `source` | `caller` |
|----------|----------|---------------|
| 直接调用 HTTP API，且未传 caller 信息 | `http_api` | `http_client` |
| Doclib SDK | `sdk` | `sdk` |
| `mineru` CLI，由 Agent 进程调用 | `cli` | `agent` |
| `mineru` CLI，由用户 shell 调用 | `cli` | `user` |
| `mineru` CLI，无法可靠判断父进程 | `cli` | `unknown` |
| MCP Server | `mcp` | `agent` |
| Web | `web` | `web` |
| 桌面 App | `app` | `app` |
| watch 自动发现 | `watch` | `system` |
| compaction / cleanup / health check 等后台任务 | `background` | `system` |

CLI 的 `caller` 可以通过父进程识别。若父进程是 Claude Code、Codex 等已知编程 Agent，则记为 `agent`；若父进程是普通 shell 或终端，则记为 `user`；无法可靠判断时记为 `unknown`。不得通过文件路径、命令文本、query、prompt、文档内容或具体会话内容推断。

`tier` 同时表达请求档位和实际解析档位:

- 用户未指定 tier，且尚未解析到实体 tier 时，使用 `default`。
- 用户显式指定实体 tier，或任务已经完成且没有必要保留默认来源时，使用 `flash`、`medium` 或 `high`。
- 用户未指定 tier，且已经解析到实体 tier，同时仍需要保留来源信息时，使用 `medium(default)` 或 `high(default)`。
- 默认选择不会解析为 `flash`，因此不定义 `flash(default)`。
- parse-server 相关 metric 也使用同一个 `tier` 维度，不再单独使用 `parse_server_tier`。

`server` 表达解析执行位置和服务形态:

- `local(flash)`: 本地 flash 解析，不经过 parse-server。
- `local(managed)`: doclib 管理的本地 parse-server。
- `local(self-hosted)`: 用户自行启动的本地或自部署 parse-server。
- `remote(official)`: 官方 `mineru.net/api`。
- `remote(custom)`: 用户显式配置的其他远端兼容 parse-server。
- `none`: 当前行为没有解析服务，例如 cache hit、查询类操作或本地 parse-server disabled 配置状态。
- `unknown`: 无法可靠判断。

`parse_server.*` metric 不使用 `server=local(flash)`，因为 flash 不经过 parse-server。`parse.*` metric 可以使用 `server=local(flash)` 表示显式 flash 本地解析。

## 11. Metric 候选全集

Telemetry 记录业务行为，不记录所有 HTTP 请求。状态查询、健康检查、列表读取和普通配置读取默认不计入 telemetry。

本节列出第一版候选全集。后续实现前可以按 P0 范围裁剪，但裁剪后的实现必须仍能回答产品使用规模、核心链路稳定性和主要性能瓶颈。

`parse.*` 与 `parse_server.*` 的语义边界:

- `parse.*` 统计 doclib 解析业务生命周期，回答用户是否发起解析、缓存是否命中、任务是否完成、处理了多少文件和页面、最终状态如何。
- `parse_server.*` 统计 doclib 解析业务生命周期中调用 parse-server 的执行片段，回答是否实际使用 parse-server、使用 local 还是 remote、使用什么模式和 tier、该执行片段的请求数、处理量、失败率和耗时。
- 一次 parse 请求可以产生 `parse.*`，但不一定产生 `parse_server.*`。例如 cache hit、显式 `flash` 本地解析、复用已有完成批次时，通常不产生 parse-server 执行指标。
- 一次 parse 请求如果实际调用了 local 或 remote parse-server，则同时产生 `parse.*` 和 `parse_server.*`。两者不能互相替代。
- `parse.duration.bucket.count` 统计 doclib parse 业务生命周期耗时；`parse_server.duration.bucket.count` 只统计 doclib 调用 parse-server 的执行片段耗时。

产品使用统计:

| Metric | 说明 |
|--------|------|
| `server.started.count` | doclib server 启动次数 |
| `server.shutdown.count` | doclib server 正常关闭次数 |
| `server.recovered_stale_lock.count` | doclib server 启动恢复 stale lock 的次数 |
| `cli.command.count` | `mineru` CLI 子命令调用次数 |
| `server.command.count` | `mineru server` 生命周期命令调用次数 |
| `ingest.requested.count` | 入库请求次数 |
| `ingest.completed.count` | 入库成功次数 |
| `ingest.duplicate.count` | 入库时发现内容已存在的次数 |
| `parse.requested.count` | parse 请求次数 |
| `parse.completed.count` | parse 完成次数，按 `status` 区分成功、失败、部分成功和取消 |
| `parse.files.count` | parse 处理文件数 |
| `parse.processed_page_count` | parse 处理页数；数值统计，不是页面列表 |
| `parse.output.requested.count` | parse 请求输出格式次数 |
| `parse.output.to_stdout.count` | parse 输出到 STDOUT 的次数 |
| `parse.output.to_file.count` | parse 输出到文件的次数 |
| `parse.marker.disabled.count` | 用户通过 `--no-marker` 禁用 marker 的次数 |
| `parse.wait_mode.count` | parse 同步或异步等待模式使用次数 |
| `parse.page_range.requested.count` | parse 请求页码范围模式使用次数 |
| `parse.force.count` | 用户通过 `--force` 请求重新解析的次数 |
| `parse.created_batch.count` | 本次请求创建的新 parse batch 数 |
| `parse.active_reused.count` | 本次请求复用 pending/parsing batch 的次数 |
| `parse.waited_batch.count` | 本次请求等待的 parse batch 数 |
| `watch.files.discovered.count` | watch 发现文件数 |
| `watch.files.parsed.count` | watch 自动解析文件数 |
| `rules.matched.count` | rule 命中次数 |
| `rules.excluded.count` | exclude rule 排除文件次数 |
| `rules.parse_rule_applied.count` | parsing-rule 被应用次数 |
| `search.performed.count` | search 次数，不采集 query |
| `find.performed.count` | find 文件名搜索次数，不采集 query |
| `doclib.content.requested.count` | 请求读取或渲染文档内容的次数 |
| `doclib.content.rendered.count` | 内容读取时成功渲染输出格式的次数 |
| `doclib.content.truncated.count` | 内容输出因 token/page budget 被截断的次数 |
| `doclib.content.continuation_requested.count` | 用户或 Agent 请求 continuation 的次数 |
| `doclib.docs.added.count` | doclib 新增文档数 |
| `search.index.updated.count` | 搜索索引更新次数 |
| `search.index.source_tier.count` | 搜索索引来源 tier 计数 |
| `parse.cache.hit.count` | parse cache 命中次数 |
| `parse.cache.miss.count` | parse cache 未命中次数 |
| `remote.requested.count` | 用户显式请求 remote 的次数 |
| `remote.allowed.count` | remote 请求被隐私策略允许的次数 |
| `config.watch.added.count` | watch 目录添加次数 |
| `config.watch.removed.count` | watch 目录移除次数 |
| `config.rule.added.count` | exclude 或 parsing rule 添加次数 |
| `config.rule.removed.count` | exclude 或 parsing rule 移除次数 |
| `config.parse_server.changed.count` | parse-server 配置变更次数 |
| `parse_server.configured.count` | doclib server 启动或配置变化时记录当前 parse-server 配置 |
| `parse_server.local.started.count` | managed 本地 parse-server 被 doclib 拉起的次数 |
| `parse_server.local.restarted.count` | managed 本地 parse-server 被重启的次数 |
| `parse_server.requested.count` | doclib 调用 parse-server 的请求次数 |
| `parse_server.completed.count` | doclib 调用 parse-server 成功完成次数 |
| `parse_server.files.count` | parse-server 处理文件数 |
| `parse_server.processed_page_count` | parse-server 处理页数；数值统计，不是页面列表 |
| `parse.compaction.completed.count` | parse 批次 compaction 成功次数 |
| `cleanup.deleted_files.count` | cleanup 删除文件数 |
| `cleanup.orphan_records.count` | cleanup 清理孤儿记录数 |

质量与稳定性:

| Metric | 说明 |
|--------|------|
| `ingest.failed.count` | 入库失败次数 |
| `parse.failed.count` | parse 失败次数 |
| `parse.partial.count` | parse 部分成功次数 |
| `parse.retry.count` | parse 重试次数 |
| `parse.timeout.count` | parse 超时次数 |
| `remote.not_allowed.count` | remote 因隐私策略未显式允许而被拒绝的次数 |
| `remote.fallback_to_local.count` | remote 失败后 fallback 到 local 的次数 |
| `parse_server.unavailable.count` | doclib 调用 parse-server 不可用次数 |
| `parse_server.failed.count` | doclib 调用 parse-server 失败次数 |
| `parse_server.local.start_failed.count` | managed 本地 parse-server 启动失败次数 |
| `parse.fallback.count` | 发生 fallback 的次数 |
| `parse.invalidate.count` | 用户或系统触发 invalidate 的次数 |
| `watch.error.count` | watch 主链路错误次数 |
| `parse.wait.timeout.count` | parse 同步等待超时次数 |
| `search.index.failed.count` | 搜索索引更新失败次数 |
| `doclib.content.render.failed.count` | 内容读取时渲染输出格式失败次数 |
| `parse.compaction.failed.count` | parse 批次 compaction 失败次数 |
| `cleanup.failed.count` | cleanup 失败次数 |

性能与规模分布:

| Metric | 说明 |
|--------|------|
| `ingest.file_size.bucket.count` | 入库文件大小分布 |
| `parse.duration.bucket.count` | parse 耗时分布 |
| `parse.queue.wait.duration.bucket.count` | parse 请求排队等待耗时分布 |
| `parse.page_count.bucket.count` | 单文件页数分布 |
| `parse.file_size.bucket.count` | 文件大小分布 |
| `parse_server.duration.bucket.count` | doclib 调用 parse-server 的耗时分布 |
| `doclib.content.render.duration.bucket.count` | 内容读取时渲染输出格式耗时分布 |
| `search.result_count.bucket.count` | search 返回结果数量分布，不采集 query 或结果内容 |
| `find.result_count.bucket.count` | find 返回结果数量分布，不采集 query 或结果内容 |
| `cleanup.requested.count` | cleanup 用户请求次数 |

Parse-server 相关 metric 必须带以下维度，便于区分用户是否使用 parse-server、使用哪类解析服务、以及对应 tier:

```text
server: local(managed) | local(self-hosted) | remote(official) | remote(custom) | unknown
tier: medium | high | medium(default) | high(default) | unknown
status: succeeded | failed | partial | canceled
```

Parse-server 配置口径:

- `parse_server.configured.count`: 记录 doclib server 启动或 parse-server 配置变化时的配置状态。
- 本地 disabled 时，使用 `server=none`、`tier=unknown`。
- 本地 managed 时，使用 `server=local(managed)`、`tier=<managed tier>`。
- 本地 self-hosted 时，使用 `server=local(self-hosted)`、`tier` 来自能力发现结果；发现失败时为 `unknown`。
- 官方远端使用 `server=remote(official)`。
- 用户显式配置的其他远端兼容 parse-server 使用 `server=remote(custom)`。
- 不记录 remote URL。

本地 parse-server 的解析数量和性能通过以下查询口径获得:

- 数量: `parse_server.requested.count`、`parse_server.completed.count`、`parse_server.files.count`、`parse_server.processed_page_count`，且 `server=local(managed)` 或 `local(self-hosted)`。
- 性能: `parse_server.duration.bucket.count`，且 `server=local(managed)` 或 `local(self-hosted)`。
- 模式: 由 `server=local(managed)` 或 `local(self-hosted)` 区分。
- tier: `tier=medium`、`high`、`medium(default)` 或 `high(default)`。

Telemetry 自身:

| Metric | 说明 |
|--------|------|
| `telemetry.enabled.count` | 用户开启 telemetry 次数 |
| `telemetry.disabled.count` | 用户关闭 telemetry 次数 |
| `telemetry.flush.succeeded.count` | telemetry 上报成功次数 |
| `telemetry.flush.failed.count` | telemetry 上报失败次数 |

## 12. P0 裁剪原则

P0 不一定实现所有候选 metric。裁剪时必须保留能回答以下问题的指标:

1. 使用规模: WAI、DAU、parse 次数、文件数、页数。
2. Agent 入口: `caller=agent` 与 `caller=user` 的使用差异。
3. 调用通道: HTTP API、SDK、CLI、MCP、Web、App、watch 和后台任务的来源分布。
4. 解析服务: 是否使用 parse-server，使用 `local(managed)`、`local(self-hosted)`、`remote(official)` 还是 `remote(custom)`，medium 还是 high。
5. 主链路质量: parse、ingest、render、search index 的成功和失败。
6. 关键性能: parse 耗时、parse-server 耗时、queue wait、render 耗时的 bucket。
7. 隐私边界: remote 是否被请求、允许、拒绝，以及 remote 失败后是否 fallback 到 local。

可优先裁剪或延后实现的指标:

- cleanup 和 compaction 维护类指标。
- server shutdown 等生命周期补充指标。
- continuation / truncation 等尚未进入第一版输出能力的指标。
- 过细且暂时没有 dashboard 消费场景的 bucket 指标。

## 13. Bucket 定义

耗时分桶:

```text
lt_1s | 1_5s | 5_30s | 30_120s | 2_10m | gt_10m
```

页数分桶:

```text
1 | 2_5 | 6_20 | 21_100 | 101_500 | gt_500
```

文件大小分桶:

```text
lt_1mb | 1_10mb | 10_50mb | 50_200mb | gt_200mb
```

search 结果数分桶:

```text
0 | 1_5 | 6_20 | 21_100 | gt_100
```

find 结果数分桶:

```text
0 | 1_5 | 6_20 | 21_100 | gt_100
```

## 14. 候选 Metric 口径

### 14.1 Ingest

Ingest 指文件进入 doclib 后计算 SHA256、提取基础 metadata、写入 `files` / `docs` / 文件名索引的阶段。

推荐维度:

```text
source: http_api | sdk | cli | mcp | web | app | watch | background | unknown
caller: agent | user | http_client | sdk | web | app | system | unknown
input_type
status
error_code
```

`ingest.duplicate.count` 表示同一 `sha256` 已存在，只新增或更新文件路径实例，不代表失败。

### 14.2 Search Index

Search index 指文件名索引和内容全文索引更新，不记录索引文本、snippet 或命中文档。

推荐维度:

```text
source: http_api | sdk | cli | mcp | web | app | watch | background | unknown
caller: agent | user | http_client | sdk | web | app | system | unknown
tier
status
error_code
```

`search.index.source_tier.count` 用于统计当前索引文本来自哪个 tier。搜索结果可以来自 `flash`，但用户主动阅读时不能默认停留在 `flash`。

### 14.3 Render / Content

Render 指 `GET /docs/{sha256}/content` 或等价 SDK/CLI 读取动作中，从 Middle JSON 转换为 Markdown、structured_content、HTML 等派生格式。

推荐维度:

```text
source
caller
output_format
tier
status
error_code
bucket
```

`doclib.content.requested.count` 记录内容读取请求。`doclib.content.rendered.count` 记录实际完成渲染。请求 `middle_json` 时可以只记录 requested，不一定记录 rendered。

### 14.4 Queue / Batch

Queue 和 batch 指 doclib parse 请求在 DB 任务队列中的创建、复用和等待行为。

推荐维度:

```text
source
caller
tier
status
```

`parse.created_batch.count` 表示请求创建了新的 parse batch。`parse.active_reused.count` 表示请求复用了已经存在的 pending/parsing batch。`parse.waited_batch.count` 表示当前请求需要等待的 batch 数；一次请求可以等待多个 batch。

`parse.queue.wait.duration.bucket.count` 只统计从请求进入等待到相关 batch 完成或失败的等待耗时，不包含实际 parse-server 执行耗时。

### 14.5 Remote Boundary

Remote metrics 只记录用户是否显式允许远端和是否发生 local fallback，不记录 remote URL、API Key 或账号信息。

推荐维度:

```text
source
caller
tier
status
error_code
```

`remote.requested.count` 表示当前请求显式带有 remote 语义。`remote.allowed.count` 表示隐私策略允许该请求使用 remote。`remote.not_allowed.count` 表示系统知道 remote 可能解决问题，但用户没有显式允许。`remote.fallback_to_local.count` 表示 remote 失败后按隐私边界 fallback 到 local。

### 14.6 Rules

Rules metrics 只记录规则是否命中，不记录 rule name、pattern、路径或文件名。

推荐维度:

```text
rule_type: exclude | parsing | unknown
tier
remote: true | false | unknown
```

`rules.excluded.count` 表示 exclude rule 阻止文件进入后续链路。`rules.parse_rule_applied.count` 表示 parsing-rule 决定了 tier、page_range 或 remote 语义。

### 14.7 Server Lifecycle

Server lifecycle metrics 记录 doclib server 和 managed local parse-server 的生命周期。

推荐维度:

```text
status
error_code
server
tier
```

`server.recovered_stale_lock.count` 用于统计启动恢复能力。`parse_server.local.start_failed.count` 和 `parse_server.local.restarted.count` 用于观察 managed local parse-server 稳定性。

### 14.8 Compaction / Cleanup

Compaction 和 cleanup 属于维护链路，用于观察 JSON 批次增长、孤儿记录和文件清理效果。

推荐维度:

```text
tier
status
error_code
```

这些指标不记录 sha256、路径或文件名。

### 14.9 CLI 行为

CLI 行为 metrics 只记录用户或 Agent 使用了哪些 `mineru` 子命令和关键模式，不记录命令参数中的路径、query、pattern、URL、API Key 或输出内容。

推荐维度:

```text
source: cli
caller
command
status
```

`cli.command.count` 的 `command` 取值:

```text
parse | invalidate | search | find | show | cleanup | server | config | telemetry | unknown
```

`server.command.count` 的 `command` 取值:

```text
start | stop | restart | status | unknown
```

`GET /server/status` 不计入 `server.command.count`，避免健康检查和轮询污染。只有用户或 Agent 通过 CLI 主动执行 `mineru server status` 时才记录。

Parse CLI 行为推荐维度:

```text
source: cli
caller
tier
server
output_format
wait_mode
wait_bucket
page_range_mode
status
```

口径:

- `parse.output.requested.count`: 记录 `--format` 请求的输出格式。
- `parse.output.to_stdout.count`: 输出到 STDOUT。
- `parse.output.to_file.count`: 指定 `--output` 写入文件；不记录输出路径。
- `parse.marker.disabled.count`: 用户传入 `--no-marker`。
- `parse.wait_mode.count`: `--no-wait` 记为 `wait_mode=async`，否则记为 `sync`。
- `parse.wait.timeout.count`: 同步等待超时。
- `parse.page_range.requested.count`: 未传页码为 `default`，显式 `all` 为 `all`，其他页码范围为 `partial`；不记录具体页码字符串。
- `parse.force.count`: 用户传入 `--force`。

配置类 CLI 行为推荐维度:

```text
source: cli
caller
config_area
server
tier
remote
status
```

口径:

- `config.watch.added.count` / `config.watch.removed.count`: 不记录 watch path、label 或 removable 设备标识。
- `config.rule.added.count` / `config.rule.removed.count`: 不记录 rule name、pattern 或 page_range。
- `config.parse_server.changed.count`: 不记录 URL 或 API Key，只通过 `server` 区分 `local(managed)`、`local(self-hosted)`、`remote(official)`、`remote(custom)`、`none`、`unknown`。

Cleanup CLI 行为推荐维度:

```text
source: cli
caller
cleanup_target
dry_run
status
```

`cleanup_target` 取值:

```text
orphan_docs | deleted_files | temp | mixed | unknown
```

## 15. Doclib API 感知范围

Telemetry 不记录所有 doclib HTTP API 调用，只记录能够代表产品使用规模或核心行为的 API。

对用户行为类 API，调用方应尽量向 doclib 传递 `caller`。这使 telemetry 可以区分 Agent 调用 `mineru` 命令和用户直接调用 `mineru` 命令。不能可靠识别时使用 `unknown`，不得通过敏感内容推断。
直接调用 HTTP API 且未传 caller 信息时，`source=http_api`、`caller=http_client`。SDK、CLI、MCP、Web 和桌面 App 应由对应入口向 doclib 传递 `source` 与 `caller`。

应记录的 API:

| API | Metric | 说明 |
|-----|--------|------|
| `POST /parses` | `parse.requested.count` 等 parse 指标；使用 parse-server 时记录 `parse_server.*` 指标 | 记录请求、cache hit/miss、完成/失败、文件数、页数、tier、耗时、大小分布和 parse-server 使用情况 |
| `POST /invalidate` | `parse.invalidate.count` | 记录 invalidate 行为，不记录路径或 sha256 |
| `GET /search` | `search.performed.count`、`search.result_count.bucket.count` | 不记录 query、命中文档或返回内容 |
| `GET /find` | `find.performed.count`、`find.result_count.bucket.count` | 与 search 分开统计，不记录 query、命中文档或返回内容 |
| `GET /docs/{sha256}/content` | `doclib.content.requested.count` | 记录内容读取或渲染次数，不记录 sha256 或内容 |

不应记录为 telemetry 的 API:

| API | 原因 |
|-----|------|
| `GET /parses` | 状态查询和轮询噪音较大 |
| `GET /parses/{id}` | 状态查询，不代表核心产品使用量 |
| `GET /docs` | 列表浏览噪音较大 |
| `GET /docs/{sha256}` | 元数据读取，P0 不作为核心指标 |
| `GET /files/by-path` | 工具性查询 |
| `GET /config` / `POST /config` | 普通配置读取和修改不计入，telemetry 开关变化除外 |
| watch / rules 配置 API | 只记录后台 watch 实际发现和解析行为 |
| cleanup API | 管理维护行为 |
| `GET /server/status` | 健康检查和轮询会污染活跃度 |
| `POST /shutdown` | 管理行为 |

## 16. 本地存储

Telemetry 用户状态、内部状态和聚合数据都存储在 doclib DB 中，不使用启动前文件配置。

Telemetry endpoint 不支持用户配置，应由代码内置。开发和测试环境如需覆盖 endpoint，应使用测试构建或专门的开发开关，不进入用户配置体系。

建议状态结构可以继续用 KV 表表达:

```toml
[telemetry]
consent_state = "enabled"
installation_id = "inst_01HX..."
last_flush_at = "2026-06-10T12:00:00Z"
```

状态语义:

- `consent_state`: 用户 telemetry 状态，取值为 `unset` / `enabled` / `disabled`。
- `installation_id`: 内部匿名安装实例 ID，用户不可配置，也不需要默认展示完整值。
- `last_flush_at`: 内部运行状态，表示最近一次成功 flush 时间，不是用户配置。

`consent_state` 语义:

| 值 | 含义 |
|----|------|
| `unset` | 用户尚未完成首次选择，doclib server 应展示或触发首次选择流程 |
| `enabled` | 用户允许记录并上报 telemetry |
| `disabled` | 用户明确关闭 telemetry |

如果 `consent_state=disabled`，doclib server 不应写入新的 telemetry 聚合数据。

Telemetry 聚合数据应使用独立表保存，至少包含:

| 字段 | 说明 |
|------|------|
| `id` | 本地聚合记录 ID |
| `period_start` | 聚合窗口开始时间 |
| `period_end` | 聚合窗口结束时间 |
| `metric_name` | metric 名称 |
| `metric_value` | 非负数值 |
| `dimensions` | metric-level dimensions JSON |
| `dimensions_hash` | dimensions JSON 的稳定 hash，用于聚合 upsert |
| `created_at` | 创建时间 |
| `updated_at` | 最近更新时间 |

聚合唯一键:

```text
period_start + period_end + metric_name + dimensions_hash
```

同一唯一键重复写入时，累加 `metric_value`。聚合 key 不包含 context；flush 时读取当前 context 并组装 telemetry batch。服务端确认成功接收后，删除对应聚合记录。网络失败或 5xx 时保留记录以便后续重试。

Flush 并发与幂等采用 P0 简单方案:

1. doclib 同一时间只允许一个 flush 任务执行。
2. flush 开始时获取本地 flush lock；已有 lock 且未过期时，本次 flush 直接跳过。
3. flush 从 DB 读取一批未上报聚合记录，生成 `batch_id` 并 POST 到 telemetry endpoint。
4. 服务端确认成功接收后，本地删除这些聚合记录。
5. 如果 POST 成功但本地删除前 doclib 崩溃，后续可能重复上报同一批 metrics。
6. P0 接受少量重复。服务端应基于 `batch_id` 在短窗口内做幂等去重；超过窗口的重复数据由 cleaned metrics 层处理。

Flush lock 可以放在 telemetry 内部状态中，例如:

```toml
[telemetry]
flush_locked_at = "2026-06-10T12:00:00Z"
```

## 17. 服务端防污染

匿名 telemetry endpoint 无法完全避免伪造数据。服务端必须把本地 telemetry 视为半可信数据，只用于趋势分析。

服务端必须实现:

- schema version 校验
- metric name 白名单
- context key/value 白名单
- dimension key/value 白名单
- `value` 非负数校验
- batch 大小限制
- period 时间范围限制
- `installation_id` 限流
- IP 限流
- 异常数据 quarantine
- raw metrics 与 cleaned metrics 分层

Dashboard 默认展示 cleaned metrics。mineru.net API 服务端日志属于 trusted usage，本地 anonymous telemetry 属于 anonymous product analytics，两类数据必须分开标记和分析。

## 18. P0 实现边界

P0 需要完成:

- doclib server telemetry DB 配置读取与首次选择状态管理。
- 匿名 `installation_id` 生成并持久化到 doclib DB。
- telemetry metrics 聚合存储到 doclib DB。
- 小时级 period 聚合与 metric upsert。
- flush 成功后删除 DB 中已确认上报的 telemetry 聚合数据。
- `mineru` CLI、Doclib SDK、MCP、Web、桌面 App 和直接 HTTP API 到 doclib server 的 `source` / `caller` 透传。
- P0 metric 白名单与 dimension 白名单。
- 周期 flush 与失败重试。
- `mineru telemetry status|enable|disable|preview|flush`。
- 服务端 endpoint schema。

P0 不需要完成:

- 每事件实时上报。
- 详细硬件画像。
- 账号级归因。
- GPU 型号、显存、驱动和 CUDA 版本采集。
- exception message / traceback 上报。
- telemetry 结果查询 API。
