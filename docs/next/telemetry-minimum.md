# Telemetry Minimum Design

状态: Draft
日期: 2026-06-18
范围: doclib 第一版 telemetry 最小设计和指标集
非目标: CLI 早失败漏斗、rules / cleanup / compaction、完整 watch event 统计、详细硬件画像、blocking wait API

本文档是第一版 telemetry 实现的准确定义。`telemetry.md` 作为背景和候选设计参考；当两者不一致时，以本文档为准。

## 1. 第一版目标

第一版 telemetry 只回答两个问题:

1. 用户和 Agent 在以什么方式、什么频率使用 MinerU。
2. 从结果、速度、等待时间和出错情况评估用户满意度。

第一版只统计进入 doclib 后的业务行为。暂不统计 `mineru` CLI 在进入 doclib 之前的失败，例如参数错误、文件不存在、server 未启动。这类入口漏斗后续可用 `cli.request.count` 单独补充。

## 2. 采集边界

只有 doclib server 记录、聚合和上报 telemetry。

`mineru` CLI 和 `DoclibClient` 可以向 doclib server 传递调用上下文或 observation，但不直接向外部 telemetry endpoint 上报。

以下纯工具不具备 telemetry 能力:

- parser SDK，包括 `mineru.parser.parse()` 和 `parse_async()`。
- `mineru-kit parse`。
- `mineru-kit api-server`。
- 其他不依赖 doclib server 的专家工具。

这个边界避免 parser/tool 层在作为库或底层工具使用时包含 telemetry 配置、存储或网络行为。

### 实现模块边界

第一版 telemetry 作为 doclib 内部独立模块实现:

```text
mineru/doclib/telemetry/
  __init__.py
  constants.py
  context.py
  caller.py
  buckets.py
  store.py
  payload.py
  observation.py
  service.py
```

职责:

- `constants.py`: metric name、dimension key/value 白名单、metric allowed dimensions 内存表、consent 枚举和 telemetry endpoint 常量。
- `context.py`: `contextvars` 上下文、HTTP header 解析、request context 设置。
- `caller.py`: 当前进程 caller 推断，只输出 `agent | user | unknown`。
- `buckets.py`: duration、pages、file size、results 分桶。
- `store.py`: telemetry state 和 aggregate 的 DB 读写。
- `payload.py`: preview 和 flush payload 组装。
- `observation.py`: `/observations` request 校验和转换。
- `service.py`: telemetry 对外主入口，提供 record、preview、flush、enable、disable 能力。

`AppState` 持有:

```text
state.telemetry_store
state.telemetry_svc
state.telemetry_flush_worker
```

startup 初始化顺序:

```text
DatabaseManager.initialize()
  -> TelemetryStore(db)
  -> TelemetryService(store, context collector)
  -> ParseService(..., telemetry_svc=state.telemetry_svc)
  -> ScanService(..., telemetry_svc=state.telemetry_svc)
  -> SearchService(..., telemetry_svc=state.telemetry_svc)
  -> background workers
  -> TelemetryFlushWorker(...)
```

传递规则:

- `ParseService`、`ScanService`、`SearchService` 构造函数显式接收 `telemetry_svc`。
- `DoclibServer` route 通过 `self.state.telemetry_svc` 记录 route-level metric。
- background worker 不创建 telemetry service，只使用已注入到 service 的 telemetry。
- telemetry service 所有 record 方法内部吞掉异常，最多记录 debug / warning log，不影响主流程。
- 不使用全局单例或动态从模块级变量读取 telemetry service。

`TelemetryFlushWorker` 放在 doclib background 模块:

```text
mineru/doclib/background/telemetry_flush.py
```

形态:

```python
class TelemetryFlushWorker:
    def __init__(
        self,
        telemetry_svc: TelemetryService,
        *,
        interval_sec: int = 7200,
    ) -> None: ...

    async def run(self) -> None: ...
    async def stop(self) -> None: ...
```

worker 规则:

- 启动后先尝试一次 flush，但只有 `consent_state=enabled` 才真正外发。
- 之后每 2 小时尝试一次 flush。
- shutdown 时调用 `stop()`。
- 手动 flush 和后台 flush 复用同一个 `TelemetryService.flush()`。
- 一个 HTTP 请求只发送一个 period。
- 一次 flush 最多发送 10 个 period，即最多 10 个 HTTP 请求。
- `preview` 只展示最早一个将发送的 period batch。

## 3. 用户选择与开关

首次启动 doclib server 时，如果用户尚未选择，允许先进入 `unset` 状态。

状态:

```text
consent_state = unset | enabled | disabled
```

语义:

- `unset`: 用户尚未完成首次选择。doclib server 可以本地记录 telemetry 聚合数据，但不得 flush 到外部 endpoint。
- `enabled`: 用户允许记录并上报 telemetry。
- `disabled`: 用户明确关闭 telemetry。

默认策略:

- 首次选择界面不预选开启或关闭。
- 用户可以关闭。
- 用户选择结果记录为 `consent_state`。
- `consent_state=unset` 时，可以写入本地 telemetry 聚合数据，但不能上报。
- 如果用户后续选择 `enabled`，此前 `unset` 状态下记录的本地聚合数据可以随之后数据一起 flush。
- `consent_state=disabled` 时，不写入新的 telemetry 聚合数据，也不上报。
- 用户选择或切换到 `disabled` 时，DB 中未上报 telemetry 数据应删除，避免后续误发。

CLI 被调用时，应 best-effort 判断是否处于交互式环境。可以使用 stdin / stdout 是否为 TTY 作为基础判断，并避免在 CI、Agent 或明确非交互环境中提示。若处于交互式普通用户环境且 `consent_state=unset`，第一版 CLI 应提示用户完成选择。若开始使用后一直是非交互或 Agent 场景，则默认维持 `unset`，直到有机会让用户选择。

CLI `unset` prompt 规则:

- prompt 只在 CLI 层实现，不放入 `DoclibClient`，避免 SDK 用户被提示。
- `mineru telemetry status|enable|disable|preview|flush` 命令自身不触发 prompt。
- `mineru server start` 第一版暂不触发 prompt；后续业务命令会提示。
- prompt 触发条件必须同时满足:
  - `consent_state == "unset"`
  - stdin 和 stdout 都是 TTY
  - 当前不是 `--help` 帮助模式
  - 当前不是 `--json` 输出模式
  - 不在 CI 环境
  - caller 推断不是 `agent`
  - doclib server 已可访问
- 不满足条件时不提示，保持 `unset`。
- 用户输入 Enter / `y` / `yes` 时，调用 `POST /telemetry/actions/enable`。
- 用户输入 `n` / `no` 时，调用 `POST /telemetry/actions/disable`。
- EOF 或其它非法输入保持 `unset`，并继续原命令。非法输入第一版不循环追问。
- Ctrl-C 保持 `unset`，并按 CLI 原取消逻辑退出。
- 只要用户选择过，server 里的 `consent_state` 变成 `enabled` 或 `disabled`，后续不再提示。

推荐提示文案:

```text
Help improve MinerU by sending anonymous, locally aggregated usage and diagnostic data.

Collected: command names, MinerU version, OS, architecture, Python version, install channel, coarse CPU/GPU categories, success/failure status, error categories, tiers, and performance timing buckets.
Not collected: document contents, extracted text/images, file names, file paths, raw URLs, search queries, prompts, snippets, tracebacks, exception messages, hostnames, usernames, account IDs, API keys, or exact CPU/GPU models.

Press Enter or type Y to enable, or type N to disable.
You can change this later with `mineru telemetry enable` or `mineru telemetry disable`.
Preview what would be sent with `mineru telemetry preview`.

Enable telemetry? [Y/n]:
```

用户必须可以通过命令查看和修改状态:

```bash
mineru telemetry status
mineru telemetry enable
mineru telemetry disable
mineru telemetry preview
mineru telemetry flush
```

`preview` 展示本地待 flush 的 telemetry request body。它等价于一次外部 telemetry HTTP 请求的 body 预览，但不触发 flush。
`installation_id` 不是敏感隐私字段，`preview` 可以展示完整 `installation_id`。

doclib server 提供最小 telemetry 管理 API:

```text
GET /telemetry/status
GET /telemetry/preview
POST /telemetry/actions/{action}
```

`action` 取值:

```text
enable | disable | flush
```

`enable` 将 `consent_state` 置为 `enabled`。`disable` 将 `consent_state` 置为 `disabled`，并清除本地未上报聚合数据。`flush` 触发一次手动 flush；如果当前不是 `enabled`，应返回 accepted 但不执行外部上报。

管理 API response:

```python
TelemetryConsentState = Literal["unset", "enabled", "disabled"]
TelemetryAction = Literal["enable", "disable", "flush"]

class TelemetryStatusResponse(DoclibModel):
    consent_state: TelemetryConsentState
    installation_id: str
    pending_aggregate_count: int
    pending_period_count: int
    last_flush_at: int | None = None
    flush_locked_at: int | None = None


class TelemetryFlushResult(DoclibModel):
    accepted: bool
    executed: bool
    reason: str | None = None
    sent_batch_count: int = 0
    sent_metric_count: int = 0


class TelemetryActionResponse(DoclibModel):
    action: TelemetryAction
    accepted: bool
    executed: bool
    consent_state: TelemetryConsentState
    reason: str | None = None
    flush_result: TelemetryFlushResult | None = None
```

API 语义:

- `GET /telemetry/status` 返回 `TelemetryStatusResponse`。
- `pending_aggregate_count` 表示当前 `telemetry_aggregates` 中待上报 aggregate row 的数量，不是 `metric_value` 总和。
- `pending_period_count` 表示当前待上报的 distinct period 数量，即 distinct `(period_start, period_end)` 数量。
- `last_flush_at` 表示最近一次至少有一个 batch 被 telemetry endpoint 2xx 确认接收的完成时间；此前从未成功 flush 时返回 `null`。
- `flush_locked_at` 表示当前 live flush lock 的获取时间；没有 live lock 时返回 `null`。
- `GET /telemetry/preview` 直接返回外部 telemetry request body，不额外包 metadata。
- `preview` 不触发 flush，不修改 `last_flush_at`，可以展示完整 `installation_id`。
- 如果本地没有待上报 metrics，`preview` 返回合法空 batch，`metrics=[]`。这只表示当前没有待上传 metrics，`flush` 不会上传空 batch。
- 空 preview payload 只用于 CLI 展示，不是 external endpoint 会收到的 payload。
- 如果本地有多个 period，第一版 `preview` 只返回一次实际 flush 会发送的最早一个 period 的 batch。
- `preview` 的 `batch_id` 固定为 `tb_preview`，不写入 DB。
- 空 batch preview 的 `period_start` / `period_end` 使用当前 UTC 小时 period。
- `POST /telemetry/actions/enable` 设置 `consent_state=enabled`，返回 `accepted=true`、`executed=true`。
- `POST /telemetry/actions/disable` 设置 `consent_state=disabled`，删除本地未上报聚合数据，返回 `accepted=true`、`executed=true`。
- `POST /telemetry/actions/flush` 在 `enabled` 时同步触发一次 flush，并返回 `TelemetryActionResponse`，其中 `flush_result` 填充 `TelemetryFlushResult`。
- `flush` 在 `unset` 或 `disabled` 时返回 HTTP 200，`accepted=true`、`executed=false`、`reason="telemetry_not_enabled"`。
- 未知 action 返回 HTTP 4xx。
- `TelemetryActionResponse.executed` 表示 action 已被执行，不表示 flush 成功上传；flush 结果以 `flush_result.reason` 为准。

`TelemetryService.flush()` 返回 `TelemetryFlushResult`。语义:

- `unset` / `disabled`: `accepted=true`、`executed=false`、`reason="telemetry_not_enabled"`。
- 没有 pending aggregate: `accepted=true`、`executed=false`、`reason="no_pending_metrics"`。
- flush lock 未过期: `accepted=true`、`executed=false`、`reason="flush_locked"`。
- 网络失败: 保留 rows，`executed=false`、`reason="network_error"`。
- telemetry endpoint 5xx: 保留 rows，`executed=false`、`reason="server_error"`。
- telemetry endpoint 4xx: 删除对应 batch rows，不更新 `last_flush_at`，记录 warning log，`executed=true`、`reason="invalid_payload_discarded"`。
- 如果同一次 flush 中同时出现至少一个 2xx 成功 batch，且也出现至少一个 network / 5xx / 4xx discard batch，则返回 `accepted=true`、`executed=true`、`reason="partial_success"`。
- 至少一个 batch 被 telemetry endpoint 2xx 确认接收时，删除已确认 rows，更新 `last_flush_at`。
- 全部 batch 都失败时，不更新 `last_flush_at`。
- 4xx discard 不算成功，不触发 `last_flush_at` 更新。
- `sent_batch_count` 表示被 telemetry endpoint 2xx 确认接收的 batch 数。
- `sent_metric_count` 表示被 telemetry endpoint 2xx 确认接收的 aggregate row 数总和，不是尝试发送的 row 数，也不是 `metric_value` 总和。

## 4. 隐私边界

Telemetry 禁止采集:

- 文档内容
- 文件名
- 文件路径
- 原始 URL
- search query
- prompt
- sha256
- parse_id / scan_id / job_id
- traceback
- exception message
- hostname
- 用户名
- 邮箱
- 账号 ID
- CPU/GPU 具体型号
- GPU 显存、驱动版本、CUDA 版本

允许采集的数据必须来自白名单字段。不得把请求体、异常对象、配置对象、`kwargs` 或任意 dict 直接透传到 telemetry。

`/observations` 可以接收 `parse_ids` 作为本地查询引用，但这些 ID 只允许用于 doclib server 本地补齐低基数字段，不得写入 telemetry 聚合或外部上报 payload。

## 5. 本地聚合与上报

第一版使用匿名 `installation_id` 作为安装实例口径。

- `installation_id` 在 doclib server 第一次启动时自动生成。
- `installation_id` 存储在 doclib DB 的 telemetry 内部状态中。
- `installation_id` 不是敏感隐私字段。
- 不依赖账号体系。
- 即使用户尚未选择 telemetry，或当前没有上报需求，也可以生成 `installation_id`。
- 用户选择或切换到 `disabled` 时，不删除 `installation_id`，避免后续重新启用后安装实例 ID 改变。
- 用户删除配置或重装后，会被视为新的安装实例。
- 第一版 `installation_id` 格式定义为 `inst_<32 lowercase hex>`，例如 `inst_7f6e0c6a0d174dc6a3f1b6f6a36d7f73`。

第一版采用本地聚合 metrics 上报，不采用每个事件即时上报。

运行时行为:

```text
doclib server runtime
  -> record metric delta
  -> update telemetry aggregate in doclib DB
  -> periodic flush
  -> POST telemetry endpoint
  -> delete flushed rows from doclib DB after success
```

聚合和 flush 策略:

- 本地 telemetry 配置和 telemetry 聚合数据都存储在 doclib DB 中。
- 本地按 1 小时 period 聚合，period 边界使用 UTC 整点。
- period 采用左闭右开区间，即 `[period_start, period_end)`。
- 一次 flush 可以处理多个未 flush 的小时级 period，但每个 period 独立生成一个 HTTP request。
- flush 发送顺序按 `period_start ASC, period_end ASC`。
- `consent_state=enabled` 时，启动后尝试 flush 一次。
- `consent_state=enabled` 时，运行中每 2 小时尝试 flush 一次。
- `consent_state=unset` 或 `disabled` 时，不进行外部 flush。
- 网络失败或服务端 5xx 时保留本地聚合结果，后续重试。
- 服务端 4xx 表示 payload 不合法，对应 batch rows 应删除，不更新 `last_flush_at`，记录 warning log，不应无限重试。
- flush 成功后，已被服务端确认接收的 telemetry 聚合数据应从 doclib DB 中删除。
- 同一个 period 内 context 变化可以接受；flush 时使用当前 context 上报该 period 的聚合 metrics。
- 一个 HTTP 请求只发送一个 period。
- 一次 flush 最多发送 10 个 period。

主流程不得等待 telemetry 上报。telemetry 失败不得影响 parse、watch、search 或 doclib API。

Telemetry endpoint 使用代码内置值，用户不可配置:

```http
POST https://telemetry.mineru.net/v1/metrics
Content-Type: application/json
```

开发和测试环境如需覆盖 endpoint，应使用测试构建或专门的开发开关，不进入用户配置体系。

flush 使用 `httpx.AsyncClient`，请求 timeout 固定为 10 秒。测试环境可以通过内部测试 hook 或构造参数替换 endpoint；该能力不进入用户配置。

现阶段不用考虑 migration，可以直接修改 `mineru/doclib/migrations/001_init.sql`。新增表应使用 `CREATE TABLE IF NOT EXISTS`，使已有 DB 在启动时也能补建 telemetry 表。

Telemetry state 不使用现有 `config` 表。`config` 表语义是用户可配 runtime config override；`installation_id`、`last_flush_at`、`flush_locked_at` 不是用户配置。

本地状态使用独立 KV 表表达:

```sql
CREATE TABLE IF NOT EXISTS telemetry_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);
```

`telemetry_state` keys:

| key | value |
|-----|-------|
| `consent_state` | `unset | enabled | disabled` |
| `installation_id` | `inst_...` |
| `last_flush_at` | epoch ms timestamp |
| `flush_locked_at` | epoch ms timestamp |

初始化规则:

- server startup 时，如果没有 `installation_id`，生成并写入。
- server startup 时，如果没有 `consent_state`，写入 `unset`。
- server startup 时，如果没有 `last_flush_at` 或 `flush_locked_at`，写入字符串 `"0"`，表示当前为空值。
- server startup 时固定初始化所有 telemetry state keys，包括 `installation_id`、`consent_state`、`last_flush_at`、`flush_locked_at`，避免后续 flush lock 或 status 查询处理 key 缺失。
- 即使用户选择或切换到 `disabled`，也不删除 `installation_id`。
- 用户选择或切换到 `disabled` 时，删除 `telemetry_aggregates` 中全部未上报聚合数据。

Telemetry 聚合数据使用独立表保存:

```sql
CREATE TABLE IF NOT EXISTS telemetry_aggregates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start    INTEGER NOT NULL,
    period_end      INTEGER NOT NULL,
    metric_name     TEXT    NOT NULL,
    metric_value    INTEGER NOT NULL DEFAULT 0,
    dimensions      TEXT    NOT NULL,
    dimensions_hash TEXT    NOT NULL,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    UNIQUE(period_start, period_end, metric_name, dimensions_hash)
);

CREATE INDEX IF NOT EXISTS idx_telemetry_aggregates_period
ON telemetry_aggregates(period_start, period_end);
```

字段说明:

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

`metric_value` 使用 `INTEGER`。第一版所有 metric 都是 count、bucket count、files/pages 总数或其它非负整数值。以后如果需要 ratio / float metric，再单独扩展存储模型。

时间格式:

- DB 中时间使用 epoch ms。
- 外部 payload 中的 `period_start` / `period_end` 使用 UTC ISO 8601，例如 `2026-06-22T10:00:00Z`。
- preview payload 也使用同样的 UTC ISO 8601 时间格式。

聚合唯一键:

```text
period_start + period_end + metric_name + dimensions_hash
```

同一唯一键重复写入时，累加 `metric_value`。聚合 key 不包含 context；flush 时读取当前 context 并组装 telemetry batch。

`dimensions_hash` 生成规则:

1. 对 dimensions 先做 normalize 和 whitelist 过滤。
2. 使用 `json.dumps(dimensions, sort_keys=True, separators=(",", ":"))` 生成稳定 JSON 字符串。
3. 对该字符串计算 SHA-256 hex。

Flush 并发与幂等采用第一版简单方案:

1. doclib 同一时间只允许一个 flush 任务执行。
2. flush 开始时获取本地 flush lock；已有 lock 且未过期时，本次 flush 直接跳过。
3. flush 从 DB 读取最多 10 个未上报 period，每个 period 生成一个 `batch_id` 并 POST 一次 telemetry endpoint。
4. 服务端确认成功接收后，本地删除这些聚合记录。
5. 如果 POST 成功但本地删除前 doclib 崩溃，后续可能重复上报同一批 metrics。
6. 第一版接受少量重复。服务端应基于 `batch_id` 在短窗口内做幂等去重；超过窗口的重复数据由 cleaned metrics 层处理。

flush lock TTL:

```text
flush_lock_ttl_sec = 1800
```

即 30 分钟。超过该时间的 lock 视为过期。

lock 存储规则:

- 获取 flush lock 时，把 `flush_locked_at` 写为当前 epoch ms。
- flush 结束时，无论成功、失败还是提前跳过，只要当前调用持有 lock，就把 `flush_locked_at` 重置为 `"0"`。
- `TelemetryStatusResponse.flush_locked_at` 中，`"0"` 映射为 `null`。

## 6. Payload 与 Context

一次上报包含一个 batch。环境信息放在 batch-level `context` 中，不重复放入每个 metric。

```json
{
  "batch_id": "tb_01HX...",
  "schema_version": 1,
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
      "name": "parse.finished.count",
      "value": 12,
      "dimensions": {
        "source": "cli",
        "caller": "agent",
        "tier": "medium(default)",
        "status": "queued"
      }
    }
  ]
}
```

规则:

- `batch_id` 由本地 flush 任务生成，用于服务端短窗口幂等。第一版格式定义为 `tb_<32 lowercase hex>`。
- `schema_version` 使用整数，第一版固定为 `1`。
- `value` 必须是非负数。
- 字符串数据只能出现在 `context` 或 `dimensions`。
- `context` 描述安装实例、运行环境和版本。
- `metrics[].dimensions` 描述当前 metric 的业务分类。
- `metrics[].dimensions` 必须来自白名单，不能包含任意字符串。
- `context` 表示 flush 时 doclib server 的当前运行环境。同一 period 内 context 变化可以接受，不要求拆分 batch。
- `metrics` 列表按稳定顺序输出，第一版使用 `name ASC`，再按 normalize 后的 `dimensions` JSON 字符串升序。

第一版 context 字段:

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
- `unknown`: 无法检测、检测失败，或发现了第一版未分类的其他加速器。

context 采集规则:

- `os`: 使用 `platform.system()` 归一化为 `macos | windows | linux | other | unknown`。
- `arch`: 使用 `platform.machine()` 归一化为 `x86_64 | arm64 | other | unknown`；`amd64` 归一化为 `x86_64`，`aarch64` 归一化为 `arm64`。
- `python_version`: 使用 `sys.version_info.major.minor`；不在白名单时记为 `other`。
- `cpu_count_bucket`: 使用 `os.cpu_count()`；`None` 或小于等于 0 时记为 `unknown`，否则按 `1_4 | 5_8 | 9_16 | gt_16` 分桶。
- `install_channel` 使用以下优先级:
  1. 若检测到当前运行在 Docker / 容器环境中，记为 `docker`。
  2. 否则若 `importlib.metadata.distribution("mineru")` 的 `direct_url.json` 表示本地目录安装，或 `dir_info.editable=true`，记为 `source`。
  3. 否则若当前 `mineru` 模块文件路径不在 `site-packages` / `dist-packages` 下，且向上能找到项目级 `pyproject.toml`，记为 `source`。
  4. 否则尝试读取 `importlib.metadata.distribution("mineru")` 的 `INSTALLER` 元数据；值为 `uv` 时记为 `uv`，值为 `pip` 时记为 `pip`。
  5. 其它情况记为 `unknown`。
- Docker / 容器检测为 best-effort，第一版允许使用 `/.dockerenv`、`/proc/1/cgroup`、`/proc/self/cgroup` 等本地信号；判断失败时降级，不向业务层抛错。
- `gpu_vendor` 使用保守检测:
  1. `platform.system()=="Darwin"` 且 `arch=="arm64"` 时记为 `apple`。
  2. Linux 上若 DRM / sysfs vendor 可明确识别为 `0x10de`，或存在 `nvidia-smi` / `/dev/nvidiactl`，记为 `nvidia`。
  3. Linux 上若 DRM / sysfs vendor 可明确识别为 `0x1002`，记为 `amd`。
  4. Linux 上若未发现任何 GPU / render device 信号，记为 `none`。
  5. Windows 上使用 `winreg` best-effort 读取 display adapter 信息；能明确识别 `NVIDIA` / `VEN_10DE` 时记为 `nvidia`，能明确识别 `AMD` / `Radeon` / `VEN_1002` 时记为 `amd`，其它情况记为 `unknown`。第一版不在 Windows 上上报 `none`。
  6. 其它无法可靠判断的情况记为 `unknown`。
- 第一版不为了 telemetry 主动 import `torch`、`paddle`、`transformers` 等重依赖做硬件探测。

第一版不采集 CPU/GPU family、具体型号、显存、驱动和 CUDA 版本。

外部 telemetry endpoint 必须实现 schema version 校验、metric name 白名单、context key/value 白名单、dimension key/value 白名单、`value` 非负数校验、batch 大小限制、period 时间范围限制，以及安装实例和 IP 级限流。

## 7. 命名原则

流程结构写入 metric name，分类结果写入 dimensions。

示例:

```text
parse.request.count
parse.finished.count
parse.wait_duration_bucket.count
parse_task.execute.count
```

不用 `phase` 或 `stage` 维度区分入口和步骤。入口请求使用 `parse.*`，后台异步任务使用 `parse_task.*`。

Metric value 口径:

- `*.request.count`、`*.finished.count`、`*.started.count`、`*.created.count` 等事件型 metric 的 `value` 表示发生次数。
- `*.duration_bucket.count`、`*.file_size_bucket.count`、`*.pages_bucket.count`、`*.results_bucket.count` 的 `value` 表示落入该 bucket 的样本数。
- `*.files.count` 的 `value` 表示文件数量总和，不表示请求或任务次数。
- `*.pages.count` 的 `value` 表示页数总和，不表示请求或任务次数。
- `scan.files.count` 的 `value` 表示对应 `result` 下的文件数量总和。

Duration bucket 口径:

- 所有 `*.duration_bucket.count` 应尽量携带与对应结果 metric 相同的 `status`。
- `parse.duration_bucket.count` 使用同一次 `parse.finished.count` 的 `status`，即 `cached | direct | reused | queued | failed`。
- `parse.wait_duration_bucket.count` 使用同一次 `parse.wait.count` 的 `status`，即 `succeeded | failed | timeout | canceled`。
- `parse_task.duration_bucket.count` 使用同一次 `parse_task.finished.count` 的 `status`。
- `parse_task.execute_duration_bucket.count` 使用同一次 `parse_task.execute.count` 的 `status`。
- `parse_task.write_duration_bucket.count` 使用同一次 `parse_task.write.count` 的 `status`。
- `ingest.duration_bucket.count`、`search.duration_bucket.count`、`find.duration_bucket.count`、`content.duration_bucket.count`、`scan.duration_bucket.count` 使用对应 `*.finished.count` 的 `status`。

## 8. Context 传递

`source` 和 `caller` 不写入每个业务 request model。

CLI 到 `DoclibClient` 是同进程调用，使用 `contextvars` 传递进程内 telemetry context，不使用环境变量。

```text
source=cli
caller=agent | user | unknown
```

进程内 context API:

```python
@dataclass(frozen=True)
class TelemetryContext:
    source: TelemetrySource
    caller: TelemetryCaller

def set_telemetry_context(source: TelemetrySource, caller: TelemetryCaller) -> Token: ...
def reset_telemetry_context(token: Token) -> None: ...
def get_telemetry_context() -> TelemetryContext | None: ...
def infer_default_client_context() -> TelemetryContext: ...
```

CLI 规则:

- CLI 入口设置 telemetry context。
- `DoclibClient` 在发请求前读取 context。
- `caller` 由 helper best-effort 检测当前进程父进程树。
- 已知 Agent 进程记为 `agent`，普通 shell / terminal 记为 `user`，无法可靠判断时记为 `unknown`。
- helper 只输出枚举值，不记录进程名原文、命令行、路径或 prompt。

`DoclibClient` 发送 header 规则:

- 如果 `contextvars` 中已有 context，使用该 context。
- 如果没有 context，默认 `source=sdk`。
- 如果没有 context，`caller` 仍由当前进程调用树 best-effort 推断。
- caller 推断失败时，`caller=unknown`。

`DoclibClient` 到 doclib server 使用 HTTP header 传递:

```text
X-MinerU-Source: cli | sdk | http_api | watch | background | unknown
X-MinerU-Caller: agent | user | sdk | http_client | system | unknown
```

server middleware 解析 header，写入 request context。route、service 和 telemetry emitter 从 request context 读取。request 结束时必须 reset contextvars token，避免上下文泄漏到后续请求。

默认规则:

- CLI 设置进程内 context 后，`DoclibClient` 发送 `source=cli`。
- SDK 直接创建 `DoclibClient` 时，默认 `source=sdk`，`caller` 仍由进程上下文 best-effort 推断。
- 外部 HTTP 请求缺少 header 时，server 默认 `source=http_api`、`caller=http_client`。
- HTTP header 值不在白名单时降级: invalid source -> `unknown`，invalid caller -> `unknown`。
- watch / background 等 server 内部任务不走 header，直接使用内部固定来源，例如 `source=watch`、`caller=system`。

caller 推断规则:

- 遍历父进程树，只读取进程名，不读取完整命令行、路径、参数或环境变量。
- 进程名统一取 basename 后转小写，再与白名单比较。
- 遍历最大深度固定为 12；遇到 `pid <= 1`、父 pid 重复、读取失败或达到最大深度时停止。
- 命中白名单时按优先级返回: `agent` 高于 `user`。即只要 12 层内出现已知 Agent 进程名，就返回 `agent`；否则若出现 shell / terminal 进程名，则返回 `user`；否则返回 `unknown`。
- 命中常见 Agent 进程名时返回 `agent`，例如 `codex`、`claude`、`cursor`、`windsurf`、`aider`、`gemini`、`qwen`、`cline`。
- 命中常见 shell / terminal 进程名时返回 `user`，例如 `bash`、`zsh`、`fish`、`Terminal`、`iTerm2`、`WindowsTerminal`。
- 权限不足、平台不支持、进程树读取失败或无法判断时返回 `unknown`。
- 进程名只用于本地枚举判断，不写入 telemetry 聚合或外部 payload。

第一版不做后台 task 的 `source/caller` 归因。`parse_task.*`、`scan.finished.count`、`scan.files.count`、`ingest.*` 等后台执行指标不要求带 `source` / `caller`。用户或 Agent 来源通过入口请求指标和 `/observations` 观察。

## 9. 公共 Dimensions

所有用户行为类 metric 应尽量带:

```text
source
caller
status
```

取值:

```text
source = cli | sdk | http_api | watch | background | unknown
caller = agent | user | sdk | http_client | system | unknown
status = <metric-specific low-cardinality status>
```

按需补充:

```text
tier = default | flash | medium | high | medium(default) | high(default) | unknown
server = local(flash) | local(managed) | local(self-hosted) | remote(official) | remote(custom) | none | unknown
error_code = <telemetry_error_code>
bucket = <metric bucket value>
result = <metric-specific low-cardinality result>
content_mode = read | parse_output | export | unknown
output_format = markdown | image | other
trigger = parse | scan | watch | show | background | unknown
```

不得把路径、文件名、sha256、query、异常消息、traceback、URL、API Key、prompt 或文档内容放入 dimensions。

`status` 表示当前 metric 的业务状态。不同 metric 可以有不同枚举，但必须在 metric 定义中写清楚，且保持低基数。

`source` / `caller` 适用范围:

- 入口请求和调用方观察类指标应带 `source` / `caller`，例如 `parse.request.count`、`parse.finished.count`、`search.*`、`find.*`、`content.*`、`scan.request.count`、`watch.add.count`、`watch.remove.count`、`parse.wait.*`。
- 后台执行类指标不要求带 `source` / `caller`，例如 `parse_task.*`、`scan.finished.count`、`scan.duration_bucket.count`、`scan.files.count`、`ingest.*`。

`error_code` 规则:

- 第一版 `error_code` 只用于失败类结果 count metric，不用于 request metric、value metric、bucket metric，也不用于任何 duration bucket。
- 第一版会带 `error_code` 的 metric 只有:
  - `parse.finished.count`
  - `parse_task.finished.count`
  - `parse_task.execute.count`
  - `parse_task.write.count`
- 成功状态不带 `error_code`，不上传 `none`。
- metric 中的 `error_code` 必须来自第一版 whitelist:

```text
internal_error

invalid_request
file_not_found
file_permission_denied
not_cached
no_accessible_file

quality_tier_unavailable
no_engine
engine_unavailable
parse_server_unavailable
tier_mismatch
parse_failed
parse_timeout

metadata_failed
parse_json_write_failed
ingest_failed
scan_failed
```

- 不在 whitelist 的稳定错误码，第一版统一降级为 `internal_error`。

`tier` 归一化规则:

- `flash` / `medium` / `high` 表示用户显式指定，或后台任务的实体 tier。
- `medium(default)` / `high(default)` 只用于入口请求层指标，表示请求未显式指定 tier，但默认选择策略最终解析到了 `medium` 或 `high`。
- `default` 只用于入口请求在默认 tier 解析完成前就失败，且调用方未显式指定 tier 的场景。
- `unknown` 表示无法安全判断。

`server` 归一化规则:

- `none`: 当前步骤不涉及 parse-server，例如 `flash` 本地解析。
- `local(managed)`: 命中 doclib 自己拉起并管理的 local parse-server，即 `parse_server.local.mode=managed`。
- `local(self-hosted)`: 命中用户自管的 local parse-server，即 `parse_server.local.mode=self_hosted`。
- `remote(official)`: 目标 URL 命中官方托管的 remote parse-server，包括 production 和 staging 等官方环境。
- `remote(custom)`: 使用 remote parse-server，但目标 URL 不属于官方托管环境。
- `local(flash)` 作为保留枚举，第一版如果 `parse_task.execute.*` 不经过 parse-server，可统一使用 `server=none`。
- 无法安全判断时记为 `unknown`。

### CLI / Client 改动

新增 CLI 文件:

```text
mineru/cli/commands/telemetry.py
```

更新:

```text
mineru/cli/commands/__init__.py
mineru/cli/main.py
```

`mineru` 顶层命令新增:

```bash
mineru telemetry status
mineru telemetry enable
mineru telemetry disable
mineru telemetry preview
mineru telemetry flush
```

`TOP_LEVEL_COMMAND_ORDER` 中放在 `config` 后:

```text
server
config
telemetry
```

输出规则:

- `status`: human 输出 consent、installation_id、pending aggregates、last_flush_at；支持 `--json`。
- `preview`: 默认 pretty JSON 输出 telemetry request body。
- `enable` / `disable` / `flush`: 输出 action result；支持 `--json`。
- server 未启动时沿用现有 CLI error 风格。

`DoclibClient` 新增方法:

```python
def get_telemetry_status(self) -> TelemetryStatusResponse: ...
def get_telemetry_preview(self) -> dict[str, Any]: ...
def telemetry_action(self, action: TelemetryAction) -> TelemetryActionResponse: ...
def submit_observation(self, request: ObservationRequest) -> ObservationResponse: ...
```

对应 route:

```text
GET /telemetry/status
GET /telemetry/preview
POST /telemetry/actions/{action}
POST /observations
```

`DoclibInterface` 和 `AsyncDoclibInterface` 也需要补对应抽象方法，保持当前接口模式。

CLI context:

- `mineru/cli/main.py` 增加 root callback。
- callback 设置 `source=cli`，`caller=infer_caller_from_process_tree()`。
- 不使用环境变量。
- 第一版实现 CLI 交互式且 `consent_state=unset` 时提示用户选择。
- CLI prompt 只在 stdin / stdout 是 TTY，且不在 CI 或明确非交互环境中触发。
- 非交互或 Agent 场景下不提示，维持 `consent_state=unset`。
- 用户也可以随时通过显式 `mineru telemetry ...` 命令查看和修改状态。

CLI prompt helper:

```python
def maybe_prompt_telemetry_consent(
    client: DoclibClient,
    *,
    json_mode: bool,
    command_name: str,
) -> None: ...
```

实现位置:

```text
mineru/cli/telemetry.py
```

职责:

- `is_interactive_cli()` 判断 stdin / stdout 是否为 TTY。
- `is_ci_environment()` 判断常见 CI 环境变量，例如 `CI`、`GITHUB_ACTIONS`、`GITLAB_CI`、`BUILDKITE`、`JENKINS_URL`。
- `maybe_prompt_telemetry_consent()` 查询 telemetry status，按规则提示用户，并调用 telemetry action。

调用时机:

- 各业务 CLI 命令创建 `DoclibClient` 后、发业务请求前调用。
- `mineru telemetry ...` 命令自身不调用。
- prompt 失败、status 查询失败或 action 调用失败时，不影响原业务命令继续执行。

`parse --wait` observation:

- `parse_cmd` 开始等待前记录 `wait_start = time.monotonic()`。
- 等到 done: 上报 `parse_wait {status=succeeded, duration_ms, parse_ids}`。
- 等到 failed: 先上报 `status=failed`，再按原逻辑报错。
- 超时: 上报 `status=timeout`。
- 用户取消，例如 Ctrl-C: 尽量上报 `status=canceled`，然后按原 CLI 取消逻辑退出。
- 上报失败静默忽略，不影响 CLI 输出。

### TelemetryService 采集 API

第一版使用少量通用 record API，不为每个 metric 单独定义专用方法:

```python
class TelemetryService:
    async def record_count(
        self,
        name: str,
        *,
        dimensions: Mapping[str, str] | None = None,
        value: int = 1,
    ) -> None: ...

    async def record_value(
        self,
        name: str,
        value: int,
        *,
        dimensions: Mapping[str, str] | None = None,
    ) -> None: ...

    async def record_duration(
        self,
        name: str,
        duration_ms: int,
        *,
        dimensions: Mapping[str, str] | None = None,
    ) -> None: ...

    async def record_bucket(
        self,
        name: str,
        bucket: str,
        *,
        dimensions: Mapping[str, str] | None = None,
        value: int = 1,
    ) -> None: ...
```

语义:

- `record_count`: 事件发生次数，例如 `parse.request.count`。
- `record_value`: 数值总和，例如 `parse.pages.count`、`scan.files.count`。
- `record_duration`: 根据 `duration_ms` 自动写入 duration bucket，调用方不传 bucket。
- `record_bucket`: 非 duration bucket，例如 pages、file size、results bucket。
- 第一版所有 `value` 必须是非负整数。
- `dimensions` 在 telemetry 层统一 normalize 和 whitelist 校验。
- unknown / invalid dimension value 降级为 `unknown` 或 `none`，不让任意字符串进入聚合。
- `consent_state=disabled` 时，所有 record 方法直接 no-op。
- `consent_state=unset` 或 `enabled` 时，record 方法写本地 aggregate。
- 本地 aggregate 写入可以 `await`，第一版不引入内存 queue。
- 本地 aggregate 写失败不重试，不影响主流程。

Metric allowed dimensions 映射:

- metric -> allowed dimensions 映射是 telemetry schema 的一部分，使用代码内存表，不放入数据库。
- `METRIC_SPECS` 放在 `mineru/doclib/telemetry/constants.py`。
- DB 只存 normalize 后的 `dimensions` JSON 和 `dimensions_hash`。
- 本地 telemetry 和外部 telemetry endpoint 都应按 `schema_version` 维护各自白名单。

示例:

```python
@dataclass(frozen=True)
class MetricSpec:
    allowed_dimensions: frozenset[str]
    required_dimensions: frozenset[str] = frozenset()


METRIC_SPECS: dict[str, MetricSpec] = {
    "parse.request.count": MetricSpec(
        allowed_dimensions=frozenset({"source", "caller"}),
    ),
    "parse.finished.count": MetricSpec(
        allowed_dimensions=frozenset({"source", "caller", "status", "tier", "error_code"}),
    ),
    "parse.duration_bucket.count": MetricSpec(
        allowed_dimensions=frozenset({"source", "caller", "status", "tier", "bucket"}),
    ),
}
```

`required_dimensions` 第一版可用于测试和开发期校验；运行时缺失时按失败兜底规则降级或跳过，不向业务层抛错。

### 失败兜底规则

- telemetry 永远不能改变业务返回、异常类型或状态码。
- 所有 `record_*` 内部 catch `Exception`，不向外抛。
- 打点代码不得读取或传递 exception message、traceback、path、filename、query、sha256、parse_id。
- `error_code` 只允许使用稳定业务错误码，例如 `MineruError.code` 或 parse / scan row 的 `error_code`。不在 telemetry whitelist 内时统一降级为 `internal_error`。
- route 层失败时，可以用 `try/except Exception as exc` 计算 telemetry status / error_code，但必须重新 `raise` 原异常。
- duration 使用 `time.monotonic()`；DB timestamp 仍使用 ms epoch。
- 如果无法补齐必需维度，使用安全枚举值，例如 `status=failed` 或 `unknown`。
- 如果无法补齐非必需规模信息，例如 pages、file size，不记录对应规模 metric，不使用 `unknown` bucket。
- route-level request metric 在进入业务 handler 后立即记录；如果后续业务失败，request 仍保留，finished 记录 failed。
- FastAPI / Pydantic 在进入业务 handler 前拒绝的请求第一版不计。
- background task 的 source / caller 不做归因，不为此修改 `parses` 或 `scans` 表存来源快照。

## 10. Observations

有些体验信息只存在于 client / CLI / SDK 侧，server 自己无法观察，例如调用方等待了多久、是否超时、是否被取消。

第一版增加一个通用 observations 接口:

```text
POST /observations
```

request 必须使用白名单 type，不允许透传任意 dict。第一版支持:

```text
type = parse_wait
```

`parse_wait` 只记录低敏字段:

```json
{
  "type": "parse_wait",
  "duration_ms": 43120,
  "status": "succeeded",
  "refs": {
    "parse_ids": [123, 124]
  }
}
```

请求约束:

| 字段 | 类型 | 约束 |
|------|------|------|
| `type` | string | 第一版只允许 `parse_wait` |
| `duration_ms` | integer | `0 <= duration_ms <= 86400000` |
| `status` | string | `succeeded` / `failed` / `timeout` / `canceled` |
| `refs.parse_ids` | integer array | 非空，最多 200 个，元素必须为正整数 |

响应:

```json
{
  "accepted": true,
  "recorded": true,
  "reason": null
}
```

响应语义:

- `accepted=true`: server 接受了 observation 请求格式。
- `recorded=true`: observation 已写入 telemetry 聚合。
- `recorded=false`: 请求格式有效，但没有写入聚合，例如 telemetry disabled、`parse_ids` 全部查不到，或无法补齐必要低基数字段。
- `reason` 只在 `recorded=false` 时使用，第一版取值为 `telemetry_disabled | parse_refs_not_found | missing_dimensions`。
- schema 不合法时返回 HTTP 4xx，例如未知 `type`、字段类型错误、`duration_ms` 超限、`refs.parse_ids` 为空或过长。

server 收到 observation 后，根据 `parse_ids` 查询本地 DB 补齐 tier、页数、错误码等低基数字段，再写入 telemetry 聚合。

observation 写入失败不能影响主流程。`consent_state=disabled` 时，该接口应返回 `accepted=true, recorded=false, reason=telemetry_disabled`。`consent_state=unset` 时，可以写入本地聚合，但不会触发外部 flush。

第一版保持简单，不做 observation 幂等去重。client 上报失败可以丢失，重试导致的重复 observation 也可以接受。

第一版不增加 blocking wait API。`POST /parses` 继续保持非阻塞，CLI / client 通过轮询显示进度，并在等待结束后上报 `parse_wait` observation。

采集点位:

- `POST /observations` 只支持 `parse_wait`。
- schema 合法后，根据 `parse_ids` 查询 task，补齐 `tier` 和 `pages_bucket`。
- 记录 `parse.wait.count {source, caller, status, tier, pages_bucket}`。
- 记录 `parse.wait_duration_bucket.count {source, caller, status, tier, pages_bucket}`。
- `consent_state=disabled` 时返回 `accepted=true, recorded=false, reason=telemetry_disabled`。
- `parse_ids` 全部查不到时返回 `accepted=true, recorded=false, reason=parse_refs_not_found`。
- 无法补齐必要低基数字段时返回 `accepted=true, recorded=false, reason=missing_dimensions`。

## 11. 使用方式与频率

这些 metric 回答“谁在用、从哪里用、用什么能力、频率如何”。

```text
parse.request.count
scan.request.count
watch.add.count
watch.remove.count
search.request.count
find.request.count
content.request.count
```

口径:

- `parse.request.count`: doclib 收到一次 parse 业务请求。
- `scan.request.count`: doclib 收到一次 scan 请求，包括手动 scan、watch rescan、watch initial scan。
- `watch.add.count`: doclib 收到一次 watch add 配置请求。
- `watch.remove.count`: doclib 收到一次 watch remove 配置请求。
- `search.request.count`: doclib 收到一次全文搜索请求，不记录 query。
- `find.request.count`: doclib 收到一次文件名搜索请求，不记录 query。
- `content.request.count`: doclib 收到一次读取或导出内容请求，不记录内容。

除 `scan.request.count` 外，`*.request.count` 的计数边界是进入对应 doclib 业务 handler 之后。FastAPI / Pydantic 在进入业务 handler 前拒绝的 schema 错误、路由不存在、认证或连接层错误，第一版不计入 request metric。进入业务 handler 后发生的业务失败仍计入 request metric，并通过对应 `*.finished.count {status=failed}` 表达结果。`scan.request.count` 额外覆盖 internal watch-triggered scan 创建。

这些 metric 不带 `command` 维度。能力类型已经由 metric name 表达。

采集点位总则:

- request metric 默认由 `DoclibServer` route 层记录，不在 service 层重复记录；`scan.request.count` 是唯一例外，internal watch-triggered scan 也记。
- route 进入业务 handler 后立即记录对应 `*.request.count`。
- route 层负责入口体验指标，例如 finished、duration、results bucket、content mode。
- service / worker 层负责后台执行指标，例如 `parse_task.*`、`scan.finished.count`、`scan.files.count`、`ingest.*`。

## 12. 用户满意度: Parse 请求层

`parse.*` 表示调用方感知到的 parse 请求体验。一次请求可能 cache hit、复用已有任务、创建新任务、等待任务完成，或直接返回 queued。

```text
parse.finished.count
parse.duration_bucket.count

parse.wait.count
parse.wait_duration_bucket.count

parse.files.count
parse.pages.count
parse.file_size_bucket.count
parse.pages_bucket.count

parse.invalidate.count
```

口径:

- `parse.finished.count`: `POST /parses` 请求返回给调用方的状态。
- `parse.duration_bucket.count`: 从进入 doclib parse 请求到 `POST /parses` 返回的耗时分布。
- `parse.wait.count`: 调用方等待后台 parse task 的结果，由 `parse_wait` observation 采集。
- `parse.wait_duration_bucket.count`: 调用方等待后台 task 的真实耗时分布，由 `parse_wait` observation 采集。
- `parse.files.count`: 请求涉及的文件数。第一版通常为 1。
- `parse.pages.count`: 请求涉及的页数，只记录数量，不记录页码列表。
- `parse.file_size_bucket.count`: 请求文件大小分布。
- `parse.pages_bucket.count`: 请求页数分布。
- `parse.invalidate.count`: 用户主动让已完成 parse 结果失效。

推荐状态:

```text
parse.finished.count:
  status = cached | direct | reused | queued | failed

parse.wait.count:
  status = succeeded | failed | timeout | canceled

parse.invalidate.count:
  status = succeeded | failed | skipped
```

说明:

- `cached` 表示请求所需内容已在本地 cache 中，`POST /parses` 立即返回可读取结果。
- `direct` 表示请求无需创建后台 parse task，doclib 直接完成，例如 plain text 文件不需要异步解析。
- `reused` 表示请求复用了已有 pending / parsing task，没有创建新的 parse task。
- `queued` 表示请求创建了新的后台 parse task。
- 当前代码映射规则: `cache_hit=True` -> `cached`；plain text response -> `direct`；`reused_parse_ids` 非空且 `created_parse_ids` 为空 -> `reused`；`created_parse_ids` 非空 -> `queued`；`ParseResponse.status=failed` 或 route 抛出异常 -> `failed`。
- `parse.finished.count {status=queued}` 已表示本次请求需要后台 task，因此不再用 `parse.wait.count` 记录 `required` / `not_required`。
- `parse.wait.count {status=succeeded|failed|timeout|canceled}` 和 `parse.wait_duration_bucket.count` 只由 `parse_wait` observation 记录。
- `parse.wait.*` 只覆盖使用 CLI / `DoclibClient` wait helper 的调用方。直接 HTTP 调用方如果只调用 `POST /parses` 而不上报 observation，不会产生 wait 指标。
- `parse.wait.*` 不是所有 `parse.finished.count {status=queued}` 的完整子集统计，只表示客户端明确上报过 observation 的等待行为。
- 第一版 `parse.wait.*` 不带 `error_code`。
- `parse.wait_duration_bucket.count` 应带 `tier`、`pages_bucket`。这两个维度从 `parse_ids` 对应的 task 中读取。
- 同一次 wait observation 包含多个 `parse_ids` 时，`pages_bucket` 使用这些 task 页数之和；通常这些 task 属于同一 tier。如果发现多个 tier，`tier=unknown`。
- `cached`、`direct`、`reused` 通常不会产生新的 `parse_task.created.count`。
- `force=True` 会跳过 cache lookup，通常表现为 `queued` 或 `reused`。
- `parse.files.count`、`parse.pages.count`、`parse.file_size_bucket.count`、`parse.pages_bucket.count` 只在对应信息可获得时记录。如果请求在 ingest 或 metadata 可用前失败，缺失的规模指标不记录，不使用 `unknown` bucket。
- `parse.finished.count {status=failed}` 的 `tier` 规则:
  - 调用方显式指定 tier 且请求失败时，使用该显式 tier。
  - 调用方未显式指定 tier，且在默认 tier 解析完成前就失败时，使用 `default`。
  - 调用方未显式指定 tier，但默认选择策略已明确解析到了实体 tier 后才失败时，使用 `medium(default)` 或 `high(default)`。

采集点位:

- `DoclibServer.ensure_parse` 进入 handler 后记录 `parse.request.count {source, caller}`。
- `DoclibServer.ensure_parse` 正常返回后记录 `parse.finished.count {source, caller, status, tier}`。
- `DoclibServer.ensure_parse` 正常返回或抛出异常时记录 `parse.duration_bucket.count {source, caller, status, tier}`。
- `DoclibServer.ensure_parse` 抛出异常时，`parse.finished.count` 的 `status=failed`，`error_code` 使用稳定业务错误码；没有则使用 `internal_error`。
- `DoclibServer.ensure_parse` 能拿到文件 metadata 时记录 `parse.files.count`、`parse.pages.count`、`parse.file_size_bucket.count`、`parse.pages_bucket.count`。
- `DoclibServer.invalidate` 成功且 `invalidated_count > 0` 时记录 `parse.invalidate.count {status=succeeded}`。
- `DoclibServer.invalidate` 成功但 `invalidated_count = 0` 时记录 `parse.invalidate.count {status=skipped}`。
- `DoclibServer.invalidate` 抛出异常时记录 `parse.invalidate.count {status=failed}`，然后重新抛出原异常。

## 13. 用户满意度: Parse 后台任务层

`parse_task.*` 表示 doclib worker 实际执行的异步 parse 任务。它对应 `parses` 表中的任务实体，不等同于入口请求。

```text
parse_task.created.count
parse_task.started.count
parse_task.finished.count
parse_task.duration_bucket.count

parse_task.execute.count
parse_task.execute_duration_bucket.count
parse_task.write.count
parse_task.write_duration_bucket.count

parse_task.files.count
parse_task.pages.count
```

口径:

- `parse_task.created.count`: doclib 创建新的后台 parse task。
- `parse_task.started.count`: worker 获取 task 并开始执行。
- `parse_task.finished.count`: 后台 task 完成，按成功或失败区分。
- `parse_task.duration_bucket.count`: 从 worker 开始执行到 task 完成的耗时分布。
- `parse_task.execute.count`: 实际解析执行步骤结果。
- `parse_task.execute_duration_bucket.count`: 实际解析执行步骤耗时分布。
- `parse_task.write.count`: 写 Middle JSON、更新 FTS、更新 doc metadata 的步骤结果。
- `parse_task.write_duration_bucket.count`: write 步骤耗时分布。
- `parse_task.files.count`: 后台 task 实际处理文件数。第一版通常为 1。
- `parse_task.pages.count`: 后台 task 实际处理页数。

推荐状态:

```text
parse_task.finished.count:
  status = succeeded | failed

parse_task.execute.count:
  status = succeeded | failed

parse_task.write.count:
  status = succeeded | failed
```

采集要求:

- `parse_task.created.count` 必须在所有 `parses` row 创建点记录，包括显式 `POST /parses` 和 ingest 自动创建的初始 parse task。
- `parse_task.started.count` 应在 worker acquire task 成功后记录。
- `parse_task.finished.count` 应覆盖 service 内失败和 worker fallback 异常处理。
- 第一版 `parse_task.*` 不带 `source` / `caller`。不需要为后台 task 增加来源快照存储。
- `parse_task.execute.*` 覆盖 `_parse_via_local` / `_parse_via_api` 的实际解析执行。
- `parse_task.write.*` 覆盖解析后所有后续操作，包括 Middle JSON 写入、FTS 更新和 docs metadata 更新。
- 如果 FTS 更新或 docs metadata 更新失败，应同时记录 `parse_task.write.count {status=failed}` 和 `parse_task.finished.count {status=failed}`。
- step metric 只在 step 已开始后记录。execute 失败时不记录 write；write 未开始时不记录 `parse_task.write.*`。

`parse_task.execute.*` 应带 `server` 维度:

```text
server = local(flash) | local(managed) | local(self-hosted) | remote(official) | remote(custom) | unknown
```

因此第一版不再单独使用 `parse_server.*` metric。parse-server 使用情况通过:

```text
parse_task.execute.count {server=local(managed)}
parse_task.execute.count {server=remote(official)}
parse_task.execute_duration_bucket.count {server=local(self-hosted)}
```

表达。

采集点位:

- `ParseService.request_parse` 新建 `parses` row 后记录 `parse_task.created.count {tier}`。
- `ParseService.ingest_file` 自动创建初始 parse row 后记录 `parse_task.created.count {tier}`。
- `ParseService.acquire_task` 成功返回 task 后记录 `parse_task.started.count {tier}`。
- `ParseService.process_doc` 从开始执行到 task 成功或失败，记录 `parse_task.finished.count {status, tier, error_code}`。
- `parse_task.finished.count {status=failed}` 的 `error_code` 来源优先级:
  1. 如果 execute 步骤失败，使用 execute 步骤的失败码。
  2. 否则如果 write 步骤失败，使用 write 步骤的失败码。
  3. 否则如果 worker fallback 或其它未归类异常导致 task failed，使用 `parse_failed`；若没有稳定失败码可用，再降级为 `internal_error`。
- `ParseService.process_doc` 从开始执行到 task 成功或失败，记录 `parse_task.duration_bucket.count {status, tier}`。
- `ParseService.process_doc` 能拿到 task 页数时记录 `parse_task.files.count` 和 `parse_task.pages.count`。
- `_parse_via_local` / `_parse_via_api` 周围记录 `parse_task.execute.count {status, tier, server, error_code}`。
- `_parse_via_local` / `_parse_via_api` 周围记录 `parse_task.execute_duration_bucket.count {status, tier, server}`。
- JSON 写入、FTS 更新、docs metadata 更新作为一个 write 步骤，记录 `parse_task.write.count {status, tier, error_code}`。
- write 步骤记录 `parse_task.write_duration_bucket.count {status, tier}`。
- execute 失败时不记录 write。
- write 未开始时不记录 `parse_task.write.*`。
- FTS 更新或 docs metadata 更新失败时，同时记录 `parse_task.write.count {status=failed}` 和 `parse_task.finished.count {status=failed}`。
- `ParseWorkerPool` fallback exception 路径也必须记录 `parse_task.finished.count {status=failed}`，避免 service 外异常漏记。

## 14. Ingest

`ingest.*` 表示 doclib 发现文件后读取文件状态、计算 sha256、提取 metadata、创建 doc/file 记录和默认 parse task 的过程。

```text
ingest.finished.count
ingest.duration_bucket.count
```

口径:

- `ingest.finished.count`: ingest 完成结果。
- `ingest.duration_bucket.count`: ingest 耗时分布。

推荐状态:

```text
status = succeeded | failed | skipped
```

说明:

- ingest 可由 parse、scan、watch、show file 等路径触发。
- ingest 失败会影响用户是否能 parse / search / read，因此进入第一版。
- `ingest.*` 应带 `trigger` 维度，表示触发来源: `parse | scan | watch | show | background | unknown`。
- `trigger=parse`: `POST /parses` 路径触发同步 ingest。
- `trigger=scan`: scan service 刷新文件时触发 ingest。
- `trigger=watch`: watch filesystem event 直接刷新文件时触发 ingest。
- `trigger=show`: `show file`、`get_file_by_path` 或 `get_doc_by_path` 这类状态查询触发 ingest。
- `trigger=background`: ingest worker 处理 backlog 文件。
- 不记录路径、文件名或具体 metadata 内容。

采集点位:

- `ParseService.refresh_file`、`ensure_ingested`、`ingest_file` 增加 keyword-only `trigger` 参数，默认 `unknown`。
- `POST /parses` 路径调用 ingest 时传 `trigger=parse`。
- scan service 刷新文件时传 `trigger=scan`。
- `get_doc_by_path`、`get_file_by_path` 等状态查询触发 ingest 时传 `trigger=show`。
- watch filesystem event 直接 refresh 文件时传 `trigger=watch`。
- ingest worker 处理 backlog 文件时传 `trigger=background`。
- `ingest_file` 完成后记录 `ingest.finished.count {trigger, status}`。
- `ingest_file` 完成后记录 `ingest.duration_bucket.count {trigger, status}`。
- 第一版只在进入 ingest path 后记录 `ingest.*`。
- 不支持、无需 ingest 或已有 sha256 且未变化时，如果已经进入 ingest path，可以记为 `status=skipped`。
- 普通 `refresh_file` 判断为 known 且未调用 `ingest_file` 时，不强制记录 `ingest.finished.count {status=skipped}`，避免 scan 噪音。
- ingest 抛出异常时记录 `status=failed`，然后重新抛出原异常。

## 15. Search / Find / Content 体验

这些 metric 回答非 parse 请求是否成功、耗时如何、返回规模如何。

```text
search.finished.count
search.duration_bucket.count
search.results_bucket.count

find.finished.count
find.duration_bucket.count
find.results_bucket.count

content.finished.count
content.duration_bucket.count
```

口径:

- `search.finished.count`: 全文搜索请求结果，不记录 query 或结果内容。
- `search.duration_bucket.count`: 全文搜索耗时分布。
- `search.results_bucket.count`: 全文搜索返回数量分布。
- `find.finished.count`: 文件名搜索请求结果，不记录 query 或结果内容。
- `find.duration_bucket.count`: 文件名搜索耗时分布。
- `find.results_bucket.count`: 文件名搜索返回数量分布。
- `content.finished.count`: 内容读取或导出请求结果，不记录内容。
- `content.duration_bucket.count`: 内容读取或导出耗时分布。
- 第一版 content 指标不记录输出大小。即使 `output_format=image` 生成本地 temp asset，也只记录 `content_mode`、`output_format`、`tier`、`status` 和耗时。

推荐状态:

```text
status = succeeded | failed
```

`content.*` 应带:

```text
tier
content_mode = read | parse_output | export
output_format = markdown | image | other
```

`content_mode=read` 表示 `/content` locator 读取，`parse_output` 表示 `/docs/{sha256}/content` 读取 parse 输出，`export` 表示 `/docs/{sha256}/exports` 写出文件。

`output_format` 映射规则:

- `GET /docs/{sha256}/content` 根据调用方传入的 raw format 归一化：`markdown -> markdown`，`image -> image`，其它值 -> `other`。当前实现里只有 `markdown` 会成功，`image` 和 `other` 都会按原逻辑报错。
- `GET /content` 使用 `request.format`，当前合法值是 `markdown | image`。
- `POST /docs/{sha256}/exports` 当前实现只支持导出 markdown 文件；`request.format=="markdown"` 记为 `markdown`，`request.format=="image"` 记为 `image`，其余值记为 `other`。
- `other` 主要用于 route 已进入 handler，但请求中的 format 不是第一版已知枚举的失败场景；不是一个鼓励成功使用的输出格式。

采集点位:

- `DoclibServer.search` 进入 handler 后记录 `search.request.count {source, caller}`。
- `DoclibServer.search` 返回或抛出异常时记录 `search.finished.count {source, caller, status}`。
- `DoclibServer.search` 返回或抛出异常时记录 `search.duration_bucket.count {source, caller, status}`。
- `DoclibServer.search` 正常返回后记录 `search.results_bucket.count {source, caller, bucket}`，bucket 根据返回结果总数计算。
- `DoclibServer.find` 同理记录 `find.request.count`、`find.finished.count`、`find.duration_bucket.count`、`find.results_bucket.count`。
- `DoclibServer.get_doc_content` 使用 `content_mode=parse_output`。
- `DoclibServer.read_content` 使用 `content_mode=read`。
- `DoclibServer.export_doc_content` 使用 `content_mode=export`。
- content route 进入 handler 后记录 `content.request.count {source, caller, content_mode, output_format}`。
- content route 返回或抛出异常时记录 `content.finished.count {source, caller, content_mode, output_format, tier, status}`。
- content route 返回或抛出异常时记录 `content.duration_bucket.count {source, caller, content_mode, output_format, tier, status}`。
- search / find 不记录 query、snippet、filename、path 或结果内容。
- content 不记录正文、图片内容、输出路径或输出大小。

## 16. Scan / Watch 轻量体验

第一版记录 scan 入口请求、复用、最终结果和文件处理结果。watch 只记录配置请求，不展开完整 watch event 流。

```text
scan.finished.count
scan.duration_bucket.count
scan.reuse.count
scan.files.count

watch.add.finished.count
watch.remove.finished.count
```

口径:

- `scan.finished.count`: scan task 完成结果，包括 manual scan 和 watch scan。
- `scan.duration_bucket.count`: scan task 耗时分布。
- `scan.reuse.count`: scan 请求是否复用已有 pending / running scan task。
- `scan.files.count`: scan task 处理的文件数，按结果分类累计。
- `watch.add.finished.count`: watch add 配置请求结果。
- `watch.remove.finished.count`: watch remove 配置请求结果。

推荐状态:

```text
scan.finished.count:
  status = succeeded | failed

scan.reuse.count:
  status = hit | miss

scan.files.count:
  result = seen | refreshed | new | changed | deleted | unreachable | error | unsupported | excluded

watch.add.finished.count:
  status = succeeded | failed

watch.remove.finished.count:
  status = succeeded | failed
```

说明:

- `mineru watch rescan` 记为 `scan.request.count` / `scan.finished.count`，不记为 `watch.add.count` 或 `watch.remove.count`。
- watch loop initial scan 也记为 scan，`source=watch`、`caller=system`。
- watch filesystem event 直接 refresh 文件，第一版不展开为 `watch_task.*`。
- `scan.files.count` 的 `value` 为对应 `result` 的文件数，不是 scan task 次数。

采集点位:

- `DoclibServer.create_scan` 进入 handler 后记录 `scan.request.count {source, caller}`。
- watch initial scan 和 watch rescan 在内部创建 scan task 时也记录 `scan.request.count {source=watch, caller=system}`。
- `ScanService.create_scan` 发现已有 pending / running scan 并复用时，记录 `scan.reuse.count {status=hit}`。
- `ScanService.create_scan` 新建 scan row 时，记录 `scan.reuse.count {status=miss}`。
- `ScanService.process_scan` 完成后记录 `scan.finished.count {status=succeeded|failed}`。
- `ScanService.process_scan` 完成后记录 `scan.duration_bucket.count {status=succeeded|failed}`。
- `ScanService.process_scan` 完成后根据 `_ScanCounters` 记录 `scan.files.count {result=...}`，value 为对应文件数量。
- `_ScanCounters.files_seen` -> `scan.files.count {result=seen}`。
- `_ScanCounters.files_refreshed` -> `scan.files.count {result=refreshed}`。
- `_ScanCounters.files_new` -> `scan.files.count {result=new}`。
- `_ScanCounters.files_changed` -> `scan.files.count {result=changed}`。
- `_ScanCounters.files_deleted` -> `scan.files.count {result=deleted}`。
- `_ScanCounters.files_unreachable` -> `scan.files.count {result=unreachable}`。
- `_ScanCounters.files_error` -> `scan.files.count {result=error}`。
- `_ScanCounters.files_unsupported` -> `scan.files.count {result=unsupported}`。
- `_ScanCounters.files_excluded` -> `scan.files.count {result=excluded}`。
- `DoclibServer.add_watch` 进入 handler 后记录 `watch.add.count {source, caller}`。
- `DoclibServer.add_watch` 返回或抛出异常时记录 `watch.add.finished.count {source, caller, status}`。
- `DoclibServer.remove_watch` 进入 handler 后记录 `watch.remove.count {source, caller}`。
- `DoclibServer.remove_watch` 返回或抛出异常时记录 `watch.remove.finished.count {source, caller, status}`。

## 17. Bucket

耗时:

```text
lt_1s | 1_5s | 5_30s | 30_120s | 2_10m | gt_10m
```

页数:

```text
1 | 2_5 | 6_20 | 21_100 | 101_500 | gt_500
```

文件大小:

```text
lt_1mb | 1_10mb | 10_50mb | 50_200mb | gt_200mb
```

搜索结果数:

```text
0 | 1_5 | 6_20 | 21_100 | gt_100
```

Bucket 边界统一使用左闭右开区间，最后一个 bucket 右侧无界。例如 `1_5s` 表示 `[1s, 5s)`，`gt_10m` 表示 `[10m, +inf)`。

## 18. 测试与验收

第一版测试分四层，不要求做真实外网端到端测试。

Unit tests:

- `buckets.py`: duration、pages、file size、results bucket 边界。
- `context.py` / `caller.py`: contextvars set/get/reset、missing header 默认值、invalid header 降级、caller 推断失败返回 `unknown`。
- `store.py`: 初始化生成 `installation_id`、默认 `consent_state=unset`、aggregate upsert 累加、disabled 后 record no-op、disable 清空 aggregates 但不删除 `installation_id`。
- `payload.py`: preview payload schema、dimensions 稳定排序/hash、不包含 forbidden fields、空 metrics 仍返回合法 batch。

Route / service tests:

- telemetry management API: status、preview、enable、disable、flush 在 unset/disabled 返回 `executed=false`。
- observations: valid `parse_wait` -> `recorded=true`；disabled -> `recorded=false, reason=telemetry_disabled`；invalid duration / empty parse_ids -> HTTP 4xx；parse_ids 查不到 -> `recorded=false, reason=parse_refs_not_found`；缺少必要维度 -> `recorded=false, reason=missing_dimensions`。
- parse route: `parse.request.count`、`parse.finished.count` 的 cached/direct/reused/queued/failed 映射、`parse.duration_bucket.count` 带 status。
- scan/search/content/watch route: request / finished / duration 基本打点、search/find results bucket、content mode 映射、watch add/remove success/failed。

CLI prompt tests:

- unset + TTY + user -> prompt。
- unset + `--help` -> no prompt。
- unset + `--json` -> no prompt。
- unset + CI -> no prompt。
- unset + caller agent -> no prompt。
- user chooses yes / Enter -> calls enable。
- user chooses no -> calls disable。
- EOF / invalid input -> remains unset and continues original command。
- Ctrl-C -> remains unset and exits via original CLI cancellation path。

Worker / service tests:

- parse task created / started / finished。
- execute succeeded / failed。
- write succeeded / failed。
- FTS 或 docs metadata 失败时同时记录 `parse_task.write.count {status=failed}` 和 `parse_task.finished.count {status=failed}`。
- scan files counters 正确映射到 `scan.files.count`。

Flush tests:

- unset 不外发。
- disabled 不外发。
- enabled + no metrics 不外发。
- enabled + metrics 调用 fake endpoint。
- 2xx 删除 rows。
- 5xx 保留 rows。
- 4xx 删除对应 batch rows，不更新 `last_flush_at`，记录 warning log。
- 至少一个 batch 2xx 成功时更新 `last_flush_at`。
- 全部 batch 失败时不更新 `last_flush_at`。
- `sent_batch_count` 和 `sent_metric_count` 只统计 2xx 成功接收的 batch / aggregate rows。
- flush lock 存在时跳过。
- 一次 flush 最多发送 10 个 period。
- 一个 HTTP 请求只包含一个 period。

验收标准:

- telemetry disabled 时不新增 aggregate rows。
- telemetry unset 时能写 aggregate，但不会外发。
- telemetry enabled 时能 preview 和 flush。
- 所有 telemetry 写入失败不影响业务 API。
- external payload 不包含 path、filename、query、sha256、parse_id、exception message、traceback。
- CLI `mineru telemetry status|enable|disable|preview|flush` 可用。
- CLI `parse --wait` 成功、失败、超时、取消路径都会尽量提交 `parse_wait` observation。

## 19. 第一版不做

以下指标延后:

- `cli.request.count`: CLI 进入 doclib 前的入口漏斗和早失败。
- `rules.*`: exclude / parsing rule 命中统计。
- `cleanup.*`: cleanup 用户请求和清理结果。
- `compaction.*`: parse batch compaction 维护指标。
- `watch_task.*`: watch loop 自动事件、文件发现、自动解析。
- `parse_server.*`: 使用 `parse_task.execute.*` + `server` 维度替代。
- `server.shutdown.count`、`server.recovered_stale_lock.count`: server 生命周期补充指标。
- blocking wait API: 第一版用 client polling + `/observations` 采集等待体验，不让 `POST /parses` 长时间阻塞。

这些后续可以作为新增 metric 加入，不影响第一版 schema。
