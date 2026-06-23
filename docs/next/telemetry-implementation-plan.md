# Telemetry Implementation Plan

状态: Draft
日期: 2026-06-22
依据: `docs/next/telemetry-minimum.md`

本文档记录 doclib 第一版 telemetry 的实现计划。指标定义、隐私边界和协议语义以 `telemetry-minimum.md` 为准；本文只描述开发顺序、代码挂点和实现约束。

## 1. 已确认实现口径

第一版只实现 doclib telemetry:

- doclib server 负责本地记录、聚合和 flush。
- CLI / `DoclibClient` 只传递 context 或 observation，不直接外发 telemetry。
- parser SDK、`mineru-kit parse`、`mineru-kit api-server` 不纳入第一版。
- `parse.wait.*` 不带 `error_code`。
- `error_code` 只出现在失败类结果 count metric:
  - `parse.finished.count`
  - `parse_task.finished.count`
  - `parse_task.execute.count`
  - `parse_task.write.count`
- duration bucket 不带 `error_code`。
- 成功状态不带 `error_code`，不上传 `none`。

第一版外部 metrics API 先按 `aaa.py` 中 staging 接口实现:

```text
POST https://staging.mineru.org.cn/metrics/v2/metrics
```

请求规则:

- body 使用 compact JSON raw bytes，不能使用 HTTP client 的 `json=...` 自动序列化。
- body 包含 `api_version="v2"`。
- `schema_version` 使用字符串 `"1"`。
- header 包含 `Content-Type: application/json`、`X-Track-App-Key`、`X-Track-Ts`、`X-Track-Sign`。
- `X-Track-Sign = hmac_sha256(APP_SECRET, APP_KEY + ts + raw_body)`。
- `APP_KEY` / `APP_SECRET` 第一版内置在代码常量中，值从 `aaa.py` 同步。
- `preview` 只返回 request body，不返回签名 header。
- flush 不解析 response body，只看 HTTP status。

flush response 处理:

- `2xx`: batch accepted，删除对应 aggregate rows。
- `4xx`: 删除对应 batch rows，不更新 `last_flush_at`，返回 `invalid_payload_discarded`，记录 warning。
- `5xx`: 保留 rows，返回 `server_error`。
- network / timeout: 保留 rows，返回 `network_error`。
- 同一次 flush 中既有 2xx，又有失败 batch，返回 `partial_success`。
- response body 不写 DB、不进入 preview、不进入 CLI 输出、不进入日志。

official remote 判定:

- host 等于或后缀匹配 `mineru.net`、`mineru.org.cn` 时为 official。
- 不使用裸 substring 匹配，避免误判。

CLI unset prompt 覆盖:

- 触发: `parse`、`read`、`scan`、`watch`、`search`、`find`、`list`、`show`、`invalidate`、`forget`、`cleanup`。
- 不触发: `server`、`config`、`telemetry`。
- `--json`、CI、agent caller、非 TTY、server 不可访问时不提示。

## 2. 开发阶段

### Phase 1: Core 模块与 DB

新增模块:

```text
mineru/doclib/telemetry/
  __init__.py
  constants.py
  buckets.py
  context.py
  caller.py
  store.py
  payload.py
  observation.py
  service.py
```

实现内容:

- `constants.py`: metric name、dimension whitelist、`METRIC_SPECS`、consent state、endpoint、API version、schema version、staging app key/secret。
- `buckets.py`: duration、pages、file size、results bucket。
- `context.py`: `contextvars`、HTTP header parse、request context set/reset。
- `caller.py`: parent process tree inference，最大深度 12，只读进程名。
- `store.py`: `telemetry_state` 和 `telemetry_aggregates` 读写。
- `payload.py`: preview / flush payload 组装，metrics 稳定排序。
- `service.py`: `record_*`、`status`、`preview`、`enable`、`disable`、`flush`。

DB 改动:

- 修改 `mineru/doclib/migrations/001_init.sql`，增加 `telemetry_state`、`telemetry_aggregates`。
- 修改 `mineru/doclib/app.py` 的 `REQUIRED_SCHEMA_TABLES`。
- 不做单独 migration，现阶段直接改 `001_init.sql`。

测试重点:

- bucket 边界。
- dimension normalize / whitelist / hash。
- state 初始化: `installation_id`、`consent_state=unset`、`last_flush_at=0`、`flush_locked_at=0`。
- disabled 后 record no-op，disable 清空 aggregates 但保留 `installation_id`。
- preview payload 不包含 forbidden fields。

### Phase 2: Context 传递与管理 API

接口模型:

- 在 `mineru/doclib/types.py` 增加 telemetry response / request models。
- 在 `mineru/doclib/base.py` 和 `AsyncDoclibInterface` 增加抽象方法。
- 在 `mineru/doclib/client.py` 增加 sync client 方法。
- 在 `mineru/doclib/server.py` 增加 route 实现。

新增 API:

```text
GET /telemetry/status
GET /telemetry/preview
POST /telemetry/actions/{action}
POST /observations
```

context 传递:

- `DoclibClient._request_model` 读取 contextvars，并发送:
  - `X-MinerU-Source`
  - `X-MinerU-Caller`
- server middleware 解析 header，缺失时默认 `source=http_api`、`caller=http_client`。
- request 结束后 reset context token。

测试重点:

- missing / invalid header 降级。
- SDK 默认 `source=sdk`，caller best-effort 推断。
- telemetry status / preview / enable / disable / flush 行为。
- `/observations` schema 校验和 disabled/unset 行为。

### Phase 3: Flush Worker

新增:

```text
mineru/doclib/background/telemetry_flush.py
```

`AppState` 增加:

```text
state.telemetry_store
state.telemetry_svc
state.telemetry_flush_worker
```

startup 顺序:

```text
DatabaseManager.initialize()
  -> TelemetryStore(db)
  -> TelemetryService(store, context collector)
  -> business services with telemetry_svc
  -> background workers
  -> TelemetryFlushWorker
```

flush 行为:

- 启动后先尝试一次 flush。
- 每 2 小时尝试一次 flush。
- 只有 `consent_state=enabled` 才外发。
- 一次 flush 最多 10 个 period。
- 每个 HTTP request 只发送一个 period。
- period 按 `period_start ASC, period_end ASC`。
- lock TTL 1800 秒。
- 发送使用 `httpx.AsyncClient`，timeout 10 秒。
- 使用 compact JSON raw bytes + HMAC header。

测试重点:

- unset / disabled 不外发。
- no pending metrics 不外发。
- 2xx 删除 rows。
- 4xx 删除 rows，不更新 `last_flush_at`。
- 5xx/network 保留 rows。
- partial success。
- lock 存在时跳过。
- 一次最多 10 个 period。

### Phase 4: Route-level Metrics

先接入口体验层，不碰后台 worker 内部步骤。

parse route:

- `parse.request.count`
- `parse.finished.count`
- `parse.duration_bucket.count`
- `parse.files.count`
- `parse.pages.count`
- `parse.file_size_bucket.count`
- `parse.pages_bucket.count`
- `parse.invalidate.count`

search / find / content:

- `search.request.count`
- `search.finished.count`
- `search.duration_bucket.count`
- `search.results_bucket.count`
- `find.request.count`
- `find.finished.count`
- `find.duration_bucket.count`
- `find.results_bucket.count`
- `content.request.count`
- `content.finished.count`
- `content.duration_bucket.count`

scan / watch route:

- `scan.request.count` route 部分。
- `watch.add.count`
- `watch.add.finished.count`
- `watch.remove.count`
- `watch.remove.finished.count`

实现约束:

- route-level request metric 在进入业务 handler 后立即记录。
- FastAPI / Pydantic 进入 handler 前拒绝的请求不计。
- 失败路径可以计算 telemetry status / error_code，但必须 re-raise 原异常。
- telemetry 写入失败不能影响业务返回。

### Phase 5: Service / Worker Metrics

`ParseService`:

- 构造函数显式接收 `telemetry_svc`。
- 接 `parse_task.created.count`、`parse_task.started.count`。
- 在 `process_doc` 周围接:
  - `parse_task.finished.count`
  - `parse_task.duration_bucket.count`
  - `parse_task.files.count`
  - `parse_task.pages.count`
- 在 `_parse_via_local` / `_parse_via_api` 周围接:
  - `parse_task.execute.count`
  - `parse_task.execute_duration_bucket.count`
- JSON 写入、FTS 更新、docs metadata 更新作为 write 步骤:
  - `parse_task.write.count`
  - `parse_task.write_duration_bucket.count`

`ParseWorkerPool`:

- fallback exception 路径也必须记录 `parse_task.finished.count {status=failed}`。

`ScanService`:

- 构造函数显式接收 `telemetry_svc`。
- 接:
  - `scan.reuse.count`
  - `scan.finished.count`
  - `scan.duration_bucket.count`
  - `scan.files.count`
- internal watch-triggered scan 创建时记录 `scan.request.count {source=watch, caller=system}`。

Ingest:

- `refresh_file` / `ensure_ingested` / `ingest_file` 增加 keyword-only `trigger`，默认 `unknown`。
- 触发来源: `parse | scan | watch | show | background | unknown`。
- 记录:
  - `ingest.finished.count`
  - `ingest.duration_bucket.count`

测试重点:

- parse task created / started / finished。
- execute succeeded / failed。
- write succeeded / failed。
- write failed 同时导致 task finished failed。
- scan counters 映射到 `scan.files.count`。
- ingest trigger 和 skipped 规则。

### Phase 6: Observations 与 CLI Wait

`POST /observations`:

- 第一版只支持 `type=parse_wait`。
- 根据 `parse_ids` 查询 task，补齐 `tier` 和 `pages_bucket`。
- 不补齐、不上报 `error_code`。
- 记录:
  - `parse.wait.count`
  - `parse.wait_duration_bucket.count`

`DoclibClient`:

- 增加 `submit_observation()`。

CLI parse wait:

- wait 前记录 `wait_start = time.monotonic()`。
- done -> `status=succeeded`。
- failed -> `status=failed`。
- timeout -> `status=timeout`。
- Ctrl-C / cancel -> 尽量 `status=canceled`。
- observation 失败静默忽略。

### Phase 7: CLI Telemetry 命令与 Unset Prompt

新增:

```text
mineru/cli/commands/telemetry.py
mineru/cli/telemetry.py
```

`mineru` 顶层命令:

```text
mineru telemetry status
mineru telemetry enable
mineru telemetry disable
mineru telemetry preview
mineru telemetry flush
```

`TOP_LEVEL_COMMAND_ORDER`:

```text
server
config
telemetry
```

prompt helper:

- `is_interactive_cli()`
- `is_ci_environment()`
- `maybe_prompt_telemetry_consent(client, *, json_mode, command_name)`

触发范围:

- 触发: `parse`、`read`、`scan`、`watch`、`search`、`find`、`list`、`show`、`invalidate`、`forget`、`cleanup`。
- 不触发: `server`、`config`、`telemetry`。
- `--json`、CI、agent caller、非 TTY、server 不可访问时不提示。

## 3. 建议提交边界

建议拆成 5 个开发包:

1. `telemetry core`: DB、store、service、payload、buckets、context。
2. `telemetry api`: management API、observations API、client/base/types。
3. `telemetry flush`: worker、flush lock、staging metrics API。
4. `telemetry metrics`: parse/search/content/scan/watch/ingest/parse_task 打点。
5. `telemetry cli`: commands、prompt、parse wait observation。

## 4. 高风险点

- `DoclibClient` 是 sync client，server 侧 telemetry service 是 async，不要混用。
- route decorator 要求 `base.py`、`client.py`、`server.py` 接口保持一致。
- `DatabaseManager` 当前每次操作新建连接，aggregate upsert 和 flush lock 要依赖 SQL 原子性。
- `error_code` 只在四个失败 count metric 上出现，不得加到 duration bucket。
- telemetry record 失败必须吞掉，业务异常必须原样返回。
- HMAC 签名必须使用发送出去的 exact raw body。
- preview 只展示 body，不展示 app key、secret、签名或 response。
