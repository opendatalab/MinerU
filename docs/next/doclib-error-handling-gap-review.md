# Doclib Error Handling Gap Review

本文总结当前 doclib 在报错、错误码、错误处理和重试方面的代码现状、已处理事项和剩余问题。目标是给后续实现任务提供可落地的边界，而不是替代正式错误码规格。

范围:

- `mineru/errors.py`
- `mineru/doclib/app.py`
- `mineru/doclib/server.py`
- `mineru/doclib/client.py`
- `mineru/doclib/services/*`
- `mineru/doclib/background/*`

不纳入范围:

- `mineru/doclib/OLD/*`
- mineru.net 远端 API 的完整错误规格
- parser/tool 层内部 backend 错误细分

## 当前已处理

### E-001 全局错误码注册

状态: 已处理。

当前 `mineru/errors.py` 已补充 doclib 当前会持久化或抛出的主要错误码，包括:

- `stat_failed`
- `ingest_failed`
- `metadata_failed`
- `scan_failed`
- `parse_empty`
- `parse_json_write_failed`
- `no_accessible_file`
- `quality_tier_unavailable`
- `scan_not_found`
- `watch_not_found`
- `rule_not_found`

同时新增 `registered_error_codes()`，用于测试和后续 telemetry / 文档生成复用。

剩余风险:

- 当前测试中的 doclib 错误码集合仍是手写列表，不是从代码或 schema 自动生成。后续新增错误码时，仍可能忘记注册。

建议:

- P1 增加静态检查或更集中的错误码定义模块，让写入 DB 的错误码只能来自统一常量。

### E-002 HTTP 状态码映射统一

状态: 已处理。

当前 `mineru/errors.py` 提供 `http_status_for(code, error_type)`，`doclib/app.py` 和 `doclib/server.py` 复用同一套 HTTP 状态码规则。

当前关键规则:

| code / type | HTTP |
| --- | ---: |
| `invalid_request_error` | 400 |
| not found 类 code | 404 |
| `authentication_error` | 401 |
| `permission_error` | 403 |
| `rate_limit_error` | 429 |
| `engine_error` | 503 |
| `api_error` | 500 |

已修正:

- watch 缺失使用 `watch_not_found`。
- rule 缺失使用 `rule_not_found`。
- scan 缺失使用 `scan_not_found`。

剩余风险:

- `internal_error` 仍会把原始异常 message 暴露给客户端。
- SDK client 抛出的 `MineruError` 不保留 HTTP status code。

## 当前仍存在的问题

### E-003 ingest 失败不自动重试，但缺少显式恢复入口

状态: 部分解决，仍需设计。

当前代码语义:

- ingest worker 只领取:
  - `sha256 IS NULL`
  - `status = active`
  - `error_code IS NULL`
- 任何已有 `files.error_code` 的 file 都不会被 ingest worker 自动重试。
- `file_permission_denied`、`stat_failed`、`ingest_failed` 都会阻止自动领取。

这是有意设计: 默认不重试，只有明确有重试意义的错误才重试。

剩余问题:

1. P0 没有显式 retry ingest 入口。
2. `ingest_failed` 且文件 `mtime/size` 未变化时，普通 scan/refresh 会保留错误态。
3. 用户或 Agent 无法表达“我知道文件没变，但现在想重新尝试 ingest”。

建议:

- `retry_ingest_failures()` 不进入 P0，放到 P1 再讨论。
- 如果实现，应作为显式命令或 API，而不是 worker 自动重试。
- 推荐命令方向:
  - `mineru scan <path>` 仍只做文件状态刷新。
  - 另设显式 retry 行为，例如 `mineru retry ingest <path>` 或 `mineru cleanup/retry` 子命令，避免 scan 语义膨胀。

完成边界:

- 能按 path / watch / all 选择失败 file。
- 只清除允许重试的 `files.error_code`。
- 不重试 `file_permission_denied` / `stat_failed`，除非先 refresh 后状态已恢复。
- 输出被重试、被跳过、仍失败的数量。

### E-004 ingest 错误缺少 retryability 元数据

状态: 未处理。

当前 DB 只有:

- `files.error_code`
- `files.error_msg`

没有:

- `retryable`
- `error_stage`
- `error_at`
- `retry_count`
- `next_retry_at`

在当前“默认不自动重试”的策略下，`retry_count` 和 `next_retry_at` 可以暂缓。但 `error_stage` 和 `error_at` 对状态解释有价值。

建议:

- P0 可暂不加字段。
- P1 如果 telemetry / status 需要更好的错误解释，再增加:
  - `files.error_at`
  - `parses.error_at`
  - `scans.error_at`

### E-005 parse worker 外层兜底会丢失阶段信息

状态: 未处理。

当前 `ParseService.process_doc()` 已经能写入一些精确错误:

- `no_accessible_file`
- `parse_empty`
- `parse_json_write_failed`
- parse-server 相关错误

但如果异常发生在以下阶段:

- FTS 更新
- docs metadata 更新
- 其他未被局部捕获的持久化阶段

最终可能被 `parse_worker.py` 外层兜底统一写成:

- `parse_failed`

问题:

- 失去阶段信息。
- 难以判断错误是否来自解析引擎、索引、DB 或输出 JSON。

建议:

- 将 parse 完成后的持久化阶段拆出明确错误码:
  - `parse_fts_update_failed`
  - `parse_doc_metadata_update_failed`
  - `parse_persist_failed`
- 这些错误应进入 `mineru/errors.py` 注册。

完成边界:

- 每个 parse 后处理阶段失败时写不同 code。
- 不把非解析引擎错误泛化成 `parse_failed`。
- 测试覆盖 FTS 和 docs metadata 更新失败。

### E-006 parse failed 没有显式 retry 语义

状态: 未处理。

当前行为:

- `pending/parsing` 任务 lock 超时后可以重新被 worker 领取。
- `failed` 是终态。
- 后续同一文件再次 parse 时，可能创建新的 batch。

问题:

- 没有明确的 `retry parse` API 或 CLI。
- 不同错误的可重试性不同，但当前都只是 `status=failed`。

建议:

- 先不做自动 retry。
- 后续如果加显式 retry，应区分:
  - 可重试: `engine_unavailable`、`parse_server_unavailable`、`parse_json_write_failed`
  - 不应直接重试: `tier_mismatch`、`no_accessible_file`
  - 需要人工判断: `parse_failed`

完成边界:

- retry 不覆盖旧 done batch。
- retry 创建新的 parse batch 或重置指定 failed batch，需要二选一并文档化。
- retry 结果必须按 parse id 查询，不能被旧 done 结果掩盖。

### E-007 服务启动时 parsing 任务被置 failed 但缺少错误码

状态: 未处理。

当前 doclib app startup 会将遗留的 `parsing` parse row 改成 `failed`，并清理 lock。

问题:

- 没有写入明确 `error_code` / `error_msg`。
- 用户只能看到 failed，不知道是服务重启中断、worker 崩溃，还是解析本身失败。

建议:

- 使用明确错误码，例如:
  - `parse_interrupted`
- 注册到 `mineru/errors.py`。
- message 可为 `Parse interrupted by server restart.`

完成边界:

- startup 恢复逻辑写入 `error_code=parse_interrupted`。
- 测试覆盖 app startup recovery。

### E-008 compaction 仍有 DB / JSON 不一致风险

状态: 未处理，高优先级。

当前 compaction 逻辑大致是:

1. 删除旧 done parse rows。
2. 插入 compacted parse rows。
3. 读取旧 JSON。
4. 删除旧 JSON。
5. 写新 JSON。

并且 JSON 读取、删除、写入中有异常吞掉的路径。

风险:

- DB 已经指向 compacted rows。
- 但 compacted JSON 没有成功写入。
- 后续读取内容时找不到对应 parsed JSON。

这和已修复的 `parse_json_write_failed` 是同类一致性问题，但发生在 compaction。

建议:

- 改为 JSON-first / DB-after:
  1. 读取旧 JSON。
  2. 写新 JSON 到临时文件。
  3. fsync/rename 成最终 JSON。
  4. DB transaction 中替换 parse rows。
  5. 最后清理旧 JSON。
- 如果任何一步失败，不改变旧 done rows。

完成边界:

- compaction JSON 写入失败时，旧 parse rows 和旧 JSON 仍可读。
- 不吞异常，应记录日志并保留状态。
- 测试覆盖写 JSON 失败、读 JSON 失败、删除旧 JSON 失败。

### E-009 watch loop 异常不可观测

状态: 未处理。

当前 watch loop 中一些异常被直接 `pass`:

- initial scan fallback 异常
- watch task 异常退出

问题:

- 用户无法从 server status 或 API 知道某个 watch 已经异常退出。
- watch 可能失效但没有结构化错误。

建议:

- 为 `watches` 增加错误字段，或复用 scan task 记录 watch scan 失败:
  - `error_code`
  - `error_msg`
  - `last_error_at`
- watch task 异常退出时应更新 watch 状态或创建 scan failed 记录。

完成边界:

- watch task 异常不再静默消失。
- `mineru server status` 能看到 watch 错误摘要。
- 测试覆盖 watch task 抛异常后的可观测状态。

### E-010 removable device 恢复后没有自动 rescan

状态: 未处理。

当前 device monitor 在 removable watch root 恢复可访问时，会把:

- watch status 改回 active
- unreachable files 改回 active

但不会自动创建一次 watch scan。

问题:

- 设备拔出期间文件可能增删改。
- 仅恢复 active 状态不足以确认文件真实状态。

建议:

- device 从 unreachable 恢复 active 时创建一个 watch scan。
- 该 scan 负责重新发现 deleted / changed / new / stat error。

完成边界:

- 恢复事件只 enqueue scan，不同步遍历所有文件。
- 不在同步 API 调用中做大规模扫描。
- 测试覆盖 unreachable -> active 后创建 scan。

### E-011 scan failed 没有 retry 策略

状态: 设计上暂可接受，仍需文档明确。

当前行为:

- scan 异常会写:
  - `scans.status=failed`
  - `error_code=scan_failed`
  - `error_msg`
- failed scan 不自动重试。

问题:

- 对 manual scan，这是合理的。
- 对 watch initial scan / internal scan，是否应自动 retry 尚未明确。

建议:

- P0 先保持 manual scan 不自动 retry。
- watch/internal scan 是否 retry，放到 watch 可靠性设计中统一处理。

完成边界:

- 文档明确 scan failed 是否需要用户重新发起。
- 如果 watch scan 自动 retry，需要防止重复创建 scan task。

### E-012 internal_error 暴露原始异常文本

状态: 未处理。

当前 `doclib/app.py` 和 `doclib/server.py` 的 unexpected exception handler 会将 `str(exc)` 放入 error response。

风险:

- 可能暴露本地路径。
- 可能暴露 SQL 或内部实现细节。
- 未来如果 HTTP API 暴露给外部客户端，不适合作为默认行为。

建议:

- 默认 response 使用通用 message。
- 详细异常只写日志。
- 需要 debug 时通过配置开启详细错误。

完成边界:

- `internal_error.message` 默认不包含 traceback、SQL、绝对路径。
- 日志仍保留完整异常。
- 测试覆盖 unexpected exception response。

### E-013 SDK client 不保留 HTTP 状态码和可操作信息

状态: 未处理。

当前 `DoclibClient` 收到 error envelope 后会抛 `MineruError(code, message, param)`。

问题:

- 不保留 HTTP status。
- 不保留 `retryable`。
- 不保留 `user_action`。
- 未来 CLI 需要输出下一步建议时信息不足。

建议:

- 短期不改异常结构。
- 后续统一错误模型时增加:
  - `status_code`
  - `retryable`
  - `user_action`
  - `request_id`

完成边界:

- SDK client 能保留 response status code。
- CLI 能根据错误输出稳定建议，而不是解析 message。

## 当前错误处理原则

### 文件级错误

写入位置: `files.error_code` / `files.error_msg`

适用范围:

- 文件路径可访问性。
- stat / permission 问题。
- ingest 阶段无法完成的 file 级错误。

当前原则:

- `deleted` / `unreachable` 是文件状态，不是 file error。
- permission/stat error 不应标记为 deleted。
- worker 不主动重试已有 file error。

### 文档级错误

写入位置: `docs.error_code` / `docs.error_msg`

适用范围:

- 内容身份 `sha256` 级别的问题。
- 当前主要用于 metadata 提取失败。

当前原则:

- `metadata_failed` 不阻断入库。
- metadata 后续被更高质量 parse 纠正后，可以清除 doc error。

### Parse batch 错误

写入位置: `parses.error_code` / `parses.error_msg`

适用范围:

- 某个 `sha256 + tier + pages` 批次的问题。
- parse-server 不可用。
- parse 返回空。
- parsed JSON 写入失败。

当前原则:

- failed parse batch 不自动重试。
- 后续 parse 请求可以创建新 batch。
- done batch 只有在对应 parsed JSON 存在时才参与覆盖判断。

### Scan task 错误

写入位置: `scans.error_code` / `scans.error_msg`

适用范围:

- 一次 scan task 的执行失败。

当前原则:

- scan 和 ingest 职责分离。
- scan done 不表示 ingest done。
- scan failed 不自动 retry。

## 建议优先级

| 优先级 | 项目 | 原因 |
| --- | --- | --- |
| P0 | E-008 compaction DB / JSON 一致性 | 有数据一致性风险 |
| P0 | E-007 startup interrupted parse 错误码 | 当前 failed 无原因，不利于 Agent 判断 |
| P0 | E-012 internal_error 信息泄漏 | 错误响应契约应尽早稳定 |
| P1 | E-003 显式 retry ingest 入口 | P0 不做，需要先定 CLI/API 形态 |
| P1 | E-005 parse 后处理阶段错误码 | 提升诊断能力 |
| P1 | E-009 watch 异常可观测 | 提升稳定性和可维护性 |
| P1 | E-010 removable 恢复后自动 scan | 与文件生命周期一致性相关 |
| P1 | E-013 SDK 错误增强 | 依赖整体错误模型扩展 |
| P2 | E-004 error_at / retry metadata | 可随 telemetry/status 一起做 |
| P2 | E-011 scan retry 策略 | 当前可接受手动重试 |

## 下一步建议

建议先处理 `E-008 compaction DB / JSON 一致性`。

原因:

- 这是当前剩余问题中最接近真实数据损坏的一项。
- 它和已经修复的 `parse_json_write_failed` 属于同类问题。
- 修复边界清晰，可以用单元测试验证。
