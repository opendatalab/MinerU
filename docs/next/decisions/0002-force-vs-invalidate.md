# ADR-0002: Force 与 Invalidate 缓存语义

状态: Accepted
日期: 2026-06-09
相关文档: ../architecture.md, ../workflows.md, ../cli/mineru-parse.md, ../sdk/doclib-client.md
补充决策: 0003-parse-request-wait-batches.md, 0004-doclib-http-api-resources.md

## 背景

doclib 只持久化按页组织的 Middle JSON 批次文件。同一文档、同一解析档位可以因为增量页码解析、重新解析或后台合并而存在多个 done batch。

因此需要明确两个容易混淆的动作:

- `--force`: 用户要求本次请求重新解析。
- `invalidate`: 用户或系统要求已有缓存不再参与后续读取、搜索和覆盖判断。

如果二者语义混在一起，重新解析失败时可能丢失可用旧结果；如果读取逻辑不区分有效批次和失效批次，缓存覆盖、搜索索引和 compaction 都会出现不一致。

## 决策

`--force` 与 `invalidate` 是两个不同动作。

`--force` 的语义:

1. 对本次请求的 `sha256 + tier + page_range` 忽略已有缓存覆盖判断。
2. 不忽略 `pending` / `parsing` 的 active parse；已经覆盖请求页码的 active parse 可以复用并提升 priority。
3. 只为未被 active parse 覆盖的页码创建新的 parse record。
4. 不删除、不标记失效、不提前 supersede 旧的 done batch。
5. 本次请求返回 `wait_parse_ids`、`created_parse_ids` 和 `reused_parse_ids`。调用方等待 `wait_parse_ids`，而不是轮询文档聚合状态。
6. wait parse 成功后，读取同一页时按有效 done batch 的 `done_at` 选择最新页面内容。
7. wait parse 失败时，本次 force 请求失败；旧的 done batch 仍然有效，仍可被读取、搜索和后续 compaction 使用。

`invalidate` 的语义:

1. 第一版通过 `POST /invalidate` 执行，要求 `target="parses"`。
2. 将指定文档和可选 tier 的已有 done parse 标记为失效；第一版不提供页码或单 batch 粒度。
3. invalidate 本身不要求立即重新解析。
4. 失效 parse 不参与缓存覆盖判断、读取时页合并、搜索索引刷新和 compaction 的“最新页”选择。
5. invalidate 后需要刷新搜索索引：若仍有有效 done parse，则从最高有效 tier 重建 FTS；若没有有效 done parse，则删除对应 FTS 内容。
6. 对应 JSON 文件的物理删除应由 cleanup 或 compaction 异步完成，不作为 invalidate 的同步完成条件。

读取合并规则:

1. 只扫描有效的 done batch。
2. 忽略 JSON 文件缺失的 batch。
3. 按 `page_idx` 合并页面。
4. 同一页出现多次时，选择 `done_at` 最新的有效 batch。

## 替代方案

### 方案 A: `--force` 前删除旧缓存

拒绝。重新解析失败时会丢失原本可用的结果，不符合缓存作为稳定可读结果的定位。

### 方案 B: `--force` 开始时先把旧 batch 标记为 superseded

拒绝。它仍然会让失败的重新解析破坏旧结果可用性。只有新 batch 成功后，读取层按 `done_at` 自然选择新页面即可，不需要提前改变旧 batch 状态。

### 方案 C: force 严格总是创建新 parse

拒绝。已有 active parse 覆盖请求页码时，重复创建 parse 会浪费算力。`--force` 的核心语义是跳过 done cache，不是强制制造并发重复任务。

### 方案 D: `invalidate` 自动触发重新解析

拒绝。失效缓存和创建新解析任务是两个独立意图。调用方可以在 invalidate 后再显式 parse，但 invalidate 不应隐式消耗算力。

## 影响

实现影响:

- parse cache coverage 需要过滤失效 batch。
- read-time render 需要按有效 done batch 合并页面，并用 `done_at` 解决重复页。
- compaction 只处理有效 done batch，不能把失效 batch 合入新文件。
- 搜索索引刷新不能从失效 batch 读取文本；invalidate 后需要删除或降级重建 FTS。
- `force=True` 的 parse 请求只绕过 done 缓存命中，不改变旧 batch 生命周期。
- `POST /parses` 需要返回 `wait_parse_ids`、`created_parse_ids`、`reused_parse_ids`。

测试影响:

- 覆盖 `force=True` 复用 active parse，并只为未覆盖页创建新 parse。
- 覆盖 force wait parse 失败后本次请求失败，但旧结果仍可读取。
- 覆盖重复页读取时最新有效 `done_at` 胜出。
- 覆盖 invalidated batch 不参与缓存命中、读取合并和 compaction。
- 覆盖 invalidate 后 FTS 删除或降级重建。

用户体验影响:

- `--force` 是低风险重新解析动作；它跳过 done cache，但不制造重复 active parse。
- invalidate 是显式缓存生命周期管理动作，应通过 SDK、管理 API 或维护命令暴露，不应隐藏在普通 parse 行为里。

## 后续动作

1. 在 SQLite schema 和 parse row 状态中使用 `status=superseded` 表达已失效缓存。
2. 更新 parse cache coverage、read-time render、search refresh 和 compaction 的过滤条件。
3. 更新 `POST /parses` 和 CLI wait，等待 `wait_parse_ids`。
4. 为 `force` 和 `invalidate` 增加独立测试。
