# ADR-0003: Parse 请求等待批次语义

状态: Accepted
日期: 2026-06-09
相关文档: ../architecture.md, ../workflows.md, ../cli/mineru-parse.md, 0002-force-vs-invalidate.md, 0004-doclib-http-api-resources.md

## 背景

doclib 的解析缓存以 `sha256 + tier + pages + done_at` 表示，一个用户请求可能被拆成多个 parse batch。

典型场景:

1. 请求页码的一部分已经被 `pending` / `parsing` batch 覆盖。
2. 请求页码的另一部分没有被任何 active batch 覆盖，需要新建 batch。
3. `--force` 跳过 done cache，但仍可能复用已存在的 active batch，避免重复消耗算力。
4. 旧 done cache 仍然有效，因此不能用聚合状态判断本次 force 请求是否成功。

如果 CLI 只轮询 `sha256 + tier` 的聚合状态，force batch 失败时可能被旧 done cache 掩盖，用户会误以为本次重新解析成功。

## 决策

parse 请求返回本次请求需要等待的 batch 集合，而不是单个 `parse_id`。

`POST /parses` 返回字段:

```json
{
  "sha256": "...",
  "tier": "standard",
  "pages": "1~10",
  "status": "pending",
  "wait_parse_ids": [12, 13],
  "created_parse_ids": [13],
  "reused_parse_ids": [12]
}
```

字段语义:

| 字段 | 含义 |
|------|------|
| `wait_parse_ids` | 本次请求需要等待的 parse batch id 集合。 |
| `created_parse_ids` | 本次请求新创建的 parse batch id 集合。 |
| `reused_parse_ids` | 本次请求复用并提权的 `pending` / `parsing` batch id 集合。 |

规则:

1. 非 force 请求如果 done cache 已覆盖全部请求页，返回 `status=done` 且 `wait_parse_ids=[]`。
2. `--force` 跳过 done cache，但不跳过 active batch。
3. 已有 active batch 覆盖的页码不重复创建 batch，而是复用该 batch 并提升 priority。
4. 未被 done cache 或 active batch 覆盖的页码创建新 batch。
5. `wait_parse_ids = reused_parse_ids + created_parse_ids`。
6. CLI `--wait` 必须等待 `wait_parse_ids` 中的所有 batch，而不是轮询聚合状态。
7. 任一 wait batch 失败，本次请求即失败；旧 done cache 是否可读是另一个问题，不应把本次请求显示为成功。
8. 所有 wait batch 成功后，CLI 再读取目标 `sha256 + tier + pages` 的内容；读取层按有效 done batch 的 `done_at` 选择最新页面。

## 状态查询

需要提供 parse record 级状态查询能力。

建议 endpoint:

```http
GET /parses?ids=12,13
```

返回:

```json
{
  "parses": [
    {"id": 12, "sha256": "...", "tier": "standard", "pages": "1~5", "status": "done"},
    {"id": 13, "sha256": "...", "tier": "standard", "pages": "6~10", "status": "parsing"}
  ]
}
```

不再保留 `/parse/status`。所有状态查询都进入 `/parses`:

- `GET /parses?ids=...` 回答“本次请求关联的 parse record 是否完成”。
- `GET /parses?sha256=...&tier=...&pages=...` 回答“文档在某个 tier 下的覆盖状态”。

CLI wait 必须使用 id 级状态查询。

## 替代方案

### 方案 A: 返回单个 `parse_id`

拒绝。一个请求可能同时复用多个 active batch，并为未覆盖页创建新 batch。单个 `parse_id` 不能表达本次请求完整等待边界。

### 方案 B: force 严格总是创建新 batch

拒绝。语义简单，但会在已有同页 active batch 时重复消耗算力，尤其对 `standard` / `pro` 和 remote 解析成本较高。

### 方案 C: CLI 继续轮询聚合状态

拒绝。聚合状态会被旧 done cache 影响，无法表达本次 force 请求是否失败。

## 影响

API 影响:

- `POST /parses` response 增加 `wait_parse_ids`、`created_parse_ids`、`reused_parse_ids`。
- `GET /parses` 提供 id 查询、状态过滤和覆盖查询。

CLI 影响:

- `mineru parse --wait` 根据 `wait_parse_ids` 等待。
- `wait_parse_ids=[]` 时直接读取缓存。
- 任一 wait batch failed 时返回本次请求失败，并可提示旧缓存仍可能可读。
- 超时时输出仍未完成的 batch id 和状态。

服务端实现影响:

- active batch 覆盖判断需要保留 batch id，而不是只保留 pages set。
- 新建 parse row 后需要返回新 row id。
- batch 状态查询需要按 id 返回 `status`、`pages`、`error_code`、`error_msg`、`done_at` 等必要字段。

测试影响:

- cache hit 返回空 `wait_parse_ids`。
- force 复用 active batch 并返回 `reused_parse_ids`。
- 部分页码复用 active、部分页码创建新 batch。
- force batch failed 时 CLI 不被旧 done cache 掩盖。
- 所有 wait batch done 后读取内容使用最新有效页面。

## 后续动作

1. 更新 API / CLI / SDK 文档中的 parse response schema。
2. 实现 `GET /parses` 状态查询 endpoint。
3. 更新 CLI wait 流程，改为等待 `wait_parse_ids`。
4. 为 force、active batch 复用、部分页补齐和失败可观测性补测试。
