# ADR-0010: Doclib Watch CLI 与 Rescan 边界

状态: Accepted
日期: 2026-06-12
相关文档: ../cli/mineru-library.md, ../workflows.md, 0009-doclib-scan-task.md

## 背景

`watch` 是 doclib 的持久资源，用于定义本地文档库需要持续监听的目录。此前 CLI 将 watch 管理放在:

```bash
mineru config watch add/list/rm
```

这个位置容易把 watch 误解为普通配置项。但 watch 实际参与文件发现、设备可达性、scan、ingest、search 可观测性等核心工作流。它与 `scan`、`forget`、`cleanup`、`search`、`parse` 一样，是文档库生命周期的一等入口。

同时，doclib 已引入 `scan` 后台任务。需要明确:

- `mineru scan <path>` 的一次性 path scan。
- `mineru watch rescan <watch>` 的 watch target 重扫。

二者复用同一个 `ScanService + ScanWorker`，但资源语义和状态副作用不同。

## 决策

将 watch 从 `mineru config` 下提升为顶层子命令:

```bash
mineru watch add <path>
mineru watch list
mineru watch remove <path>
mineru watch rescan <watch-path-or-id>
```

`mineru config watch ...` 不作为 NEXT 第一版公开入口。

`rm` 改为 `remove`。NEXT 第一版尚未发布，不保留 `rm` alias。

## 命令边界

### `mineru watch`

`watch` 面向 doclib 的持久 watch target 资源。

职责:

- 添加 watch target。
- 列出 watch targets。
- 移除 watch target。
- 对已配置 watch target 触发完整 rescan。

`watch` 不负责:

- 管理 exclude / parsing rules。
- 管理 parse-server 配置。
- 扫描任意非 watch path。

这些分别属于 `mineru config ...` 和 `mineru scan <path>`。

### `mineru config`

`config` 保留真正的配置和规则管理:

```bash
mineru config show
mineru config set ...
mineru config exclude ...
mineru config parsing-rules ...
mineru config parse-server ...
```

## `scan` 与 `watch rescan`

### `mineru scan <path>`

语义:

> 一次性检查这个 path 当前的文件状态，并让 doclib 更新对应 `files` 状态。

适用场景:

- 用户或 Agent 临时给一个文件做发现。
- 用户或 Agent 临时给一个目录做一次性扫描。
- path 不属于任何 watch。
- path 属于某个 watch，但调用方只想处理这个具体 path。

规则:

- 不创建 watch。
- 不要求 path 是 watch root。
- 不更新 `watches.last_scan_at` / `last_scan_files`。
- 写入 `scans.kind = manual`。
- 扫描目录时应用 exclude rules。
- 扫描文件时不应用 exclude rules，因为这是显式点名文件。

### `mineru watch rescan <watch-path-or-id>`

语义:

> 对一个已配置 watch target 重新执行完整扫描，并记录为该 watch 的一次 scan。

适用场景:

- watch initial scan 失败或不完整后，用户手动重跑。
- removable 设备恢复后，系统或用户触发该 watch 的恢复扫描。
- 用户调整 exclude / parsing rules 后，希望对某个 watch root 重新发现状态。
- Agent 需要确认某个 watch target 当前是否与 doclib 状态一致。

规则:

- 目标必须是已配置 watch target。
- 支持通过 watch id 或 watch root path 指定。
- 不接受 watch 下的子目录；子目录一次性扫描应使用 `mineru scan <path>`。
- 写入 `scans.kind = watch`。
- 绑定 `watch_id`。
- 使用 watch 的 enabled / status / removable 语义。
- 更新 `watches.last_scan_at` / `last_scan_files`。

## `watch rescan` 目标解析

`watch rescan` 支持:

```bash
mineru watch rescan 3
mineru watch rescan ~/Documents
```

解析规则:

- 如果参数是纯数字，优先按 watch id 查找。
- 否则按 path normalize 后精确匹配 watch root。
- 找不到 watch 时返回 `watch_not_found`。
- 参数同时匹配 id 和 path 的极端情况，以 id 优先。

P0 不支持:

- 对 watch 子目录 rescan。
- 对多个 watch 同时 rescan。
- `--force` 创建重复 running scan。

如果同一 watch 已有 pending / running watch scan，复用已有 scan task。

## HTTP / SDK 影响

不新增专用 HTTP endpoint。

`watch rescan` 通过已有 scan API 实现:

```http
POST /scans
```

请求体示例:

```json
{
  "path": "/Users/me/Documents",
  "kind": "watch",
  "source": "cli",
  "watch_id": 3
}
```

CLI / SDK 可以通过 `list_watches()` 解析 watch path / id，再调用 `create_scan()`。

## 替代方案

### 继续使用 `mineru config watch`

拒绝。watch 是文档库核心资源，不是普通配置项。放在 `config` 下会降低可发现性，也不利于 Agent 正确选择命令。

### 使用顶层 `mineru rescan`

拒绝。`rescan` 的目标必须是已配置 watch target，放在 `watch` 下能明确资源边界。

### 让 `mineru scan <watch-root>` 同时更新 watch stats

拒绝。`scan` 是 path 操作，不应因为 path 恰好等于 watch root 就改变 watch 统计语义。需要 watch 语义时应显式使用 `watch rescan`。

### 保留 `watch rm`

拒绝。NEXT 第一版还没有兼容负担，统一使用更明确的 `remove`。

## 影响

- CLI 需要新增顶层 `watch` group。
- CLI 需要移除 `config watch` group。
- `watch remove` 取代 `watch rm`。
- `watch rescan` 需要复用 `ScanService.create_scan(kind="watch")`。
- 文档中所有 `mineru config watch add/list/rm` 应迁移为 `mineru watch add/list/remove`。
- `forget` 提示语中应使用 `mineru watch remove <path>`。

## 后续动作

1. 修改 CLI 注册结构，新增 `mineru watch`。
2. 将 `config watch add/list/rm` 迁移到 `watch add/list/remove`。
3. 实现 `watch rescan <watch-path-or-id>`。
4. 更新 CLI 文档、设计文档和 forget 文案。
5. 增加测试覆盖:
   - `watch add/list/remove`。
   - `watch rescan` 按 id 解析。
   - `watch rescan` 按 path 解析。
   - `config watch` 不再注册。
