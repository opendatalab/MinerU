# ADR-0008: Doclib Forget Path 语义

状态: Accepted
日期: 2026-06-12
相关文档: ../architecture.md, ../workflows.md, ../cli/mineru-library.md, ../everydoc-doclib-gap-review.md

## 背景

doclib 已经区分 `files` 与 `docs`: `files.path` 表示一个本地路径实例，`docs.sha256` 表示内容身份。

用户或 Agent 可能需要让 doclib 忘记某个路径，例如:

- 某个路径不应再出现在本地文档库中。
- 某个目录下的文件记录需要批量移出 doclib。
- 已删除或不可达文件的历史 path 记录需要被显式清理。

这个动作不同于:

- 删除磁盘文件。
- 移除 watch。
- invalidate parse cache。
- cleanup orphan docs。

因此需要一个独立的 `forget` 语义，避免用 `remove`、`delete`、`unindex` 等动词造成误解。

## 决策

引入 `forget path` 能力。

CLI 动词使用:

```bash
mineru forget <path>
```

SDK / Interface 方法建议命名:

```python
forget_path(path: str, dry_run: bool = True) -> ForgetPathResponse
```

HTTP API 建议使用维护动作:

```http
POST /forget
```

请求体:

```json
{
  "path": "/Users/me/Documents/project",
  "dry_run": true
}
```

### 文件 path

如果 `path` 匹配一个 file row:

- 删除该 `files` row。
- 删除对应 `fts_filenames`。
- 不删除真实文件。
- 不删除 `docs`、`parses`、parsed JSON 或 `fts_contents`。
- 如果对应 doc 没有任何 file row 关联，它成为 orphan doc，由 `cleanup orphan-docs` 后续处理。

### 目录 path

如果 `path` 是目录，或 DB 中存在以 `path/` 为前缀的 file rows:

- 忘记该目录树下所有已入库 file rows。
- 目录永远递归处理。
- 不提供 `recursive` / `no-recursive` 参数。

原因是 doclib 的 `files` 表记录文件 path，不记录目录 path。目录不递归几乎没有可用语义；“只处理一层文件”属于后续 `depth` 能力，不进入 P0。

### 不存在的 path

如果磁盘 path 不存在:

- 精确匹配 DB 中的 file path 时，按文件 path 处理。
- 前缀匹配 DB 中的 `path/` 时，按目录 path 处理。
- 两者都没有时，返回 0 个匹配记录，不报错。

### watch root

如果 `path` 是已配置 watch root，P0 默认拒绝:

```text
Path is a configured watch root. Use mineru watch remove <path> to remove the watch first.
```

原因:

- `forget` 不应隐式删除 watch 配置。
- 如果 watch 继续存在，下一轮 scan 会重新发现这些文件。
- `watch remove` 与 `forget` 是不同动作: 前者停止监控并解绑文件，后者删除 file rows。

### watch 下的文件或子目录

允许 forget active watch 下的文件或子目录。

如果 path 位于 active watch root 下，应返回 warning:

```text
Path is under an active watch and may be rediscovered on the next scan.
```

这不是错误。`forget` 不是 ignore rule，不阻止未来重新发现。

### scan status

`forget` 支持所有 file status:

- `active`
- `deleted`
- `unreachable`

原因是用户可能需要忘记已删除历史记录或 removable 设备上的不可达记录。

### dry-run

CLI 和 API 默认 dry-run:

```bash
mineru forget <path>
mineru forget <path> --no-dry-run
```

dry-run 返回将会忘记的 file row 数量、匹配方式和 warning，但不修改 DB。

## 返回结构

建议响应:

```json
{
  "path": "/Users/me/Documents/project",
  "matched_as": "directory",
  "forgotten_files": 12,
  "dry_run": true,
  "warnings": [
    "Path is under an active watch and may be rediscovered on the next scan."
  ]
}
```

`matched_as` 取值:

```text
file | directory | none
```

## 替代方案

### 使用 `remove-file`

拒绝。`remove-file` 容易被理解为删除真实磁盘文件。

### 使用 `unindex`

拒绝。`unindex` 暗示只删除搜索索引，但实际动作是删除 `files` row，并可能让 doc 进入 orphan 状态。

### 使用 `ingest` / `index`

拒绝。这两个词暴露内部实现阶段，不符合 MinerU 当前对用户隐藏 index 概念的产品语义。

### 目录支持 `--no-recursive`

拒绝。doclib 不记录目录 row，目录非递归没有清晰效果。后续如需要只处理一层文件，可单独设计 `depth`。

### forget watch root 时自动 remove watch

拒绝。`forget` 与 `watch remove` 是两个不同生命周期动作。隐式移除 watch 配置会造成不可预期副作用。

## 影响

- 需要新增 Interface / SDK / Client / Server 方法。
- 需要新增 CLI `mineru forget <path>`。
- 需要实现 `files` row 与 `fts_filenames` 的一致删除。
- 不修改 `docs` / `parses` / `fts_contents` / parsed JSON 生命周期。
- orphan doc cleanup 仍由 `mineru cleanup orphan-docs` 显式执行。
- active watch 下 forget 的路径可能被后续 scan 重新发现，必须通过 warning 告知。

## 后续动作

1. 在 doclib Interface 增加 `forget_path()`。
2. 在 HTTP server 增加 `POST /forget`。
3. 在 client 增加对应方法。
4. 在 CLI 增加 `mineru forget <path>`。
5. 增加测试覆盖:
   - forget 单个文件。
   - forget 目录树。
   - forget 不存在但 DB 中有历史记录的 path。
   - watch root 默认拒绝。
   - watch 下 path 返回 warning。
   - forget 不删除 docs/parses/fts_contents。
