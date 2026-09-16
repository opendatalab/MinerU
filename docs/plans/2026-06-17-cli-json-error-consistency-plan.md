# CLI JSON 模式报错一致性整改计划

**日期**: 2026-06-17
**状态**: 待实施
**范围**: `mineru/cli_next` 中所有已支持 `--json` 的命令，在业务错误和“服务未运行/无法连接”等场景下统一输出结构化 JSON；同时收口 `mineru server status --json` 在 server 未运行时的输出。
**非目标**:
- 不为当前尚未支持 `--json` 的 mutation 命令补加 `--json`
- 不改变 `parse --json`、`read --json` 的成功响应顶层 shape
- 不处理 Typer / Click 参数解析阶段的框架默认错误输出

## 1. 背景

当前 `cli_next` 中大多数命令已经提供 `--json`，但错误路径并不统一：

- `mineru parse --json` 已会输出结构化 JSON error。
- `read`、`search`、`find`、`watch`、`config`、`show`、`cleanup`、`forget`、`server status` 等命令，在业务错误下仍会输出纯文本错误。
- `mineru server status --json` 在 server 未运行时当前会输出 `"Server is not running."` 文本，而不是稳定 JSON。

这使得 `--json` 不能被稳定地当作机器接口使用。

## 2. 目标

统一规则如下：

1. 已支持 `--json` 的命令，在**进入命令实现之后**发生的业务错误，stdout 必须输出：

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "some_code",
    "message": "...",
    "param": "..."
  }
}
```

2. `stderr` 不得混入额外的人类解释文本；stdout 不得混入 `Error:`、`Written to`、Rich 样式文本或 traceback。

3. `mineru server status --json` 在 server 未运行时，不走 error JSON，而是返回稳定的状态 JSON，例如：

```json
{
  "running": false
}
```

理由：这不是异常，而是命令查询到的合法状态。

4. Typer / Click 在参数解析阶段产生的错误维持框架默认行为，当前阶段不在本计划内统一。

## 3. 影响命令清单

本计划覆盖当前已支持 `--json` 的命令：

- `mineru parse`
- `mineru read`
- `mineru scan`
- `mineru watch add/list/remove/rescan`
- `mineru search`
- `mineru find`
- `mineru list files/docs/parses/scans`
- `mineru show file/doc/parse/scan`
- `mineru server status`
- `mineru config show/get`
- `mineru config exclude-rules add/list`
- `mineru config parsing-rules add/list`
- `mineru forget`
- `mineru cleanup deleted-files/orphan-docs/temp`

不覆盖当前未支持 `--json` 的命令：

- `mineru invalidate`
- `mineru config set/unset`
- `mineru config exclude-rules remove`
- `mineru config parsing-rules remove`

## 4. 实施策略

### 4.1 抽取统一 helper

在 `mineru/cli_next` 中抽取统一的 JSON 错误输出 helper，职责：

- 接受 `Exception`
- 尝试复用 `parse` 已有的 `MineruError` / `error_response` 映射逻辑
- 在 `json_mode=True` 时输出结构化 JSON error
- 在 `json_mode=False` 时维持现有 `print_error(...)`

该 helper 应避免每个命令重复手写：

```python
except Exception as exc:
    print_error(str(exc))
    raise typer.Exit(1)
```

### 4.2 server status 特判

`mineru server status` 在 `json_mode=True` 且 server 未运行时：

- 直接输出稳定状态 JSON
- 不走 `print_info("Server is not running.")`
- 不走 error JSON

`json_mode=False` 时维持当前人类可读提示。

### 4.3 保持成功响应 shape 不变

本计划不调整成功响应结构，只整改错误路径：

- `parse --json` 继续使用 `{parse, content}`
- `read --json` 继续使用 `DocContentResponse`
- `list/search/show/... --json` 继续直接输出其 response model

## 5. 分步任务

1. 新增统一 JSON error helper，并将 `parse` 当前逻辑并入公共实现。
2. 将 `read`、`search/find`、`watch`、`scan`、`show`、`config`、`cleanup`、`forget` 接入该 helper。
3. 单独收口 `server status --json` 在 server 未运行时的稳定 JSON 输出。
4. 为各类命令补测试：
   - 业务错误下 stdout 为可解析 JSON
   - 不包含 `Error:` 文本
   - `server status --json` 在未运行时返回状态 JSON
5. 更新 CLI 文档和 E2E 用例中对 `--json` 错误行为的预期。

## 6. 测试矩阵

至少覆盖以下场景：

- `read --json` 读取不存在 locator
- `search --json` 非法 tier 或后端错误
- `watch add --json` 路径不存在
- `show parse --json` 不存在的 parse id
- `config get --json` 不存在的 key
- `cleanup temp --json` 非法 `older_than`
- `server status --json` 在 server 未运行时

断言重点：

- stdout 可直接 `json.loads(...)`
- 退出码符合预期
- stdout 不混入人类文本前缀

## 7. 风险与注意事项

1. 当前部分异常只是一段普通字符串，需要先尽量映射到稳定错误码；若无法映射，也应统一落到 `api_error` 或等价兜底 code，而不是把纯文本直接塞到 stdout。
2. 不能为了统一错误输出而改变成功输出 shape。
3. 不能把“server 未运行”误归类为错误 JSON；该场景属于状态查询结果。
