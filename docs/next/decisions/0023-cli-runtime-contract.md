# ADR-0023: CLI Runtime 输出、错误与退出码集中契约

状态: Accepted
日期: 2026-06-30
相关文档: ../cli.md, ../cli/mineru.md, ../errors.md, 0015-cli-output-json-composition.md

## 背景

MinerU CLI 目前包含顶层命令和多个子命令，例如 `parse`、`read`、`scan`、`watch rescan`、`show scan`、`config`、`cleanup` 等。这些命令都需要处理 stdout / stderr、`--json`、错误码和 exit code，但实现分散在各个 command 模块中。

已经出现过几类不一致：

- `--json` 模式下 stdout 混入人类可读文本，导致 stdout 不是合法 JSON。
- 部分错误路径直接 `print_error(...)` 后 `typer.Exit(1)`，绕过 JSON error envelope。
- 服务端结构化错误在 CLI 侧被 fallback 成 `api_error`。
- 触发并等待的任务最终 `failed` 时仍 exit `0`。
- 查询型命令和触发型命令都复用同一个输出函数，容易把资源状态和命令执行结果混淆。

这些问题靠 code review 很难长期避免。后续新增命令时，开发者需要同时记住 stdout / stderr 约定、JSON 输出形状、错误 envelope、连接失败映射、任务状态到 exit code 的转换规则，维护成本高。

因此需要把 CLI 输出、错误和退出码决策集中到一个 runtime 层。命令实现只表达业务动作和结果，不直接决定如何输出和退出。

## 决策

引入 CLI runtime 层，集中控制 stdout / stderr、JSON mode、错误 envelope 和 exit code。

目标模块结构：

```text
mineru/cli/
  runtime.py        # 统一输出、错误、退出码和命令包装
  contracts.py      # CliContext、CliResult、CliTaskResult 等类型
  output.py         # 底层输出原语，仅 runtime 使用
  commands/*.py     # 参数归一化、调用 client、就近定义一次性 renderer
```

普通命令模块不应直接调用：

- `print(...)`
- `typer.echo(...)`
- `print_error(...)`
- `print_json(...)`
- `print_success(...)`
- `emit_result(...)`
- `raise typer.Exit(...)`

普通命令模块应通过 `run_cli()` 执行业务 action。简单命令的 action 直接返回业务数据 `T`，复杂命令可以返回 `CliResult[T]` 或 `CliTaskResult[T]`。`run_cli()` 统一捕获异常、输出结果并处理 exit code。

### 输出契约

stdout 只承载命令核心结果。

规则：

- `--json` 成功时，stdout 必须只包含一个合法 JSON 对象。
- `--json` 错误时，stdout 必须只包含 JSON error envelope。
- 非 JSON 成功时，stdout 输出核心结果文本。
- notice、progress、warning、verbose 文本始终写 stderr。
- 非 JSON 错误时，stderr 输出人类可读错误文本。
- Typer / Click 参数解析错误可以保留框架默认行为和 exit code `2`。

JSON error envelope 继续使用既有形状：

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "file_not_found",
    "message": "...",
    "param": "path"
  }
}
```

### Exit Code 契约

CLI exit code 使用以下规则：

| 场景 | Exit code |
|------|-----------|
| 命令请求成功完成 | `0` |
| 命令本身失败，或触发并等待的任务最终失败 | `1` |
| Typer / Click 参数解析错误 | `2` |

资源状态不能直接等同于命令失败。必须区分查询型命令和触发型命令：

- `mineru show scan <failed_scan_id>` 查询成功，应 exit `0`。
- `mineru scan <path> --wait ...` 创建并等待的 scan 最终 `failed`，应 exit `1`。
- `mineru watch rescan <target> --wait ...` 创建并等待的 scan 最终 `failed`，应 exit `1`。
- `mineru parse <path> --no-wait` 返回 `pending` / `parsing` 是成功提交任务，应 exit `0`。
- `mineru server status --json` 在 server 未运行时返回 `running=false`，这是状态查询，应 exit `0`。

### Runtime API

使用泛型绑定 `data` 和 `render` 的类型。

```python
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar

from rich.console import ConsoleRenderable

T = TypeVar("T")
RichObject: TypeAlias = ConsoleRenderable
PlainObject: TypeAlias = str
RenderableObject: TypeAlias = RichObject | PlainObject
RenderableOutput: TypeAlias = RenderableObject | Iterable[RenderableObject]
CliRenderer = Callable[[T], RenderableOutput | None]


@dataclass(frozen=True)
class CliContext:
    json_mode: bool
    verbose: bool = False


@dataclass(frozen=True)
class CliResult(Generic[T]):
    data: T | None
    render: CliRenderer[T] | None = None
    exit_code: int = 0
    warnings: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CliTaskResult(CliResult[T]):
    status: str | None = None
    fail_statuses: frozenset[str] = frozenset({"failed"})
    fail_if_final_failed: bool = False
```

`data` 允许为 `None`，便于迁移没有结构化成功载荷的命令。runtime 规定：当 `data is None` 时不得调用 `render`。

公开给普通命令使用的构造函数：

```python
def cli_ok(
    data: T | None = None,
    *,
    render: CliRenderer[T] | None = None,
    exit_code: int = 0,
    warnings: list[str] | None = None,
    notices: list[str] | None = None,
) -> CliResult[T]:
    ...


def cli_task(
    data: T,
    *,
    status: str | None,
    render: CliRenderer[T] | None = None,
    fail_if_final_failed: bool,
    fail_statuses: frozenset[str] = frozenset({"failed"}),
    warnings: list[str] | None = None,
    notices: list[str] | None = None,
) -> CliTaskResult[T]:
    ...
```

runtime 内部或顶层包装使用：

```python
def run_cli(
    ctx: CliContext,
    action: Callable[[], T | CliResult[T]],
    *,
    render: CliRenderer[T] | None = None,
    warnings: Callable[[T], list[str]] | None = None,
    notices: Callable[[T], list[str]] | None = None,
    exit_code: int | Callable[[T], int] = 0,
) -> None:
    ...


def emit_result(ctx: CliContext, result: CliResult[T]) -> None:
    ...


def emit_error(ctx: CliContext, exc: Exception) -> NoReturn:
    ...
```

`emit_error()` 不作为普通命令的推荐公共 API。普通命令应通过抛出 `MineruError`、`InvalidRequestError`、client 异常或其它异常，让 `run_cli()` 统一捕获。`emit_error()` 只用于 runtime 内部、兼容旧代码和少量无法立即迁移的特殊入口。

不提供 `cli_error()`。错误路径统一使用异常，避免命令作者在 `return error` 和 `raise error` 之间做选择。

`server_not_running` 不通过 fallback 文案推断。Doclib client 应抛出 `ServerNotRunningError`，runtime 也可以识别明确的连接异常，例如 `httpx.ConnectError` / `httpx.ConnectTimeout`。

### 命令写法

迁移后的普通命令应接近以下结构：

```python
def show_file_cmd(path: str, *, json_mode: bool) -> None:
    ctx = CliContext(json_mode=json_mode)
    run_cli(
        ctx,
        lambda: DoclibClient(timeout=30).get_file_by_path(normalize_cli_path(path)),
        render=_render_file_info,
    )


def _render_file_info(data: FileInfoResponse) -> str:
    ...
```

简单命令不应为了包装 `cli_ok()` 额外定义 `_show_file() -> CliResult[FileInfoResponse]` 这类样板函数。只有当命令逻辑足够复杂、需要独立测试，或需要返回 `CliTaskResult` 时，才保留私有业务 helper。

触发并等待任务的命令应显式声明任务策略：

```python
def _scan(path: str, *, wait: int, no_wait: bool) -> CliTaskResult[ScanInfo]:
    scan_info = client.create_scan(...)
    if not no_wait and wait > 0:
        scan_info = wait_for_scan(...)
    return cli_task(
        scan_info,
        status=scan_info.status,
        render=render_scan,
        fail_if_final_failed=not no_wait,
    )
```

查询型命令即使展示 failed 状态，也使用 `cli_ok()`：

```python
run_cli(ctx, lambda: client.get_scan(scan_id), render=_render_scan)
```

### Renderer 规则

renderer 只把结构化数据转换为可渲染对象，不执行 IO，不抛 `typer.Exit`。可渲染对象包括字符串、Rich `ConsoleRenderable`、这些对象的 iterable 或 generator。renderer 返回 `None` 表示“非 JSON 模式下不输出核心结果”，不是表示 renderer 自己已经完成输出。

一次性 renderer 应就近放在对应 command 模块中：

```python
def _render_scan(scan: ScanInfo) -> Table | str:
    ...
```

跨命令复用的子格式化函数可以在后续需要时集中到单独模块；当前优先把一次性 renderer 放在调用处附近。

runtime 决定渲染结果写 stdout 还是 stderr；renderer 不关心 JSON mode。

## 替代方案

### 方案 A：继续依赖约定和 code review

不采用。

已有问题说明，分散约定很容易被遗漏。新增命令时需要记住太多细节，长期会继续产生不一致。

### 方案 B：只修当前发现的 bug

不采用作为长期方案。

逐点修复可以缓解当前问题，但无法降低未来新增命令的错误概率。CLI 输出和退出语义仍分散在各命令模块中。

### 方案 C：一次性重写所有 CLI 命令

不采用。

风险过高，容易误改查询型命令、触发型命令和 server lifecycle 命令之间的语义差异。迁移应分阶段进行，保证每一步都能独立验证。

### 方案 D：把 `emit_error()` 作为普通命令公共 API

不采用。

普通命令如果既可以 `return` 结果、又可以直接 `emit_error()`，业务逻辑和输出副作用仍会混在一起。错误路径应优先通过异常表达，由 `run_cli()` 统一捕获。

## 影响

### 对实现

- 新增 `contracts.py` 和 `runtime.py`。
- 现有 `output.py` 保留为底层输出原语，普通命令不直接调用。
- 一次性人类文本渲染保留在对应 command 模块中，便于就近阅读。
- 顶层单命令由 `main.py` 直接注册对应 command 模块函数，避免 `main.py wrapper -> *_cmd` 的重复函数层。

### 对命令开发

新增或修改普通命令时，开发者只需要：

1. 归一化 CLI 参数。
2. 调用 `DoclibClient` 或本地服务。
3. 调用 `run_cli()` 并提供业务 action 与 renderer。
4. 本地语义错误抛 `MineruError` / `InvalidRequestError`。

开发者不应手写 JSON、直接 print、直接 `typer.Exit`。

### 对兼容性

- 成功 JSON 的顶层 shape 不应因为 runtime 迁移改变。
- 既有错误 JSON envelope 形状保持不变。
- 非 JSON 文本可以在不改变语义的前提下微调，但不应破坏文档化命令输出。
- 部分之前错误 exit `0` 的触发型命令会收紧为 exit `1`，这是有意修复。

### 特殊例外

当前仅允许以下特殊路径保留输出副作用：

- `mineru/cli/runtime.py`
- `mineru/cli/output.py`
- `mineru/cli/telemetry.py` 的首次交互式 consent prompt；它必须跳过 `--json`、help、CI 和 agent caller 场景。

`mineru/cli/commands/parse.py` 仍有少量内部 `emit_result()`，这是剩余架构债，不属于可扩散模式。例外必须进入 contract test 或专项测试，不能隐式扩散。

## 执行计划

### Phase 1：建立 runtime 与契约测试

状态：已完成。

1. 新增 `mineru/cli/contracts.py`，定义 `CliContext`、`CliResult[T]`、`CliTaskResult[T]`。
2. 新增 `mineru/cli/runtime.py`，实现 `cli_ok()`、`cli_task()`、`run_cli()`、`emit_result()`、`emit_error()`。
3. 新增 `tests/unittest/test_cli_runtime_contract.py`，覆盖：
   - JSON 成功 stdout 为单个 JSON。
   - JSON 错误 stdout 为 error envelope。
   - 非 JSON 错误写 stderr。
   - notices / warnings 写 stderr。
   - `CliTaskResult` 在 `fail_if_final_failed=True` 且状态为 `failed` 时输出结果后 exit `1`。
   - `fail_if_final_failed=False` 时 failed 状态 exit `0`。
   - renderer 支持字符串、Rich 对象、iterable、generator 和 `None`。

### Phase 2：迁移已知高风险命令

状态：已完成。

已完成：

- `scan --wait` / `watch rescan --wait` 最终 failed 时 exit `1`。
- `show scan <failed>` 保持 exit `0`。
- `parse` / `read` 内容输出边缘错误已改为 JSON error envelope。
- config remove 和 invalidate 的 tuple 风格错误字符串已收敛为结构化错误输出。
- doclib server 未运行不再通过 fallback 文案推断，而是通过 `ServerNotRunningError` / 连接异常识别。

### Phase 3：收紧静态约束

状态：已完成。

新增 AST contract test，扫描 `mineru/cli/commands/**/*.py`：

- 禁止普通命令直接调用 `print(...)`。
- 禁止普通命令直接调用 `typer.echo(...)`。
- 禁止普通命令直接调用 `print_error(...)`、`print_json(...)`、`print_success(...)`。
- 禁止普通命令直接 `raise typer.Exit(...)`。

allowlist 仅包含本 ADR 的特殊例外。迁移期可以临时 allowlist 尚未迁移的命令，但每次新增 allowlist 都需要测试说明。

已增加的约束：

- 已迁移命令不得直接输出。
- 已迁移命令不得新增 `_xxx() -> CliResult[...]` 样板 helper。
- renderer 函数不得使用 `Any` 注解。
- command 模块不允许直接调用 `emit_result()`；`parse.py` 的内部调用作为后续收敛项单独处理。

### Phase 4：迁移剩余命令并清理兼容层

状态：已完成。

已迁移：

- `search`
- `find`
- `list parses/scans/files/docs`
- `show parse/scan/file/doc`
- `cleanup deleted-files/orphan-docs/temp`
- `forget`
- `telemetry status/enable/disable/flush`
- `config show/get/set/unset`
- `config exclude-rules add/list/remove`
- `config parsing-rules add/list/remove`
- `watch add/list/remove`
- `scan`
- `watch rescan`
- `read`
- `invalidate`
- `server start/stop/status/shutdown`
- `telemetry preview`
- `version`

后续清理：

- 收敛 `parse.py` 内部多次 `emit_result()` 为更标准的 runtime 结果返回。
- 若跨命令 renderer 复用明显增加，再引入共享 renderer/formatter 模块。

## 验证矩阵

迁移期间和迁移完成后，测试必须覆盖以下矩阵：

| 命令类型 | 示例 | 成功 exit | 业务 failed 状态 | JSON 错误 stdout |
|----------|------|-----------|------------------|------------------|
| 查询 / 展示 | `show scan` | `0` | `0` | envelope |
| 列表 / 搜索 | `list files`, `search` | `0` | 不适用 | envelope |
| 触发并等待 | `scan --wait`, `watch rescan --wait` | `0` | `1` | envelope |
| 只提交任务 | `parse --no-wait` | `0` | 不适用 | envelope |
| 内容输出 | `parse`, `read` | `0` | `1` | envelope |
| 配置变更 | `config set`, `config * remove` | `0` | `1` | envelope 若支持 `--json` |
| 状态查询 | `server status` | `0` | `0` | JSON 状态 |

Golden cases：

- `scan --wait --json` 返回 failed scan JSON 后 exit `1`。
- `show scan <failed> --json` 返回 scan JSON 后 exit `0`。
- `parse --json` 内容缺失返回 JSON error 后 exit `1`。
- `read --format image --output out.png --json` asset 缺失返回 JSON error 后 exit `1`。
- 所有进入命令实现后的 JSON 错误 stdout 都可 `json.loads`。
- verbose / progress / warning 不进入 JSON stdout。
- 连接失败统一映射为 `server_not_running`。

## 后续动作

1. 收敛 `parse.py` 内部直接 `emit_result()`。
2. 继续用命令级 golden tests 覆盖新增命令的 JSON / stderr / exit code 行为。
3. 更新 CLI 开发文档，说明新增命令必须使用 runtime。
