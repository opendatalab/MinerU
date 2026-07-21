# ADR-0021: Doclib 本地 Transport 与 Endpoint Discovery

状态: Accepted
日期: 2026-06-22
相关文档: ../architecture.md, ../config.md, ../cli/mineru-server.md, ../sdk/doclib-client.md

## 背景

doclib server 是本地常驻进程，服务 CLI、SDK、MCP、桌面端和未来 Web UI。当前实现默认通过 Unix domain socket 连接 doclib，同时可以额外启用 TCP listener。

这个模型在 macOS 和 Linux 上工作良好，但面向 Windows 用户时存在风险:

- Windows 版本本身可能支持 Unix domain socket，但用户安装的 Python distribution build 未必启用 `socket.AF_UNIX`。
- 当前配置在 import 阶段就会计算默认 UDS path，如果 UDS 不可用可能导致 CLI / SDK 在启动前失败。
- 当前 server 启动时无条件 bind UDS，不能只使用 TCP。
- 当前 client 默认只使用 UDS，无法自动切换到 TCP。
- TCP 端口可能因为占用而 fallback 到随机端口，client 需要一种稳定方式发现实际 endpoint。

为了面向 Windows 用户大规模推广，doclib 不能要求用户电脑上的 Python 一定支持 UDS。需要将 UDS 从“必需 transport”调整为“优先 transport”，并提供跨平台 fallback。

## 决策

doclib 本地连接采用 “UDS 优先，TCP loopback 自动兜底” 的策略。

### 配置命名

将启动前配置中的 `doclib.http` 改名为 `doclib.tcp`。不兼容旧配置项，因为当前仍处于第一版实现阶段。

原因:

- UDS 和 TCP 上运行的都是 HTTP + JSON 协议。
- `http.enabled` 容易被误解为是否启用 HTTP API。
- 该配置实际控制的是 TCP listener，命名为 `tcp` 与 `uds` 对称。

配置结构:

```yaml
doclib:
  uds:
    enabled: true
    path: ~/.mineru/doclib.sock
    permission: 0o600
  tcp:
    enabled: false
    host: 127.0.0.1
    port: 15980
    strict_port: false
    backlog: 128
    timeout: 600
```

### 默认启用规则

`enabled` 字段继续使用 `bool`，不引入字符串三态，也不使用 `None` 表示 auto。

`doclib.uds.enabled` 和 `doclib.tcp.enabled` 的配置类型为 `auto | true | false`，默认值均为 `auto`。当前第一版只在启动前解析 `auto`，不在 UDS bind 失败后继续 fallback:

- 如果当前 Python runtime 支持 UDS:
  - `doclib.uds.enabled=auto` 解析为 `true`
  - `doclib.tcp.enabled=auto` 解析为 `false`
- 如果当前 Python runtime 不支持 UDS:
  - `doclib.uds.enabled=auto` 解析为 `false`
  - `doclib.tcp.enabled=auto` 解析为 `true`

用户显式配置 `true` / `false` 时，server 尊重用户配置。

如果用户显式或默认结果导致 UDS 和 TCP 都关闭，server 启动失败，并提示至少需要启用一个 local transport。

如果用户显式启用 UDS，但当前 Python runtime 不支持 UDS，server 启动失败，并提示可以关闭 UDS 或启用 TCP。

### UDS path 不再承担能力检测

默认 UDS path 始终可以计算:

```text
$MINERU_HOME/doclib.sock
```

`_default_uds_path()` 不再因为 UDS 不可用而抛错。UDS 能力检测只影响 `uds.enabled=auto` 和 `tcp.enabled=auto` 的启动前解析结果，以及 server 启动时的 bind 逻辑。

### TCP listener

TCP 第一版只作为本地 loopback transport:

- 默认 host 为 `127.0.0.1`。
- 默认 port 为 `15980`。
- `strict_port=false` 时，如果默认端口被占用，server 可以绑定随机可用端口。
- 实际端口通过 endpoint discovery 文件暴露给 client。

如果用户显式配置非 loopback host，第一版允许，但视为高级配置。第一版不额外实现 token 或远程访问防护。

### Endpoint discovery 文件

server 启动成功后写入:

```text
$MINERU_HOME/doclib.endpoint.json
```

该文件描述当前 server 实例可用的本地连接方式。

示例:

```json
{
  "version": 2,
  "pid": 12345,
  "server_id": "df971716-36c7-4e4d-b585-b798331ec7f4",
  "transports": [
    {
      "type": "uds",
      "path": "/Users/me/.mineru/doclib.sock"
    },
    {
      "type": "tcp",
      "base_url": "http://127.0.0.1:15980"
    }
  ]
}
```

规则:

- 只写入实际成功绑定的 transports。
- `server_id` 是每次 server 启动生成的随机 UUID，并由 `/server/status` 返回。
- TCP 使用随机端口时，`base_url` 必须写入真实端口。
- 文件写入应使用 atomic write: 先写临时文件，再 `replace`。
- server shutdown / stop 时 best-effort 删除该文件。
- stale endpoint 文件不能被视为 server 一定运行；client 连接失败后仍应报 `ServerNotRunningError`。

### Client 连接策略

`DoclibClient()` 默认走 endpoint resolver，而不是固定使用 UDS。

公开构造参数调整为:

```python
class DoclibClient:
    def __init__(
        self,
        *,
        endpoint_path: str | Path | None = None,
        socket_path: str | Path | None = None,
        base_url: str | None = None,
        timeout: int = 60,
        api_prefix: str = "/api/v1",
    ) -> None: ...
```

参数语义:

- `endpoint_path`: endpoint discovery 文件路径。默认 `None` 表示 `$MINERU_HOME/doclib.endpoint.json`。
- `socket_path`: 显式指定 UDS path。传入后不读取 endpoint discovery。
- `base_url`: 只表示 TCP 模式下的真实 base URL，例如 `http://127.0.0.1:15980`。不再承担 UDS dummy URL 的公开语义。
- `timeout`: HTTP client timeout。
- `api_prefix`: doclib API path prefix。

`socket_path` 和 `base_url` 都是显式 endpoint override，不能同时传入；如果同时传入，client 应抛出 `ValueError`。

UDS 模式下 httpx 仍然需要一个 base URL。这个 dummy URL 使用内部常量，例如:

```python
DOCLIB_UDS_BASE_URL = "http://mineru"
```

普通用户不需要关心该常量。

默认连接顺序:

1. 如果调用方显式传入 `socket_path`，使用该 UDS path。
2. 如果调用方显式传入 `base_url`，使用该 TCP endpoint。
3. 否则读取 `$MINERU_HOME/doclib.endpoint.json`。
4. endpoint 中同时存在 UDS 和 TCP 时，优先尝试 UDS。
5. UDS 不存在、不可用或连接失败时，尝试 TCP。
6. endpoint 不存在、无效或不包含可用 transport 时，不根据当前 config 推导 transport。
7. 第一次使用候选 transport 前，通过 `/server/status` 校验其 `server_id` 与 endpoint 一致。
8. 身份不匹配时继续尝试下一个 transport；所有候选均不匹配时抛出 `server_instance_mismatch`。
9. 没有候选 endpoint 或所有候选 endpoint 都无法连接时，抛 `ServerNotRunningError`。

显式 `socket_path` 或 `base_url` 代表调用方明确选择的 server，不执行 endpoint `server_id` 校验。

CLI 的 `server status`、`server stop` 和业务命令都应使用同一套 resolver，不再依赖 socket 文件是否存在判断 server 是否运行。

### Server status

状态模型中的 `http` 字段改为 `tcp`，CLI human-readable 输出也显示 `TCP`。状态中的 `server_id` 标识当前
server 进程，并用于默认 endpoint discovery 的身份校验；`pid` 和 `mineru_home` 只用于诊断。

`server status --json` 应继续返回:

- `socket_path`: UDS path，可能为空或仅表示配置 path。
- `tcp.enabled`
- `tcp.host`
- `tcp.port`

其中 `tcp.port` 表示实际绑定端口；如果 TCP 未启用则为 `null`。

### 本地 TCP 不实现 token

第一版本地 TCP 不实现 token，不写入 token，不要求 client 携带 token header。

安全边界由以下约束提供:

- 默认只监听 `127.0.0.1`。
- 本地 doclib 主要面向同一用户会话内的 CLI / SDK / Agent。
- 非 loopback host 需要用户显式配置，属于高级使用场景。

后续如果要支持更开放的本地服务场景，可以在新 ADR 中讨论 token、mTLS、系统凭据或 named pipe。

## 替代方案

### 方案 A: 继续要求 UDS

拒绝。Windows 用户的 Python distribution build 不一定启用 `AF_UNIX`，要求 UDS 会让 CLI / SDK 在部分 Windows 环境下不可用。

### 方案 B: 将 `enabled` 改为字符串三态

示例:

```python
Literal["auto", "enabled", "disabled"]
```

拒绝。它能明确表达 auto，但会改变现有配置字段类型，用户配置也更啰嗦。第一版只需要“未配置时自动选择，显式配置时尊重用户”，用 bool + default factory 足够。

### 方案 C: 使用 `bool | None` 表示 auto

示例:

```python
enabled: bool | None = None
```

拒绝。它保留字段名，但 `None` 的业务语义不够自解释，后续实现容易在多个调用点散落 `is None` 判断。第一版选择把 auto 收敛到默认值生成逻辑中。

### 方案 D: Windows 使用 Named Pipe

暂不采用。Named Pipe 的安全属性更接近 UDS，但当前 doclib 基于 uvicorn / httpx 的 HTTP transport 实现，Named Pipe 会引入较多平台特化代码。TCP loopback fallback 成本更低，覆盖面更好。

### 方案 E: 本地 TCP 强制 token

暂不采用。token 可以降低本机低成本误连或误发风险，但也会增加 endpoint 文件、client header、调试和文档复杂度。第一版先限制默认 host 为 `127.0.0.1`，不实现 token。

## 影响

配置影响:

- `doclib.http.*` 改为 `doclib.tcp.*`。
- 环境变量从 `MINERU_DOCLIB_HTTP_*` 改为 `MINERU_DOCLIB_TCP_*`。
- 不兼容旧配置项。

Server 影响:

- server 启动时按 `uds.enabled` 和 `tcp.enabled` 分别创建 socket。
- server 允许只启 TCP。
- server 写入并清理 `doclib.endpoint.json`。
- UDS 不可用不再导致 config import 失败。

Client / CLI 影响:

- `DoclibClient()` 默认使用 endpoint resolver。
- CLI 不再用 socket 文件存在性判断 server 运行状态。
- `server status`、`server stop`、业务命令使用同一套连接发现逻辑。

文档影响:

- 更新 architecture / config / server CLI / SDK 文档中的 UDS-only 描述。
- 明确 UDS 与 TCP 都承载 HTTP + JSON 协议。
- 明确 Windows fallback 行为。

测试影响:

- 增加 UDS 可用时默认只启 UDS 的配置测试。
- 增加 UDS 不可用时默认启 TCP 的配置测试。
- 增加 server 只启 TCP 的启动测试。
- 增加 endpoint 文件写入、随机端口写入、shutdown 清理测试。
- 增加 client resolver 优先 UDS、fallback TCP、stale endpoint 失败测试。
- 更新 `MINERU_DOCLIB_HTTP_*` 相关测试为 `MINERU_DOCLIB_TCP_*`。

## 后续动作

1. 修改配置模型: `HTTPConfig` 改为 `TCPConfig`，`DoclibConfig.http` 改为 `DoclibConfig.tcp`。
2. 为 `UDSConfig` 和 `TCPConfig` 使用 `auto | true | false` 类型，默认值为 `auto`，并在启动时解析为布尔值。
3. 移除 `_default_uds_path()` 中的 UDS 能力检测和异常。
4. 修改 server bind 逻辑，允许 UDS / TCP 独立启用。
5. 实现 `$MINERU_HOME/doclib.endpoint.json` 的 atomic write 和 cleanup。
6. 实现 `DoclibClient` endpoint resolver。
7. 修改 CLI server lifecycle 使用 resolver。
8. 修改 status schema 和 CLI 输出: `http` -> `tcp`。
9. 更新文档和 E2E 测试用例。
