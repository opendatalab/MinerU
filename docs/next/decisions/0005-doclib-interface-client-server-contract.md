# ADR-0005: Doclib Interface、Client 与 Server 契约一致性

状态: Accepted
日期: 2026-06-11
相关文档: ../sdk/doclib-client.md, ../architecture.md, 0004-doclib-http-api-resources.md

## 背景

doclib server 的能力同时被多类入口使用:

- `mineru` CLI。
- Python SDK client。
- 本地 HTTP API。
- 未来的 MCP、Web 和桌面 App。

当前实现中，doclib 的能力定义分散在 client、FastAPI routes、service 和部分 Pydantic types 中。长期看会产生几个问题:

- Client 方法、Server route 和文档中的 API 容易漂移。
- 请求体和返回体没有一个稳定的主定义位置。
- 后续 Agent 实现功能时，需要从多处代码反推 doclib 能力边界。
- 同步 SDK 和异步 server 的运行模型不同，不能用单一接口粗暴统一。

需要定义一套更清晰的契约层，使 doclib 提供的能力、请求模型、响应模型、Client 实现和 Server 实现都可以被静态检查和运行时检查约束。

## 决策

doclib 引入同步接口和异步接口两套一等契约:

- `DoclibInterface`: 同步接口，面向同步 SDK client。
- `AsyncDoclibInterface`: 异步接口，面向 async SDK、FastAPI server 实现和异步 in-process 实现。

两套接口保持同名、同参数语义和同返回模型。差异只在同步/异步调用方式:

```python
class DoclibInterface(ABC):
    @abstractmethod
    def ensure_parse(self, request: ParseRequest) -> ParseResponse: ...


class AsyncDoclibInterface(ABC):
    @abstractmethod
    async def ensure_parse(self, request: ParseRequest) -> ParseResponse: ...
```

Interface 层只定义 doclib 能力契约，不包含 HTTP route 元数据，不依赖 FastAPI，也不出现 `Request`、`Response` 等 HTTP 实现概念。

### Schema 层

所有稳定请求和响应模型集中定义在 doclib interface 包中。建议按领域拆分:

```text
mineru/doclib/interface/
  __init__.py
  base.py
  schemas.py        # 或聚合导出
  parse.py
  docs.py
  search.py
  config.py
  server.py
  cleanup.py
  routes.py
```

复杂能力优先使用请求模型作为入参:

```python
def ensure_parse(self, request: ParseRequest) -> ParseResponse: ...
def invalidate(self, request: InvalidateRequest) -> InvalidateResponse: ...
```

GET 语义的方法不使用 request body model，必须使用 path/query 参数风格:

```python
def list_parses(
    self,
    *,
    ids: list[int] | None = None,
    sha256: str | None = None,
    tier: str | None = None,
    status: str | None = None,
    pages: str | None = None,
    include_superseded: bool = False,
) -> ListParsesResponse: ...
def get_parse(self, parse_id: int) -> ParseInfo: ...
def get_doc(self, sha256: str, *, expand_files: bool = False) -> DocInfo: ...
def get_doc_content(
    self,
    sha256: str,
    *,
    tier: str,
    pages: str | None = None,
    format: str = "markdown",
    output: str | None = None,
    no_marker: bool = False,
) -> DocContentResponse: ...
def search(
    self,
    query: str,
    *,
    file_type: str | None = None,
    tier: Tier | None = None,
    min_tier: Tier | None = None,
    limit: int = 20,
    offset: int = 0,
) -> SearchResponse: ...
def find(self, query: str, *, ext: str | None = None, limit: int = 50) -> FindResponse: ...
def get_file_info(self, path: str) -> FileInfoResponse: ...
```

第一版不以裸 `dict` 作为主契约。确实需要携带扩展字段时，应在 response model 中显式保留扩展字段，例如 `extra: dict[str, Any] = {}`。

### Route 元数据

HTTP route 元数据只挂在具体实现方法上，不挂在 interface 方法上。

Server 与 Client 使用同一个 `@route` 装饰器:

```python
@dataclass(frozen=True)
class RouteInfo:
    method: str
    path: str
    tags: tuple[str, ...] = ()


def route(method: str, path: str, *, tags: tuple[str, ...] = ()) -> Callable:
    def decorator(func: Callable) -> Callable:
        func._route_info = RouteInfo(method=method, path=path, tags=tags)
        return func

    return decorator
```

装饰器只挂载 `_route_info`，不注册 FastAPI route，不发起 HTTP 请求，也不区分 server/client。

Server 实现:

```python
class DoclibServer(AsyncDoclibInterface):
    @route("POST", "/parses", tags=("parse",))
    async def ensure_parse(self, request: ParseRequest) -> ParseResponse:
        ...
```

Client 实现:

```python
class MineruClient(DoclibInterface):
    @route("POST", "/parses", tags=("parse",))
    def ensure_parse(self, request: ParseRequest) -> ParseResponse:
        return self._request_typed(self.ensure_parse, request, ParseResponse)
```

这样做的原则是:

- 打开 Client 或 Server 方法时，可以直接看到 HTTP method 和 path。
- Interface 保持纯能力契约，不被 HTTP 绑定。
- Client 和 Server 的 route 字符串允许重复书写，但必须由测试强制保持一致。

### Server 注册方式

doclib server 启动时扫描 `DoclibServer` 实例上的实现方法:

1. 找到带 `_route_info` 的 callable。
2. 读取 method、path、tags。
3. 注册为 FastAPI endpoint。
4. endpoint 内部负责将 HTTP path/query/body 映射为 interface schema。
5. 调用 `DoclibServer` 实现方法。
6. 将 Pydantic response model 转换为 HTTP JSON response。

`DoclibServer` 可以继续调用现有 async service。Interface 只约束公开能力，不要求 service 层也继承 interface。

### Client 调用方式

`MineruClient` 继承 `DoclibInterface`。每个 HTTP-backed 方法也带同一个 `@route` 装饰器。

Client 方法通过自身方法对象读取 `_route_info`:

```python
def _request_typed(
    self,
    method_func: Callable,
    request: BaseModel | None,
    response_type: type[T],
) -> T:
    route_info = get_route_info(method_func)
    ...
```

Client 不从 Server 实现类读取 route 元数据，避免 SDK client 依赖 server 层或 FastAPI 相关模块。

### 一致性检查

需要增加运行时测试，作为契约漂移的硬约束。

必须检查:

- `DoclibInterface` 与 `AsyncDoclibInterface` 方法清单一致。
- 两套 interface 的同名方法参数一致，忽略 `self` 和 `async` 差异。
- 两套 interface 的同名方法返回模型一致。
- `MineruClient` 实现 `DoclibInterface` 的所有 abstract methods。
- `DoclibServer` 实现 `AsyncDoclibInterface` 的所有 abstract methods。
- `MineruClient` 与 `DoclibInterface` 的同名方法签名一致。
- `DoclibServer` 与 `AsyncDoclibInterface` 的同名方法签名一致。
- `MineruClient.method._route_info == DoclibServer.method._route_info`。
- 所有公开 interface 方法在 Client 和 Server 实现上都有 route 元数据。
- Server route 注册结果与实现方法上的 route 元数据一致。

签名比较规则:

- 忽略 `self`。
- 同步/异步差异不参与比较。
- 参数名、参数顺序、默认值、类型注解必须一致。
- 返回类型注解必须一致。
- Server 实现方法不得额外暴露 `Request`、`Response`、`BackgroundTasks` 等 FastAPI 注入参数。

### 能力范围

第一版 interface 覆盖 doclib 当前对外能力，包括用户能力和管理能力。

Parse / Docs:

- `ensure_parse`
- `list_parses`
- `get_parse`
- `invalidate`
- `list_docs`
- `get_doc`
- `get_doc_content`

Search / Info:

- `search`
- `find`
- `get_file_info`

Config:

- `get_config`
- `set_config`
- `add_watch`
- `list_watches`
- `remove_watch`
- `add_exclude_rule`
- `list_exclude_rules`
- `remove_exclude_rule`
- `add_parsing_rule`
- `list_parsing_rules`
- `remove_parsing_rule`

Server:

- `get_server_status`
- `shutdown_server`

Cleanup:

- `cleanup_deleted_files`
- `cleanup_orphan_docs`
- `cleanup_temp_files`

后续新增 MCP、Web 或 App 能力时，应优先复用这套 interface，而不是直接新增绕过 interface 的 HTTP route。

## 替代方案

### 方案 A: route 元数据挂在 interface 方法上

拒绝。这样会让 interface 层绑定 HTTP 语义，不利于 in-process、MCP 或非 HTTP 实现复用。

### 方案 B: route 只集中放在 registry 中

拒绝。集中 registry 一致性强，但读具体方法实现时看不到 HTTP method/path，不够直观，后续由 Agent 维护时容易漏看 registry。

### 方案 C: Client 从 Server 实现类读取 route 元数据

拒绝。这样会让 SDK client 依赖 server 实现层，可能间接引入 FastAPI 或 server-only 依赖，破坏 client/server 边界。

### 方案 D: Client 和 Server 各自写 route 字符串，无检查

拒绝。直观但缺少硬约束，不能解决 client/server 漂移问题。

## 影响

代码影响:

- 新增 `mineru/doclib/interface/` 包。
- `MineruClient` 改为继承 `DoclibInterface`，返回 typed response models。
- 新增 `DoclibServer` 或等价 async server implementation，继承 `AsyncDoclibInterface`。
- FastAPI routes 从手写 router 函数逐步迁移为扫描 server implementation 方法注册。
- 旧的分散 request/response models 迁移到 interface schema 层。

API 影响:

- HTTP path 仍遵循 ADR-0004 的资源模型。
- route 元数据成为 Client 和 Server 实现层的显式约束。
- 文档中的请求体、查询参数和返回体应从 interface schema 同步维护。

测试影响:

- 新增 interface sync/async 签名一致性测试。
- 新增 client/server route 元数据一致性测试。
- 新增 server route 注册 smoke test。
- 新增 typed response parse/serialization 测试。

兼容性影响:

- NEXT 版尚未发布，不需要保留旧 alias 或旧 response shape。
- 迁移期间可以保留旧 routes 文件作为 adapter，但稳定契约应以 interface 层为准。

## 后续动作

1. 创建 `mineru/doclib/interface/` 包，定义 `DoclibInterface`、`AsyncDoclibInterface`、`RouteInfo` 和 `route`。
2. 将当前 `mineru/doclib/types.py` 和 routes 内部 Pydantic models 迁移到 interface schema。
3. 按能力范围补齐请求/响应模型。
4. 改造 `MineruClient` 继承 `DoclibInterface` 并使用 `@route`。
5. 新增 `DoclibServer` 实现 `AsyncDoclibInterface` 并使用 `@route`。
6. 改造 FastAPI app 创建逻辑，扫描 `DoclibServer` 注册 routes。
7. 增加签名一致性、route 一致性和 OpenAPI smoke tests。
8. 删除或收敛旧 `routes/*.py` 中与新 server implementation 重复的 handler。
