# MinerU SDK 设计（初稿）

> **状态**：初稿，待团队评审。

---

## 1. 两层 SDK

| SDK 层 | 代码位置 | 职责 | 消费者 |
|--------|---------|------|--------|
| **Tool SDK** | `mineru.parser` | 纯解析能力（调引擎、格式转换），无状态 | `mineru-kit` CLI、Server 内部 parse worker |
| **Product SDK** | `mineru.client` | 与本地 Server 通信（提交解析、查缓存、搜索、管理） | `mineru` CLI、MCP Server、桌面端 |

两层 SDK 解析接口的返回 JSON 本质相同，CLI 只做格式化（rich/marker/截断等）。

---

## 2. 包结构

```
mineru/
├── __init__.py
├── version.py
├── errors.py                      # [新建] 错误码定义（对应 NEXT-ERROR.md）
├── constants.py                   # [新建] 枚举、文件类型白名单
├── types.py                       # [新建] 共享 Pydantic 模型（请求/响应契约）
├── client.py                      # [新建] Product SDK（httpx over UDS）
│
├── parser/                        # [从 api/ 重构] Tool SDK
│   ├── __init__.py                # 导出 parse(), ParseResult 等
│   ├── pdf.py                     # PdfPipelineParser, PdfVlmParser, PdfHybridParser
│   ├── office.py                  # DocxParser, PptxParser, XlsxParser
│   ├── html.py                    # HtmlParser
│   ├── engines.py                 # 引擎发现、tier → backend 映射
│   ├── result.py                  # ParseResult
│   ├── types.py                   # PageInfo, Block, Line, Span
│   └── api_server.py              # mineru-kit api-server（无状态 HTTP 解析服务，兼容 SaaS API 格式）
│
├── doclib/                         # 本地文档库（原 server/）
│   ├── __init__.py
│   ├── server.py                   # FastAPI app factory, lifecycle
│   ├── config.py                  # doclib 配置（UDS 路径、端口、SQLite 路径等）
│   ├── routes/                    # HTTP 路由
│   │   ├── parse.py
│   │   ├── search.py
│   │   ├── info.py
│   │   └── config.py
│   ├── services/                  # 业务逻辑
│   │   ├── parse_svc.py
│   │   ├── search_svc.py
│   │   └── config_svc.py
│   ├── core/                      # 数据层
│   │   ├── db.py                  # SQLite + migrations
│   │   ├── fts.py                 # FTS5 操作
│   │   └── file_io.py             # SHA-256, metadata
│   └── background/                # 后台任务
│       ├── watch.py               # 文件系统监控
│       ├── parse_worker.py        # 解析任务 worker
│       └── device_monitor.py      # 可插拔设备检测
│
├── cli_next/                      # [新建，迁移期临时] 新 CLI
│   ├── __init__.py
│   ├── mineru_cmd.py              # `mineru` 主命令（typer）
│   ├── kit_cmd.py                 # `mineru-kit` 主命令（typer）
│   ├── output.py                  # Rich 格式化 + JSON 输出切换
│   └── commands/                  # 子命令分组
│       ├── parse.py               # mineru parse
│       ├── server.py              # mineru server start/stop/status
│       ├── search.py              # mineru search / find
│       ├── info.py                # mineru info
│       ├── config.py              # mineru config
│       └── kit_parse.py           # mineru-kit parse
│
├── cli/                           # [现有，迁移完成后删除]
├── schema/                        # [新建] JSON Schema + 校验
│   ├── middle_json/
│   │   ├── v1.json
│   │   └── v0.json
│   ├── validate.py
│   └── migrate.py
├── backend/                       # [不动] 四个解析 backend
├── model/                         # [不动] ML 模型定义
├── utils/                         # [不动] 工具函数
└── resources/                     # [不动] 静态资源
```

---

## 3. 依赖方向

```
cli_next/mineru_cmd  →  client       →  doclib/  →  parser/  →  backend/
                                                     │
                                                     ├── flash tier: 直接调用
                                                     └── standard/pro tier: HTTP 调用
                                                           ├─ mineru-kit api-server (本地)
                                                           └─ mineru.net/api (远程)
cli_next/kit_cmd     →  parser/      →  backend/
mineru-kit api-server → parser/api_server.py     →  parser/  →  backend/
```

- `mineru-kit` 只依赖 `parser/`，不碰 `client` 和 `doclib/`
- `mineru` 通过 `client` 与 doclib 通信，doclib 内部调用 `parser/` 做 flash 解析，或通过 HTTP 调用 parse-server 做 standard/pro 解析
- `parser/api_server.py` 是无状态 HTTP 解析服务（`mineru-kit api-server`），与 `doclib/` 完全独立。它实现 NEXT-API.md 的 Files/Uploads/Jobs 端点，被 doclib 的 ParseWorker 通过 HTTP 调用
- `errors.py`、`constants.py`、`types.py` 是共享基础模块，被各层引用

---

## 4. 从现有代码的迁移路径

| 现有 | → 目标 | 策略 |
|------|--------|------|
| `mineru/api/` | `mineru/parser/` | 重构接口签名（加入 tier），保留解析逻辑 |
| `mineru/api/api_parser.py` | `mineru/parser/remote.py` | MinerUApiParser 移入，改用新 API 格式 |
| `mineru/cli/` (click) | `mineru/cli_next/` (typer) | 新写，迁移完成后 cli/ 删除、cli_next/ 改名 cli/ |
| `mineru/cli/fast_api.py` | 删除 | 被 `doclib/` 替代 |
| `mineru/parse_server.py` | `mineru/parser/api_server.py` | 移入 parser 包，属于 mineru-kit api-server |
| `mineru/doclib/` (原 server/) | 已实现 | — |
| `mineru/config.py` | 已实现 | 全局启动配置模块，包含 doclib 启动前配置 |

---

## 5. Tool SDK 接口（mineru.parser）

### 5.1 公开接口

```python
from mineru.parser import parse, ParseResult, PageInfo, Block, Line, Span

def parse(
    path: str | Path,
    *,
    tier: str | None = None,      # flash / standard / pro; None means default selection
    page_range: str | None = None,     # 页码范围，如 "1~5" / "all"
    lang: str = "ch",
    formula: bool = True,
    table: bool = True,
    image_analysis: bool = True,
    output_dir: str = "./output",
    # 高级参数（mineru-kit 暴露，mineru 隐藏）
    backend: str | None = None,   # 指定后覆盖 tier 映射
    method: str = "auto",
    server_url: str | None = None,
) -> ParseResult: ...
```

### 5.2 tier → backend 映射（engines.py）

```python
TIER_MAP = {
    "flash": "cpu-extract",
    "standard": "pipeline",
    "pro": "hybrid-auto-engine",
}

def resolve_engine(tier: str, backend: str | None = None) -> str:
    """确定实际使用的 backend。backend 显式指定时覆盖 tier。"""
    if backend:
        return backend
    return TIER_MAP[tier]
```

### 5.3 ParseResult

```python
class ParseResult:
    """解析结果，提供多种输出格式。"""

    def markdown(self, *, page_range: str | None = None, marker: bool = True) -> str: ...
    def text(self, *, page_range: str | None = None) -> str: ...
    def json(self) -> dict: ...          # middle_json
    def html(self) -> str: ...
    def pages(self) -> list[PageInfo]: ...
    def page_count(self) -> int: ...
```

---

## 6. Product SDK 接口（mineru.client）

```python
from mineru.client import MineruClient

class MineruClient:
    """与本地 mineru server 通信的客户端。"""

    def __init__(self, socket_path: str | None = None): ...

    # 解析
    def parse(
        self,
        path: str | Path,
        *,
        tier: str | None = None,
        page_range: str | None = None,
        force: bool = False,
        wait: int = 60,
    ) -> ParseResponse: ...

    # 查询
    def info(self, path: str | Path) -> FileInfo: ...
    def search(
        self,
        query: str,
        *,
        file_type: str | None = None,
        tier: str | None = None,
        min_tier: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[SearchResult]: ...
    def find(self, query: str, *, ext: str | None = None, limit: int = 50) -> list[FileResult]: ...

    # Server 管理
    def server_status(self) -> ServerStatus: ...

    # 配置
    def config_show(self) -> dict: ...
    def config_watch_add(self, path: str, *, removable: bool = False) -> None: ...
    def config_watch_list(self) -> list[WatchTarget]: ...
    def config_watch_rm(self, path: str) -> None: ...
```

---

## 7. 待补充

- 数据库 DDL（完整建表 SQL，multi-tier page_range 覆盖设计）
- Server 内部路由表（UDS 端点清单）
- 测试策略（fixture 列表、CI 配置）
- 依赖声明（新增 typer、httpx、watchfiles 等）
