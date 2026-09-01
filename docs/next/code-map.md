# 模块地图

状态: Draft
读者: 核心开发、编程 Agent、首次接触 4.x 代码的贡献者
范围: next 分支（4.x Alpha）的模块职责、入口与两条主调用链
基准: 本文件以 commit `317428b8`（2026-09-01，next 分支）为基准核实。

**维护规则**：结构性改动（模块新增/删除/职责迁移、入口或调用链变化）合并时必须同步更新本文件；纯函数级改动不需要。

## 1. 入口声明

`pyproject.toml:137-140` 声明 4 个 console script：

| 入口 | 指向 |
|---|---|
| `mineru` | `mineru.cli.main:main` |
| `mineru-kit` | `mineru.kit.main:main` |
| `mineru-router` | `mineru.kit.router.cli:main` |
| `mineru-gradio` | `mineru.cli_old.gradio_app:main` |

## 2. 模块职责表

| 模块 | 职责 | 关键锚点 |
|---|---|---|
| `mineru/cli/` | 面向 agent 的文档中心 CLI，16 个顶层命令（parse/read/scan/watch/search/find/usage/list/show/telemetry/server/config/invalidate/forget/cleanup/version，注册顺序见 `TOP_LEVEL_COMMAND_ORDER`） | `cli/main.py:22-40`、`cli/commands/*.py` |
| `mineru/doclib/` | 本地文档库常驻服务：入库、SHA256 去重、解析调度、缓存、搜索、配置。FastAPI + SQLite(WAL) + FTS5，路由→`services/`→`core/` 三层 | `doclib/server.py`（2439 行）；`doclib/services/{parse,search,scan,config,cleanup}_svc.py`；`doclib/core/{db,fts,file_io}.py` |
| `mineru/doclib/background/` | 后台 asyncio 任务：ingest（入库）、parse_worker（解析执行）、scan_worker / watch（目录监听）、parse_server_health（parse-server 托管与健康探测）、device_monitor、compaction（批次合并）、telemetry_flush | 各文件顶部 docstring；`background/parse_worker.py:15` |
| `mineru/parser/` | parse-server 与解析门面：`api_server.py`（2495 行，独立进程）、`api_client.py`（HTTP 客户端）、`tier.py`（tier/backend 解析）、`mineru_parser.py`（`MinerUParser` 统一解析入口）、`base.py`（Middle JSON schema） | `parser/mineru_parser.py`、`parser/tier.py` |
| `mineru/backend/` | 解析引擎：`analyze.py`（`doc_analyze` 统一门面）→ `analysis/`（按格式分派：`pdf/`、`csv.py`、`epub.py`、`html.py`、`ofd.py`、`office.py`）→ `postprocess/`（Middle JSON 组装、段落/标题/列表/表格合并、`llm_aided.py`、`legacy_schema_adapter.py`） | `backend/analyze.py`、`backend/analysis/pdf/pipeline.py` |
| `mineru/model/` | 模型层：`flash/`（纯 CPU 本地结构化解析器族：pdf / office / epub / html / ofd / csv）；`runtime/`（`device.py` 设备与 light/full 栈、`hybrid.py` 原子模型分派、`onnx.py`）；`layout/ mfr/ ocr/ table/ vlm/`（ML 模型实现）；`_internal/pytorchocr` | `model/runtime/device.py:48`、`model/runtime/hybrid.py` |
| `mineru/render/` | 输出渲染：markdown / content_list(v2) / structured_content / docx / epub / html / pdf | `render/api.py`（唯一渲染入口）、`render/contracts.py` |
| `mineru/kit/` | 服务端工具 CLI：`parse` / `api-server` / `vlm-server` / `router` / `models`，面向自部署/多卡部署 | `kit/main.py`、`kit/commands/*.py` |
| `mineru/cli_old/` | 3.x CLI 原样迁入的兼容层（client / fast_api / gradio_app / router 等），仅改导入路径与裁剪 | 仅 `mineru-gradio` 入口引用 |
| 顶层类型/配置 | `types.py`（1317 行，Tier / locator / 各 Response 模型）、`errors.py`（错误协议）、`config.py`（MINERU_HOME、UDS/TCP、model stack）、`filetypes.py`（扩展名→tier 语义） | `config.py`（`Config`）、`errors.py:20` |
| `docs/next/` | 4.x 全套设计文档：architecture / tiers / errors / middle-json + 33 篇 ADR（`decisions/0001-0033`）+ 实施计划 `docs/plans/` | `docs/next/README.md` |
| `tests/unittest/` | 约 2372 个测试函数 | — |

## 3. 主调用链

### 3.1 解析链（`mineru parse <file>`）

```
cli/commands/parse.py parse_cmd（--remote 选项 :100）
  → DoclibClient（doclib/client.py）经 UDS（$MINERU_HOME/doclib.sock；UDS 不可用回退
    TCP 127.0.0.1:15980，见 config.py 与 doclib/endpoint.py）
  → doclib/server.py → ParseService.submit（services/parse_svc.py:726：校验 tier/privacy/
    page_range/扩展名；privacy = "remote"|"local" :754）写入 parses 表任务队列
  → ParseWorkerPool（background/parse_worker.py:15）→ process_doc（parse_svc.py:931）按 tier 分派：
    · flash   → 进程内本地 flash 解析（parse_svc.py:971 附近，"local(flash)"）
    · basic/standard/advanced → _parse_via_api（parse_svc.py:1146）三选一：
        managed（doclib 自动拉起 `mineru-kit api-server --tier … --no-flash --preload-models`，
                 background/parse_server_health.py:98-113；TCP 127.0.0.1:16580）
        self_hosted（parse_server.local.self_hosted_url）
        remote（parse_server.remote.url）
      均不可用时抛 no_engine（parse_svc.py:1239）
  → parse-server 侧：parser/api_server.py → MinerUParser（parser/mineru_parser.py，按后缀路由）
    → doc_analyze（backend/analyze.py）→ analysis/<格式> → postprocess → MiddleJson(schema 2.0)
    → 解析产物写回 doclib（background/compaction.py 合并批次）
```

### 3.2 读取链（`mineru read <locator>`）

```
cli/commands/read.py:25 read_cmd
  → DoclibClient.read_content
  → doclib/server.py:684 _build_read_plan_from_locator（locator 解析在 :1484 _parse_doc_locator）
  → 从缓存页取内容、按 --limit 截断
  → 生成 ContentNextRequest（server.py:2142 _next_content_request；模型定义 doclib/types.py:293）
  → CLI 侧拼 `<!-- Next: mineru read … -->` 续读标记（cli/commands/read.py:164）
```

## 4. 相关文档

- 陷阱与兼容面联动：[pitfalls.md](pitfalls.md)
- 不变量红线：[invariants.md](invariants.md)
- 系统架构与设计：[architecture.md](architecture.md)、[decisions/README.md](decisions/README.md)
