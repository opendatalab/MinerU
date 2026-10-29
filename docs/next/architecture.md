# 当前架构与实现归属

核对版本：4.1.2。具体文件入口见 [模块地图](code-map.md)，公共行为见双语用户文档。
本页描述现有调用关系，不混入未实现的产品方案。

## 文档与推理

DocVortex 拥有原生格式解析、PDF 基础访问、schema、确定性后处理、素材、七种通用 renderer 与导出。
MinerU 拥有 OCR/VLM/Hybrid、模型生命周期、可选 LLM 增强、tier 策略、CLI/API/Gradio/Doclib，以及 Content List V1/V2。
依赖单向从 MinerU 指向 DocVortex 的公开接口。

`MinerUParser` 准备本地输入并调用 `backend/analyze.py` 的统一同步／异步门面。
原生格式进入 DocVortex；PDF 按 effort 与 parse_mode 选择原生或推理路径。
确定性后处理形成 MiddleJson 后才执行 MinerU 可选增强。ModelJson 只用于内部分析。

## 两类服务

| 服务 | 入口 | 状态范围 |
|---|---|---|
| V1 API Server | `mineru/parser/api_server.py`、`mineru/tasks/` | 上传、逐文件执行、结果 Files、租约；任务和索引在当前进程内 |
| Router | `mineru/kit/router/` | worker 发现、认证、调度、公共 ID 映射、源缓存与结果租约 |
| file-parse | `mineru/kit/file_parse/` | SQLite 保存 request_id 及上游引用，支持有条件恢复 |
| Doclib server | `mineru/doclib/app.py`、`server.py`、`client.py` | 本地文档身份、路径、解析缓存、索引和运行时设置 |

V1 与 Doclib HTTP 接口不是同一协议。Doclib 使用 `/api/v1`，客户端通过 endpoint 文件发现 UDS 或 loopback TCP。
文件类型、请求默认档位和缓存读取的选择规则见 [档位](tiers.md)。

## Doclib 服务分工

| 模块 | 责任 |
|---|---|
| `services/ingest_svc.py` | 路径入库、源内容身份与基础信息 |
| `services/parse_queue_svc.py` | 请求校验、缓存覆盖与排队 |
| `services/parse_execution_svc.py` | 调用解析后端、保存产物并更新索引 |
| `services/content_read_svc.py` | 缓存读取、locator、分页内容与导出 |
| `background/` | 入库、解析、扫描、watch、compaction、健康检测和 telemetry flush |
| `core/middle_json.py` | 已持久化历史 Middle JSON 的唯一兼容读取边界 |

缓存保存 Middle JSON 和需要的 sidecar，内容读取按需渲染。
Doclib 图像读取可在源文件仍可访问时按定位信息生成图像；这不意味着通用 ParseResult 持有源 PDF 对象。
`read` 不自动新建解析。缓存协议、索引和几何变更需验证实际读取及导出结果。

## 配置和生命周期

启动配置按 [配置优先级](config.md) 读取；Doclib 运行时 KV 和规则存储在 SQLite。
AppState 持有服务实例，跨实例配置显式传入。只有获得所属资源的实例才负责释放。
上传截止、执行状态、结果保留、下载活动引用和取消各有独立生命周期；取消后迟到结果不能改写终态。

公开错误按 [接口边界](errors.md) 处理。没有远端许可时，不以错误恢复为由上传文档。
未完成设计集中在 [待办](backlog.md)，历史过程记录保存在仓库档案。
