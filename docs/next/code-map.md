# 模块地图

核对版本：MinerU 4.1.2。模块职责变化时同步更新本页；文件长度、测试总数与旧提交行号不作为定位依据。

## 入口和职责

| 模块 | 当前职责与入口 |
|---|---|
| `mineru/cli/` | 文档库 CLI；`main.py` 显式组织 parse/read/search/scan/watch/server 等入口 |
| `mineru/kit/commands/` | parse、api-server、file-parse-server、vlm-server、models、hardware、webui 等工具命令 |
| `mineru/parser/` | 本地与远程解析门面；`base.py` 的 ParseResult、`mineru_parser.py`、`api_client.py`、`api_server.py` |
| `mineru/tasks/` | `contracts.py` 共享协议、`service.py` 任务编排、`execution.py` 执行、`http.py` 路由 |
| `mineru/parser/task_backend.py`、`task_outputs.py` | 本地解析与 Tasks 的适配、多格式结果打包 |
| `mineru/backend/` | `analyze.py` 统一分析门面；PDF OCR/VLM/Hybrid；原生格式通过 DocVortex 公开接口；后处理追加可选 LLM 增强 |
| `mineru/model/` | layout/OCR/公式/表格/VLM 推理；`runtime/` 管理设备、ONNX 选型和模型生命周期；registry/download 管理模型资产 |
| `mineru/integrations/docvortex.py`、`mineru/types.py` | 宿主产品元数据校验和共享文档类型重导出；原生解析、PDF 访问及共享 schema 的实现属于 DocVortex |
| `mineru/render/` | 九种输出门面；Content List V1/V2 属于 MinerU，其他七种渲染与通用导出能力复用 DocVortex |
| `mineru/doclib/` | app/server/interface/client、SQLite/FTS、transport discovery 和有状态文档库 |
| `mineru/doclib/services/` | ingest_svc 刷新入库；parse_queue_svc 排队与失效；parse_execution_svc 执行与索引；content_read_svc 缓存读取与导出 |
| `mineru/doclib/core/` | DB、FTS、parse_artifacts；middle_json 是历史缓存的唯一读取入口 |
| `mineru/doclib/background/` | ingest、parse、scan/watch、设备监测、服务健康检查、compaction 与 telemetry flush |
| `mineru/kit/router/` | 多 worker 调度、认证、源缓存、结果路由及租约 |
| `mineru/kit/file_parse/` | Tasks 同步包装、SQLite 恢复记录和附件传输 |
| `mineru/kit/gradio/` | UI 构建、转换编排、document_io、素材与预览；由配置的 API 提供解析能力 |
| `mineru/config.py`、`errors.py` | 惰性配置加载、来源信息与内部 CLI/Doclib 错误分类 |

## 本地文档库调用链

```text
mineru parse → DoclibClient → doclib server
  → IngestService / ParseQueueService
  → ParseWorkerPool → ParseExecutionService
  → 本地 Flash，或配置的 managed / self-hosted / remote API
  → 缓存 Middle JSON 与素材 → ContentReadService → 阅读与导出
```

Doclib cache hit、覆盖计算、读取和 compaction 统一通过 `core/middle_json.py` 的历史缓存边界。
自托管状态和配置由 AppState 持有，不通过模块级隐式状态传递。

## 无状态解析与 Tasks 调用链

```text
mineru-kit parse / Python parse → MinerUParser → doc_analyze
  → backend.analysis → DocVortex 原生能力或 MinerU OCR/VLM/Hybrid
  → DocVortex 确定性后处理 → MinerU 可选 LLM 增强 → ParseResult

HTTP / WebUI → Tasks → 执行后端 → 同一解析门面
  → task_outputs 多格式打包 → Files 下载
```

原始 ModelJson 仅服务分析和后处理，不进入公开结果包。
公开协议见 [Tasks](https://opendatalab.github.io/MinerU/zh/usage/tasks/)、[Middle JSON](middle-json/README.md)；
后续工作见 [待办](backlog.md)，长期选择见 [ADR](decisions/README.md)。
