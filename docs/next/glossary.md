# 开发术语

本页只解释当前代码中的概念。具体字段和默认值由对应专题维护，避免在术语表复制接口规格。

| 术语 | 当前含义 | 参考 |
|---|---|---|
| tier | flash/basic/standard/advanced 产品档位 | [档位](tiers.md) |
| effort | 分析内部的 flash/medium/high/xhigh，与 tier 显式映射 | `mineru/parser/tier.py` |
| small backend | ONNX 或 Torch 小模型执行后端 | [设备策略](onnx-device.md) |
| VLM engine | llama.cpp、vLLM、LMDeploy 或显式 MLX；远程地址存在时走 HTTP 推理 | [模型配置](config.md#remote-vlm) |
| API Server / parse-server | `/v1/tasks` 无状态解析服务；进程内任务状态不保证跨重启恢复 | [API](api/README.md) |
| Router | 聚合 worker，管理上传、调度、结果映射和租约 | [架构](architecture.md) |
| file-parse | 独立 multipart 单文件同步包装服务，持久化 request_id 与上游任务引用 | [正式参考](https://opendatalab.github.io/MinerU/zh/usage/file_parse/) |
| Doclib | 本地有状态文档库，提供入库、缓存、搜索、扫描和读取 | [架构](architecture.md) |
| file / path | Doclib 的路径实例，可有可达性与错误状态 | `mineru/doclib/types.py` |
| doc / sha256 | Doclib 的内容身份；多个路径可对应同一内容 | `mineru/doclib/core/` |
| short_id | Doclib 的短文档标识，用于 locator；不同于 Tasks file_id | [定位符决策](decisions/0012-doclib-block-locator.md) |
| Task | HTTP 父任务与有序 files 清单 | [Tasks](https://opendatalab.github.io/MinerU/zh/usage/tasks/) |
| file_id | Tasks 从创建到成功产物保持的公共逐文件 ID | [Files](api/files.md) |
| parse record | Doclib 的持久化解析批次，状态及缓存语义独立于 HTTP Task | [Doclib SDK](sdk/doclib-client.md) |
| ModelJson | 内部 raw pages 与 page_index_map，不进入公共结果包 | [Middle JSON](middle-json/README.md) |
| MiddleJson | 带 schema 身份、版本、metadata 与 extensions 的有序页面语义树 | [协议](middle-json/envelope.md) |
| ParseResult | 持有 MiddleJson 的 SDK 结果对象，提供渲染与保存 | [SDK 结果](sdk/parse-result.md) |
| Structured Content | 派生内容字典，没有独立 schema 身份 | [输出格式](https://opendatalab.github.io/MinerU/zh/reference/output_files/) |
| locator | Doclib 的文档／档位／页／块／字符定位表达式 | [读取命令](cli/mineru-read.md) |
| page_idx / page_no | 内部从 0 开始的索引／面向用户从 1 开始的页码 | [页码命名](page-naming.md) |
| page_range | PDF 页范围字符串，区别于 pages 页面数组 | [范围语法](page-ranges.md) |
| metadata.producer | 文档协议的生产者，区别于源文件属性 | [元数据](metadata.md) |
| access_level | 当前 API 认证视角，不由本仓库承诺云端套餐 | [用量](api/usage-limits.md) |

历史术语如 Jobs、独立 Upload、model.stack 和顶层 effort 字段不可用于新请求。
迁移说明保留它们用于解释升级，不能据此恢复旧接口。
