# Middle JSON 规范

状态: Implemented

共享类型由 `docvortex.schema` 唯一定义，`mineru.types` 重新导出。当前协议身份为
`docvortex.middle` 或 `docvortex.model`，`schema_version="2.0"`；Python 属性 `schema_id`
序列化为 JSON 键 `schema`。`ParseResult` 只持有 MiddleJson，不保存或导出 ModelJson。

## 现行契约

| 主题 | 规范 |
|---|---|
| 页面、Block 与 InlineSpan | [当前事实标准](current-medium.md) |
| 协议身份、metadata 和产品扩展 | [统一外层协议](envelope.md) |
| 历史读取边界 | [Doclib 缓存兼容](doclib-compatibility.md) |
| 渲染、素材与导出 | [Rendering Contract](rendering.md)、[用户输出参考](https://opendatalab.github.io/MinerU/zh/reference/output_files/) |
| 稳定引用 | [Doclib 定位符](../decisions/0012-doclib-block-locator.md)、[读取命令](../cli/mineru-read.md) |

`ModelJson` 保存 raw pages 和 page_index_map；`MiddleJson` 保存有序 PageInfo。
PageInfo 只包含 page_idx 与 blocks。自然语言 InlineSpan 不携带字体或几何信息。
MinerU 产品扩展严格限定为 `extensions["mineru"]` 下的 tier 与 parse_mode；生产者版本在 metadata.producer.version。

通用 ParseResult 和 HTTP/ZIP 客户端严格读取当前协议。只有 Doclib 持久化缓存经
`read_cached_middle_json(path)` 转换受支持的历史结果；不能把缓存兼容扩大成 SDK/API 承诺。

Structured Content 是派生输出，目前不携带独立 schema 身份。
[带版本身份的旧提案](https://github.com/opendatalab/MinerU/blob/master/docs/proposals/structured-content-schema.md)
没有成为当前输出协议。旧结构盘点与迁移记录已进入历史档案。

公开字段或渲染行为发生变化时同步这些规范、相关 ADR、双语输出文档和消费者测试。
未完成工作只登记在 [当前待办](../backlog.md)。
