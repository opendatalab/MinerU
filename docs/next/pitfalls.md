# 开发陷阱

本页描述当前 4.1 系列的实现边界；历史设计和旧 API 不能作为兼容性承诺。

## 不同入口有不同的默认档位

Doclib 主动解析 PDF/图片按 Standard → Basic 选择质量档位。Tasks 及 Gradio 按
Standard → Basic → Flash → Advanced 选择；仅开放 Advanced 时才默认选择它。
原生文档、已有缓存读取与硬件推荐各有独立规则，见 [档位](tiers.md)。不要复用名称相似但语义不同的选择器。

## 产品档位与模型后端分开

公开解析入口使用 tier；small_backend 与 VLM engine 是独立模型配置。
不恢复 `--backend`、`model.stack` 或旧 backend 别名。远程 VLM 是 OpenAI 兼容推理服务，
与接收文档的 V1 Tasks 服务不同；vLLM 只是部署选择之一。

## 历史兼容只在缓存读取边界

ParseResult 与 HTTP/ZIP 严格验证共享 schema 2.0。旧格式只通过 Doclib 的
`read_cached_middle_json(path)` 转换。缓存读取、覆盖、命中和 compaction 必须使用同一入口。
修改几何或素材后同时验证序列化、素材引用和真实导出，不能只比较结构数量。

## 错误与取消不能跨层套用

公共 Tasks 错误只有 code、message 和可选 param；内部 CLI/Doclib 仍保留 type 分类。
file-parse 的状态和错误对象另有契约。`retryable/user_action/docs_url` 是待讨论扩展，不能假定响应已经包含。
HTTP 200 不代表任务解析成功；客户端等待超时不代表取消。底层不可中断计算允许收尾，迟到结果不得覆盖取消终态。

## 发布后的接口变更需要迁移说明

4.x 已是正式发布系列。集成依赖应固定经过验证的版本；破坏性变更同步更新双语迁移文档、
Tasks/SDK 消费者和测试。不能因 HTTP 仍为 `/v1` 就假定不同 MinerU 版本协议兼容。

## 代码约定

新增函数提供中文说明和必要类型注解；MinerU 内部使用相对导入。公开 API import 不触发重模型加载。
模块导入不读取配置或创建服务。Issue 关联使用 `Refs #n`，不使用自动关闭关键词。
