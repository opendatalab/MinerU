# Unified API

状态: Draft
读者: API 使用者、服务端开发者、SDK 开发者
范围: MinerU v1 API 的资源模型、认证、上传、文件、任务、对话、用量和限流
非目标: 本地 CLI 参数说明；SDK 封装细节
来源: 由根目录旧 Unified API 底稿迁移整理而来；修订记录和历史接口对比未迁移

## 当前定位

Unified API 文档描述两类部署形态下的同一套 API:

- `mineru.net` 官方远程 API。
- 本地启动的 Local Parse Server API。

两者的 endpoint、请求结构和主要响应结构尽量保持一致。新文档默认先描述 `mineru.net` 的完整行为，再在每个主题末尾说明本地 server 的简化或差异。

## 目录

1. [API 总览](api/README.md): 资源模型、端点地图、认证、通用约定。
2. [响应与错误](api/responses.md): 成功响应、错误 envelope、请求追踪。
3. [Health、Models 与 Tiers](api/health-models.md): 服务能力、模型列表、解析档位。
4. [Uploads 与 Files](api/uploads-files.md): 上传生命周期、文件对象、产物下载。
5. [Parse Jobs](api/parse-jobs.md): 解析任务、同步等待、轮询、SSE。
6. [Chat 与 Responses](api/chat.md): OpenAI-compatible 文档对话接口。
7. [Webhooks](api/webhooks.md): 异步回调协议。
8. [Usage 与 Limits](api/usage-limits.md): access level、限流、用量查询。
9. [端到端示例](api/examples.md): 常见调用流程。

## 整理原则

- 不迁移底稿开头的历史变更说明。
- 不迁移底稿末尾面向历史接口的对比内容。
- 官方 API 是主线；本地 server 差异只在对应主题中补充。
- 可观察行为优先，内部实现细节留给架构或配置文档。

## 相关文档

- Tier 语义见 [解析 Tier](tiers.md)。
- 错误码体系见 [错误码体系](errors.md)。
- SDK 封装策略见 [SDK 设计](sdk.md)。
- 本地服务边界见 [系统架构](architecture.md)。
