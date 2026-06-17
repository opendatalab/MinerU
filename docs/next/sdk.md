# SDK 设计

状态: Draft
读者: SDK 开发者、集成方、核心开发者
范围: Tool SDK、API-backed parser、Doclib SDK 的分层、公开接口和迁移路径
非目标: HTTP API 全量字段说明；CLI 命令手册
底稿: `../../NEXT-SDK.md`

## 当前定位

SDK 文档定义代码集成入口。它既要让普通开发者知道如何解析一个文件，也要让核心开发者知道 parser、parse-server、doclib client 之间的边界。

当前 SDK 设计按三层组织:

- `mineru.parser`: 无状态解析工具层。
- `mineru.parser.MinerUApiParser`: 通过 v1 API 委托解析的 parser。
- `mineru.doclib.client.MineruClient`: 本地 doclib 的 Product SDK。

## 目录

1. [SDK 分层](sdk/README.md): 三层 SDK 的职责、消费者和依赖方向。
2. [Tool SDK: parser](sdk/parser.md): `mineru.parser` 的公开入口和 parser 类。
3. [API-backed Parser](sdk/api-parser.md): `MinerUApiParser` 与 parse-server / mineru.net 的关系。
4. [Doclib Client](sdk/doclib-client.md): `MineruClient` 的本地文档库能力。
5. [ParseResult](sdk/parse-result.md): 解析结果对象、输出格式和保存行为。
6. [Tier 与错误](sdk/tiers-errors.md): SDK 层 tier 语义、隐私策略和错误映射。
7. [迁移路径](sdk/migration.md): 从当前代码走向目标公开契约的步骤。

## 与其他文档的关系

- HTTP 行为见 [Unified API](api.md)。
- CLI 行为见 [CLI 规格](cli.md)。
- 返回结构见 [Middle JSON](middle-json.md)。
- 错误模型见 [错误码体系](errors.md)。
- Tier 产品语义见 [解析 Tier](tiers.md)。

## 整理原则

- 公共 import path 优先于内部文件路径。
- 当前已实现能力与目标公开契约分开描述。
- SDK 默认遵守隐私优先，不隐式上传本地文件到远端。
- Tool SDK 不依赖 doclib；doclib 可以依赖 Tool SDK。
- 对外 API 层应避免 import 时加载重依赖。
- 其他语言 SDK 暂无开发计划；如果未来开发，只覆盖 v1 Unified API，不覆盖本地 doclib UDS 能力。
