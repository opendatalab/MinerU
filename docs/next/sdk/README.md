# SDK 分层

状态: Draft
读者: SDK 开发者、集成方、核心开发者
范围: SDK 层次、职责、消费者和依赖方向
底稿: `../../../NEXT-SDK.md`

## 三层模型

当前 SDK 不应只分成 Tool SDK 和 Product SDK。代码里已经形成三个需要分别描述的入口:

| 层 | 公开入口 | 职责 | 主要消费者 |
|----|----------|------|------------|
| Tool SDK | `mineru.parser` | 无状态解析文件，返回 `ParseResult`。 | `mineru-kit parse`、parse-server worker、需要直接嵌入解析能力的开发者 |
| API-backed Parser | `mineru.parser.MinerUApiParser` | 仍实现 `DocumentParser`，但通过 v1 API 委托解析。 | 需要用同一 parser 接口连接 local parse-server 或 mineru.net 的开发者 |
| Doclib SDK | `mineru.doclib.client.MineruClient` | 连接本地 doclib，使用缓存、搜索、watch、配置和服务状态能力。 | `mineru` CLI、MCP Server、桌面端、本地自动化 |

这三层返回的数据可以共享中间结构和错误模型，但生命周期不同:

- Tool SDK 是无状态调用。
- API-backed Parser 是无状态 API client。
- Doclib SDK 是有状态本地文档库 client。

Doclib SDK 与 doclib 本地 HTTP + JSON API 使用同一套方法和协议语义。doclib 本地 API 可以运行在 UDS 或 TCP loopback transport 上，Python client 默认通过 `$MINERU_HOME/doclib.endpoint.json` 发现实际 endpoint。项目内部除 `mineru.doclib.client` 外，不应直接通过 HTTP 调用 doclib API；外部客户端未来可以直接依赖 doclib HTTP API。

其他语言 SDK 暂无开发计划。如果未来开发，只覆盖 v1 Unified API，不覆盖本地 doclib transport discovery 能力；本地 doclib 能力由 Python Doclib SDK 和 MinerU 自身入口承载。

## 依赖方向

```text
mineru-kit parse
  -> mineru.parser
      -> backend / render / types

mineru-kit api-server
  -> mineru.parser.api_server
      -> mineru.parser
          -> backend / render / types

mineru doclib
  -> mineru.doclib.client
      -> doclib server over UDS or TCP loopback
          -> doclib services
              -> mineru.parser
              -> MinerUApiParser -> local parse-server or mineru.net
```

约束:

- `mineru.parser` 不依赖 `doclib`。
- `mineru.doclib.client` 不直接调用 heavy backend，只通过 doclib server 通信。
- `MinerUApiParser` 可以在 Tool SDK 中存在，因为它遵守 `DocumentParser` 接口，但它必须显式要求 `api_url`。
- SDK 不应在 import 阶段启动 server、连接数据库、读取配置或加载模型。

## 包边界

| 包 | 是否面向用户 | 说明 |
|----|--------------|------|
| `mineru.parser` | 是 | Tool SDK 主入口。 |
| `mineru.parser.api_server` | 主要面向运行时 | `mineru-kit api-server` runtime 和 v1 API contract。 |
| `mineru.doclib.client` | 是 | 本地 doclib Product SDK。 |
| `mineru.doclib.services` | 否 | doclib 内部业务逻辑。 |
| `mineru.backend` | 否 | parser 内部依赖，不直接作为 SDK。 |
| `mineru.types` | 是 | 共享结构类型，可由 `ParseResult` 暴露。 |

## 命名约定

文档中使用以下名称:

- **Tool SDK**: `mineru.parser`。
- **API-backed parser**: `MinerUApiParser`。
- **Doclib SDK**: `MineruClient`。
- **parse-server**: 实现 v1 Unified API 的无状态解析服务。
- **doclib server**: 本地文档库服务，通过 UDS 或 TCP loopback 暴露本地产品能力。

## 未决问题

顶层别名集中维护在 [开放问题清单](../open-questions.md)。`MinerUApiParser` 位置已收敛，保持 `from mineru.parser import MinerUApiParser`。
