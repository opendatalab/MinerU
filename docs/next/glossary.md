# 术语表

状态: Draft
读者: 核心开发者、SDK/API/CLI 设计参与者、文档作者
范围: Next MinerU 文档体系中的核心术语、推荐命名和易混边界
非目标: 定义 API 字段级 schema；替代各专题文档的完整说明

## 1. 定位

本术语表用于统一 `docs/next/` 中的命名。后续新增 API、SDK、CLI、配置、Middle JSON 或架构文档时，应优先使用本文件中的推荐术语。

原则:

- 面向用户和公开接口时，优先使用产品术语。
- 面向实现和调试时，可以使用实现术语，但必须清楚它不等于产品承诺。
- 同一个文档中不要把 `tier`、`backend`、`parser`、`parse-server` 混作同义词。
- `engine` 不作为新的独立产品概念；在已有错误码或历史命名中出现时，按 `backend` 或解析服务能力理解。
- 历史底稿中的旧称可以保留在底稿里，新文档应逐步收敛到本术语表。

## 2. 推荐术语速查

| 推荐术语 | 中文含义 | 使用场景 | 不应混用为 |
|----------|----------|----------|------------|
| Tier | 解析档位 | CLI/API/SDK 用户可见参数 | backend、parser、model |
| Backend | 解析后端 | kit 专家参数、核心开发、Middle JSON、render 差异 | tier |
| Parser | SDK 中执行解析的对象或函数 | Tool SDK、代码接口 | parse-server |
| Parse Server | 解析服务 | v1 API、local/remote 解析服务、配置项 | doclib server |
| Local Parse Server | 本地解析服务 | doclib 调用、本机隐私解析 | doclib server |
| Remote Parse Server | 远端解析服务 | `mineru.net/api` 或 config 中的自定义远端地址 | local parse-server |
| doclib server | 本地文档库服务 | 文件入库、缓存、搜索、watch、配置 | parse-server |
| Tool SDK | `mineru.parser` | 无状态解析工具层 | Doclib SDK |
| API-backed Parser | `MinerUApiParser` | 用 parser 接口调用 v1 API | Doclib SDK |
| Doclib SDK | `DoclibClient` | 连接本地 doclib 的产品 SDK | v1 API client |
| Middle JSON | 解析中间结构 | backend 到 render / SDK / Agent 的结构 | Markdown、Content List |
| ParseResult | SDK 结果对象 | Tool SDK 返回值 | Middle JSON envelope |
| Watch | 文件自动发现机制 | 本地文档库后台任务 | 主动 parse |
| Ingest | 文件入库阶段 | 计算 SHA256、基础 metadata、写 DB | parse |
| Parse | 内容解析阶段 | 生成 Middle JSON、Markdown、Content List | ingest |
| Privacy | 用户请求的隐私偏好 | local/remote 选择 | via |
| Via | 实际执行路径 | 解析完成后的审计字段 | privacy |

## 3. Tier / Backend / Parser

### 3.1 Tier

`tier` 是推荐公开术语，中文可写作“解析档位”。

它表达用户能理解的质量、速度、资源消耗和隐私路径承诺。当前定义:

| Tier | 类型 | 含义 |
|------|------|------|
| `flash` | 实体 tier | CPU-only、快速、低质量，用于 watch、发现和索引 |
| `basic` | 实体 tier | 面向普通本地硬件的模型解析能力 |
| `standard` | 实体 tier | 绝大多数场景足够好的高质量解析能力，可来自 `mineru.net/api` 或高端本地硬件 |
| `advanced` | 实体 tier | 最高质量解析能力，比 `standard` 消耗更多算力和时间 |

规范:

- CLI/API/SDK 的用户可见参数应使用 `tier`。
- CLI 不传 `--tier`、HTTP API 省略 `tier` 或传 JSON `null`、Python SDK 传 `None` 表示使用默认选择策略。
- 任务入队、缓存目录、产物 metadata 应记录实际使用的实体 tier，即 `flash`、`basic`、`standard` 或 `advanced`。
- PDF/image 的默认选择策略不能解析为 `flash`；Office/HTML 这类仅支持 flash tier 的输入归一规则见 [ADR-0024](decisions/0024-file-type-tier-normalization.md)；text 直接读取。

完整产品语义见 [解析 Tier](tiers.md)。

### 3.2 Backend

`backend` 中文统一为“解析后端”。它是内部实现术语，表示具体解析实现族或模型管线。

`engine` 和 `backend` 在含义上非常接近，都是某种解析能力的实现后端。新文档应优先使用 `backend`；只有在已有错误码、历史命名或兼容接口中才保留 `engine`。

典型 backend:

| Backend | 含义 |
|---------|------|
| `pipeline` | Pipeline PDF 解析实现 |
| `vlm` | VLM PDF 解析实现 |
| `hybrid` | Hybrid PDF 解析实现 |
| `office` | Office 文档解析实现 |
| `html` | HTML 解析实现 |
| `flash` | 快速 CPU PDF 解析 backend；同时也是 `flash` tier 的默认实现 |

规范:

- 对普通用户显示 `tier`，不要把 `backend` 作为主选择项。
- `backend` 只应暴露在 kit 或核心开发层，例如 `mineru-kit parse --backend`。
- Tool SDK 的直接 parser 可以接受专家 `backend` 参数；API-backed parser、Doclib SDK、doclib server API 和 v1 API 不应要求用户理解或选择 `backend`。
- `mineru-kit api-server` 使用单值 `--tier flash|basic|standard` 表示能力上限，不暴露 `--backend`；启动后的 HTTP API 通过 `/v1/tiers` 发布展开后的请求 tier。
- Middle JSON 中 `_meta.backend` 表示产物来源实现，不表示用户请求的 `tier`。
- `backend` 不应承担隐私语义；隐私由 `privacy` / `remote` / `via` 描述。

### 3.3 Engine

`engine` 不作为独立术语继续扩展。遇到已有 `engine` 命名时，按下面规则理解:

| 已有命名 | 推荐理解 |
|----------|----------|
| `engine_error` | 解析后端、解析服务或解析执行相关错误 |
| `no_engine` | 当前请求的解析档位没有可用解析服务或解析后端 |
| `engine_unavailable` | 解析服务或解析后端暂不可用 |

后续新增文档、API 字段和配置项时，优先使用 `backend`、`parse-server`、`tier` 或更具体的错误码，不再扩大 `engine` 的使用范围。

### 3.4 Parser

`parser` 是 SDK 和代码接口术语，表示执行解析的对象、函数或类。

典型 parser:

| Parser | 含义 |
|--------|------|
| `parse()` | Tool SDK 的便捷函数 |
| `DocumentParser` | Parser 抽象接口 |
| `PdfPipelineParser` | 旧 SDK 兼容类，内部委托 Hybrid medium effort |
| `PdfVlmParser` | 调用 VLM backend 的 parser |
| `PdfHybridParser` | 调用 hybrid backend 的 parser |
| `DocxParser` / `PptxParser` / `XlsxParser` | Office parser |
| `HtmlParser` | HTML parser |
| `MinerUApiParser` | 通过 v1 API 委托解析的 parser |

规范:

- `parser` 不等于 `parse-server`。
- `MinerUApiParser` 是一个 Python parser 封装，不是 server。
- `mineru.parser` 不应依赖 doclib，也不应在 import 时启动服务或加载重模型。

## 4. 服务与进程

### 4.1 doclib

`doclib` 是本地文档库能力的统称。它包括文件入库、SHA256 去重、解析任务调度、缓存、搜索、watch、配置和本地 API。

文档中可以写:

- `doclib`: 指产品能力或模块。
- `doclib server`: 指运行中的本地服务进程。
- `Doclib SDK`: 指连接 doclib server 的 SDK 层。

不要把 `doclib` 称为 parse-server。doclib 可以调度解析，但它不是无状态解析服务。

### 4.2 doclib server

`doclib server` 中文统一为“本地文档库服务”。它是本地常驻服务，默认通过 Unix Domain Socket 向 `mineru` CLI、MCP Server、桌面端和 Doclib SDK 暴露本地文档库能力。

职责:

- 管理 files/docs/parses 等本地状态。
- 执行 watch、ingest、parse worker、search、config。
- 调用本地或远端 parse-server。
- 管理缓存和解析产物。

不负责:

- 直接对外提供 `mineru.net/api` 官方 API。
- 自身加载所有 heavy model。
- 在未授权时上传用户文档。

### 4.3 Parse Server

`parse-server` 是最终术语，中文统一为“解析服务”。它是无状态解析服务，提供 v1 Unified API，并且已经作为配置命名的一部分使用。

它接收文件或本地路径，执行 `basic` / `standard` / `advanced` 等解析能力，并返回解析任务与产物。它不管理长期文档库、watch、搜索索引或用户本地配置。

文档中推荐写法:

- `parse-server`: 泛指这类服务。
- `Local Parse Server`: 本地或自部署 parse-server。
- `Remote Parse Server`: 远端 parse-server。
- `mineru.net API`: 官方远程 API，属于 Remote Parse Server 的默认实现。

### 4.4 Local Parse Server

`Local Parse Server` 中文统一为“本地解析服务”，指运行在用户可信环境中的 parse-server。

它可以有三种管理模式:

| 模式 | 含义 |
|------|------|
| `disabled` | 不使用本地 parse-server |
| `managed` | 由 doclib server 启动和停止 |
| `self_hosted` | 用户自行启动，doclib 只连接 URL 和探活 |

Local Parse Server 的 API 尽量兼容官方 v1 API，但可以简化 Webhook、CDN、鉴权和上传实现。

### 4.5 Remote Parse Server

`Remote Parse Server` 中文统一为“远端解析服务”，指用户显式允许调用的远端 parse-server。

默认远端是 `https://mineru.net/api`。也可以是用户配置的兼容 v1 API 的远端 URL。

规范:

- 远端 URL 或 API Key 存在，不等于允许上传。
- 只有当前请求显式 `--remote`、SDK 显式 `remote=True`，或规则显式允许 remote，才可上传用户文档。
- 远端失败后的 fallback 不能改变用户隐私偏好。

### 4.6 mineru.net API

`mineru.net API` 是 MinerU 官方远程 API。

它是 v1 Unified API 的主线定义来源，也是当前 `standard` tier 的默认远端能力来源。

不要把 `mineru.net API` 写成 `doclib API`。doclib 是本地产品服务，`mineru.net API` 是官方远程解析和对话 API。

## 5. SDK 层

### 5.1 Tool SDK

`Tool SDK` 指 `mineru.parser`。

它面向需要直接在进程内解析文件的开发者，特点是无状态、轻量入口、返回 `ParseResult`。

职责:

- 提供 `parse()` 和 parser 类。
- 根据 `tier` 或专家 `backend` 选择具体 parser。
- 返回统一 `ParseResult`。

不负责:

- watch、搜索、长期缓存。
- doclib server 生命周期。
- 隐式远端上传。

### 5.2 API-backed Parser

`API-backed Parser` 指 `MinerUApiParser`。

它实现 `DocumentParser` 接口，但解析工作委托给 v1 API。调用方传入 `api_url` 即表示已经选择目标服务；它不负责判断是否允许 remote。

### 5.3 Doclib SDK

`Doclib SDK` 指 `DoclibClient`。

它连接本机 doclib server，面向产品能力:

- parse。
- search。
- find。
- watch/config。
- server status。
- cache and invalidation。

Doclib SDK 不是 v1 Unified API client，也不应直接加载 heavy backend。

## 6. 数据结构与产物

### 6.1 Middle JSON

`Middle JSON` 是解析中间结构，用于连接 backend、render、SDK 和 Agent 能力。

当前事实标准来自 `PageInfo`、`Block`、`Line`、`Span` 等 typed structure。下一阶段目标是统一 envelope、metadata、backend 差异和 Agent citation 字段。

规范:

- Middle JSON 不是最终 Markdown。
- Middle JSON 不是 Content List。
- Middle JSON 可以被 `ParseResult` 包装，但两者不是同一个概念。

### 6.2 ParseResult

`ParseResult` 是 Tool SDK 的统一结果对象。

它应能承载:

- typed pages。
- Markdown。
- Content List。
- Content List v2。
- images。
- source metadata。

`ParseResult` 面向 SDK 使用者；Middle JSON envelope 面向持久化、API、跨进程传输和 schema validation。

### 6.3 Markdown

`Markdown` 是面向阅读和导出的文本产物。它可以由 Middle JSON render 得到。

Markdown 不应反向成为规范源。需要结构化信息时，应使用 Middle JSON 或 Content List。

### 6.4 Content List

`Content List` 是结构化内容列表，适合下游程序消费。

它与 Markdown 都是 render output。长期目标是让 render 主要依赖 block type 和 unified schema，而不是 backend-specific dispatch。

### 6.5 Artifact

`artifact` 表示一次解析或导出的产物集合。不同上下文中的含义不同:

| 上下文 | artifact 含义 |
|--------|---------------|
| doclib 持久化 | 按页组织的 Middle JSON 批次文件 |
| SDK `save()` / CLI 导出 | Markdown、Content List、HTML、images 等从 Middle JSON 渲染得到的导出文件 |
| v1 API 文件资源 | 上传源文件或服务端生成的结果文件 |

在 doclib 中，artifact 按 `sha256 + 实际使用的 tier + page_range + done_at` 隔离存储；非 JSON 格式默认读取时转换，不作为 doclib 的持久化产物。

## 7. 文件、任务与索引

### 7.1 Source File

`source file` 指用户提供或 watch 发现的原始文件。

在本地 doclib 中，同一个 source file path 会先进入 `files`，再通过 SHA256 关联到 `docs`。

### 7.2 File Resource

`file` 在 v1 API 中是平台资源，由 `file_id` 标识。它可以表示上传的源文件，也可以表示解析产物。

不要把 API `file` 资源和本地文件路径混用。API `file_id` 是不透明 ID，客户端不得从中推断路径、租户或时间。

### 7.3 Upload

`upload` 是 v1 API 的上传会话。

官方 API 可以使用分片上传和对象存储；Local Parse Server 可以简化实现，但资源模型应保持兼容。

### 7.4 Job

`job` 是 v1 API 中的一次解析任务。

doclib 内部的 `parses` row 也是解析任务和缓存记录，但它不是 v1 API `job` 的同一个资源。文档中需要区分:

| 术语 | 所属系统 | 含义 |
|------|----------|------|
| `job` | v1 API | API 解析任务 |
| `parses` row | doclib | 本地解析任务和缓存记录 |

### 7.5 Watch

`watch` 是自动发现文件的机制。

默认使用 `flash` 进行低成本处理，用于文件发现、预览和搜索索引。watch 不是用户主动阅读文档，也不代表最终阅读质量。

### 7.6 Ingest

`ingest` 是入库阶段。

职责:

- 发现或接收文件路径。
- 计算 SHA256。
- 提取基础 metadata。
- 写入 `files` / `docs`。
- 建立文件名索引。
- 根据规则创建后续 parse 任务。

Ingest 不等于 parse。Ingest 可以不产生完整 Middle JSON。

### 7.7 Parse

`parse` 是内容解析阶段。

职责:

- 选择或解析 tier。
- 调用本地 parser 或 parse-server。
- 生成 Middle JSON 和 render output。
- 更新 parse 状态、缓存和内容索引。

主动阅读场景下，未指定 tier 应使用默认选择策略，并解析到可用的非 `flash` 质量 tier。

## 8. 隐私与执行路径

### 8.1 Local

`local` 可以有两个含义，文档中需要明确上下文:

| 上下文 | 含义 |
|--------|------|
| 隐私偏好 | 不允许上传到远端 |
| 执行路径 | 实际由本机 parser 或 Local Parse Server 执行 |

如果可能，隐私偏好使用 `privacy=local`，实际路径使用 `via=local`。

### 8.2 Remote

`remote` 也有两个含义:

| 上下文 | 含义 |
|--------|------|
| 用户授权 | 当前请求允许上传到远端 |
| 执行路径 | 实际由 Remote Parse Server 执行 |

文档中不要只写“remote 可用所以会上传”。必须写清楚“用户显式允许 remote 后才可上传”。

### 8.3 Privacy

`privacy` 表示用户请求偏好。

推荐取值:

| 值 | 含义 |
|----|------|
| `local` | 不允许上传远端 |
| `remote` | 允许调用远端 |

`privacy` 应由当前请求、规则或显式 SDK 参数决定。全局 remote URL 或 API Key 不应自动改变 privacy。

### 8.4 Via

`via` 表示实际执行路径。

推荐取值:

| 值 | 含义 |
|----|------|
| `local` | 实际在本地执行 |
| `remote` | 实际在远端执行 |

`via` 是审计和结果 metadata，不应替代用户请求的 `privacy`。

### 8.5 Fallback

`fallback` 指某条执行路径失败后选择另一条路径。

规范:

- Fallback 不能扩大隐私边界。
- `remote` 失败后可以 fallback 到 `local`，因为不会增加隐私暴露。
- `local` 失败后不能自动 fallback 到 `remote`，除非用户已经显式允许 remote。
- Tier fallback 不能把默认选择或主动阅读场景静默降级到 `flash`。

## 9. 配置术语

### 9.1 启动前配置

启动前配置是 doclib server 启动之前必须知道的配置。

典型内容:

- UDS 路径。
- HTTP 监听。
- SQLite 路径和 pragma。
- data dir。
- worker 数。
- log 路径。

这类配置通常来自文件或内置默认值，不能依赖 doclib DB。

### 9.2 运行时配置

运行时配置是 doclib server 启动后可读取和修改的配置。

典型内容:

- parse-server mode / URL / API key。
- parse-server mode。
- watch target。
- exclude rule。
- parsing rule。
- remote URL / API Key。

运行时配置存在不等于当前请求一定采用。会改变隐私边界的配置必须由当前请求或规则显式授权。

### 9.3 Parsing Rule

`parsing rule` 是针对路径或文件集合的自动解析规则。

它可以指定 tier、page_range、remote 等策略。规则如果允许远端上传，必须显式写出 remote 语义。

### 9.4 Exclude Rule

`exclude rule` 是文件发现阶段的排除规则。

它影响 watch 和扫描，不应影响用户显式指定的单文件 parse，除非 CLI/API 规格另行规定。

## 10. API 术语

### 10.1 Official API

`Official API` 指 `mineru.net/api` 提供的远程 API。

API 文档每个 endpoint 应先描述 Official API，再描述 Local Parse Server 的差异。

### 10.2 Local Server Differences

`Local Server Differences` 指 Local Parse Server 相对 Official API 的简化或变更。

典型差异:

- 鉴权可选。
- Webhook 可不实现。
- 上传可简化。
- 文件下载可直接返回 body。
- 支持 `local` source。

不要把 local difference 写成另一套不兼容 API。目标是复用同一套客户端。

### 10.3 Access Level

`access level` 指官方 API 的访问级别，例如 anonymous、registered 或更高权限。

Local Parse Server 默认不按公网 access level 做售卖或配额区分。

## 11. 命名规范

### 11.1 大小写

正文中推荐:

| 推荐写法 | 用法 |
|----------|------|
| `tier` | 字段名、参数名、泛指解析档位 |
| Tier | 章节标题或概念名 |
| `flash` / `basic` / `standard` / `advanced` | tier 枚举值 |
| parse-server | 最终术语，泛指解析服务 |
| Local Parse Server | 本地解析服务 |
| Remote Parse Server | 远端解析服务 |
| doclib server | 本地文档库服务 |
| backend | 解析后端 |
| Tool SDK | SDK 层名 |
| Doclib SDK | SDK 层名 |
| Middle JSON | 中间结构名 |

### 11.2 连字符

推荐:

- `parse-server`: 服务类别。
- `doclib server`: 本地文档库服务。
- `API-backed parser`: SDK 层中的 parser 类型。

避免:

- `parse server` 和 `parse-server` 在同一文档混用。
- `doclib-server` 作为新文档主写法。

### 11.3 字段名

公开字段优先使用小写 snake_case:

| 推荐字段 | 含义 |
|----------|------|
| `tier` | 实际使用的解析档位；如果用户未指定，则记录系统实际选择的实体 tier |
| `backend` | 实际实现来源 |
| `privacy` | 用户隐私偏好 |
| `via` | 实际执行路径 |
| `source` | API 输入来源 |
| `file_id` | API file 资源 ID |
| `job_id` | API job 资源 ID |

## 12. 易混用法对照

| 不推荐说法 | 推荐说法 | 原因 |
|------------|----------|------|
| 默认 tier 是某个 engine | 默认选择策略 | 默认选择是一段选择逻辑，不是一个解析后端 |
| Standard backend | `standard` tier 或具体 backend | `standard` 是解析档位，不是解析后端名称 |
| Flash 是默认解析 | watch 默认 `flash`；主动阅读不指定 tier 时走默认选择策略 | 避免误解质量策略 |
| parser server | parse-server | 服务类别统一写法 |
| doclib API 等同 v1 API | doclib server API / v1 Unified API | 两者资源模型不同 |
| local 表示一定不上传 | `privacy=local` 表示不上传，`via=local` 表示本地执行 | local 有两个上下文 |
| remote URL 配好了所以能上传 | 只有显式 remote 才能上传 | 隐私边界必须显式 |
| backend_version 是 tier | backend version 是实现版本，tier 是产品档位 | 维度不同 |
| Middle JSON 是 ParseResult | ParseResult 包装或暴露 Middle JSON | 对象层级不同 |

## 13. 文档写作约束

新增或修改 `docs/next/` 文档时遵守:

1. 用户可见选择写 `tier`，不要写 `backend`。
2. 讨论执行服务时，明确是 `doclib server` 还是 `parse-server`。
3. 讨论本地/远端时，明确是隐私偏好还是实际执行路径。
4. 讨论解析实现时，明确是 `parser`、`backend` 还是 `model`。
5. 讨论 API 时，先描述 Official API，再描述 Local Parse Server 差异。
6. 讨论 Middle JSON 时，区分 schema、typed object、render output 和 SDK result。
7. 不把 `engine` 当成独立概念扩展；已有 `engine_error` 等命名按解析后端或解析服务错误理解。

## 14. 已确认约定

- `flash` 长期同时作为 tier 和 PDF 解析 backend 名称存在。
- API、SDK、doclib 和 Middle JSON 结果中默认只需要记录实际使用的 `tier`。
- 不记录 `requested_tier` / `resolved_tier` 双字段；用户未指定 tier 时，以系统实际使用的解析档位为准。
