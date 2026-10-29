# ADR-0017: MinerU Kit API Server Command

状态: Accepted，已实现
日期: 2026-06-17
相关文档:
- ../cli/mineru-kit.md
- ../cli/mineru-kit-api-server.md
- ../api/health-models.md

## 背景

`mineru-kit api-server` 是 local parse-server 的正式启动入口。它需要与以下边界保持清楚：

- 与 `mineru-kit parse` 区分：前者启动自部署解析服务，后者执行一次性无状态解析。
- 与 doclib 区分：doclib 负责文件、缓存、任务和搜索；api-server 只负责 v1 parse API 和模型生命周期。
- 与旧命令区分：未来正式入口应收敛到 `mineru-kit api-server`，而不是继续散落在旧脚本名中。

同时，需要明确 self-hosted 与 managed 的边界、tier/backend 语义、参数分层，以及它实际覆盖的 API 范围。

## 决策

### 1. 命令定位

`mineru-kit api-server` 是未来唯一正式的 parse-server 启动入口。

它用于启动 **self-hosted parse server**：

```bash
mineru-kit api-server [flags]
```

它提供的是 **v1 API（非 doclib API）**。

### 2. self-hosted 与 managed

`mineru-kit api-server` 对用户只对应 self-hosted 场景。

- self-hosted：用户手工启动 `mineru-kit api-server`
- managed：由 doclib / `mineru server` 在运行时拉起和管理 parse-server 进程

managed 是生命周期管理方式，不是用户直接操作的独立命令产品形态。

因此不再把 `remote-compatible` 作为第三种部署模式写入正式文档。

### 3. 单进程 tier 规则

一个 API 服务进程使用 `--tier auto|flash|basic|standard|advanced` 表示 PDF/图片的最高开放档位，上限包含自身及全部较低档位；省略等价于 auto，取硬件推荐集合的最高档。Standard 上限不再隐含 Advanced。`--tiers`、`--no-flash`、`--no-advanced` 均直接移除。

自动推荐先按有效小模型后端分流：CPU 可用逻辑核至少 4 时开放到 Basic，否则只开放 Flash。GPU 经既有选型确认可用且选定设备显存／预算至少 5.5 GiB 时，独立于 CPU 核数开放全部四档。Torch、Apple Metal 和符合既有绑定规则的独显保留免测速路径；其他 ONNX GPU 仍须校准胜出。CPU 至少 4 核或存在内存达标的兼容 GPU 候选时，Windows/Linux 在本次启动内完成适用选型；最终 CPU 回退不能被静态库存覆盖。显式启动上限跳过推荐。不引入新的采样阈值或 VLM 实测，依赖预检与可选预加载仍执行。Advanced 复用 Standard 模型配置，请求默认仍优先 Standard。`hardware detect` 与服务使用相同结构化推荐及设备内存证据。

`/v1/tiers` 仅发布 PDF/图片可选档位。所有服务都支持现有原生文档格式，固定使用 flash；原生支持不再从公开 flash 档位推断，不新增原生能力声明。Tasks 请求 tier 继续可选，PDF/图片按 `standard -> basic -> flash` 默认选择，advanced 仅显式使用。

Python 工厂保留非空精确集合 `create_app(tiers=...)`，供内嵌部署和 SaaS 模拟使用；`tiers={"standard"}` 仍支持原生文档。WebUI、Router 和 Doclib 托管配置统一按上限解释，增加显式 advanced 选项；客户端不再补起本地 Flash 服务。

详细规则及迁移见 [API 启动文档](../cli/mineru-kit-api-server.md)。

### 4. backend 的公开边界

backend 不是普通 API 协议层字段。

规则：

- 启动参数不公开 `--backend`
- 运行中的普通 API 响应不暴露 backend
- `GET /v1/tiers` 也不新增 backend 字段
- 调用方如需推断实现，只能从 `current_model` 做弱推断

因此 backend 只属于内部实现概念。

### 5. 参数分层

#### 稳定公开参数

- `--host`
- `--port`
- `--tier`
- `--api-key`

#### 稳定解析参数

- `--language`
- `--disable-image-analysis`
- `--concurrency`
- `--upload-dir`
- `--url-timeout`

### 6. 不进入正式契约的参数

OCR 模式使用 V1 任务级 `ocr_mode` 参数，省略时默认 `auto`。服务启动参数 `--ocr-mode` 已移除，不保留全局默认值或兼容透传。

`--reload` 不进入正式命令设计。

原因：

- 它只是开发热重载参数
- 对部署者和普通使用者没有正式价值
- 不应和 tier、语言、鉴权等长期稳定参数混在一起

实现层如需保留开发模式能力，应另行处理，不写入正式 CLI 契约。

### 7. 与 doclib 的职责边界

#### doclib 负责

- 文件入库
- SHA256 缓存
- 任务调度
- 解析产物存储
- 搜索索引
- managed lifecycle
- parse-server 健康检查与选择

#### api-server 负责

- 暴露 tier 能力
- 接收 v1 parse API 请求
- 执行解析
- 返回解析结果
- 管理模型下载、预热、失败重试和退避

api-server 不负责 doclib 的文档库资源语义。

### 8. API 覆盖范围

`mineru-kit api-server` 的目标是实现：

- **v1 API（非 doclib API）中的绝大多数 path**

当前明确排除：

- chat 的两个 path

同时明确不实现：

- doclib 的 `/docs`
- `/parses`
- `/search`
- `/invalidate`
- 其它本地文档库资源接口

### 9. 本地安全边界

- 默认监听 loopback
- 可通过 `--api-key` 配置固定 API key
- 默认不设置 API key

## 替代方案

### 1. 把 managed 作为第三种正式部署模式写入用户命令文档

没有采用。managed 是 doclib 生命周期管理方式，不是用户直接执行的命令形态。

### 2. 在 `GET /v1/tiers` 中额外公开 backend

没有采用。虽然能力发现理论上可以承载 backend，但当前决定保持 v1 API 文档不改动，backend 只属于内部实现概念。

### 3. 保留 `--reload` 作为正式参数

没有采用。它只服务开发热重载，不进入面向部署者和集成方的正式契约。

## 影响

- `mineru-kit api-server` 的用户入口和 doclib managed 行为边界更清楚。
- 参数体系可与 `mineru-kit parse`、parser SDK 保持一致。
- v1 API 与 doclib API 的职责分层更清楚。
- 旧的 `mineru-api` 等脚本后续应被视为兼容入口或迁移对象，而不是长期主入口。

## 后续动作

- 更新 `docs/next/cli/mineru-kit-api-server.md`
- 更新 `docs/next/cli/`
- 更新 `docs/next/decisions/README.md`
- 后续实现 `mineru-kit api-server` 顶层脚本与参数解析
