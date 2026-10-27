# mineru-kit api-server

状态: Implemented
读者: 服务部署者、核心开发者、`mineru server` 集成开发者
范围: `mineru-kit api-server` 的定位、self-hosted 边界、与 doclib 的协作和参数契约
非目标: 统一 REST API 的字段级定义；模型服务内部实现
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru-kit api-server` 是正式的 self-hosted parse-server 启动入口。`--tiers` 接受逗号分隔的精确档位集合（`flash/basic/standard/advanced`），或 `auto`；不传时按硬件自动推荐开放集合。

`mineru doclib` 可以通过 HTTP 调用它执行解析任务。

## 2. 与 doclib 的协作

```text
mineru CLI
  -> doclib
    -> local parse-server (`mineru-kit api-server`)
```

doclib 负责：

- 文件入库。
- SHA256 缓存。
- 任务排队。
- 解析产物存储。
- 搜索索引。
- parse-server 健康检查。

api-server 负责：

- 暴露可用 tier。
- 接收解析请求。
- 执行模型解析。
- 返回解析产物。
- 负责模型下载、预热、重试和退避。

## 3. Tier 能力发现

api-server 必须提供能力发现接口，让 doclib 或客户端知道当前服务支持哪个 tier。

启动时确定开放集合，能力发现、请求校验和实际执行使用同一集合。显式列表不自动补齐其他档位，并跳过硬件推荐；依赖预检仍照常执行。

| api-server 启动配置 | `/v1/tiers` | PDF/image 请求未指定 tier 时 |
|---------------------|-------------|--------------------|
| `--tiers flash` | `flash` | `flash` |
| `--tiers basic` | `basic` | `basic` |
| `--tiers standard` | `standard` | `standard` |
| `--tiers flash,standard` | `flash`、`standard` | `standard` |
| `--tiers advanced` | `advanced` | `503 quality_tier_unavailable`，须显式指定 |
| `--tiers flash,basic,standard,advanced` | 全部四档 | `standard` |

Tasks 省略 tier 或传 JSON `null` 时按 `standard -> basic -> flash` 选择默认档位，advanced 必须显式指定。HTTP SDK 和 file_parse 使用同一请求策略。原生文档固定使用 flash；列表未包含 flash 时，OFD/EPUB/Office/HTML/MHTML/CSV/TSV 等需要 Flash 的输入会被拒绝。完整规则见 [Tasks](../api/tasks.md#tier默认选择与文件作用范围)。

### 自动推荐

省略 `--tiers` 等价于 `--tiers auto`：

| 硬件条件 | 推荐开放集合 |
|----------|--------------|
| 进程可用 CPU 数不足 4，且无达标 GPU | `flash` |
| 进程可用 CPU 数至少 4，且无达标 GPU | `flash,basic,standard` |
| 有可用独显、OCR 校准达标的集显，或 Metal 可用的 Apple Silicon | 全部四档 |

CPU 数按逻辑 CPU 计算，并考虑进程亲和性和 Linux cgroup 配额。集显沿用现有 OCR 校准的有效性及 GPU 时间不高于 CPU 时间 0.8 倍的判定。Apple Silicon 的 Metal 集显按已知能力直接推荐。探测失败或设备身份不明确时按 CPU 条件推荐。

推荐阶段只探测设备并读取已有校准记录，不下载模型、不执行 OCR/VLM 性能测试、不写配置，也不增加内存分级门槛。模型的正常惰性加载和显式 `--preload-models` 行为独立于推荐阶段。

真实 OCR 校准会在当前配置文件旁保存 `<配置文件路径>.onnx-calibration.json`。只有设备身份、后端版本和 CPU 配置匹配、输出一致且结果稳定的记录可用于集显推荐。旧的 `model.onnx_device=cpu/gpu` 偏好及手动 GPU 设置不作为校准证据。记录缺失、损坏或失效时集显不自动开放 advanced，用户仍可通过显式列表启用。新校准结果在下次启动时生效，运行中的服务不会动态增减档位。

远程 VLM 配置不自动扩大推荐集合；部署者可显式指定所需列表。服务开放 advanced 不代表请求默认使用 advanced。

### 旧命令迁移

旧启动参数 `--tier`、`--no-flash`、`--no-advanced` 已移除，传入直接报错。等价迁移如下：

| 旧参数 | 新参数 |
|--------|--------|
| `--tier flash` | `--tiers flash` |
| `--tier basic` | `--tiers flash,basic` |
| `--tier basic --no-flash` | `--tiers basic` |
| `--tier standard` | `--tiers flash,basic,standard,advanced` |
| `--tier standard --no-flash` | `--tiers basic,standard,advanced` |
| `--tier standard --no-advanced` | `--tiers flash,basic,standard` |
| `--tier standard --no-flash --no-advanced` | `--tiers basic,standard` |

Python 工厂使用 `create_app(tiers={"flash", "standard"})`；`tiers=None` 自动推荐。空集合、裸字符串、未知档位及 CLI 中的空项或 `auto` 混用均报错。集合去重后按 `flash/basic/standard/advanced` 顺序发布。

Doclib、WebUI 和 Router 保留现有托管配置语义，启动子进程时转换成等价列表。Doclib 的 basic/standard 分别传入 `basic`、`basic,standard,advanced`，Flash 仍由 Doclib 进程内执行；WebUI 和 Router 继续展开原有 flash/basic/standard 预设。

## 4. self-hosted 与 managed

`mineru-kit api-server` 对用户只对应 self-hosted 场景。

- self-hosted：用户自行启动，doclib 或其它客户端连接指定 URL
- managed：由 `mineru server` / doclib 在运行时拉起和停止 parse-server 进程

managed 是生命周期管理方式，不是用户直接执行的命令模式。

## 5. Usage

api-server 可自动推荐，也可显式指定精确集合：

```bash
mineru-kit api-server
mineru-kit api-server --tiers auto
mineru-kit api-server --tiers flash,standard --port 8000
mineru-kit api-server --tiers advanced --preload-models
mineru-api --tiers flash,basic,standard,advanced
python -m mineru.parser.api_server --tiers flash
```

显式集合包含 standard 或 advanced 时，依赖检查与预加载合并使用现有 Standard 模型配置；只有 basic 时使用 Basic 配置。共享模型只加载一次，发布集合仍严格等于指定列表。

OCR 模式通过每次 `POST /v1/tasks` 请求的 `ocr_mode` 设置，可选 `auto`、`txt`、`ocr`，省略时为 `auto`。启动参数 `--ocr-mode` 已移除，传入会报错；Python `create_app()` 同样不再接受该参数。

模型默认在首次解析时懒加载。`--preload-models` 会在 Basic 或 Standard 服务启动时提前初始化所需模型或 VLM 客户端，并在失败时让能力接口返回明确错误；Flash 没有本地模型，该参数对 Flash 无操作。Doclib managed parse-server 会自动启用模型预加载。

### 服务日志级别

`mineru-kit api-server`、`mineru-api` 和 `python -m mineru.parser.api_server` 均支持 `--log-level`，
可选 `critical`、`error`、`warning`、`info`、`debug`、`trace`，大小写不敏感。未显式传入时使用全局
`log.level`（默认 `info`，可被 `MINERU_LOG_LEVEL` 覆盖）。该参数仅控制服务侧日志：Uvicorn 的启停、
HTTP 访问、ASGI 日志和 API 服务自身日志。模型加载/解析阶段的 Loguru 默认 stderr sink 不受该参数
影响，始终跟随全局 `log.level`（即 `MINERU_LOG_LEVEL` > YAML `log.level` > 默认 `info`）；宿主程序
显式添加的 Loguru sink 和 tqdm 进度条同样保留各自原有设置。

```bash
mineru-kit api-server --log-level warning
mineru-api --log-level error
python -m mineru.parser.api_server --log-level debug
```

Doclib（包括命令行解析时使用的本地服务及自动重启）、Gradio 和 Router 自动拉起的 api-server
均显式使用 `--log-level warning`，保留警告与错误并隐藏服务 info 日志（含每条 HTTP 请求的访问日志）；
这些托管子进程的模型/解析日志仍由 `MINERU_LOG_LEVEL` 控制，例如排障时设
`MINERU_LOG_LEVEL=debug` 后启动 `mineru-kit webui`，可在控制台看到模型 DEBUG 日志而不会被
HTTP 访问日志刷屏。连接外部服务时，其日志级别由服务部署者控制。
`mineru-kit parse` 的本地模式直接调用 parser，不启动 api-server。

### 使用远程 MinerU VLM 服务

```bash
mineru-kit api-server --tiers flash,standard --vlm-server-url http://127.0.0.1:30000/v1
mineru-api --vlm-server-url https://vlm.example.com/proxy/v1 --vlm-model mineru-model --preload-models
```

两个命令和 `python -m mineru.parser.api_server` 支持相同的连接参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vlm-server-url` | 空 | 远程服务地址，空值选择本地 VLM |
| `--vlm-api-key` | 空 | 远程 VLM 服务的 Bearer Key |
| `--vlm-model` | 空 | 上游模型名；为空时通过 `/v1/models` 唯一发现 |
| `--vlm-http-timeout` | 600 | VLM HTTP 读写超时，单位秒，必须为正整数 |
| `--vlm-max-concurrency` | 100 | VLM 推理请求并发，必须为正整数 |

表中为未配置时的默认值。命令行只覆盖显式传入的字段，其余读取全局 `model.vlm`；
环境变量覆盖 YAML。详见 [远程 VLM 配置](../config.md#远程-vlm)。例如：

```bash
export MINERU_MODEL_VLM_SERVER_URL=http://127.0.0.1:30000/v1
export MINERU_MODEL_VLM_API_KEY=your-vlm-key
mineru-kit api-server --vlm-http-timeout 120 --vlm-max-concurrency 16
# 本次启动覆盖全局地址，恢复本地 VLM：
mineru-kit api-server --vlm-server-url ""
```

`--api-key` 认证解析 API 的调用方，`--vlm-api-key` 认证 API server 对上游 VLM 的请求，
两者独立。远程连接只用于 Standard/Advanced 的 VLM 推理，仍需本地 Hybrid 模型；
启动依赖预检不要求本地 VLM 引擎。Basic/Flash 不会连接远程 VLM。

启用 `--preload-models` 时，Standard 服务连接上游进行模型发现或校验，并预加载本地
Hybrid 模型；否则上游连接推迟到首次 Standard/Advanced 解析。连接、认证或模型校验失败
通过已有预加载错误或解析任务错误返回，不自动回退本地 VLM。推理重试继续使用底层默认设置。

Python 可使用 `create_app(vlm_config=VlmConfig(...))` 完整覆盖全局配置。配置由服务实例
传入其任务，不写回全局设置。此功能不增加 HTTP 请求字段；`/v1/models` 和 `/v1/tiers`
继续发布 MinerU 逻辑模型及质量档位，上游 serving 模型别名仅用于推理请求。

启动完成后，HTTP API 不暴露 backend。`GET /v1/tiers` 也不新增 backend 字段；调用方如需推断实现，只能从 `current_model` 做弱推断。

裸 `vlm` / `hybrid` 不是合法的 api-server 启动 backend；它们只可作为 Middle JSON 来源标记或内部分类概念。

正式参数分层：

### 稳定公开参数

- host / port
- tier，单值 `flash` / `basic` / `standard`
- no-flash
- no-advanced
- preload-models
- API key
- vlm-server-url / vlm-api-key / vlm-model / vlm-http-timeout / vlm-max-concurrency

### 稳定解析参数

- disable-image-analysis
- concurrency
- upload-dir
- retention-seconds（默认 86400 秒，0 关闭自动回收）
- url-timeout
- allow-http-source

### 专家参数

- 当前 `mineru-kit api-server` 命令层不暴露 `--backend`

`--reload` 不进入正式命令设计。

本地 api-server 默认监听 loopback。它可以通过 `--api-key` 设置固定 API Key；默认不设置 API Key。设置后，客户端必须发送 `Authorization: Bearer <api-key>`。

## 6. API 覆盖范围

`mineru-kit api-server` 目标是实现 v1 API（非 doclib API）中的绝大多数 path。

当前明确排除：

- chat 的两个 path

同时明确不实现：

- doclib 的 `/docs`
- `/parses`
- `/search`
- `/invalidate`

完整设计背景见 [ADR-0017](../decisions/0017-mineru-kit-api-server-command.md)。

## 资源保留、引擎状态与离线文档

API 创建参数 `create_app(retention_seconds=...)`、API/Router 的 `--retention-seconds`
及环境变量 `MINERU_API_RETENTION_SECONDS` 使用同一设置，显式参数优先，默认 86400 秒。
零关闭自动回收；上传的显式 `expires_after` 期限继续生效。每 300 秒扫描一次，已到期资源
立即对新的读取/提交返回 404，磁盘删除可延迟至下一次扫描。任务和产物的保留时间从任务
取消或推理结束后的实际清理完成时计算，排队和运行时间不计入。长时间运行任务已产生的
产物在清理完成前不会到期。

```bash
mineru-kit api-server --retention-seconds 172800
mineru-kit router --retention-seconds 172800
# 持久保留：
mineru-kit api-server --retention-seconds 0
```

源文件的多个名字视图和多个任务共享内容寻址字节。排队、运行、哈希预检、下载、写入
及跨 worker 传输期间保护字节，最后一个引用释放后才允许回收。元数据保持内存模式；
服务重启后旧资源 ID 不恢复，启动扫描只清理超过保留时间的受管无引用文件。一个上传目录
只能由一个 API 服务实例占用；多实例应分别配置目录。Router 同步回收过期路由及源缓存，
等待重试的输入副本清理仍由后台处理，过期资源不重新向用户开放。

明确的本地引擎死亡（例如 vLLM `EngineDeadError`）使健康检查和后续提交返回现有错误
封装及 HTTP 503，错误码为 `engine_dead`。当前任务取消后等清理完成，排队任务失败，
用户主动取消仍为 `canceled`。普通解析错误或远程超时不会判定本地引擎死亡。Router
立即剔除故障 worker；托管进程按现有启停超时重建，失败按 5/10/20/30 秒退避；外部
worker 由其管理方恢复。旧 generation 的未完成任务保留失败快照供查询至到期。

`/docs`、`/redoc`、`/openapi.json` 默认启用，JS/CSS/图标位于 `/docs/assets` 下并来自
安装包。Swagger 的 Bearer 授权和 OAuth 回调支持代理 `root_path`；ReDoc 的品牌图标也
使用本地资源。`MINERU_API_ENABLE_FASTAPI_DOCS=false` 同时关闭文档入口及资源挂载。

Torch 小模型首次初始化时限制 CPU 线程，并传播到后续工作线程。有效 OMP/MKL 显式
配置优先，否则依次使用 `MINERU_CPU_NUM_THREADS`、`MINERU_INTRA_OP_NUM_THREADS`、8。
按 CPU 亲和性及 cgroup 配额裁剪，且不调高已有更低值。单独设置 OpenBLAS 不视为 Torch
线程限制。纯 ONNX/Flash 的启动路径保持惰性加载。

### 文件准备与解析排队

Tasks 使用 waiting-file 表示等待上传、服务端下载或校验，准备完成后进入 pending，取得解析名额后进入 running。
URL 输入由独立准备阶段处理，其并发上限沿用 concurrency；等待解析名额的已准备输入也占准备预算，防止源字节无界缓冲。准备阶段不占解析名额，已上传就绪的文件可以正常执行。
文件 duration_ms 包含实际下载、校验、解析和导出耗时，不包含准备或解析队列等待时间。
