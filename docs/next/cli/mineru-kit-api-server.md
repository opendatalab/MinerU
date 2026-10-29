# mineru-kit api-server

状态: Implemented
读者: 服务部署者、核心开发者、`mineru server` 集成开发者
范围: `mineru-kit api-server` 的定位、self-hosted 边界、与 doclib 的协作和参数契约
非目标: 统一 REST API 的字段级定义；模型服务内部实现
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru-kit api-server` 是正式的 self-hosted parse-server 启动入口。`--tier` 是 PDF/图片的最高开放档位，接受 `flash/basic/standard/advanced/auto`；省略时根据硬件推荐上限。所有服务都支持现有原生文档格式，固定使用 flash。

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

启动上限包含自身及全部较低档位。`/v1/tiers` 仅发布 PDF/图片可选档位，不用于判断原生文档能否解析。

| api-server 启动配置 | PDF/图片公开档位 | PDF/image 请求未指定 tier 时 |
|---------------------|-------------|--------------------|
| `--tier flash` | `flash` | `flash` |
| `--tier basic` | `flash`、`basic` | `basic` |
| `--tier standard` | `flash`、`basic`、`standard` | `standard` |
| `--tier advanced` | 全部四档 | `standard` |

Tasks 的请求 `tier` 仍可选，只对 PDF/图片生效。省略或 JSON `null` 时按 `standard -> basic -> flash -> advanced` 选择，仅开放 advanced 时也可使用默认值。所有服务均支持 OFD/EPUB/Office/HTML/MHTML/CSV/TSV 等现有原生格式，固定使用 flash，不检查公开列表中是否有 flash。非法枚举或空字符串仍报错。完整规则见 [Tasks](../api/tasks.md#tier默认选择与文件作用范围)。

### 自动推荐

省略 `--tier` 等价于 `--tier auto`。启动时取推荐集合的最高档作为上限并展开，运行中不重新选档：

| 硬件条件 | 启动上限 | PDF/图片公开集合 |
|----------|----------|------------------|
| 进程可用 CPU 数不足 4，且无达标 GPU | flash | `flash` |
| 进程可用 CPU 数至少 4，且无达标 GPU | basic | `flash,basic` |
| GPU 已确认可用，且选定设备显存／预算至少 5.5 GiB（CPU 核数不限） | advanced | 全部四档 |

CPU 数按逻辑 CPU 计算，并考虑进程亲和性和 Linux cgroup 配额。集显沿用有效 OCR 校准的 GPU 时间不高于 CPU 时间 0.8 倍的判定。探测失败或设备身份不明确时按 CPU 条件推荐。显式上限可以覆盖推荐，随后仍执行正常依赖预检。

5.5 GiB 按 `5_905_580_032` 字节比较，不四舍五入、不累加多卡，也不使用 `MINERU_VIRTUAL_VRAM_SIZE`。独显读取专用显存总量；Apple 统一内存使用 Metal 推荐工作集，Windows/Linux 集显分别使用 DXGI 本地预算与 Vulkan `heapBudget`。预算比较使用上限，不减去当前用量；未知预算不以系统总内存替代。容量证据可通过 `mineru-kit hardware detect --json` 的 `gpu_memory` 查看。

GPU 类型与 OCR、llama.cpp 和硬件诊断使用共享分类；目标运行后端可见且原生身份确认的独显可以免测速，但仍须满足 5.5 GiB 内存门槛才能开放 Standard/Advanced。ggml 的通用 GPU 标志、未知类型及虚拟 GPU 均不直接视为独显；同型号多卡无法唯一关联时保持保守结果。各后端可以选择不同 GPU，共用分类和优先级，不强制跨后端同卡。

自动推荐先解析有效小模型后端。Torch（包括 auto 最终选择 Torch）验证实际设备可用性及容量：选中 CUDA/MPS/XPU 等设备的显存／预算至少 5.5 GiB 时推荐 advanced；显式 CPU 或设备不可用时按 CPU 条件推荐，不下载 ONNX 校准模型或改写其配置/记录。Torch 推荐不依赖 llama.cpp 库存。

ONNX 分支先探测硬件并读取已有校准记录。如果 CPU 至少 4 核或存在内存达标的兼容 GPU 候选，Windows/Linux 会在发布服务能力前同步执行现有 ONNX 设备选择流程，再根据实际选择和容量确定最终上限。CPU 不足 4 核且无内存达标候选、macOS 及其他平台跳过这一步；显式 `--tier` 和 Python 精确集合也不触发这项启动校准。

启动选型继续遵守 `model.onnx_device`：`auto` 才按设备策略决定是否测速，显式或已保存的 `cpu/gpu` 不强制重新校准。Windows 确认的独显，以及 Linux 通过兼容检查、由 Vulkan 唯一识别并配置物理绑定的独显，均跳过 OCR 校准；首次真实 GPU 会话成功后才保存设备偏好。集显或类型无法确认的候选继续按 0.8 阈值测速。需要测速时会按正常模型生命周期准备或下载 OCR rec 权重，并复用已有进程缓存、文件锁、采样阈值及失败回退。下载和模型准备不计入 15 秒校准子进程超时。校准失败或结果不稳定不会阻止后续推荐，但不作为 GPU 胜出证据。

实际 OCR 校准把证据保存在 `<配置文件路径>.onnx-calibration.json`，仅复用设备、运行时版本与 CPU 配置匹配的有效结果；旧 cpu/gpu 偏好及手动 GPU 设置不视为校准证据。启动阶段产生的新证据在本次最终推荐中生效，之后 `/v1/tiers` 在服务生命周期内固定。

远程 VLM 配置不自动扩大推荐上限或放宽本机内存门槛，也不增加 VLM 性能测试。上述 OCR 选型不要求 `--preload-models`；该参数仍另外控制完整模型或 VLM 客户端的提前初始化。

`mineru-kit hardware detect` 使用同一结构化推荐流程；`hardware info` 只读输出诊断报告，详见 [hardware 命令](mineru-kit-hardware.md)。API 工厂的 VLM 配置覆盖也参与推荐，避免与实际运行配置不一致。

### 内嵌部署与 SaaS

Python 工厂保留 `create_app(tiers={"standard"})` 的非空精确集合接口。这里的集合只约束 PDF/图片：上述服务只发布 standard，却仍能解析原生文档并返回实际 flash 结果。`tiers=None` 使用硬件推荐上限。空集合、裸字符串和未知档位均报错。

SaaS 可按共同契约仅发布 standard，原生文档由同一服务解析。无需新增原生能力声明、额外请求字段或本地补充服务。Router 对原生文件仅检查 worker 健康状态、来源和输出格式，转发时省略 tier；PDF/图片继续携带已确定的档位。

### 参数与托管迁移

`--tiers`、`--no-flash`、`--no-advanced` 均不接受，也不保留别名。

| 原完整集合 | 新启动上限 |
|------------|------------|
| `--tiers flash` | `--tier flash` |
| `--tiers flash,basic` | `--tier basic` |
| `--tiers flash,basic,standard` | `--tier standard` |
| `--tiers flash,basic,standard,advanced` | `--tier advanced` |

非连续精确集合只用于 Python 工厂或部署方自己的服务配置，不能通过本地 CLI 表达。原 `--tiers standard` 只开放 standard，与新 `--tier standard` 的上限展开语义不同。

WebUI、Router、Doclib 的托管 standard 均只开放到 standard，不再隐含 advanced；需要时显式配置 advanced。WebUI 默认配置为 auto，Router、Doclib 仍为 standard。Doclib 的 `parse_server.local.managed_tier` 支持 basic、standard、advanced，其原生执行路径保持现状。Advanced 的模型检查与下载使用 Standard 模型配置。

## 4. self-hosted 与 managed

`mineru-kit api-server` 对用户只对应 self-hosted 场景。

- self-hosted：用户自行启动，doclib 或其它客户端连接指定 URL
- managed：由 `mineru server` / doclib 在运行时拉起和停止 parse-server 进程

managed 是生命周期管理方式，不是用户直接执行的命令模式。

## 5. Usage

启动示例：

```bash
mineru-kit api-server
mineru-kit api-server --tier auto
mineru-kit api-server --tier standard --port 8000
mineru-kit api-server --tier advanced --preload-models
mineru-api --tier basic
python -m mineru.parser.api_server --tier flash
```

Standard/Advanced 上限复用 Standard 模型配置，Basic 上限使用 Basic 配置，共享模型只预加载一次。原生文档始终保留独立 Flash 执行路径，内部 Flash 能力不会额外开放 PDF 的 Flash 请求。

OCR 模式通过每次 `POST /v1/tasks` 请求的 `ocr_mode` 设置，可选 `auto`、`txt`、`ocr`，省略时为 `auto`。启动参数 `--ocr-mode` 已移除，传入会报错；Python `create_app()` 同样不再接受该参数。

模型默认在首次解析时懒加载。`--preload-models` 会在 Basic、Standard 或 Advanced 服务启动时提前初始化所需模型或 VLM 客户端，并在失败时让能力接口返回明确错误；Flash 没有本地模型，该参数对 Flash 无操作。Doclib managed parse-server 会自动启用模型预加载。

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
mineru-kit api-server --tier standard --vlm-server-url http://127.0.0.1:30000/v1
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
- tier，单值 `auto` / `flash` / `basic` / `standard` / `advanced`
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
