# 文档分析的原生异步运行时

PDF 的 high/xhigh 分析在使用本地 vLLM 或远程 HTTP 时，通过常驻运行时执行原生异步推理。
`doc_analyze`、`aio_doc_analyze`、`parse`、`parse_async` 的参数与结果协议保持一致。

## 运行时与共享边界

- 相同有效配置的同步分析、异步分析与 API 预加载共享一个运行时。vLLM 托管路径统一使用 AsyncLLM，
  不同时加载离线 LLM 和 AsyncLLM 两套权重。
- 每个运行时拥有一个事件循环线程。同步入口等待线程安全提交的结果，异步入口等待可取消的 Future；
  调用方可以多次创建、关闭自己的事件循环。
- HTTP 连接池在所属运行时循环中持续复用。地址、模型、凭据及连接配置仍保持隔离。
- API 应用拥有独立租约。关闭一个应用只释放自身租约；其他应用或独立解析仍持有的实例不会因此关闭。
  独立 Python 调用使用进程级模型缓存，退出时清理；显式模型关闭仍由 `shutdown_cached_models()` 完成。

## 并发与取消

- API 的 `--concurrency` 控制同时运行的 job 数，默认仍为 1。同一个 job 内的文件顺序执行。
- `model.vlm.max_concurrency` 控制同一有效配置运行时内的 VLM 在途请求总数，默认 100；
  多份文档共享该额度，不各自获得 100 个名额。
- 单文档按既有窗口顺序执行。准备与回填阶段在线程中运行，本地共享模型阶段按设备串行保护；
  等待 VLM 时释放保护，允许其他文档推进。PDFium 继续使用 DocVortex 的共享锁。
- 取消会传播到实际任务，但只有底层请求及同步阶段清理完成后，才关闭图片/PDF 和释放 job 名额。
  排队取消不会开始解析，已经取消的任务不会再发布结果。
- HTTP 取消保证本地停止等待与清理，远端是否立即停止生成取决于服务端；低频 GPU 指标不能证明即时 abort。
- 本地 LMDeploy、MLX、llama.cpp、Flash/basic 和非 PDF 保持受控线程回退。
  取消这些路径需要等待在途同步阶段结束，不承诺立即中断模型计算。

## 分层实现

MinerU 拥有模型构造、缓存、运行时、文档编排和 API 任务生命周期。mineru-vl-utils 提供原生异步抽取及
HTTP 连接与健康探测清理接口，不关闭调用方提供的引擎或 executor。DocVortex 的基础访问、语义协议和确定性处理职责不变。

## 引擎健康与故障传播

依赖适配层将 vLLM 与 LMDeploy 的明确死亡统一为 `EngineUnavailableError`，保留原始异常链。
普通请求失败和探测超时不使用该异常；批量容错和 MFR 回退不能吞掉明确死亡。

| 后端 | `health_probe_mode` | 健康入口 |
| --- | --- | --- |
| llama.cpp | `sync` | `probe_health(timeout)`，读取原生循环终止状态 |
| LMDeploy | `sync` | `probe_health(timeout)`，在 Pipeline 循环查询 |
| vLLM AsyncLLM | `async` | `aio_probe_health(timeout)`，在运行时所属循环查询 |
| vLLM 离线 LLM、MLX | `none` | 不提供探测；不通过私有对象伪造健康 |

MinerU 托管的 vLLM 同步调用仍使用 AsyncLLM，所以同样具备空闲健康探测。
API 只探测已加载的实例，默认周期和期限均为五秒，每实例最多一个在途探测。
异步探测的外部等待期限不依赖引擎循环，因此引擎循环阻塞时 API 仍能返回暂不可用。

同步句柄与异步代理共享 `RuntimeLifecycle`。明确死亡永久锁存，并通知全部 owner；缓存命中及后续调用拒绝
复用旧实例。健康和接单统一返回 `503 engine_dead`，未完成任务失败并清理结果。托管进程由既有 Router／Doclib
机制替换，外部 worker 仅摘除。CLI 非零退出，SDK 调用方自行释放并重建。

探测超时返回 `503 engine_unavailable`，拒绝新推理但允许继续探测；恢复健康后可解除，不能覆盖已锁存的死亡。
MLX 未知运行时异常只使当前请求失败，不进行 MFR 回退或永久隔离，后续请求仍可使用该实例。
取消依旧等待各后端完成必要清理，不将 Future 取消等同于 GPU 或线程工作结束。

同步与异步 PDF 窗口共同使用准备输入和结果回填函数，保持原生表格优先、文本/公式处理、图像开关与页序。
high 等待外部布局抽取，xhigh 等待两阶段抽取；后置 LLM 增强使用现有协程。

## 验收与性能采集

在项目虚拟环境中执行 `scripts/benchmark_async_analysis.py`；脚本还需要 `psutil`。
配置文件或环境变量提供鉴权信息，不把密钥写入命令行或报告。

```bash
.venv1/bin/python scripts/benchmark_async_analysis.py demo/pdfs/demo1.pdf demo/pdfs/demo2.pdf \
  --server-url http://localhost:30000 --tier standard --ocr-mode ocr \
  --concurrency 1 2 4 8 --repeats 4 --output output/native_async/http.json
```

本地 vLLM 验收使用 `--engine vllm`，不传 `--server-url`，并确保配置中未设置远程地址。
`--mode sync-thread` 用于同步线程调用；比较旧实现时应使用独立基线源码，固定依赖、输入、模型、窗口和采样配置。
`--save-results` 可保存预热结果用于语义回归；`--cancel-after` 记录异步取消返回耗时。

报告包括文档/页吞吐、解析 P50/P95、含排队的延迟、事件循环延迟、线程及 RSS 峰值。
本地 NVIDIA 环境通过 nvidia-smi 采集解析进程及子进程显存；远端显存缺失时保留空值。
单次样本和模拟后端不能替代 GPU 性能验收；源码位置、文件哈希及配置必须与结果一起保留。

上线性能目标为代表性多文档负载吞吐提升至少 25%，或同吞吐下 P95 降低至少 20%，
同时单文档耗时回退不超过 5%。未达到目标时不扩大默认并发，不宣称性能提升。
本地嵌入式 vLLM 必须另做 Linux CUDA 实机验收；远程 LMDeploy HTTP 通过不能代替该项。

## 两仓交付

本次依赖 mineru-vl-utils 新增的 `MinerUClient.aclose()` 与取消清理能力。
开发验证使用配套的本地 editable 源码。正式交付必须先发布包含这些改动的 mineru-vl-utils 版本，
确认包索引可解析后，再提高 MinerU 的依赖下限；未完成设备验收前保留发布门槛。

llama.cpp 要求 `mineru-llama-cpp>=0.2.1`。只有明确循环退出或显式关闭映射为公共致命异常，普通 `EngineError` 仍按请求处理。底层保留原始错误类型、错误码和详情；取消后的原生线程排空由底层负责，运行时沿用共享 owner 通知与缓存拒绝规则。健康快照不提供解码进度心跳，也不将无进度推断为死亡；进程级 abort 仍由宿主管理恢复。

LMDeploy 的健康接口可能使用独立执行器，取消协程不等于工作线程已退出。MinerU 的同步、异步及取消关闭都等待同一健康探测实际排空，再释放 Pipeline；不把依赖库的局部关闭超时当作已排空。托管 worker 的总关闭期限、终止和强制退出仍由现有 Router／Doclib 管理器控制。直接 SDK 调用方也必须保持 Pipeline 及其循环存活直至客户端成功关闭。
