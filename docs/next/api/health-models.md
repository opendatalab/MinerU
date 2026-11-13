# Health、Models 与 Tiers

实现：`mineru/parser/api_server.py`、`mineru/kit/router/app.py`。
这些发现接口不要求 API Key，但仍可能因预加载失败或无健康 worker 返回 503。
Router 仅聚合健康且管理状态为 active 的节点；draining 节点仍可服务已有任务，但不再贡献对外发现能力。参见 [upstream 管理](router.md)。
它们描述当前服务，不能推断公网 SaaS 的套餐或部署版本。

## GET `/v1/health`

返回 `status="ok"`、服务 `version` 和 `features`。
Python Server 的 features 包含 `task_output_formats=["default","docx","html","latex"]`、
`sources=["upload","url"]`。Router 从健康 worker 合并格式和输入类型，没有健康 worker 时返回 503。
健康检查不执行完整推理；成功不代表任意输入必定解析成功。

Python Server 已完成设备选型时，附加 `runtime: {"device": "cpu"}` 或
`runtime: {"device": "gpu"}`。该值来自不可变进程快照，表示默认本地推理模式；
gpu 包含 MPS、XPU、NPU 等非 CPU 加速器，不代表每个模型、显式覆盖或远程 VLM 的实际设备。
模型局部 CPU 回退不会改写该模式。

尚未选型时省略 `runtime`，不填充 CPU 占位。显式 tier 的惰性启动可能没有此字段；
自动 tier 在启动时完成选型后即能返回。health 请求本身不探测设备、下载模型或执行校准。
客户端应接受字段缺失，并将其解释为未提供设备信息；旧服务同样可能缺少该字段。
Router 不返回单一 `runtime.device`，因为其健康 worker 可能混合 CPU/GPU，Router 自身设备也不能代表 worker。

模型预加载失败使用 `model_preload_dependency_missing/model_preload_files_missing/model_preload_device_unavailable/model_preload_failed` 等错误码。

## GET `/v1/models` 与 `/v1/models/{model}`

列表为 `{"object":"list","data":[...]}`。Python Server 发布一个虚拟模型：
`id="mineru-<服务版本>"`，`object="model"`，`created` 为注册时的 Unix 秒，`owned_by="mineru"`，附带 description。
ID 标识服务版本，不是推理引擎，也不是 Tasks 请求参数。
查询不存在或旧版本的 ID 返回 `404 model_not_found`。

Router 合并健康 worker 的模型列表并按 ID 去重。多个版本 ID 表示 worker 版本不同，不是多种解析 tier。

## GET `/v1/tiers`

返回 `object="list"` 与 data 数组，每项包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | 四档枚举 | 该服务允许 PDF/图片请求使用的档位 |
| `description` | string | 档位说明 |
| `current_model` | string 或 null | 当前虚拟模型 ID；未知时可空 |

| Python API 启动配置 | 开放集合 |
|---|---|
| `--tier flash` | flash |
| `--tier basic` | flash、basic |
| `--tier standard` | flash、basic、standard |
| `--tier advanced` | 全部四档 |
| `--tier auto` 或省略 | [硬件推荐](../cli/mineru-kit-hardware.md)的上限及较低档位 |
| `create_app(tiers={"advanced"})` | 仅 advanced；Python 参数是精确集合 |

该列表只约束 PDF／图片。正常可用服务的原生文档仍使用 Flash，不要求列表中存在 flash。
Tasks 省略 tier 时使用 standard → basic → flash → advanced；只开放 Advanced 时才自动选中它。
显式不支持的 PDF/图片档位报错，不改档。完整语义见 [Tasks](https://opendatalab.github.io/MinerU/zh/usage/tasks/)。

## 已加载 LMDeploy 的运行时健康

解析服务每 5 秒探测本应用已创建的 LMDeploy 运行时，单次探测期限 5 秒，后端最多一个在途探测。探测不会初始化惰性模型，也不会为 Flash 服务下载 VLM。

- 确认引擎死亡后锁存 `engine_dead`，健康、能力发现与新任务入口返回 503；运行与排队任务沿既有清理边界失败，不发布未完成结果。原实例不会在同一次请求中自动重建或降级。
- 探测超时、睡眠或暂时不可用返回 503 / `engine_unavailable`，拒绝新推理但不永久标记死亡；后续健康探测恢复后解除。
- Router／Doclib 管理的本地进程沿既有控制通道、关闭超时和恢复策略重建；外部 worker 只停止分配，不由 MinerU 擅自结束其进程。SDK 传播错误，由调用方负责释放和重建。

同一运行时实例的全部关联 owner 都会收到失效状态，缓存命中不能继续交付该死亡实例。健康状态恢复只适用于暂时不可用；永久失效必须释放旧实例，由其宿主管理新的实例。故障请求本身不会静默重试、重建或降低档位。

引擎后台任务退出不一定导致 Python 宿主进程退出。宿主存活时，仍需完成上述错误传播和任务清理；整个托管 worker 退出时，由进程管理者发现退出、清除旧工作状态并重建。CLI 以非零状态退出，SDK 向调用者抛出明确异常；共享库不自行终止调用方进程。

普通单片识别失败仍可回退 MFR。此机制不替代原始引擎崩溃原因调查，也不以解决所有后端崩溃根因为交付前提。日志保留首次错误的异常链；Windows 验收脚本位于 `scripts/acceptance/`，只以受支持的 TurboMind 后端验证故障处理流程。
