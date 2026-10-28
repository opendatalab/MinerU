# mineru-kit hardware

状态: Implemented

`hardware detect` 推荐当前环境的 API 启动上限；`hardware info` 生成只读的问题反馈报告。两者默认输出可读文本，`--json` 输出同一份结构化数据。日志与进度进入 stderr，stdout 只保存结果。

```bash
mineru-kit hardware detect
mineru-kit hardware detect --json
mineru-kit hardware info
mineru-kit hardware info --json > hardware-info.json
```

`hardware` 不带子命令时显示帮助。没有 `--quiet` 或 `auto-tier` 别名。配置沿用 `MINERU_CONFIG`、`MINERU_HOME` 和现有后端/设备环境变量。

## 自动推荐

命令与 `api-server --tier auto` 共享结构化推荐结果。先解析有效小模型后端，再确定设备与上限：

| 有效后端 | 行为 |
|----------|------|
| Torch + 可用 CUDA/MPS/XPU 等加速设备 | 跳过 OCR 测速；选定设备的显存／预算至少 5.5 GiB 时推荐 advanced，否则按 CPU 条件推荐 |
| Torch + 显式 CPU 或加速设备不可用 | 跳过测速，进程可用逻辑 CPU 至少 4 时推荐 basic，否则 flash |
| ONNX | CPU 至少 4 核或存在内存达标的兼容 GPU 候选时，在 Windows/Linux 同步完成现有设备选型；实际选中 GPU 且内存达标时开放全部四档，不再额外检查性能证据 |

自动推荐的 CPU 条件只开放 Basic：进程可用逻辑核数至少 4 时开放 `flash,basic`，默认请求 Basic，否则只开放 Flash。计数遵守进程亲和性和容器 CPU 配额。GPU 经现有选型确认可用且显存／预算至少 `5_905_580_032` 字节时，独立于 CPU 核数开放全部四档；默认请求仍为 Standard，Advanced 必须显式请求。

内存检查只使用确认设备的原始字节值，不四舍五入、不累加多卡，也不读取 `MINERU_VIRTUAL_VRAM_SIZE`。独显按专用显存总量判断；Apple Metal 按推荐工作集预算，Windows 集显按 DXGI 本地预算，Linux 集显按 Vulkan `heapBudget` 判断。预算比较使用上限，不减去当前用量。不能唯一关联设备、容量未知、预算扩展不可用或查询失败时按 CPU 条件推荐；系统总内存和共享内存理论上限不能替代 GPU 预算。

`small_backend=auto` 最终选择 Torch 时同样免校准。显式设备字符串必须经过实际可用性检查；检查失败会给出保守结果及原因，不修改设备配置。Torch 推荐不依赖 llama.cpp 库存，不下载 ONNX 校准权重，也不写入 `model.onnx_device` 或 OCR 校准记录。

ONNX 分支保留既有兼容策略、0.8 阈值和进程缓存。Windows 确认的独显，以及 Linux 通过兼容检查、由 Vulkan 唯一识别并配置物理绑定的独显，均免 OCR 校准，报告 `status=skipped, reason=discrete_gpu`；首次真实 GPU 会话成功后才保存 `gpu`，硬件推荐本身不生成校准记录。`onnx_device=auto` 时，集显或类型无法确认的候选继续沿用现有设备选型测速，首次可能下载 OCR rec 模型；下载和派生图准备不计入 15 秒校准 worker 超时。校准失败、超时或不稳定导致本次设备选型回退 CPU 时，按 CPU 条件推荐，并携带警告。

`onnx_device=gpu` 无论来自用户配置还是之前的自动选型，都不触发测速；实际选中兼容 GPU 后，仅检查该设备的显存／预算是否达到 5.5 GiB。校准记录缺失、损坏、过期或记录 GPU 较慢，均不阻止内存达标的 GPU 开放 Standard/Advanced。此时推荐原因是 `configured_gpu`，校准状态为 `status=skipped, reason=configured_gpu`。内存不足或未知时报告相应内存原因。记录摘要只供诊断，读取失败会附带警告，不改变设备和档位结论。`onnx_device=cpu` 仍遵循 CPU 规则。

llama.cpp 自动集显路径同样按有效小模型后端分流：Torch 使用加速可用性结果，不再绕回 OCR 校准；ONNX 保留原有性能结果和设备身份匹配。原生独显优先、显式 CPU/GPU 参数、物理绑定和兼容处理保持不变。

OCR、llama.cpp、档位推荐与诊断共用 GPU 分类和双向唯一身份匹配。先按各后端的实际可见性和兼容性过滤，再按独显、集显、虚拟 GPU、未知类型排序；后端之间无需选择同一张卡。ggml 的 GPU/IGPU 标志不再直接作为独显判据，Windows 使用 D3D12 UMA、Linux 使用 Vulkan 原生分类。无法唯一关联的设备保留未知类型，不授予独显免校准或仅凭类型开放 Advanced。

JSON 报告保留 `schema_version=1`，包含 `ceiling`、`pdf_image_tiers`、`default_request_tier`、`reason`、`cpu_count`、`small_backend`、`vlm_engine`、`device`、`calibration` 和 `warnings`，并增补 `gpu_memory`。内存证据包含 `device`、`identity`、`memory_bytes`、`kind`（`dedicated`／`budget`／`unknown`）、`source`、`warning`；没有候选设备时为 `null`。校准状态来自执行结果；Torch 使用 `status=skipped, reason=torch_backend`，不将可用性结论标成性能实测通过。

本次实际选型落在 CPU 或选型失败时，不会凭静态显卡库存重新开放高档位。性能判断归属 ONNX 自动设备选型，auto tier 不增加独立测速或历史性能证据门槛。设备容量信息不进入 OCR 校准指纹，每次启动读取当前选定设备的内存容量。每个服务在启动阶段形成一次最终开放集合，运行过程中不动态升降档位。

开放 Advanced 后，请求省略 tier 仍默认 Standard。推荐不代表模型已准备好，API 服务继续执行依赖预检；显式启动上限和 Python 精确集合不进入自动推荐。API 工厂的 VLM 配置覆盖参与共享推荐，远程 VLM 显示为 `http-client`，不自动扩大本机上限。

## 环境报告

`info` 不调用自动推荐或 OCR 校准，不下载、加载模型权重或写配置。运行时导入和原生 GPU 探测使用独立子进程，每项最多等待 15 秒。某项失败时继续报告其他信息，并区分未安装、导入失败和超时。

- 系统、Python 版本与解释器路径，MinerU 版本与实际模块路径。
- CPU 型号、物理/逻辑/进程可用核数，基础/当前/最大频率（MHz）；内存总量及可用量（bytes）。
- 系统可探测到的全部独显、集显和虚拟 GPU，以及当前进程运行时可用设备。保留名称、身份、驱动及可获得的显存/共享内存信息。
- GPU 的 `kind` 与选型共用同一套分类；`kind_source` 标识分类来源，`source` 保留硬件资料的采集来源，`unified_memory` 单独记录已知的统一内存属性。
- 已安装的软件包版本与位置：MinerU、DocVortex、mineru-llama-cpp、mineru-vl-utils、torch、transformers、ONNX Runtime及EP、vLLM、LMDeploy、mlx-vlm、pypdfium2。未安装的包不列入软件版本列表。
- Torch 导入、编译使用的 CUDA/ROCm 版本和后端可用性；ORT provider 与相关引擎的探测结果。
- 后端、设备、线程及可见设备限制的有效配置与来源；OCR 校准记录的存在状态、环境匹配、稳定性、耗时和阈值。

未知值为 JSON `null`，文本显示“未知”。例如 Apple Silicon 通常不报告可可靠读取的 CPU 主频，统一内存也不作为独立显存。系统或容器限制导致库存不完整时保留错误说明；不从型号名称猜测设备类型或频率。

配置报告使用明确白名单，不输出认证信息、完整远程服务地址或整个环境变量集合。诊断报告中的本地路径用于确认解释器和实际加载位置。

## 退出码

能形成推荐结果或环境报告时退出码为 0，包括带警告的保守推荐和局部探测失败。参数、配置错误或无法形成结果时返回非零退出码。
