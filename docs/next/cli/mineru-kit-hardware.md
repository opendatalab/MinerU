# mineru-kit hardware

状态: Implemented

`hardware detect` 推荐当前环境的 API 启动上限；`hardware info` 生成只读的问题反馈报告。两者默认输出可读文本，`--json` 输出同一份结构化数据。日志与进度进入 stderr，stdout 只保存结果。

```bash
mineru-kit hardware detect
mineru-kit hardware detect --json
mineru-kit hardware info
mineru-kit hardware info --json > hardware-info.json
```

`hardware` 不带子命令时显示帮助。根帮助及各级硬件帮助按需加载命令描述，不读取用户配置或启动硬件探测。没有 `--quiet` 或 `auto-tier` 别名。配置沿用 `MINERU_CONFIG`、`MINERU_HOME` 和现有后端/设备环境变量。

## 自动推荐

命令与 API、WebUI 的自动启动共享结构化推荐结果。先解析有效小模型后端，再直接采用其 GPU/CPU 设备结论，不查询显存容量：

| 设备结论 | 进程可用逻辑核数 | PDF/图片开放档位 | 请求默认档位 |
|----------|------------------|------------------|--------------|
| GPU | 不限 | flash、basic、standard、advanced | standard |
| CPU | 至少 12 | flash、basic、standard | standard |
| CPU | 4–11 | flash、basic | basic |
| CPU | 不足 4 | flash | flash |

CPU 计数沿用进程亲和性和 Linux cgroup 配额限制。读取失败回退为 1 核并记录警告，不影响已确认的 GPU 结果。CPU 核数再多也不自动开放 Advanced。

Torch（包括 `small_backend=auto` 最终选择 Torch）使用经过可用性验证的加速设备结论，包括 CUDA/ROCm、MPS、XPU 等。显式 CPU、设备不可用或探测失败时按 CPU 规则推荐，并保留原因与警告。该分支不调用 ONNX 选型、不下载 OCR 校准权重，也不修改 ONNX 配置或校准记录。

ONNX 分支在所有 CPU 核数下调用现有 `resolve_onnx_device()`，复用进程缓存、兼容策略、显式配置和校准。Windows 确认的独显，以及 Linux 通过兼容检查、由 Vulkan 唯一识别并配置物理绑定的独显，保留免 OCR 校准路径。`onnx_device=auto` 下需要测速的候选沿用 0.8 阈值；首次可能准备或下载 OCR rec 模型。下载和派生图准备不计入 15 秒校准 worker 超时。校准失败、超时或不稳定时使用本次 CPU 结果推荐。

`onnx_device=gpu` 无论来自用户配置还是先前自动选型，都不强制测速。实际选中 GPU 就开放全部四档；校准记录缺失、损坏、过期或记录 GPU 较慢均不额外阻止开放。此时推荐原因是 `configured_gpu`，校准状态为 `status=skipped, reason=configured_gpu`。记录摘要仅供诊断，读取失败附带警告，不改变设备和档位结论。`onnx_device=cpu` 及不受支持平台的 ONNX CPU 回退均遵循 CPU 规则；ONNX 分支不额外探测 Apple Metal。

推荐与 llama.cpp 自动模式复用相同的小模型设备选择入口，不读取 llama 专用参数，也不额外探测 ggml 库存或匹配显卡身份。VLM 使用其他引擎或远程服务不会自动扩大推荐上限。

JSON 报告保持 `schema_version=1`，包含 `ceiling`、`pdf_image_tiers`、`default_request_tier`、`reason`、`cpu_count`、`small_backend`、`vlm_engine`、`device`、`calibration` 和 `warnings`。兼容保留 `gpu_memory` 字段，固定为 `null`；文本不再展示内存门槛。Torch 校准状态为 `status=skipped, reason=torch_backend`，不将可用性标成性能实测通过。

每个服务在启动阶段形成一次最终开放集合，运行中不动态升降档位。选型落在 CPU 或失败时按 CPU 核数推荐，不凭静态显卡库存升级。显式启动上限和 Python 精确集合跳过自动推荐；依赖预检、可选预加载和原生文档固定 Flash 的行为保持不变。开放全部四档时请求默认仍为 Standard。

## 环境报告

`info` 不调用自动推荐或 OCR 校准，不下载、加载模型权重或写配置。运行时导入和原生 GPU 探测使用独立子进程，最多并发 4 项，并与系统信息采集重叠进行；每项子进程启动后最多等待 15 秒。某项失败时继续报告其他信息，并区分未安装、导入失败和超时。

默认文本先输出系统、Python 和 MinerU 版本，再依次输出 CPU/内存、已安装软件、完成合并的显卡库存、运行时探测、有效配置及设备选择与校准信息。每节完成后立即刷新，重定向到管道时也无需等到整份报告完成。`--json` 仍一次输出完整对象，原有字段和结果顺序保持稳定，`schema_version=1`；ONNX 运行时结果补充 `selected_device`，记录只读选型及原因。

- 系统、Python 和 MinerU 版本；JSON 额外保留解释器与实际模块路径。
- CPU 型号、物理/逻辑/进程可用核数，基础/当前/最大频率（MHz）；内存总量及可用量。文本容量统一按 `1024³` 换算为三位小数的 GiB，JSON 保留原始字节数。
- 系统可探测到的全部独显、集显和虚拟 GPU，以及当前进程运行时可用设备。保留名称、身份、驱动及可获得的显存/共享内存信息。
- GPU 的 `kind` 与选型共用同一套分类；`kind_source` 标识分类来源，`source` 保留硬件资料的采集来源，`unified_memory` 单独记录已知的统一内存属性。
- 已安装的软件包版本：MinerU、DocVortex、mineru-llama-cpp、mineru-vl-utils、torch、transformers、ONNX Runtime及EP、vLLM、LMDeploy、mlx-vlm、pypdfium2。JSON 额外保留安装位置，未安装的包不列入软件版本列表。
- 运行时摘要：实际小模型后端与 VLM 引擎、各运行时状态和版本；Torch 只展示选中设备及其型号/容量，ONNX 展示可用 provider 和选中候选，llama.cpp 展示可见设备名称。
- 后端、设备、线程及可见设备限制的有效配置与来源；OCR 校准记录的存在状态、环境匹配、稳定性、耗时和阈值。

运行时文本不直接展开内部 `selection`、`backends` 或适配器字典。Torch 的 CUDA/ROCm 构建版本仅在选中相应设备且版本已知时展示；CPU、MPS 及 NPU/GCU/MUSA/MLU/SDAA 等扩展设备直接显示实际设备名，不列出其他不可用后端，也不套用无关构建字段。这些构建版本来自 Torch 自身，不代表系统驱动或另行安装的工具链版本。

Windows DXGI 和 Linux Vulkan 的原生库存用于补充“系统显卡”，不在运行时文本中重复展开；macOS 使用系统工具采集，不执行 DXGI/Vulkan 这两项库存探测。采集错误仍进入提示，完整后端矩阵、所有适配器及 LUID/PCI/厂商标识等诊断字段继续保留在 `--json` 中。

“设备选择与校准”展示当前小模型后端、设备及选择依据。Torch 直接使用运行时的设备检查结果，并说明不适用 ONNX OCR 校准；ONNX 复用实际选型的候选规则，解释显式 CPU/GPU、独显免校准、不可用或待校准的情况。只读检查不会初始化 GPU 推理会话；需要测速时，设备显示“待校准”（JSON `selected_device.device` 为 `null`），不预先声称 CPU 或 GPU 胜出。

存在有效校准记录时展开详细信息；没有详细记录时仅展示设备选择与原因，不列出空的测速字段。损坏或不可读的记录保留路径及错误。当前为 Torch 时，已有 ONNX 记录明确标注为不参与当前选择。JSON 的原始 `calibration` 对象仍保留全部字段。

未知值为 JSON `null`，文本显示“未知”。例如 Apple Silicon 通常不报告可可靠读取的 CPU 主频，统一内存也不作为独立显存。系统或容器限制导致库存不完整时保留错误说明；不从型号名称猜测设备类型或频率。

独立显存采集能力和 Windows PCI 身份修复继续保留，其结果不参与自动 tier。Windows 原生查询不依赖 Torch：DXGI 库存通过 `D3DKMTOpenAdapterFromLuid` / `D3DKMTQueryAdapterInfo(KMTQAITYPE_ADAPTERADDRESS)` 补充物理 PCI 位置，内存仍由对应适配器的 DXGI 接口读取。身份匹配优先使用 LUID 和已知 PCI 坐标，不猜测缺失字段；查询失败或多个同等匹配保留未知。读取预算成功不代表模型推理已验证。

配置报告使用明确白名单，不输出认证信息、完整远程服务地址或整个环境变量集合。文本隐藏解释器、模块和软件安装路径，保留配置文件及校准记录路径；JSON 中的完整本地路径仍可用于确认解释器和实际加载位置。

## 退出码

能形成推荐结果或环境报告时退出码为 0，包括带警告的保守推荐和局部探测失败。参数、配置错误或无法形成结果时返回非零退出码。
