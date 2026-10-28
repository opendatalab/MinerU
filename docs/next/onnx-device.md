# ONNX 设备选择、DirectML 与 WebGPU

`model.onnx_device` 控制 ONNX 小模型栈的 CPU / Windows DirectML / Linux WebGPU 选择：

```yaml
model:
  small_backend: onnx
  onnx_device: auto
```

环境变量 `MINERU_MODEL_ONNX_DEVICE` 高于文件配置。`small_backend=auto` 仍按原有规则选择
Torch 或 ONNX；此选项不强制切换模型栈，也不改变 Torch / VLM 的设备。

### 共享 GPU 分类与选型

OCR、llama.cpp、auto tier 和硬件诊断共用 `discrete / integrated / virtual / unknown` 分类。
Windows 的分类依据为 D3D12 UMA，Linux 为 Vulkan 原生设备类型，macOS 复用现有系统显示设备信息。
统一内存属性单独保存，不将 Apple Silicon 的集显分类解释为低性能设备。

各后端先应用显式配置、设备可见性及兼容限制，再按“确认独显 → 确认集显 → 虚拟 GPU → 类型未知”排序；
同类设备保留 Windows 高性能排序、Vulkan 原生顺序或 llama.cpp 的单后端多设备分组规则。
OCR 与 llama.cpp 可以因支持范围不同选择不同显卡；不在目标后端候选中的独显不影响当前设备的校准或推荐。

ggml 的 GPU/IGPU 枚举仅用于确定 llama.cpp 可见的 GPU 库存，分类来自与平台原生设备的唯一关联。
身份匹配优先使用双方已有的 LUID、PCI 地址，再使用双方唯一的厂商/设备型号；名称只用于原生库存的唯一关联，
不用于猜测类型。重复库存、同型号多卡无法唯一关联或原生查询失败时保留未知，不凭此开放 Advanced 或免校准。
本实现不增加 UUID、PCI 地址等身份采集。Torch 的加速可用性推荐和 Apple Silicon/MPS 特例保持不变。

| 值 | Windows DML | Linux WebGPU |
|---|---|---|
| `cpu` | 全部使用 CPU，不枚举显卡、不测速 | 全部使用 CPU，不加载插件、不探测 GPU |
| `gpu` | 首次使用时发现 GPU，独显优先，不测速 | 通过 Intel 架构检查后使用 GPU，不测速；旧 Intel 回退 CPU |
| `auto` | 独显直接使用；集显或类型未知执行 rec 测速 | 通过架构检查且唯一确认的独显直接使用；集显或类型未知执行自适应 rec 测速 |

Windows 使用 DXGI 高性能顺序、D3D12 UMA 属性和 LUID 匹配；适配器编号不写入配置。
已保存的 `gpu` 使用 Windows 高性能排序及无设备创建的 D3D12 能力检查，避免逐张创建设备查询 UMA；
`auto` 才需要 UMA 分类。旧系统缺少高性能排序接口时仍执行分类以保证独显优先。
`gpu` 模式下移除显卡会在本次使用 CPU，仍保留配置中的 `gpu`。
其他操作系统、驱动探测错误、运行时缺少所需 EP 或 GPU 会话初始化失败时，日志会说明 CPU 回退原因。

Linux 使用原生 `onnxruntime-ep-webgpu`，通过 Dawn/Vulkan 执行。
插件在每个进程首次需要 GPU 时注册；只保留可哈希、可序列化的设备描述，原生 EP 句柄在创建会话时重新获取。
WebGPU 0.4 的 ORT 设备枚举项用于注册 EP，并不绑定 Dawn 的物理显卡。通常仍使用
`powerPreference=high-performance`；参数固定为 `preferredLayout=NHWC`、`enableGraphCapture=0`、
`dawnBackendType=Vulkan`，其他计算参数沿用运行时默认值。
独显/集显分类读取 Vulkan 的 `VkPhysicalDeviceProperties.deviceType`，不依据厂商名称、显存大小或 ORT 元数据猜测。
只有 ORT 描述与 Vulkan 可见设备唯一匹配、类型为 `DISCRETE_GPU` 且会话采用物理设备约束时，才跳过自动校准。
同型号多卡、重复 EP 描述或无法确认类型时保留校准；虚拟 GPU 不视为独显。

### Linux Intel 架构策略

按 `mineru.model.runtime.intel_gpu` 的显式 PCI ID 家族表分类（`device_id & 0xff00`），
不解析商品名称、不按 ID 大小或 CPU 发布时间猜测架构。映射依据为
[Dawn GPU 表](https://github.com/google/dawn/blob/main/src/dawn/gpu_info.json)，以下范围只对 vendor `0x8086` 的 GPU 生效：

| PCI ID 家族前缀 | 架构 / 策略 |
|---|---|
| 01、04、0a、0c、0d、0f、16、22、19、31、3e、5a、59、87、9b、8a、45、4e、98 | 旧架构，CPU |
| 46、4c、49、9a、a7 | Xe-LP，CPU（包含 DG1 独显） |
| 4f、56 | Xe-HPG，允许 GPU；按 Vulkan 类型决定是否校准 |
| 7d、b6 | Xe-LPG 系，允许 GPU；按 Vulkan 类型决定是否校准 |
| 64、e2 | Xe2-LPG / Xe2-HPG，允许 GPU；按 Vulkan 类型决定是否校准 |
| b0、fd | Xe3-LPG / Xe3-LPG XS，允许 GPU；按 Vulkan 类型决定是否校准 |
| 未识别的 Intel PCI ID 家族 | 保守使用 CPU，补充显式映射后才启用 |

例如 `8086:4680`、`8086:a780`、`8086:9a49` 直接走 CPU；`8086:7d51`、`8086:7d55`、
`8086:56a0`、`8086:64a0`、`8086:e20b`、`8086:b080` 进入 GPU 选型，其中确认的独显免校准。
AMD/NVIDIA 不参与 Intel 分类；Windows DML 的设备规则和 metacommands 保持原样。
这是 MinerU 的产品支持策略，不是宣称旧 Intel 硬件不支持 WebGPU。

检查发生在下载校准模型、创建推理会话和校准之前，覆盖 auto、显式 gpu 以及已经保存的 gpu。
架构回退记录 `webgpu_architecture_policy`，不改写用户原来的 auto/gpu 配置；低层会话工厂也拒绝旧 Intel 描述符。
显式 cpu 仍然不发现 GPU、不加载插件。

### 混合显卡的物理设备约束

Linux GPU 选型查询 Vulkan 实际可见设备，尊重用户的 ICD/驱动筛选；全部架构受支持的机器同样执行此查询。
该查询仅创建实例、枚举属性，不创建 logical device、不编译 shader、不进行 GPU 推理。
从实际可见且架构允许的设备中优先选择独显，避免“发现一张 UHD 就禁用整个 WebGPU”。
例如 UHD 770＋3060 Ti 仍选择独显；若用户通过 `VK_DRIVER_FILES` 仅开放 Intel ICD，则直接走 CPU，
不会把 ORT 中仍然存在的 NVIDIA 枚举项误当成可用物理独显。

确认的独显及多卡场景在每次 WebGPU 会话创建期间临时设置 Vulkan 厂商/设备 ID 过滤，并在创建 Dawn 前重新枚举，
确认只剩目标设备才继续。优先使用 Vulkan-Loader 1.4.326 起支持的 `VK_LOADER_VENDOR_ID_FILTER` /
`VK_LOADER_DEVICE_ID_FILTER`；旧 loader 尝试 Mesa device-select 的 `vendor:device!`，同样必须验证结果。
若两种约束都不可用或被忽略，则记录原因、回退 CPU，不保存独显 GPU 偏好，也不进行可能落到其他卡的模型推理。
此时需升级 Vulkan loader 或安装/启用 Mesa device-select；不依赖 `vulkaninfo` 命令或新增 Python 包。

过滤值随设备描述传入校准子进程；临时环境变量在成功和异常退出时均恢复。
MinerU 的 WebGPU 运行时锁覆盖过滤与会话初始化，后续每次会话创建仍重设约束，以覆盖原生 context 的释放/重建。
同进程初始化后不能切换物理设备约束，需新进程；不会把 provider 的 `deviceId` 误用为物理选卡参数。
这些环境变量是进程级的，宿主若另有独立 Vulkan 消费者，应避免在同一进程同时初始化它们。
宿主也不应在 MinerU 之前自行初始化同一插件的默认 Dawn context；已有外部 WebGPU 会话时需使用新进程，
本策略无法重新绑定外部提前创建的原生 context。
单张集显保持默认物理选择。若原生类型查询失败且 ORT 库存中的架构均受支持，则保留默认候选并继续校准，
不会据此授予独显免校准资格；存在受限 Intel 架构时，查询失败直接回退 CPU。

本机双卡及仅 Intel ICD 的验证见 [Intel 架构过滤报告](onnx-intel-architecture-policy-validation.md)。

通常在首次实际使用 ONNX 模型时选型。API 使用 `--tier auto`（或省略），有效小模型后端为 ONNX，且可用逻辑 CPU 至少 4 核或存在显存／预算达到 5.5 GiB 的兼容 GPU 候选时，Windows/Linux 会提前同步调用相同的设备选择流程，并直接采用本次设备选型结果，不依赖 `--preload-models`。实际选中 GPU 且内存达标时开放 Standard/Advanced；auto tier 不增加独立测速，也不额外检查历史校准是否胜出。`onnx_device=gpu` 无论来自用户偏好还是自动选型写入，均免测速，校准记录缺失、损坏、过期或 GPU 较慢不阻止内存达标的已选 GPU 开放高级档位。CPU 不足 4 核且无内存达标 GPU 候选、显式启动档位或 macOS 不增加这项启动选型。有效小模型后端为 Torch 时采用实际设备可用性和内存证据并跳过 OCR 校准，不修改 ONNX 设备偏好或证据；llama.cpp 自动集显路径也遵循这处分流。

`mineru-kit hardware detect` 与 API 自动启动共享推荐逻辑；`hardware info` 只读报告环境与校准记录，不测速。界面启动本身、导入 SDK、查询配置和模型列表不会运行测速；界面托管的 API 服务仍遵循其启动参数。
自动决策成功后原子更新有效配置文件中的 `model.onnx_device`，保留其余键；进程间通过文件锁协调。
校准证据显式保存设备、运行时与线程指纹；分类来源等诊断字段不进入指纹，原有有效记录可继续复用。
实际设备或物理绑定改变后，诊断报告会显示旧记录不再匹配；这不会覆盖配置中的设备偏好，也不会单独限制已选 GPU 的档位。
Windows/Linux 免校准独显都只在首次真实 GPU 会话成功后保存 `gpu`；仅运行硬件推荐时仍保留 `auto`，
不下载校准权重、不生成性能校准记录。初始化错误或静默回退 CPU 不保存 GPU 偏好。
环境变量覆盖不写入文件；只读配置保留本进程选择。测速失败不保存为 CPU 性能优胜，同进程不重复尝试。写入 `onnx_device=gpu` 不要求达到 5.5 GiB，该内存门槛只用于 auto tier 开放 Standard/Advanced；较小的 GPU 仍可用于 ONNX 加速。
Linux 插件缺失、没有可用设备、物理绑定失败、校准超时或输出不一致时保留 `auto`。
通过环境变量强制 `auto` 时，每个新进程重新选择，并使用选择锁避免同时测速。
显式或已保存的 `cpu` 不会因后来接入独显而改变。

要重新测量，将配置改回 `auto` 并重启应用。修改程序阈值也不主动覆盖已保存的选择。

## 模型预设

| 模型 | Windows DML | Linux WebGPU |
|---|---|---|
| layout / OCR rec / MFR / UNet | GPU | GPU |
| 普通 OCR det | CPU | GPU |
| 印章检测 / 表格分类 / SLANet Plus | CPU | CPU |

普通文字检测和印章检测在模型入口通过 `ocr_det` / `seal_det` 明确区分。
UNet 与其他模型共享 `model.onnx_device` 的进程内选择，不单独测速；ONNX 栈的 UNet 不受旧 `MINERU_TABLE_DEVICE` 覆盖。
Torch 栈中的表格仍使用其原有 CPU/CUDA 策略。旧实验引擎和实验开关已经移除。

| 厂商 | layout | MFR | rec | UNet |
|---|---|---|---|---|
| Intel | 关 | 开 | 关 | 关 |
| AMD | 开 | 开 | 开 | 开 |
| NVIDIA | 开 | 开 | 关 | 关 |
| 其他 | ORT 默认 | ORT 默认 | ORT 默认 | ORT 默认 |

厂商表中的开关只适用于 DML 的 metacommands，关闭对应 `disable_metacommands=true`；WebGPU 不使用这些预设。
Intel layout 优先首轮响应；NVIDIA layout 采用后续补测结论。
MFR 只有实际批量为 8 时使用 DML/WebGPU，1–7 张使用惰性创建、复用的 CPU 会话，不补齐。
自定义批量小于 8 时全部使用 CPU；依据实际 EP 判断是否分流，GPU 初始化已回退 CPU 时不创建多余尾批会话。
每个 DML 会话的 `run` 都串行化，不依赖宿主是否打开全局推理锁。
WebGPU 的所有会话创建和 `run` 共用进程内锁，因为多个会话共享 Dawn 状态；CPU 会话不使用此锁。
参见 [上游并发问题](https://github.com/microsoft/onnxruntime/issues/32561)。
GPU 会话必须实际启用目标 EP，关闭 Python 层整会话静默回退；不支持的图中算子仍可由 ORT 分配给 CPU。
WebGPU 初始化失败时使用未绑定插件的新 SessionOptions 创建 CPU 会话，并保留线程、内存及 profiling 设置。

## 六图测速和阈值

### OCR rec 精简输出

CPU、DML 和 WebGPU 的 ONNX rec 共用 `ctc-ids-scores-v1` 派生图。保留原始 Softmax，在图末端增加
ArgMax / ReduceMax，只返回 `[batch,time]` 的 int64 字符 ID 与 float32 最大置信度。
CTC 去重、blank 删除、置信度平均和输入顺序还原继续由识别器处理；Torch 路径保持现有实现。
对于 `[6,3,48,320]` 输入，原始 `[6,40,18710]` 概率输出约 17.13 MiB，精简输出仅 2.81 KiB。

原始下载权重不被修改。首次使用时惰性加载 `onnx`，在 `model.base_dir/.derived/ocr-rec/` 中按
源文件 SHA256 和输出格式版本生成派生模型；文件锁、校验值和原子替换保护并发创建与缓存复用，
损坏的缓存会重建。当前转换支持内嵌权重的 float32、三维概率输出、固定字符数和 opset ≥ 11 的 rec 图。
正常会话和 CPU 回退都使用同一输出契约；精简模型不是 ORT 特定 EP 的预编译文件。

自动选型也使用该派生图，图转换发生在 15 秒 worker 预算之外。报告记录源模型、派生模型 SHA256
及 `output_format`，阈值工具拒绝混合完整概率输出和精简输出的报告。已有保存的 cpu/gpu 不会被覆盖；
需要重新测速时设置 `auto`。历史完整概率输出测速保留作为优化前对照，不据此重新推导精简图阈值。
本机阶段耗时、输出一致性和 EP 验证见 [精简 rec 生产验证](onnx-compact-rec-validation.md)。

### 测量协议

图片和 manifest 位于包内 `resources/onnx_calibration`，包含原始中英数字句子、来源和 SHA256。
重新生成工具 `scripts.make_onnx_calibration_samples` 需要显式提供字体；部署时直接读取固定 PNG，
不需要系统字体。输入复用生产 rec 的 BGR、动态宽度和归一化逻辑；CPU/GPU 使用同一份 batch6。

CPU/GPU 先各预热 1 次、计时 3 次，顺序为 CPU→GPU；默认线程为 4/1，生产显式线程配置同时应用于测速。
初步比值为 GPU 三次中位数 / CPU 三次中位数。以下任一条件触发追加：

- 比值与当前后端阈值的绝对距离 ≤ 0.1（阈值 0.8 时为 0.7–0.9，含边界）；
- 任一后端的 `(max-min)/median > 0.15`；
- 观测范围 `[min(gpu)/max(cpu), max(gpu)/min(cpu)]` 包含阈值。该范围不是统计置信区间。

追加时复用原来的两个会话，继续按 CPU→GPU 串行各运行 6 次，使每端总计 10 次：
前 3 次重新归为预热，后 7 次取中位数。已有第 4 次调用保留，不重新建会话或丢弃整个短测。
校准工具的 GPU-first 对照对两个阶段都使用相反顺序；生产始终 CPU-first。

长测使用包含端点的四分位距 / 中位数衡量波动，并比较前3、后3个正式样本的中位数变化 / 整段中位数。
两者均须 ≤ 0.15。一个孤立慢采样不会自动否决中位数，但持续漂移或大幅波动会令 `decision_valid=false`：
当次使用 CPU、保留 auto，不保存为 CPU 性能胜出。稳定长测最终仍按中位数比值 ≤ 0.8 选择 GPU；
有限采样不保证阈值附近的机器在不同运行中总选同一设备。

worker 报告 schema v5，`measurement` 保存完整自适应规则；`sampling` 保存实际采用的 1＋3 / 3＋7、
追加原因与最初采样；`decision_threshold` 保存本次使用的阈值。`cpu_elapsed` / `gpu_elapsed` 为最终中位数，
每端同时保留全部实际预热、计时、波动与漂移指标。每一次推理均检查输出及重复解码，两个后端解码必须一致。
阈值推导拒绝混合不同测量策略或决策阈值；同一自适应策略内的短测/长测可共同记录，持续不稳定的结果不能用于推导。

本机实测及 NHWC 缩小复现见 [自适应校准与 UHD 770 诊断](onnx-adaptive-calibration-uhd770-validation.md)。
此前固定 3＋7 与整本测试见 [轮次和预打包报告](onnx-i7-12700-nhwc-no-prepack-validation.md)。
UHD 770 关闭预打包仍存在内容错误及挂起，生产 WebGPU 布局和预打包参数未因本次校准改动而调整。
阈值在 `mineru.model.runtime.onnx_device` 中集中维护。
阈值按后端独立维护：`dml=0.8`、`webgpu=0.8`，即 GPU 推理耗时至少比 CPU 低 20%。
Linux 8060S 六图预检比值为 0.420、0.502、0.523，解码一致；目前没有跨硬件证据支持另设 WebGPU 数值。
原三机校准推导值为 0.69639683662382，保留为历史参考；已有 24 次记录按 0.8 判断仍得到相同选择。
详细数据见 [DML 三机校准报告](onnx-selection-validation.md) 和 [Linux WebGPU 生产验证](onnx-webgpu-validation.md)。

下载和设备发现不计入 worker 预算。独立隐藏子进程的总超时为 15 秒，包含解释器启动、加载及推理；
1–2 秒只是小样本推理目标。结果记录 backend、模型/输入校验值、ORT/EP 版本、参数、provider、初始化、预热和正式计时。
CPU/GPU 解码 token 必须一致，整会话回退 CPU 或计时无效都视为失败。

### 发布前校准

在 Ultra 5 / Arc 130T、i7-12700 / UHD 770、MAX395 / 8060S 上使用相同代码、权重、样本和 ORT：

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_onnx_selection devices
.\.venv\Scripts\python.exe -m scripts.benchmark_onnx_selection run --label ultra5 --luid <集显LUID> --output output/calibration-ultra5
```

另外两台使用 `--label i7` 和 `--label max395`。i7 必须在基准工具中明确选择 UHD 770，
应用的正常独显优先规则仍会选择 RTX 3060 Ti。输出目录必须不存在。

每台运行五个新进程，交替 CPU/GPU 先后顺序。汇总命令只接受完整且协议一致的三份报告：

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_onnx_selection derive <ultra5报告> <i7报告> <max395报告> --output output/threshold.json
```

取 Ultra 5 / MAX395 所有比值的最大值为 L，取 i7 最小比值与 1.0 的较小值为 U。
只有 L < U 才给出中点 `(L + U) / 2` 作为统计建议；当前发布值按用户要求采用 0.8，
不自动用统计建议覆盖正式常量。随后各平台以 `run --threshold <待验证值>` 和新目录运行三个
独立验证进程，要求 Ultra 5、MAX395 全部选 GPU，i7 全部选 CPU。没有有效区间或验证失败时不发布阈值。
阈值为 `None` 表示三机校准尚未完成；此时自动集显路径使用 CPU、保留 `auto`，显式 GPU 及独显仍可使用。

Linux 使用同一个工具（从项目根目录运行）：

```bash
.venv/bin/python -m scripts.benchmark_onnx_selection devices
.venv/bin/python -m scripts.benchmark_onnx_selection run --label linux-8060s --expect gpu --output output/webgpu-calibration
.venv/bin/python -m scripts.benchmark_onnx_selection run --label linux-8060s --threshold 0.8 --expect gpu --output output/webgpu-validation
```

第一条 run 启动 5 个新进程、交替 CPU/GPU 顺序；第二条启动 3 个独立验证进程。
Linux 标签可自定义，独立验证必须显式提供 `--expect cpu|gpu`；`--luid` 仅用于 DML。
WebGPU 推导新阈值需要至少三台不同标签机器的完整五进程报告，同时有 CPU/GPU 优胜组，
各组通过 `--expect` 按独立性能证据标记。不能混合 DML/WebGPU、EP 版本或 WebGPU 参数来推导阈值。
报告 schema v2 增加 backend/EP 字段；推导工具仍接受历史 v1 DML 报告。
WebGPU 阈值为 `None` 时，需要校准的 auto 路径保留 `auto` 并临时使用 CPU；显式 `gpu` 和确认的独显不受此限制。

## 安装

Windows x64 的直接依赖为 `onnxruntime-directml>=1.23,<2`，其他平台使用 `onnxruntime>=1.24.4,<2`，
不按 Python 版本拆分条件。DirectML 包本身包含 CPU provider。

项目要求 Python >=3.11。Linux x86_64 同时安装 `onnxruntime-ep-webgpu>=0.4.0,<1`；
这是与 ORT 并存的插件，不替换 `onnxruntime`。Linux ARM64、macOS 和 Windows 不安装此插件。
Linux 还需系统提供 `libvulkan.so.1` 和可用的 GPU Vulkan 驱动。使用 uv 管理环境：

```bash
.venv/bin/uv pip install --python .venv/bin/python -e .
.venv/bin/uv pip check --python .venv/bin/python
```

插件注册前，`ort.get_available_providers()` 不一定列出 WebGPU；用上面的 `devices` 命令检查插件发现能力。

已有环境从 CPU 包迁移时，先卸载 `onnxruntime`，再安装 DirectML，避免两个发行包覆盖同一模块。
安装后用 `import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_available_providers())` 检查实际能力。

文件识别使用 `mineru-magika>=1.0.3,<1.1.0`，Python 导入名仍为 `magika`，由 DocVortex 声明依赖。
迁移已有环境时须先卸载旧 `magika`，再安装新发行包，避免两个包覆盖同一模块。

2026-10-18 已使用 DocVortex 提交 `545570c4bffa2a668b7db03b5d3117e7a13faf8c` 完成
Windows x64 的完整依赖安装、发行包身份、真实分类、rec 与文档解析验收。
该提交的版本号仍为 `0.5.16`，当前 PyPI 同版本仍是旧依赖；因此验收使用固定提交构建的 wheel。
在兼容版本发布到 PyPI 并更新最低版本前，源码安装应显式指定该提交：

```powershell
.\.venv\Scripts\uv.exe pip install --python .venv/Scripts/python.exe -e . "docvortex @ git+https://github.com/myhloli/DocVortex.git@545570c4bffa2a668b7db03b5d3117e7a13faf8c"
.\.venv\Scripts\uv.exe pip check --python .venv/Scripts/python.exe
```

不能把两个 ORT 发行包或两个 Magika 发行包同时存在视为有效安装。
版本、校验值、平台矩阵与实测结果见 [依赖验收报告](onnx-dependency-validation.md)。
