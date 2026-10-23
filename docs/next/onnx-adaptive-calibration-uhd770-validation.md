# 自适应 ONNX 校准与 UHD 770 NHWC 缩小复现

2026-10-21。i7-12700 / UHD 770（8086:4680、i915）/ RTX 3060 Ti（nouveau / Mesa NVK）。
Linux 7.0.0-38、Mesa 26.0.8、ORT 1.31.0、WebGPU EP 0.4.0。沿用 ONNX 4/1 线程与 NHWC。
本轮实现自适应校准；没有改动生产 WebGPU 布局、预打包、缓存或提交参数，没有运行 NCHW 诊断。

## 自适应规则

CPU 和 GPU 各先执行 1 次预热、3 次正式采样。每次检查输出有效性及设备内重复 CTC token，一阶段结束即检查 CPU/GPU token 一致。
以下任一条件触发追加：

- 初步中位数比值与阈值的绝对距离 ≤ 0.1；当前 0.8 对应闭区间 0.7–0.9。
- 任一后端三次采样的 `(max-min)/median > 0.15`。
- `[min(gpu)/max(cpu), max(gpu)/min(cpu)]` 包含阈值。它是观察范围，不是统计置信区间。

两个会话保留并串行执行；追加各 6 次，总计各 10 次。原来的第2、3次重新归入预热，第4次保留为正式样本。
最终使用前3次预热、后7次计时的中位数。生产顺序为 CPU4→GPU4→CPU6→GPU6；短测提前结束时只有前两个阶段。
GPU-first 诊断对两个阶段都反向执行。没有重新创建会话，没有并发 CPU/GPU 测速。

长测用 inclusive IQR / 中位数与前3、后3个正式样本的中位数变化 / 全段中位数检查稳定性，均须 ≤ 0.15。
孤立慢采样可由中位数抵抗；持续漂移或较大波动则报告 `decision_valid=false`，当次回退 CPU、保留 `auto`。
稳定结果仍按 GPU 中位数 / CPU 中位数 ≤ 0.8 选 GPU；两个后端阈值不变，15 秒 worker 预算不变。
输出错误、超时和整会话回退均不能形成性能选择。

worker schema v5 的 `measurement` 描述自适应规则，`sampling` 保存所选阶段、追加原因和原始短测；
`decision_threshold`、`decision_valid` 与全部计时保留。父进程核验采样数、有限正数、中位数、比值和稳定性，
并拒绝应追加却提前结束的结果。阈值工具区分测量策略和决策阈值，拒绝使用不稳定记录。

自适应目标是以较少采样完成设备选择；提前结束时的比值不应解释为精确稳态性能。

## 同机串行对照

使用真实 `resolve_onnx_device()` 和其独立 worker，环境变量 auto 保证不写用户配置。
固定协议来自上一轮保留的 3＋7 源码副本；两种协议按轮次轮换先后顺序。
每种缓存条件、每种协议各 5 个新进程，共 20 次。所有进程串行，CPU/GPU 计时期间未混跑其他性能任务。
NVIDIA 使用 `VK_DRIVER_FILES=nouveau_icd.json`；未将 ORT 枚举条目解释为 Dawn 物理设备绑定。

| 缓存 | 自适应，秒 | 固定3＋7，秒 | 节省 | 自适应短测 / 追加 |
|---|---:|---:|---:|---:|
| 已有 shader 缓存 | 1.461 | 2.600 | 1.139秒，43.80% | 4 / 1 |
| 每进程隔离 shader 缓存 | 2.033 | 3.059 | 1.027秒，33.56% | 4 / 1 |

上表为完整选择墙钟中位数，包含设备发现、worker 启动、会话创建和推理验证；不包含调用方模块导入、模型下载、首次派生图生成。
隔离仅针对 `MESA_SHADER_CACHE_DIR`，不能称为全部系统缓存冷启动。

10 次自适应和10次固定协议全部选中 GPU、输出一致。自适应已有缓存比值为 0.264–0.278，隔离缓存为 0.274–0.572。
隔离缓存的短测比值可能高于固定长测，但仍远离 0.8；这不意味着两者测得了同一个稳态速度。
已有缓存追加那次耗时2.723秒，略超过固定协议的中位数；自适应并非每次都更快。

额外验证包含首轮与最终源码各一次真实 auto、发布基准 CLI 的5次新进程（包含 GPU-first），以及隔离文件配置的一次真实校准。
本轮共 **18 次有效自适应校准：14次短测、4次追加，全部选择 GPU、输出一致，无超时**。
另一个新进程确认隔离文件已保存 gpu，返回 configured_gpu、不再校准，并保留其余配置键。
真实用户配置 SHA256 与此前报告相同，保持 `5c447bf9c2834991aba50a86cd5a435a4753a0ad6a6e21b745e685670fd59e99`。

未在另外两台笔记本或 Windows DML 上进行新硬件测试；不能由本机远离阈值的结果宣称跨硬件稳定性已完全验证。

## 使用真实权重与输入缩小 NHWC 复现

从原始 OCR rec 截取前78节点，包含36个 initializer，末端是 Conv.14＋Add.36。
使用两种真实输入：包内六张校准图的 `[6,3,48,1012]`，以及此前 demo1＋demo2 生产 observer 捕获的 `[6,3,48,320]`。
CPU 执行相同前缀，导出 Conv.14 的实际输入激活、原始 `[192,96,1,1]` 权重和 bias。
单层对照包含原 Conv＋Add、动态输入权重的 Conv、CPU 转置权重后的等价 MatMul＋Add，以及 GPU 直接转置同一权重。

每个条件使用独立进程、Intel ICD、同一生产 NHWC 参数；只改变注明的诊断选项。
每次包含 CPU 数值对照、3次 GPU 推理、实际节点 EP profile 和 DRM FD 证据。
所有完成的对照均确认真正执行 WebGPU 节点，并且仅观察到 Intel `0000:00:02.0` / i915。

| 对照 | 观测 |
|---|---|
| 真实单层 Conv.14，默认 / 关闭预打包 / 动态权重 | 校准输入的3次输出均与 CPU 一致；生产输入默认也一致 |
| 相同权重的 GPU Transpose | 三次均与 CPU 转置逐值一致，实际 Transpose 节点为 WebGPU |
| 等价 MatMul＋Add | 两种输入都与 CPU 一致；优化后实际为 WebGPU MatMulAddFusion |
| 校准输入、78节点、默认 | 本轮三次通过，最大差约8.08e-6；此前同前缀曾出错，不能认定已恢复 |
| 生产输入、78节点、默认，第一个新进程 | 三次持续错误，最大差均约32.679 |
| 相同生产输入、默认，第二个新进程 | 三次通过，最大差约1.84e-5 |
| 78节点、关闭预打包 | 本轮两种输入都通过；此前整本仍失败，不能据此作为修复 |
| 校准输入、全部权重改为动态输入 | 三次通过，最大差约1.57e-5 |
| 校准输入、仅关闭 storage cache 或 uniform cache | 本轮都通过；各只有一次新进程，默认也曾通过，无法证明开关有效 |
| 校准输入、`maxNumPendingDispatches=1` | 首次错误约3.784，随后两次通过 |
| 生产输入、`maxNumPendingDispatches=1` | 三次错误约4.888、32.683、32.683 |
| 生产输入、`maxNumPendingDispatches=4` | 首次推理发生 GPU HANG，没有有效输出 |

由此可以排除“真实 Conv.14 的形状或权重一旦单独执行就必然错误”这一假设。
前缀错误在新进程之间变化，且提交批量改变其表现，进一步支持调查整个前缀执行中的状态、同步和资源生命周期。
独立权重 Transpose 正确，并不等于已经读出了内部 prepacked buffer；后者本轮没有直接观测。
CPU 转置后的 MatMul 对照被融合为 Gemm，也不能冒充原 Conv 采用完全相同 shader 的对照。

## 新挂起现场及源码分析

生产输入、NHWC、预打包保持默认，只将 `maxNumPendingDispatches` 改为4。
新进程 PID 48983 在首次前缀推理时失败，进程墙钟约14.14秒，不是控制脚本超时杀死。
内核记录：

```text
Oct 21 21:36:54 ... i915 0000:00:02.0: [drm] GPU HANG: ecode 12:1:85dcfdfb, in python [48983]
```

随后 Vulkan 返回 DEVICE_LOST，ORT 在 Download 的等待处报错。
DRM FD 仅为 i915，记录到驻留约97,324 KiB；没有依据将其归为显存耗尽。
启动日志明确已有 context 的 maxNumPendingDispatches=4，随后请求默认16被忽略；本次目标值4实际保留。
这里的 context 警告与之前“目标参数被忽略”的实验不能混为一谈。

发现新挂起后停止所有后续 Intel GPU 运行。原定更多缓存、robustness、逐权重和更短前缀对照没有执行，不计作通过或失败。
这次已将挂起从完整文档或长 det 序列缩小到一个含真实权重、真实输入的 rec 前缀；仍未定位到唯一失败 shader。
`/sys/class/drm/card1/error` 读取被拒绝，未取得完整 i915 error-state。

核对实际发布标签 `plugin-ep-webgpu/v0.4.0`（提交 f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7）：

- [WebGpuKernel::PrePack](https://github.com/microsoft/onnxruntime/blob/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7/onnxruntime/core/providers/webgpu/webgpu_kernel.cc) 已在成功预打包后执行 Flush，再允许 ORT 释放原 initializer；不能简单断言遗漏提交。
- [WebGpuContext::Flush](https://github.com/microsoft/onnxruntime/blob/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7/onnxruntime/core/providers/webgpu/webgpu_context.cc) 会编码 deferred dispatch、提交 queue，并刷新 pending buffer；该处提交不等于等待所有 GPU 工作完成。
- [BufferManager](https://github.com/microsoft/onnxruntime/blob/f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7/onnxruntime/core/providers/webgpu/buffer_manager.cc) 区分 initializer 与普通缓存管理。改变 storage cache 不能独立覆盖 initializer/prepack 的整个生命周期。

这些机制本身不证明违规：GPU 命令持有资源引用与同队列顺序可以提供正确生命周期。
本轮未对内部 packed buffer 逐字节读回、未在 native PrePack 边界注入完成 fence、未重编译原生插件。
因此只能将后续重点放在预打包与执行阶段的跨提交资源使用、Dawn/Tint 和 Mesa/i915；尚不能划定具体责任层。
`maxNumPendingDispatches=1/4` 都不是修复；生产参数不变。

## 验证、数据和复现

相关回归覆盖快速 GPU/CPU 选择、阈值边界追加、两种执行顺序、会话复用、预热重分类、异常值与持续漂移、
初始/追加阶段输出变化、CPU/GPU 文本不同、无效计时、worker 协议与文件配置保留。
**136 项回归通过；Ruff 格式与 lint、git diff --check 通过。** 原始日志保存在 `demo/output/adaptive-nhwc-20261021/`。

[机器可读报告](benchmarks/onnx-adaptive-calibration-uhd770-linux-20261021.json)包含20次成对测量、额外校准与配置验证、
18次 Intel 诊断尝试、实际节点 EP、物理设备、挂起、源码和模型校验值。
已从采样重算全部中位数/比值，并用最终生产协议验证本轮18次自适应记录。

原始目录：`demo/output/adaptive-nhwc-20261021/`。`benchmark_adaptive.py` 是串行对照，`prepare_repro.py` 提取真实数据，
`run_repros.py` 包含挂起监测并在新挂起后停止，`build_report.py` 仅读取证据生成报告。
实验目录不可覆盖；重跑应复制脚本到新的目录并核验输入路径。

约616 KB的 `uhd770-nhwc-repro.zip` 包含78节点原始权重图、真实生产输入、独立 ORT 脚本和 SHA256。
独立脚本已用 `--cpu-only` 检查；为保留停止边界，没有在新挂起后再运行 Intel。
包内 README 标明 `--max-pending 4` 曾触发系统 GPU HANG，以及复现命令。它是失败复现包，不是生产补丁。
