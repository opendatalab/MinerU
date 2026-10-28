# i7-12700 / UHD 770 / RTX 3060 Ti：basic＋OCR 与 WebGPU 自动选型验证

日期：2026-10-21。原始生产代码为 `a2a4960b088ac3744a64d2c46dbda4e02b9efa57`。

## 结论

- CPU 4 线程平均 **97.061 秒**；RTX 3060 Ti WebGPU 平均 **49.657 秒**，加速 **1.955×**，耗时减少 **48.84%**。
- UHD 770 的生产 NHWC 路径两次均产生空文档，约 21 秒的失败耗时不能作为加速成绩。NCHW 隔离副本虽然恢复六图识别，却在整本 OCR det 阶段触发 i915 **GPU HANG / rcs0 reset**，没有有效整本成绩；已停止第二轮，避免再次挂起。
- 正式 auto 入口的两组共 10 次均选 GPU，但此前 10 次短校准有 2 次比值超过 0.8，会误选 CPU。**0.8 不是已完全通过稳定性验收的结论；优先改善采样稳定性，未修改正式阈值。**
- UHD 770 / NHWC 的结果一致性检查能正确拒绝错误输出；失败保留 `auto`，没有保存为 CPU 性能优胜。NCHW 六图比值为 2.15–2.27，按 0.8 全部应选 CPU。

## 环境与方法

i7-12700（12 核、20 线程），UHD 770（8086:4680，i915），RTX 3060 Ti（10de:2489，nouveau / Mesa NVK）。
Linux 7.0.0-38、Mesa 26.0.8；Python 3.14.4、MinerU 4.0.12、DocVortex 0.5.16、ORT 1.31.0、WebGPU EP 0.4.0、ONNX 1.23.2。
CPU governor 为 powersave；日常桌面环境，未锁频，逐次电源和温度状态保存在 execution.json。
独显成绩属于当前 NVK / WebGPU 路线，没有使用 CUDA 或 NVIDIA 专有驱动。

沿用前两次测试：demo1 后接 demo2，共 19 页；输入 SHA256：
`06d0e74125b5d2e1fc04e658bbe4af39422d70abfaa453a87353df4d3dad3432`。
强制 ONNX 小模型栈，`tier=basic`、`ocr_mode=ocr`、`page_range=all`，关闭标题层级与跨页表格 LLM 增强；所有路线 CPU intra/inter-op 均为 4/1。
每次使用新进程；完整解析计时包含模型初始化、首轮推理、预后处理，不含模块导入、模型下载、派生 rec 缓存生成与写盘。
正式总耗时关闭 observer 和 ORT profiler，各阶段来自额外采样进程，不能逐行相加还原正式总耗时。

正式顺序为 CPU→Intel→NVIDIA→NVIDIA→Intel→CPU，之后分别进行阶段采样。
两张显卡通过子进程 `VK_DRIVER_FILES` 限制 Vulkan ICD；默认路线不设置该变量。
EP 枚举描述不绑定 Dawn 物理显卡，故不能只凭日志中的 `WebGPUDevice` 判断实际用卡。
Intel 的 i915 render 计数增加；NVIDIA 采样进程打开了 PCI 0000:01:00.0 的 renderD129/card0。
默认 auto 完整解析打开两卡进行枚举，但 Intel render 计数在 129 次采样中始终为零；结合 NVIDIA 设备描述、NVIDIA 句柄和有效推理结果，判断实际使用独显。
nouveau 未提供 i915 式 fdinfo 引擎计数，未虚构该计数。

## 整本耗时

| 路线 | 第一次（秒） | 第二次（秒） | 平均（秒） | 有效性 |
|---|---:|---:|---:|---|
| CPU，4 线程 | 96.009 | 98.113 | 97.061 | 有效 |
| RTX 3060 Ti，生产 NHWC | 50.524 | 48.790 | 49.657 | 与 CPU 输出一致 |
| UHD 770，生产 NHWC | 21.106 | 20.922 | — | 无效，空输出 |
| UHD 770，NCHW 隔离副本 | — | — | — | 首轮 GPU 挂起；第二轮停止 |

NCHW 副本只将 `WEBGPU_OPTIONS.preferredLayout` 从 NHWC 改为 NCHW，其他源码、权重、输入和参数一致。
它位于忽略目录 `demo/output/i7-12700-three-device-20261021/nchw-source/`，不是生产代码修复。
该副本结果应单独看待，不能与 NHWC 校准报告混合推导全局阈值。
首轮从进程启动到异常共 212.338 秒，第二轮在 76.129 秒被主动停止；这些是失败/中止进程墙钟，不能与上表完整解析时间比较。

## 各阶段累计推理：生产 CPU / NVIDIA

同步 ORT run 墙钟，包含首次调用、传输与图内 CPU 回退，不含模型初始化、预后处理。

| 模型 | CPU（秒） | RTX 3060 Ti 路线（秒） | 加速 | CPU/GPU 调用次数 |
|---|---:|---:|---:|---:|
| layout | 8.499 | 2.549 | 3.33× | 10 / 10 |
| mfr | 38.470 | 17.613 | 2.18× | 19 / 19 |
| ocr_det | 1.929 | 1.538 | 1.25× | 84 / 84 |
| ocr_rec | 35.307 | 9.991 | 3.53× | 336 / 336 |
| unet | 0.429 | 0.198 | 2.17× | 5 / 5 |
| table_cls | 0.009 | 0.007 | 1.24× | 1 / 1 |
| slanet_plus | 0.270 | 0.181 | 1.49× | 2 / 2 |

NVIDIA 的 MFR 满批 18×8 调用累计 13.495 秒；尾批 7 个公式由 CPU 执行，累计 4.118 秒。
layout、OCR det/rec、MFR 满批、UNet 均确认启用 WebGpuExecutionProvider；表格分类与 SLANet Plus 走 CPU。
本轮没有追加逐节点 profiling，不将“会话启用 GPU”解释为所有图内算子均由 GPU 执行。
UHD 770 / NHWC 的额外采样只产生 10 次 layout 调用、累计 18.024 秒；OCR/公式调用均为零。
NCHW 补测没有另跑阶段 observer，以上分阶段表不包含 NCHW。

## UHD 770 异常与布局诊断

NHWC 的六图识别全部为空，CPU 正确识别六条中英数字文本。
原始 rec 与精简 rec 都复现；两者 GPU token 相同，因此不能归因于新增 ArgMax/ReduceMax。
使用 Intel EP 描述和使用首个 NVIDIA EP 描述均复现，排除了仅更换 EP 枚举条目即可修复的解释。
NCHW 六图 token 与 CPU 完全一致，置信度最大绝对差约 9.9e-6。
layout 也表现出布局相关的明显数值差异。小样本恢复不代表整本可用：NCHW 在 OCR det 的 GPU 结果下载时报 `Failed to wait for the operation:2`。
内核在 20:23:32 同时记录 `Resetting rcs0 for preemption time out`、`context reset due to GPU hang` 和 `GPU HANG: ecode 12:1:85dcfdfb, in python [19609]`。
该 PID 与首轮 NCHW 解析进程一致；第二轮尚在公式阶段即被停止。NCHW 没有生成完整结果包，不作为修复建议。
本次定位到当前 UHD 770 / Mesa / WebGPU 组合：NHWC 存在数值正确性问题，NCHW 另有整本稳定性问题；尚未定位具体上游算子或驱动缺陷。
[ORT 官方参数说明](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html#configuration-options)提供了 preferredLayout 的 NCHW/NHWC 选项。

## 输出一致性

以本机 cpu-1 为基准，各文件 SHA256 全量比较。

| 运行 | 文件数 | 逐字节一致 | Markdown 字节数 |
|---|---:|---|---:|
| cpu-1 | 50 | 是 | 80574 |
| intel-1 | 4 | 否 | 0 |
| nvidia-1 | 50 | 是 | 80574 |
| nvidia-2 | 50 | 是 | 80574 |
| intel-2 | 4 | 否 | 0 |
| cpu-2 | 50 | 是 | 80574 |
| cpu-observed | 50 | 是 | 80574 |
| intel-observed | 4 | 否 | 0 |
| nvidia-observed | 50 | 是 | 80574 |
| default-auto-observed | 50 | 是 | 80574 |

## 六图阈值与正式入口

当前正式协议为一个独立 worker，CPU/GPU 各预热一次、计时两次；
`ratio = gpu_elapsed / cpu_elapsed <= 0.8` 时选 GPU，相等也接受。
Linux 集显、独显都会测速；六图 token 一致性验证先于性能决策，失败不保存性能选择。

| 测试组 | GPU/CPU 比值范围 | 按 0.8 的选择 |
|---|---:|---|
| 初始五进程：nvidia | 0.244676–0.957603 | GPU 4/5，CPU 1/5 |
| 初始五进程：default | 0.266330–0.816957 | GPU 4/5，CPU 1/5 |
| 正式 auto 入口：nvidia | 0.273108–0.432803 | GPU 5/5 |
| 正式 auto 入口：default | 0.262054–0.484251 | GPU 5/5 |
| 诊断：预热 3 次、计时 7 次取中位数 | 0.266527–0.305778 | GPU 5/5 |
| UHD 770 / NCHW 隔离副本 | 2.150545–2.265009 | 按 0.8 判断 CPU 5/5 |

初始 NVIDIA 单卡组与双卡默认组各有 1 次会选 CPU：比值分别为 0.957603、0.816957。
异常 GPU 正式样本为 155.89/69.76 ms 和 130.64/65.81 ms；正常约 31–33 ms，CPU 约 118–123 ms。
完整文档独显明显更快，故这两次属于短样本误选，不能据后续全部选 GPU 宣称问题已消失。
加长采样仅为本轮诊断对照，没有改变生产逻辑；有限重复也不能保证所有负载下稳定。

UHD 770 / NHWC 初始五次校准全部因文本不一致失败；正式 auto 的五次也全部返回 CPU / selection_failed。
隔离配置再测两次仍保留 auto；默认双卡隔离配置首次保存 gpu，下一新进程返回 configured_gpu，未重测。
真实用户配置文件 SHA256 前后相同，生产代码和正式阈值未修改。

本机建议使用已完成一致性验证的 RTX 3060 Ti 路线；阈值继续保留 0.8，优先考虑更充分预热、更多样本、中位数与异常/边界复测。
当前 UHD 770 生产 NHWC 路线的正确性问题需要单独处理，不能靠提高耗时阈值解决。

## 验证与复现

设备选择、校准和 WebGPU 三组相关回归 **85 项通过**。
源码、模型、输入校验值、逐次校准、provider、配置保存验证与输出比较见
[机器可读报告](benchmarks/onnx-i7-12700-linux-20261021.json)。
完整日志、结果包和测试脚本位于 `demo/output/i7-12700-three-device-20261021/`。

从项目根目录执行，使用全新的输出目录：

```bash
MINERU_MODEL_SMALL_BACKEND=onnx MINERU_MODEL_ONNX_DEVICE=cpu \
MINERU_INTRA_OP_NUM_THREADS=4 MINERU_INTER_OP_NUM_THREADS=1 \
.venv/bin/python demo/output/i7-12700-three-device-20261021/benchmark.py parse \
  --input output/pdf/demo1-demo2-merged.pdf --output <新目录>
```

NVIDIA 将设备改为 gpu，并加 `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nouveau_icd.json`。
原始 Intel 失败路径使用 intel_icd.json；NCHW 的重现需在已保存的隔离源码根目录运行，并使用同样绝对输入路径。
阈值复测可运行 `scripts.benchmark_onnx_selection`；`run_variant.py` 保存了 NCHW 的完整命令与环境。

## 后续

正式校准后续已改为预热 3 次、计时 7 次取中位数，保留 0.8 阈值。
本文保留修改前的测量记录；30 次新进程验收与 prepacking / 纯 det 回放排查见
[中位数校准与 UHD 770 诊断](onnx-i7-12700-calibration-median-diagnosis.md)。
