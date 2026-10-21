# ONNX OCR rec 精简输出生产验证（2026-10-21）

Linux x86_64，AMD Radeon 8060S / RADV STRIX_HALO；Python 3.14.4、ORT 1.31.0、WebGPU EP 0.4.0。
basic＋强制 OCR，demo1+demo2 合并 PDF 共 19 页。CPU intra-op=4、inter-op=1；WebGPU 为 NHWC、Vulkan、high-performance、graph capture 关闭。

## 实现与正确性

CPU、DML、WebGPU 的 ONNX rec 共用精简图，图内 ArgMax / ReduceMax 返回 ID 和置信度。原 Softmax、字符表、CTC 去重与顺序还原语义保留。Torch 和 DocVortex / Rust 未修改。
派生图按源 SHA256 和格式版本缓存，文件锁、校验与原子替换保护多进程使用。自动设备校准使用同一派生图，报告区分输出格式。
10 个真实张量对照（CPU/WebGPU，各含 batch 1/6/7、宽 320/1012）中，精简输出的 ID、置信度与各自原图逐值一致。
四次正式解析、两次阶段采样的 Markdown、三个 JSON、46 个素材，共 50 个文件全部逐字节一致，且都与优化前结果一致。
DML 共用路径由单元回归覆盖；本机为 Linux，未获得 Windows DML 硬件耗时。
401 项相关回归测试、Ruff、git diff --check 与 uv pip check 通过。

## 完整解析耗时

每种模式两个新进程，按 CPU→GPU→GPU→CPU 执行，无额外预热，关闭采样器。计时包含模型初始化与首次推理，不含模块导入、结果写盘。派生图缓存已准备好。

| 路线 | 第一次（秒） | 第二次（秒） | 平均（秒） |
|---|---:|---:|---:|
| cpu | 58.012 | 58.368 | 58.190 |
| webgpu | 34.726 | 35.076 | 34.901 |

WebGPU 相对 CPU 加速 **1.67×**，总耗时减少 **40.0%**。

## 各模型阶段累计推理

单独使用 Python call/return observer 记录同步 ORT run，包含首次调用、数据传输与图内 CPU 算子，不含预处理/后处理。采样解析的总耗时不参与上表比较，阶段相加不会等于上表完整解析耗时。
新增的两个归约算子计入 rec 推理，因此 CPU 列也使用精简输出图。表格分类与 SLANet Plus 两条路线都使用 CPU。

| 阶段 | CPU（秒） | WebGPU 路线（秒） | CPU/GPU 调用次数 |
|---|---:|---:|---:|
| layout | 4.870 | 2.008 | 10 / 10 |
| mfr | 22.892 | 12.726 | 19 / 19 |
| ocr_det | 1.175 | 0.715 | 84 / 84 |
| ocr_rec | 18.463 | 7.277 | 336 / 336 |
| unet | 0.237 | 0.110 | 5 / 5 |
| table_cls | 0.006 | 0.006 | 1 / 1 |
| slanet_plus | 0.130 | 0.112 | 2 / 2 |

## MFR 实际分流

| 路线 | EP | 批量 | 调用次数 | 推理（秒） |
|---|---|---:|---:|---:|
| cpu | CPUExecutionProvider | 8 | 18 | 19.993 |
| cpu | CPUExecutionProvider | 7 | 1 | 2.899 |
| webgpu | WebGpuExecutionProvider | 8 | 18 | 9.774 |
| webgpu | CPUExecutionProvider | 7 | 1 | 2.952 |

## 初始化与首轮调用

未额外预热完整解析。下列首轮调用已经包含在对应阶段推理中，不能再次加总；它们含实际推理，不应全部视为预热开销。
CPU/GPU 初始化为生产 create_ort_session 墙钟；表格分类及 SLANet 的其他工厂初始化未由本 observer 采集。

| 模型 | CPU 初始化（秒） | GPU 路线初始化（秒） | CPU 首轮（秒） | GPU 路线首轮（秒） |
|---|---:|---:|---:|---:|
| layout | 0.439 | 0.408 | 0.517 | 0.271 |
| mfr | 0.475 | 0.949 | 1.041 | 3.354 |
| ocr_det | 0.027 | 0.068 | 0.080 | 0.049 |
| ocr_rec | 0.061 | 0.060 | 0.046 | 0.026 |
| unet | 0.020 | 0.026 | 0.062 | 0.031 |

单独冷生成派生缓存 0.295 秒，复用验证 0.035 秒。

## 优化前后的同机参考

优化前报告来自同一天此前的运行。其他阶段会受系统状态影响，这里保留实际数值，不将全部解析时间差归因于 rec 改动。

| 指标 | 优化前（秒） | 优化后（秒） |
|---|---:|---:|
| cpu 完整解析平均 | 56.731 | 58.190 |
| webgpu 完整解析平均 | 36.164 | 34.901 |
| cpu rec 累计推理 | 17.394 | 18.463 |
| webgpu rec 累计推理 | 8.937 | 7.277 |

## EP 与自动校准

精简 rec profile：{'WebGpuExecutionProvider': 206, 'CPUExecutionProvider': 3}；ArgMax、ReduceMax 均在 WebGPU 上执行，CPU 仅执行 Gather、Slice、Concat 三个形状算子。
五种 GPU 目标模型实际启用 WebGPU，没有整会话 CPU 回退；MFR 尾批明确为 CPU。
5 个新进程六图校准 GPU/CPU 比值：0.3845, 0.3873, 0.4016, 0.5619, 0.4011；输出一致。保持各后端阈值 0.8，本次不重新推导 DML 阈值。

## 复现

原始脚本、每次输出、捕获张量、日志、profile 与报告位于 `demo/output/compact-rec-production-20261021/`。
从项目根目录设置 `MINERU_MODEL_SMALL_BACKEND=onnx`、`MINERU_MODEL_ONNX_DEVICE=cpu|gpu`、
`MINERU_INTRA_OP_NUM_THREADS=4`、`MINERU_INTER_OP_NUM_THREADS=1`，运行该目录的 `benchmark.py parse --output <新目录>`；独立阶段观测增加 `--observe`。
[机器可读报告](benchmarks/onnx-compact-rec-linux-20261021.json) 保存源码/模型/输入校验值、完整阶段数据、EP 证据和各次校准。
