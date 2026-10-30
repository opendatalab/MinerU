# Linux WebGPU 生产接入验证（2026-10-21）

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

Linux x86_64，AMD Radeon 8060S / RADV STRIX_HALO，Mesa 26.0.8；Python 3.14.4、ORT 1.31.0、WebGPU EP 0.4.0。
CPU intra-op=4、inter-op=1；WebGPU 使用 NHWC、graph capture 关闭、Vulkan、high-performance。DocVortex 和 Rust 扩展未修改。

## 自动选择与回归

5 个新进程的六图 rec 比值依次为 0.514125, 0.507938, 0.511882, 0.522714, 0.516614，全部输出一致且低于 WebGPU 阈值 0.8。
独立配置验证：首次 auto 校准后原子保存 gpu；第二个新进程直接返回 configured_gpu，保留其他配置键。
两个真实 WebGPU 会话通过 4 个调用线程交错运行 12 次，所有解码结果一致，无并发异常。
352 项 ONNX/依赖相关回归与 29 项模型预加载、运行时、CLI 测试通过；Ruff、git diff --check、uv pip check 通过。
Windows DML 本轮通过单元回归验证，未在 Windows 硬件重新测量。

## 完整解析

demo1+demo2 合并 PDF，共 19 页，basic＋强制 OCR。CPU/GPU 均在新进程中运行两次，采用 CPU→GPU→GPU→CPU 的顺序。
下表未启用 Python observer 或 ORT profiler，计时为 parse 调用，不含结果写盘和导入模块；没有额外模型预热。

| 模式 | 第一次（秒） | 第二次（秒） | 平均（秒） |
|---|---:|---:|---:|
| CPU | 56.927 | 56.534 | 56.731 |
| WebGPU＋MFR CPU 尾批 | 36.003 | 36.325 | 36.164 |

平均加速 **1.569×**，耗时减少 **36.25%**。
四次正式运行和两次 observer 运行的 Markdown、三个 JSON 与 46 个素材，共 50 个文件全部逐字节一致。

## 各阶段累计推理

以下是独立 observer 运行中同步 ORT session.run 的累计墙钟，包含首次调用、数据传输和 EP 内部 CPU 算子，
不含模型初始化、预处理、后处理。Python profiler 通过标准 call/return 事件观测，没有替换生产函数；
observer 运行的整本解析耗时不参与上面的完整解析对比。两个相同类别会话的时间已合计。

| 模型 | CPU 推理（秒） | WebGPU 路线推理（秒） | 加速 | CPU 初始化（秒） | WebGPU 路线初始化（秒） |
|---|---:|---:|---:|---:|---:|
| layout | 4.702 | 1.973 | 2.38× | 0.429 | 0.410 |
| mfr | 21.535 | 12.283 | 1.75× | 0.470 | 0.915 |
| ocr_det | 1.154 | 0.705 | 1.64× | 0.028 | 0.071 |
| ocr_rec | 17.394 | 8.937 | 1.95× | 0.061 | 0.065 |
| unet | 0.225 | 0.110 | 2.04× | 0.019 | 0.025 |

MFR 共 151 个公式：18 个满批（每批 8 个）和一个尾批（7 个）。WebGPU 路线的 MFR 初始化包含额外 CPU 尾批会话。

| 路线 | 实际 EP | 批量 | 调用次数 | 推理（秒） |
|---|---|---:|---:|---:|
| cpu-observed | CPUExecutionProvider | 8 | 18 | 18.674 |
| cpu-observed | CPUExecutionProvider | 7 | 1 | 2.861 |
| webgpu-observed | WebGpuExecutionProvider | 8 | 18 | 9.374 |
| webgpu-observed | CPUExecutionProvider | 7 | 1 | 2.909 |

## 排除预热后的模型回放

前五行使用生产 CPU 解析中捕获的第一批真实输入；UNet 使用 4130107.pdf 的全部 24 张已有 layout 裁切图，不按表格类型分流。
每个后端预热一遍，再交替顺序测量五轮，取总推理时间中位数。初始化和预热另列；后处理不计时。
该单批 rec 输入是 [6,3,48,320]，比自动校准的 [6,3,48,1012] 窄，因此不能用其速度代替整份文档的结果。

| 模型/输入 | CPU（秒） | WebGPU（秒） | 加速 | CPU/GPU 初始化（秒） | CPU/GPU 预热（秒） |
|---|---:|---:|---:|---:|---:|
| layout | 0.516468 | 0.222966 | 2.32× | 0.452 / 0.427 | 0.505 / 0.267 |
| mfr | 0.976330 | 0.364917 | 2.68× | 0.603 / 0.449 | 1.083 / 0.379 |
| mfr_tail | 2.869976 | 5.246856 | 0.55× | 0.566 / 0.428 | 2.871 / 5.360 |
| ocr_det | 0.006753 | 0.006659 | 1.01× | 0.013 / 0.046 | 0.011 / 0.025 |
| ocr_rec | 0.017542 | 0.023186 | 0.76× | 0.035 / 0.060 | 0.027 / 0.046 |
| UNet / 24 张表 | 1.432913 | 0.501504 | 2.86× | 0.018 / 0.055 | 1.551 / 0.545 |

mfr_tail 的全 GPU 回放仅用于验证分流收益，生产代码仍将不足 8 的批次送给 CPU。
UNet：24 张表中 23 张掩码完全一致，p04-t02 有 1 个像素不同；总共 9,110,528 个掩码像素。
MFR、MFR 尾批和 rec 回放的解码 token 均一致。小尺寸 rec 输入上 GPU 较慢，完整 PDF 的 rec 累计推理仍有明显加速。

## EP 与 CPU 算子回退

单独创建 profiling 会话，在计时完成后采样首个输入。以下为该次 profile 执行到的独立节点数量，不能当作耗时占比。

| 模型 | WebGPU 节点 | CPU 节点 | CPU 算子种类 |
|---|---:|---:|---|
| layout | 1397 | 304 | Add, Cast, Concat, CumSum, Equal, Expand, Gather, GatherElements, GatherND, Less, LessOrEqual, Max, Min, Mod, Mul, Range, ReduceSum, Reshape, ScatterElements, ScatterND, Shape, Slice, Squeeze, Sub, Tile, TopK, Transpose, Unsqueeze, Where |
| mfr | 616 | 275 | Add, Cast, Concat, ConstantOfShape, Equal, Expand, Gather, Loop, Mul, Range, ReduceMax, Reshape, Shape, Slice, Unsqueeze, Where |
| ocr_det | 158 | 0 | 无 |
| ocr_rec | 204 | 3 | Concat, Gather, Slice |
| unet | 155 | 38 | Concat, Gather, Slice, Unsqueeze |

所有目标会话实际启用 WebGpuExecutionProvider；CPU 节点属于图内分配，没有整会话静默回退。
SLANet Plus、表格分类和印章检测仍走 CPU。此数据只验证当前 AMD Linux 主机，不据此修改 DML 阈值或推广为全部 GPU 的速度保证。

## 复现与原始数据

[机器可读报告](benchmarks/onnx-webgpu-linux-20261021.json) 包含逐次校准、模型/源码 SHA256、输入形状、输出差异、EP 参数及生命周期结果。
原始目录为 `demo/output/webgpu-production-20261021/`，包含 benchmark.py、analyze.py、逐次输出、捕获张量及 profile。
从项目根目录使用 `.venv/bin/python` 运行，完整解析分别设置 `MINERU_MODEL_SMALL_BACKEND=onnx`、
`MINERU_MODEL_ONNX_DEVICE=cpu|gpu`、`MINERU_INTRA_OP_NUM_THREADS=4`、`MINERU_INTER_OP_NUM_THREADS=1`；
调用 `benchmark.py parse --output <新目录>`；需要独立推理观测时增加 `--observe`。
六图校准与正式使用方法见 [ONNX 设备选择](../../next/onnx-device.md)。
