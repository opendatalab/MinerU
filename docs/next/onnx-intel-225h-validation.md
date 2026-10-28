# Intel Core Ultra 5 225H：basic＋OCR CPU / GPU 性能验证（2026-10-22）

Intel Core Ultra 5 225H（14 核、14 线程），Intel Arc 集显（PCI 8086:7d51）；Linux 7.0.0-38，Mesa 26.0.8。
Python 3.14.4、MinerU 4.0.12、DocVortex 0.5.16、ORT 1.31.0、WebGPU EP 0.4.0。
全程接电，系统 balanced，CPU governor 为 powersave；CPU/GPU 测试均保持该配置。

## 测试方法

沿用前次聊天“更新 DocVortex 并测试 demo1”的精简 rec 生产测试脚本与合并 PDF，demo1 后接 demo2，共 19 页。
输入 SHA256：`06d0e74125b5d2e1fc04e658bbe4af39422d70abfaa453a87353df4d3dad3432`。
basic＋强制 OCR，整本解析；关闭标题层级和跨页表格 LLM 增强；ONNX 小模型栈。CPU intra-op=4、inter-op=1。
GPU 使用 Vulkan WebGPU，NHWC、high-performance、graph capture 关闭。MFR 满批 8 走 GPU，不足 8 走 CPU；表格分类、SLANet Plus 仍走 CPU。
每种路线两个新进程，顺序 CPU→GPU→GPU→CPU。没有额外预热；计时包含模型初始化和首轮推理，不含模块导入、模型下载及结果写盘。
派生 rec 模型缓存提前准备；主要计时关闭 observer / ORT profiler，阶段耗时由另外两个新进程采集。

## 完整解析

| 路线 | 第一次（秒） | 第二次（秒） | 平均（秒） |
|---|---:|---:|---:|
| CPU（4 线程） | 133.466 | 153.124 | 143.295 |
| Intel 集显 WebGPU＋CPU 尾批 | 96.225 | 93.409 | 94.817 |

WebGPU 加速 **1.511×**，耗时减少 **33.83%**。
CPU 第二轮比第一轮慢 14.73%；测试在日常桌面环境运行，未锁频，以上保留逐次耗时。

## 各阶段累计推理

独立 observer 运行的同步 ORT run 墙钟，包含首轮、传输及图内 CPU 算子，不含预处理、后处理和初始化。阶段之和不等于上表完整解析耗时。

| 模型 | CPU（秒） | GPU 路线（秒） | 加速 | CPU/GPU 调用次数 |
|---|---:|---:|---:|---:|
| layout | 12.120 | 7.016 | 1.73× | 10 / 10 |
| mfr | 62.757 | 39.008 | 1.61× | 19 / 19 |
| ocr_det | 2.514 | 3.090 | 0.81× | 84 / 84 |
| ocr_rec | 53.570 | 27.233 | 1.97× | 336 / 336 |
| unet | 0.523 | 0.517 | 1.01× | 5 / 5 |
| table_cls | 0.012 | 0.013 | 0.87× | 1 / 1 |
| slanet_plus | 0.305 | 0.211 | 1.44× | 2 / 2 |

## MFR 实际分流

| 路线 | 实际 EP | batch | 调用次数 | 累计推理（秒） |
|---|---|---:|---:|---:|
| cpu | CPUExecutionProvider | 8 | 18 | 57.302 |
| cpu | CPUExecutionProvider | 7 | 1 | 5.455 |
| webgpu | WebGpuExecutionProvider | 8 | 18 | 34.332 |
| webgpu | CPUExecutionProvider | 7 | 1 | 4.676 |

## 初始化与首次调用

初始化单独列出；首次调用已包含于对应阶段推理，不能再加总。MFR 的 GPU 路线初始化及首次调用均包含额外 CPU 尾批会话。表格分类、SLANet 工厂初始化未由该 observer 采集。

| 模型 | CPU 初始化（秒） | GPU 路线初始化（秒） | CPU 首次（秒） | GPU 路线首次（秒） |
|---|---:|---:|---:|---:|
| layout | 0.992 | 1.090 | 0.939 | 0.967 |
| mfr | 1.367 | 2.055 | 2.931 | 6.110 |
| ocr_det | 0.052 | 0.121 | 0.185 | 0.238 |
| ocr_rec | 0.143 | 0.150 | 0.121 | 0.115 |
| unet | 0.039 | 0.058 | 0.140 | 0.136 |

## 输出与实际后端

| 运行 | 输出文件数 | 与本机 cpu-1 逐字节一致 | 差异文件 |
|---|---:|---|---|
| cpu-1 | 50 | 是 | 无 |
| webgpu-1 | 50 | 是 | 无 |
| webgpu-2 | 50 | 是 | 无 |
| cpu-2 | 50 | 是 | 无 |
| cpu-observed | 50 | 是 | 无 |
| webgpu-observed | 50 | 是 | 无 |

本机 cpu-1 与此前 AMD 主机结果存在差异：markdown.md, middle_json.json, model_output.json, structured_content.json。
跨机 Markdown 的差异为参考文献中的一处引号；本机 CPU/GPU 一致性以本机 cpu-1 为基准独立核验。
五种 GPU 目标模型的真实生产会话均核验启用了 WebGpuExecutionProvider；MFR 尾批明确使用 CPU，未将整会话 CPU 回退记为 GPU 成绩。
独立采集的解析进程 DRM fdinfo 显示其打开 Intel PCI 0000:00:02.0 的 renderD128，i915 render 引擎累计执行时间在两秒采样间隔内增加，确认使用硬件集显。
WebGPU EP 可将不支持的图内算子分配至 CPU；本次未额外统计节点回退耗时。

## 六图自动选型校准

5 个独立子进程 GPU/CPU 比值：0.687206, 0.722185, 0.791659, 0.759257, 0.732416；每个后端预热一次、计时两次。
成功 5 次，异常 0 次；全部解码一致。
全部比值低于现有 0.8 阈值，对本机推荐 GPU。本次使用环境变量显式选择 CPU/GPU，没有改写用户设备配置。

## 复现与数据

从项目根目录执行，原始日志、六次完整输出、张量、逐次会话和电源状态记录均位于 `demo/output/intel-225h-basic-ocr-20261022/`。

```bash
MINERU_MODEL_SMALL_BACKEND=onnx MINERU_MODEL_ONNX_DEVICE=cpu \
MINERU_INTRA_OP_NUM_THREADS=4 MINERU_INTER_OP_NUM_THREADS=1 \
.venv/bin/python demo/output/intel-225h-basic-ocr-20261022/benchmark.py parse \
  --input output/pdf/demo1-demo2-merged.pdf --output <新目录>
```

GPU 将 `MINERU_MODEL_ONNX_DEVICE` 改为 `gpu`；阶段采样另加 `--observe`。
[机器可读报告](benchmarks/onnx-intel-225h-linux-20261022.json) 包含源码/模型校验值、校准、实际 EP、输出差异、逐次计时与电源状态。

本次未修改生产推理代码。数据反映接电 balanced 模式、4 线程 CPU 与当前 WebGPU 实现的比较，未测 CPU 其他线程数或其他 GPU 后端。

## 后续诊断

[OCR det、校准稳定性与 Rust 后端诊断](onnx-intel-det-rust-diagnosis.md)进一步检查了实际 Rust 调用，并显式指定 Rust 复测 CPU/GPU；还包含 det 布局、输出绑定回放与更多校准数据。
