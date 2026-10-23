# 3 次预热 / 7 次计时中位数校准与 UHD 770 排查

日期：2026-10-21；i7-12700 / UHD 770 / RTX 3060 Ti，沿用[此前完整文档测试](onnx-i7-12700-webgpu-validation.md)的机器、权重和环境。

后续更新：[NHWC 关闭 prepacking 的完整流程验证](onnx-i7-12700-nhwc-no-prepack-validation.md)发现，UHD 770 第一遍有内容错误，第二遍仍发生 GPU HANG；本报告的小样本恢复不代表整本修复。

## 已落地的校准改动

CPU 和 GPU 均预热 3 次，再计时 7 次，各自取中位数；比较 GPU 中位数 / CPU 中位数。
DML、WebGPU 阈值仍均为 **0.8**，等于 0.8 时接受 GPU，15 秒 worker 总预算不变。
每次预热和正式采样仍检查解码 token，不能用中位数掩盖文本错误。
初始化、解码检查、模型下载和派生图生成不进入单次推理时间。

worker 报告 schema_version=4，新增 measurement={warmup_runs:3,timed_runs:7,statistic:median}。
cpu_elapsed / gpu_elapsed 现在表示各自的中位数；保留完整 warmup_s / timed_s 与 first_s / median_s。
阈值推导工具以测量协议作为签名，拒绝新旧统计协议混用；历史同协议报告仍可独立读取。
没有改动 GPU 布局、设备发现、持久化策略或真实用户配置。

## 本机正式 auto 新进程复测

ONNX 栈，4/1 线程；同一六图输入和精简 rec 模型，每次使用真实 resolve_onnx_device 与其独立 worker。
没有 monkey patch、替换测速器或绕过输出验证。通过环境变量 auto 避免写入用户配置。

| 环境 | 新进程数 | 选择 GPU | GPU/CPU 中位数比值范围 | worker 秒数范围 |
|---|---:|---:|---:|---:|
| 默认双卡 | 20 | 20 | 0.296186–0.330581 | 2.277–2.435 |
| 仅 NVIDIA ICD | 10 | 10 | 0.289179–0.330369 | 2.277–2.427 |
| 仅 Intel ICD | 5 | 0 | 文本不一致，未产生有效比值 | — |

30 次 NVIDIA 路线全部成功、文本一致、选择 GPU，无误选 CPU 或超时。
另在隔离配置文件中验证：第一次测量保存 gpu，第二个新进程返回 configured_gpu 且不重测，其余配置键保留。
对每次报告重新计算 7 个正式时间的中位数，均与 CPU/GPU elapsed 和最终 ratio 精确一致。
有限次数验证不能保证所有负载下稳定；本轮没有重新测量 Windows DML 硬件性能。

## UHD 770：NHWC 空输出

已将问题进一步收敛到**原生 WebGPU 的预打包 / 常量卷积执行路径**；尚未证明是具体哪一条 native 指令或驱动代码缺陷。

- 8 组单独小型 Conv（通道数 3/4/8/16，1×1 / 3×3）均与 CPU 一致。
- 原始 rec 的前 45 节点与 CPU 接近；后续前缀可出现明显偏差。prefix-78 的最终节点为 Conv.14＋Add.36，经优化为 NHWC Conv＋bias，默认最大误差约 5.425；禁用 prepacking 后约 8.1e-6。
- 前缀变化与进程状态会影响错误是否出现，不能据此声称这是全图唯一的首个故障节点。例如 prefix-80 在一次运行错误、另一次运行接近 CPU。
- 3 个独立进程中，NHWC＋session.disable_prepacking=1 的完整原始 rec 图均与 CPU 数值接近，最大误差约 1.0e-5；最后一轮另确认六图 CTC token 完全一致。
- 相同开关对 layout 的两页输入也恢复正常：CPU/GPU 分别均为 24、9 个有效候选，最大数值差约 4.9e-4。
- 单独将 storageBufferCacheMode 改为 lazyRelease 仍然失败，最大误差接近 1；因此不能简单归因于缓存复用，更不能将“禁用全部缓存”作为修复。
- 某些同进程 robustness 对照被 Dawn 提示已初始化、设置被忽略；这些对照不能用于判断 robustness 是否有效。

原始 rec 与精简 rec 在默认 NHWC 下都曾输出空文本，因此不是新增 ArgMax/ReduceMax 或 CTC 解码引入。
rec 与 layout 的独立张量回放不经过 PDF、DocVortex 或 MinerU 后处理，排除了这些层作为本次模型数值错误的来源。

源码参考为实际安装版本对应的 [WebGPU EP v0.4.0 Conv 实现](https://github.com/microsoft/onnxruntime/blob/plugin-ep-webgpu/v0.4.0/onnxruntime/core/providers/webgpu/nn/conv.cc)：
预打包在 GPU 转置权重并保留 packed kernel；NHWC 1×1 Conv 还会将其作为常量矩阵交给 MatMul 优化路径。
所以 disable_prepacking 的对照同时绕开了权重预处理及相关常量优化，**尚不能仅凭该开关区分“转置/同步/生命周期问题”和“常量矩阵内核问题”**。
这是有实验支持的排查范围与临时规避方向，没有据此修改生产 WebGPU 参数，也没有将小样本恢复称为整本修复。

## UHD 770：NCHW GPU HANG

通过纯 CPU 解析捕获 84 次 OCR det 输入，其中主文字检测会话 75 批；回放直接调用原生 EP，不读取 PDF、不经过 DocVortex，也不混入其他模型。

| 回放 | 完成情况 | 新挂起 |
|---|---|---|
| 上次失败位置附近的单批 [1,3,128,160] | 普通参数和组合检查参数均完成 | 无 |
| 相邻第 33–35 批（0-based 32,33,34） | 全部完成，最大差约 1.3e-4 | 无 |
| 原 75 批完整序列，普通 NCHW 参数 | 全部完成，最大差约 2.6e-4 | 无 |
| 同序列，禁用预打包/缓存并开启 validation/robustness | 完成 21 批，第 22 批 [2,3,800,704] 挂起 | 有 |

新挂起发生于 **20:35:56**，i915 记录 python[24573] GPU HANG，ecode 12:1:85dcfffb。
随后 Vulkan 返回 DEVICE_LOST，ORT 在结果下载处报 Failed to wait for the operation:2。
该下载错误是 GPU 已挂起后的表现，不能将其直接认作下载函数本身的根因。
检测到新内核挂起后自动停止剩余 det GPU 回放。

这说明 NCHW 问题可以在**原生 OCR det / Dawn-Vulkan / i915** 层复现，与 PDF、Rust 后处理和多模型混跑无关；
同时，同输入在普通参数下可通过，挂起位置也从此前的 [1,3,128,160] 附近改变，不能只用一张坏图、一个固定尺寸或单次 OOM 来解释。
多个检查选项同时改变，本轮没有逐项分离其因果作用；也没有将缓存、robustness 或 prepacking 的其中一个选项单独定为根因。
具体挂起 shader、原生 EP 与驱动各自的责任尚未定位；当前用户无法读取 i915 error-state sysfs，未获取完整 GPU 错误快照，也未重编译 native EP 验证源码修复。
本轮没有验证可发布的 NCHW 修复，不将其用作生产回退。

## 验证、资料与复现

**108 项回归通过**，覆盖中位数对异常值的处理、预热排除、CPU/GPU 执行顺序、预热/计时中的文本变化拒绝、测量协议一致性、设备策略和精简 rec。
Ruff 格式与 lint、git diff --check 通过。

[机器可读报告](benchmarks/onnx-i7-12700-median-diagnosis-linux-20261021.json)保存全部 35 次选择、逐次 3＋7 采样、源码校验、配置持久化、前缀与 det 回放结果。
原始日志、独立 ONNX 前缀、optimized graphs、ORT profiles 和检测张量位于 demo/output/i7-12700-calibration-median-20261021/。

从项目根目录重新进行校准验证：

```bash
.venv/bin/python demo/output/i7-12700-calibration-median-20261021/repeat.py
```

该脚本只使用环境变量 auto；隔离配置验证见 persistence.json。再次运行会重写本目录的实验文件，保留证据时应复制脚本到新的输出目录。
prefix / det 的具体命令、输入形状与参数保存在脚本和机器可读报告中；本轮只更新校准生产逻辑，UHD 770 调整均属于诊断对照。
