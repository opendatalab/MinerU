# Linux Xe-LPG 自动串行提交验证

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

> 2026-10-29 策略更新：本文保留历史验证结果。当前仅自动 ONNX 路径依据 ONNX 已选适配器补充串行设置；
> 已移除 llama 独立选卡、绑定复核及手动 GPU 路径的自动注入。Torch 和手动选卡需显式设置兼容变量。
> 当前行为见 [llama.cpp 自动模式与手动覆盖](../../next/onnx-device.md#llamacpp-自动模式与手动覆盖)。

2026-10-22。共享 llama.cpp 设备策略新增 Xe-LPG（PCI 家族 `7dxx`、`b6xx`），
保留 Xe-LP 及更早架构的既有规则。Xe-HPG、Xe2/Xe3 LPG、其他后续架构及未知 PCI 家族不自动开启。

策略仅用于 Linux Vulkan 实际选中的 Intel 设备，适用于所有档位。设备选择、ONNX WebGPU/DML 校准、
独显优先、显式环境值、混合设备组限制、绑定复核和临时环境恢复均沿用已有机制。
进程内 Engine 和 llama-server 共用策略，无需更新原生包。

## 回归检查

以下测试共 **369 passed**；修改的 Python 文件通过 Ruff 检查及格式检查，`git diff --check` 通过。

```bash
.venv/bin/python -m pytest \
  tests/unittest/test_llama_device_policy.py \
  tests/unittest/test_llama_server_device_policy.py \
  tests/unittest/test_llama_cpp_server.py \
  tests/unittest/test_webgpu_architecture.py \
  tests/unittest/test_webgpu_runtime.py \
  tests/unittest/test_onnx_device_selection.py \
  tests/unittest/test_onnx_calibration.py -q
```

新增及扩展覆盖包括两组 Xe-LPG PCI 家族、后续架构排除、缺失 PCI 身份、Windows/非 Vulkan 后端、
集显校准结果复用、Xe-LPG 与 NVIDIA/AMD/Xe-HPG 多卡选择、显式混合设备组拒绝、环境恢复，
以及 Engine 构造和 server 启动环境。server 同时覆盖模型在 GPU 和仅视觉编码在 GPU 的情况。

## 本机完整页验证

- 设备：Core Ultra 5 225H，Intel Graphics ARL，`8086:7d51`；Linux `7.0.0-38-generic`，i915 / Mesa 26.0.8。
- 使用当前安装的 `mineru-llama-cpp==0.1.3`、原有 Q8_0 权重、生产 MinerU 源码。
- 新进程启动前完全移除 `GGML_VK_SERIALIZE_SUBMISSIONS`，显式选择 Vulkan，ONNX 小模型固定 CPU。
- Engine 与 server 的真实设备计划均绑定 `Vulkan0` / `8086:7d51`，并自动加入串行值 `1`。
- 生产 Engine 初始化记录的环境签名包含串行值 `1`；初始化后进程临时环境恢复，变量重新不存在。
- demo1 第一页、advanced OCR 完成，得到 **1 页、19 个块**；与此前手动开启串行的 Markdown 逐字一致。
  相比 CPU 基线，仅第一条作者单位的上标后少一个空格，正文其余内容一致。
- 对应进程内核日志无 fence 超时或 GPU 重置。用户配置文件 SHA-256 前后相同；未设推理超时。

本次只验收正确性，没有追加性能对比。其他架构和平台通过回归用例验证，未新增跨平台真机测试；
server 验证了本机设备计划，启动时的环境传递通过集成单测验证。

结构化结果：`docs/next/benchmarks/llama-xe-lpg-auto-serialize-linux-20261022.json`。
原始日志、输入配置和输出：`demo/output/xe-lpg-auto-serialize-20261022/`。
