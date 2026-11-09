# ADR-0036: 进程默认设备决策

状态: Accepted，已实现
日期: 2026-11-09

## 契约

`model.onnx_device` 只作为 ONNX 选型输入及持久化偏好。运行时通过
`resolve_runtime_device()` 获得同一份不可变快照，包含小模型后端、cpu/gpu
模式及完整后端设备描述。gpu 泛指已选定的非 CPU 加速模式。

- 首次正式选型固定进程默认后端与设备；配置及环境变化在新进程生效。
- `cached_runtime_device()` 只读；未初始化或初始化中返回 None，不触发探测。
- 线程共享一次初始化；子进程不继承父进程的快照或初始化锁。
- 初始化入口的下载权限和线程选项仅用于首次选型，不构成多份缓存键。
- 底层 ONNX/Torch 选型不持有正式运行缓存。只读诊断与独立预探测不发布默认决策。
- 正式离线运行可以固定 CPU 回退；后续在线调用也复用该决策，重新选型需要新进程。
- 显式模型后端/设备覆盖只影响对应模型构造，不改写进程默认快照。
- 模型局部回退不修改默认策略。具体 provider、远程 VLM 和局部覆盖不由二值代表。
- 不提供生产环境清空默认决策的接口；模型卸载不清空设备决策。

## 依赖方向

轻量类型 → 底层后端选择 → 统一进程决策 → tier、llama、模型加载、health。
基础小模型后端选择不读取统一决策；校准直接创建显式设备会话，避免递归选型。
模型构造前确定设备，设备选型不获取模型缓存或 llama 引擎锁。

## 迁移清单

| 使用点 | 目标职责 |
|---|---|
| config.py | 配置输入和 ONNX 偏好持久化 |
| onnx_device.py / device.py | 底层选型，保留既有兼容、校准和失败规则 |
| device_state.py | 唯一正式运行缓存、局部模型覆盖 |
| hybrid.py / onnx.py | 消费具体设备，模型缓存不使用配置值 |
| registry.py / parser/tier.py | 已初始化后按稳定后端选择资产与预检依赖，查询不触发选型 |
| tier_recommendation.py / llama_device.py | 消费统一 CPU/GPU 模式 |
| hardware.py / hardware_probe.py / models 命令 | 分开展示配置、探测候选和本进程决策 |
| API health | 只读返回 runtime.device，未初始化时省略 runtime |
| Router health | 不用 Router 自身设备代表 worker；本次不增加单值聚合 |

## 验收

覆盖并发、进程隔离、无副作用查询、配置稳定性、局部覆盖、失败回退及不同入口顺序。
ONNX 校准、独显确认后持久化、Torch 免 ONNX 校准、llama 显式参数和 macOS 原生路由保持原有规则。
验证模型所有权及缓存复用，真实转换检查实际设备与输出。跨平台模拟覆盖与真实设备证据分别记录。

## 实施验证（2026-11-09）

- `.venv4/bin/python -m pytest tests/unittest -q --no-cov`：6027 通过、9 跳过。
- 全量运行后补充资产/依赖与显式上下文边界，相关设备、所有权、模型管理和 health 回归 250 通过。
- 核心变更通过 Ruff；修改文件中的既有检查问题与 HEAD 对比，无新增问题。新增函数均有中文说明。
- macOS 真实 Torch/MPS：`demo/pdfs/demo1.pdf` 第 1 页，Basic 解析成功，模式 gpu，具体设备 mps。
- macOS 真实 ONNX/CPU：同一 PDF 第 2 页，Layout/OCR 会话为 CPUExecutionProvider，Basic 解析成功。
- 两条真实路径均检查 Markdown 正文、JSON 及配置变化后的设备/上下文复用。
- ONNX 第 1 页所需公式权重本地缺失，下载未完成；ONNX 实测范围不包含公式模型。
- Windows DirectML、Linux WebGPU 为模拟回归覆盖，本轮没有相应真实 GPU 设备验证。
