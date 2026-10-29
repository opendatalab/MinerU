# Linux WebGPU Intel 架构过滤与多卡验证

> 历史归档：保留原设计或验收时的记录，不作为当前版本的安装、接口或实现指引。
> 当前入口见 [开发文档](https://opendatalab.github.io/MinerU/next/)；当时的版本、测试结论和未完成事项不追溯改写。

2026-10-21。实现用户指定的产品策略：Intel Xe-LP 及以前使用 CPU；Xe-LPG、Xe-HPG 及表内后续架构保留自适应校准。
AMD/NVIDIA 不参与 Intel 架构分类，Windows DML 原有规则保持不变。

## 显式架构映射

`mineru/model/runtime/intel_gpu.py` 使用显式 PCI 家族表，以 `device_id & 0xff00` 查询架构，依据
[Dawn GPU 映射](https://github.com/google/dawn/blob/main/src/dawn/gpu_info.json)。不依赖显卡名称、CPU 发布时间或 PCI ID 数值大小。

- 46xx / 4cxx / 49xx / 9axx / a7xx：Xe-LP，CPU。包括 UHD 770、Iris Xe、N100/N150 所属家族及 DG1 独显。
- 4fxx / 56xx：Xe-HPG；7dxx / b6xx：Xe-LPG 系，保留校准。
- 64xx / e2xx：Xe2；b0xx / fdxx：Xe3，保留校准。
- 其他已知早期家族为 legacy，CPU；未识别 Intel 家族也保守使用 CPU，等待显式补充。

架构检查覆盖 auto、普通 gpu 配置和已保存的 gpu；架构回退不改写原配置，不记录为 CPU 性能胜出。
仅有旧 Intel 时在下载校准模型之前返回 CPU；低层 WebGPU 会话工厂也拒绝旧 Intel 描述符。
显式 cpu 不枚举 GPU，模块导入不加载 Vulkan 或进行硬件探测。
完整映射及行为见 [设备说明](../../next/onnx-device.md)。

## 多卡约束

发布版 WebGPU EP 0.4.0 的 ORT 枚举设备不绑定 Dawn 的实际 GPU，不能只删除枚举列表中的 Intel 项。
当库存同时存在旧 Intel 和其他候选设备时，新增的 `vulkan.py` 直接调用稳定 Vulkan C ABI 枚举实际可见 GPU。
查询尊重已有 ICD/设备筛选，仅创建 Vulkan instance、读取设备属性，不创建 logical device 或提交 GPU 计算。

从实际可见且允许的设备中优先选择独显；会话创建期间通过 Vulkan loader 的厂商/设备 ID 过滤约束 Dawn。
创建 Dawn 之前再次枚举并验证只剩目标 GPU；验证失败时不运行模型。
本机 Vulkan loader 1.4.341 支持该功能（上游最低为1.4.326），实际筛选结果只有 `10de:2489`。
旧 loader 尝试 Mesa device-select 的 `vendor:device!`，也必须独立验证；两种方法都无效则回退 CPU并明确记录原因。

仅在混合旧 Intel 的环境中增加这些步骤，纯 AMD/NVIDIA 或全部为已支持架构的库存保持原路径。
过滤描述以可哈希的整数元组传入校准 worker，禁止传递原生句柄。
环境变量在每次会话创建后恢复，包括异常路径；每次创建都重新约束，以覆盖原生 context 的释放/重建。
同进程不得切换已初始化的设备约束。MinerU 的 WebGPU 锁覆盖会话创建；宿主自行使用的其他 Vulkan 初始化不应同时运行。
嵌入式宿主若已经直接初始化同一插件的默认 Dawn context，需要使用新进程；本策略无法重新绑定外部提前创建的 context。

上游参考：[Vulkan Loader 环境过滤说明](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderInterfaceArchitecture.md)。
没有安装新 Python 包、替换 GPU 驱动或使用未发布的 ORT adapterIndex 参数。

## 本机实测

Intel i7-12700 / UHD 770（8086:4680、i915）/ RTX 3060 Ti（10de:2489、nouveau / Mesa NVK）。
ORT 1.31.0、WebGPU EP 0.4.0、Vulkan loader 1.4.341，ONNX 4/1线程。

| 环境 / 请求 | 次数 | 实际结果 |
|---|---:|---|
| 默认双卡，auto | 5 | 全部保留3060 Ti；六图解码一致；全部1＋3结束 |
| 仅 Intel ICD，auto | 1 | CPU / webgpu_architecture_policy，0.078秒，无校准 |
| 仅 Intel ICD，gpu | 1 | CPU / webgpu_architecture_policy，0.072秒，无校准 |
| 仅 NVIDIA ICD，auto | 1 | 3060 Ti，解码一致，1＋3结束 |
| 默认双卡，gpu | 1 | 跳过性能校准；连续销毁/重建3个rec会话，输出ID与CPU一致 |
| 默认双卡，独立FD复核 | 1 | 又重建3个rec会话，确认打开NVIDIA renderD129/card0 |

“仅 Intel ICD”仍使用真实双卡主机，ORT 硬件库存仍可包含 NVIDIA；Vulkan 可见性检查阻止了错误放行。
本轮没有再次在旧 Intel 上运行模型推理。所有采样的 i915 render 计数为0，没有新增 GPU HANG。
最初 fdinfo 采样只收集含 drm-driver 的条目，NVK 不提供该字段；独立复核改为同时读取 FD 指向的设备路径，
确认了 renderD129/card0。没有将初始采样未捕获 NVIDIA 字段误认为未使用独显。

默认双卡 auto 的完整选择耗时中位数见机器可读报告；它包括实际设备发现与校准，不能与不含校准的模型时间混比。
真实用户配置 SHA256 保持 `5c447bf9c2834991aba50a86cd5a435a4753a0ad6a6e21b745e685670fd59e99`。

## 完整19页流程

使用已有 demo1＋demo2 合并PDF，basic＋强制OCR、默认双卡auto、ONNX 4/1线程；关闭LLM标题层级与跨页表格增强。
复用此前生产基准脚本，未修改生产模型、权重、输出布局或推理批次。

- 完整解析 **51.884秒**，含本次auto校准及模型初始化，排除模块导入和结果保存。
- 生成19页、50个结果文件，**全部与此前同机CPU基线逐字节一致**。
- layout、MFR、OCR det/rec、UNet 的生产日志均记录 WebGPU 和绑定的3060 Ti；CPU专用路径保持原样。
- 本次未开observer；这是一次功能验收，不是严谨的前后性能变化测量。
- 没有新增GPU HANG，采样中的Intel render计数始终为0。

## 测试与限制

**208项相关测试通过，Ruff格式/lint、git diff --check通过。**
覆盖旧/新架构、未知Intel家族、AMD/NVIDIA同数值ID不误伤、DML隔离、显式/保存gpu、跳过下载/校准、
混合卡枚举顺序、ICD不可见独显、Intel Arc独显、Vulkan过滤无效拒绝、旧loader的Mesa回退、
环境恢复、会话重建、worker JSON绑定传递、软件CPU排除及Vulkan实例异常清理。

本机没有新的Arc/AMD硬件或Windows DML硬件复测，相关路由由单测覆盖。
旧loader＋Mesa回退目前为模拟验证，本机实际使用的是新loader的ID过滤。
没有把架构允许等同于模型兼容性保证：新架构继续执行原有输出检查与自适应性能选择。

[机器可读报告](benchmarks/onnx-intel-architecture-policy-linux-20261021.json)保存规则、源码SHA256、实际选择、
过滤前后库存、六图数据、重建会话、完整输出校验、硬件FD证据摘要及配置检查。
原始脚本和日志位于 `demo/output/onnx-intel-architecture-20261021/`；`build_report.py` 仅枚举设备属性并汇总证据，不运行模型。
