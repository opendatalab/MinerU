# mineru-kit hardware

状态: Implemented

`hardware detect` 推荐当前环境的 API 启动上限；`hardware info` 生成只读的问题反馈报告。两者默认输出可读文本，`--json` 输出同一份结构化数据。日志与进度进入 stderr，stdout 只保存结果。

```bash
mineru-kit hardware detect
mineru-kit hardware detect --json
mineru-kit hardware info
mineru-kit hardware info --json > hardware-info.json
```

`hardware` 不带子命令时显示帮助。没有 `--quiet` 或 `auto-tier` 别名。配置沿用 `MINERU_CONFIG`、`MINERU_HOME` 和现有后端/设备环境变量。

## 自动推荐

命令与 `api-server --tier auto` 共享结构化推荐结果。先解析有效小模型后端，再确定设备与上限：

| 有效后端 | 行为 |
|----------|------|
| Torch + 可用 CUDA/MPS/XPU 等加速设备 | 跳过 OCR 测速，推荐 advanced |
| Torch + 显式 CPU 或加速设备不可用 | 跳过测速，进程可用 CPU 至少 4 时推荐 standard，否则 flash |
| ONNX | 沿用既有硬件初选；初选至少 Basic 时在 Windows/Linux 同步完成适用的 OCR 选型，再确定最终上限 |

`small_backend=auto` 最终选择 Torch 时同样免校准。显式设备字符串必须经过实际可用性检查；检查失败会给出保守结果及原因，不修改设备配置。Torch 推荐不依赖 llama.cpp 库存，不下载 ONNX 校准权重，也不写入 `model.onnx_device` 或 OCR 校准记录。

ONNX 分支保留既有兼容策略、0.8 阈值、进程缓存和配置保存规则。首次实际测速可能下载 OCR rec 模型；下载和派生图准备不计入 15 秒校准 worker 超时。显式或已保存的 CPU/GPU 选择不强制重测。校准失败、超时或不稳定时保守推荐，并携带警告。

llama.cpp 自动集显路径同样按有效小模型后端分流：Torch 使用加速可用性结果，不再绕回 OCR 校准；ONNX 保留原有性能结果和设备身份匹配。原生独显优先、显式 CPU/GPU 参数、物理绑定和兼容处理保持不变。

JSON 报告 `schema_version=1`，包含 `ceiling`、`pdf_image_tiers`、`default_request_tier`、`reason`、`cpu_count`、`small_backend`、`vlm_engine`、`device`、`calibration` 和 `warnings`。校准状态来自执行结果；Torch 使用 `status=skipped, reason=torch_backend`，不将可用性结论标成性能实测通过。

开放 Advanced 后，请求省略 tier 仍默认 Standard。推荐不代表模型已准备好，API 服务继续执行依赖预检；显式启动上限和 Python 精确集合不进入自动推荐。API 工厂的 VLM 配置覆盖参与共享推荐，远程 VLM 显示为 `http-client`，不自动扩大本机上限。

## 环境报告

`info` 不调用自动推荐或 OCR 校准，不下载、加载模型权重或写配置。运行时导入和原生 GPU 探测使用独立子进程，每项最多等待 15 秒。某项失败时继续报告其他信息，并区分未安装、导入失败和超时。

- 系统、Python 版本与解释器路径，MinerU 版本与实际模块路径。
- CPU 型号、物理/逻辑/进程可用核数，基础/当前/最大频率（MHz）；内存总量及可用量（bytes）。
- 系统可探测到的全部独显、集显和虚拟 GPU，以及当前进程运行时可用设备。保留名称、身份、驱动及可获得的显存/共享内存信息。
- 已安装的软件包版本与位置：MinerU、DocVortex、mineru-llama-cpp、mineru-vl-utils、torch、transformers、ONNX Runtime及EP、vLLM、LMDeploy、mlx-vlm、pypdfium2。未安装的包不列入软件版本列表。
- Torch 导入、编译使用的 CUDA/ROCm 版本和后端可用性；ORT provider 与相关引擎的探测结果。
- 后端、设备、线程及可见设备限制的有效配置与来源；OCR 校准记录的存在状态、环境匹配、稳定性、耗时和阈值。

未知值为 JSON `null`，文本显示“未知”。例如 Apple Silicon 通常不报告可可靠读取的 CPU 主频，统一内存也不作为独立显存。系统或容器限制导致库存不完整时保留错误说明；不从型号名称猜测设备类型或频率。

配置报告使用明确白名单，不输出认证信息、完整远程服务地址或整个环境变量集合。诊断报告中的本地路径用于确认解释器和实际加载位置。

## 退出码

能形成推荐结果或环境报告时退出码为 0，包括带警告的保守推荐和局部探测失败。参数、配置错误或无法形成结果时返回非零退出码。
