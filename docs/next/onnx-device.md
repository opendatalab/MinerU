# ONNX 设备选择与 DirectML

`model.onnx_device` 控制 ONNX 小模型栈的 CPU / Windows DirectML 选择：

```yaml
model:
  small_backend: onnx
  onnx_device: auto
```

环境变量 `MINERU_MODEL_ONNX_DEVICE` 高于文件配置。`small_backend=auto` 仍按原有规则选择
Torch 或 ONNX；此选项不强制切换模型栈，也不改变 Torch / VLM 的设备。

| 值 | 行为 |
|---|---|
| `cpu` | ONNX 小模型全部使用 CPU，不枚举显卡、不测速 |
| `gpu` | 每个进程首次用模型时重新发现 GPU，独显优先，再选高性能排序中的集显，不测速 |
| `auto` | 独显直接选 GPU；仅有集显时执行一次 rec 测速；没有硬件 GPU 时选 CPU |

Windows 使用 DXGI 高性能顺序、D3D12 UMA 属性和 LUID 匹配；适配器编号不写入配置。
已保存的 `gpu` 使用 Windows 高性能排序及无设备创建的 D3D12 能力检查，避免逐张创建设备查询 UMA；
`auto` 才需要 UMA 分类。旧系统缺少高性能排序接口时仍执行分类以保证独显优先。
`gpu` 模式下移除显卡会在本次使用 CPU，仍保留配置中的 `gpu`。
非 Windows、驱动探测错误、运行时缺少 DML 或 GPU 会话初始化失败时，日志会说明 CPU 回退原因。

首次实际使用模型时才选型。界面启动、导入 SDK、查询配置和模型列表不会运行测速。
自动决策成功后原子更新有效配置文件中的 `model.onnx_device`，保留其余键；进程间通过文件锁协调。
环境变量覆盖不写入文件；只读配置保留本进程选择。测速失败不保存为 CPU 性能优胜，同进程不重复尝试。
通过环境变量强制 `auto` 时，每个新进程重新选择，并使用选择锁避免同时测速。
显式或已保存的 `cpu` 不会因后来接入独显而改变。

要重新测量，将配置改回 `auto` 并重启应用。修改程序阈值也不主动覆盖已保存的选择。

## 模型预设

GPU 路线只将 layout、rec、MFR 交给 DML，检测、表格等其他 ONNX 模型保持 CPU。
Torch 栈中的表格仍使用其原有 CPU/CUDA 策略。旧实验引擎和实验开关已经移除。

| 厂商 | layout | MFR | rec |
|---|---|---|---|
| Intel | 关 | 开 | 关 |
| AMD | 开 | 开 | 开 |
| NVIDIA | 开 | 开 | 关 |
| 其他 | ORT 默认 | ORT 默认 | ORT 默认 |

表格中的开关指 metacommands，关闭对应 `disable_metacommands=true`。
Intel layout 优先首轮响应；NVIDIA layout 采用后续补测结论。
MFR 只有实际批量为 8 时使用 DML，1–7 张使用惰性创建、复用的 CPU 会话，不补齐。
每个 DML 会话的 `run` 都串行化，不依赖宿主是否打开全局推理锁。

## 六图测速和阈值

图片和 manifest 位于包内 `resources/onnx_calibration`，包含原始中英数字句子、来源和 SHA256。
重新生成工具 `scripts.make_onnx_calibration_samples` 需要显式提供字体；部署时直接读取固定 PNG，
不需要系统字体。输入复用生产 rec 的 BGR、动态宽度和归一化逻辑；CPU/GPU 使用同一份 batch6。

每种设备预热一次，再计时两次。默认线程为 4/1；生产显式线程配置同时应用于测速。
比较 `ratio = gpu_elapsed / cpu_elapsed`，当 `ratio <= ONNX_GPU_TIME_RATIO_THRESHOLD` 时选 GPU。
阈值在 `mineru.model.runtime.onnx_device` 中集中维护。
当前阈值按用户要求简化为 **0.8**，即 GPU 推理耗时至少比 CPU 低 20%。
原三机校准推导值为 0.69639683662382，保留为历史参考；已有 24 次记录按 0.8 判断仍得到相同选择。
详细数据见 [三机校准报告](onnx-selection-validation.md)。

下载和设备发现不计入 worker 预算。独立隐藏子进程的总超时为 15 秒，包含解释器启动、加载及推理；
1–2 秒只是小样本推理目标。结果记录模型/输入校验值、ORT 版本与路径、provider、初始化、预热和正式计时。
CPU/GPU 解码 token 必须一致，整会话回退 CPU 或计时无效都视为失败。

### 发布前校准

在 Ultra 5 / Arc 130T、i7-12700 / UHD 770、MAX395 / 8060S 上使用相同代码、权重、样本和 ORT：

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_onnx_selection devices
.\.venv\Scripts\python.exe -m scripts.benchmark_onnx_selection run --label ultra5 --luid <集显LUID> --output output/calibration-ultra5
```

另外两台使用 `--label i7` 和 `--label max395`。i7 必须在基准工具中明确选择 UHD 770，
应用的正常独显优先规则仍会选择 RTX 3060 Ti。输出目录必须不存在。

每台运行五个新进程，交替 CPU/GPU 先后顺序。汇总命令只接受完整且协议一致的三份报告：

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_onnx_selection derive <ultra5报告> <i7报告> <max395报告> --output output/threshold.json
```

取 Ultra 5 / MAX395 所有比值的最大值为 L，取 i7 最小比值与 1.0 的较小值为 U。
只有 L < U 才给出中点 `(L + U) / 2` 作为统计建议；当前发布值按用户要求采用 0.8，
不自动用统计建议覆盖正式常量。随后各平台以 `run --threshold <待验证值>` 和新目录运行三个
独立验证进程，要求 Ultra 5、MAX395 全部选 GPU，i7 全部选 CPU。没有有效区间或验证失败时不发布阈值。
阈值为 `None` 表示三机校准尚未完成；此时自动集显路径使用 CPU、保留 `auto`，显式 GPU 及独显仍可使用。

## 安装

Windows x64 的直接依赖为 `onnxruntime-directml>=1.23,<2`，其他平台保留 CPU 包约束，
不按 Python 版本拆分条件。DirectML 包本身包含 CPU provider。

已有环境从 CPU 包迁移时，先卸载 `onnxruntime`，再安装 DirectML，避免两个发行包覆盖同一模块。
安装后用 `import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_available_providers())` 检查实际能力。

文件识别使用 `mineru-magika>=1.0.3,<1.1.0`，Python 导入名仍为 `magika`，由 DocVortex 声明依赖。
迁移已有环境时须先卸载旧 `magika`，再安装新发行包，避免两个包覆盖同一模块。

2026-10-18 已使用 DocVortex 提交 `545570c4bffa2a668b7db03b5d3117e7a13faf8c` 完成
Windows x64 的完整依赖安装、发行包身份、真实分类、rec 与文档解析验收。
该提交的版本号仍为 `0.5.16`，当前 PyPI 同版本仍是旧依赖；因此验收使用固定提交构建的 wheel。
在兼容版本发布到 PyPI 并更新最低版本前，源码安装应显式指定该提交：

```powershell
.\.venv\Scripts\uv.exe pip install --python .venv/Scripts/python.exe -e . "docvortex @ git+https://github.com/myhloli/DocVortex.git@545570c4bffa2a668b7db03b5d3117e7a13faf8c"
.\.venv\Scripts\uv.exe pip check --python .venv/Scripts/python.exe
```

不能把两个 ORT 发行包或两个 Magika 发行包同时存在视为有效安装。
版本、校验值、平台矩阵与实测结果见 [依赖验收报告](onnx-dependency-validation.md)。
