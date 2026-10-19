# ONNX 依赖验收（2026-10-18）

Windows x64 的完整安装验收通过，使用项目 `.venv`、Python 3.14.7 和 PowerShell 7。
本轮使用已发布的 `mineru-magika` 与固定提交构建的 DocVortex，未使用 `--no-deps`、修改已安装元数据或依赖忽略规则。
可复核结果见 [验收数据](benchmarks/onnx-dependency-20261018.json)。

## 安装来源与发行包身份

| 发行包 | 验收版本 / 来源 |
|---|---|
| MinerU | 本工作区构建的 `4.0.12` wheel；项目 `.venv` 安装为 editable |
| DocVortex | `0.5.16`，提交 [`545570c4bffa2a668b7db03b5d3117e7a13faf8c`](https://github.com/myhloli/DocVortex/commit/545570c4bffa2a668b7db03b5d3117e7a13faf8c) |
| mineru-magika | PyPI `1.0.3`，wheel SHA256 `1b67e05769f48c28d4d5dda0869ece4f4acce769ec7ab6ed169ed34ee5b58343` |
| onnxruntime-directml | PyPI `1.24.4` |
| mineru-vl-utils | PyPI `2.0.7` |

DocVortex 从干净的固定提交构建纯 Python wheel，`DOCVORTEX_BUILD_NATIVE=0`，与此前开发环境一致；
本报告不包含 Rust 扩展构建验收。其他 wheel 的 SHA256 均记录在验收数据中。

迁移先卸载旧 `magika`，随后运行带完整依赖解析的：

```powershell
.\.venv\Scripts\uv.exe pip install --python .venv/Scripts/python.exe --index-url https://pypi.org/simple -e ".[dev,test]" output/onnx-dependency-acceptance-20261018/dist/docvortex-0.5.16-py3-none-any.whl
.\.venv\Scripts\uv.exe pip check --python .venv/Scripts/python.exe
```

安装解析了 121 个包，安装后的 159 个发行包通过 `uv pip check`。
发行包元数据与模块所有权检查确认：

- `onnxruntime` 模块只有 `onnxruntime-directml` 一个所有者；未安装 CPU 版 `onnxruntime`。
- `magika` 模块只有 `mineru-magika` 一个所有者；未安装旧 `magika`。
- `mineru-magika` 仅声明 `click` 依赖，没有任一种 ORT 依赖。
- 实际 ORT 位于项目 `.venv/Lib/site-packages/onnxruntime/__init__.py`，版本 `1.24.4`，
  provider 为 `DmlExecutionProvider` 和 `CPUExecutionProvider`。

依赖检查脚本现同时拒绝两种 ORT 并存、旧 Magika 与新发行包并存，以及缺少新 Magika 的求解结果。
Windows x64 平台条件补充小写 `amd64`，与 DocVortex 保持一致；条件仍不区分 Python 版本。

## 回归与真实执行

- MinerU 的 ONNX、配置、路由、协议、文件类型与依赖检查：**762 passed**，仅两个第三方弃用警告。
- DocVortex 的安装元数据矩阵和真实文件 / 代码语言识别：**71 passed**。
  真实 Magika 会话使用 DirectML 发行包提供的 CPU provider，线程为 `4/1`；惰性导入检查通过。
- 真实 HTML 分类、Flash HTML 原生解析和单页 PDF 原生文本解析通过。
- Ultra 5 / Arc 130T 使用新依赖运行三个独立 rec 验证进程：

| 进程 | CPU 两次累计 / 秒 | GPU 两次累计 / 秒 | ratio | 阈值 0.8 的选择 |
|---|---|---|---|---|
| 1 | 0.259271 | 0.141918 | 0.547375 | GPU |
| 2 | 0.261896 | 0.137173 | 0.523770 | GPU |
| 3 | 0.262668 | 0.141822 | 0.539929 | GPU |

三次均实际加载 DML provider，CPU/GPU 识别结果一致，没有超时或异常。

使用新依赖在新进程中完成 demo1+demo2 的全部 19 页 basic + 强制 OCR。
结果的完整 `pages` 数组与此前 CPU 基准完全相等，包括正文、公式、边界框、顺序和页面结构；
layout、rec、MFR 使用 DML，其他模型使用 CPU。
Markdown 文本也完全一致，统一为基准脚本的 Windows CRLF 写出方式后逐字节一致，SHA256 为
`6fa409c5986b5207a4cadff8d106a063c977180758babe615fc5ce58ebd86b49`。
本轮初始验收脚本使用 `write_bytes` 输出 LF，而基准使用 `write_text` 输出 CRLF，导致首次字节比较失败；
已确认差异仅为换行，修正脚本，并保留 LF 原件、CRLF 对照及独立比较记录。

## 平台求解与 wheel 覆盖

使用三个真实 wheel 从头求解基础安装依赖，并核对 PyPI 的 Python 范围和目标 ABI / 平台 wheel。
此处仅 Windows Python 3.14 执行了真实安装；其余格子是求解和发行文件检查，不代表在目标机器执行过安装或推理。

| 平台 | Python | 结果 | ORT / Magika 选择 |
|---|---|---|---|
| Windows x64 | 3.10 | 通过 | DirectML `1.23.0` + mineru-magika `1.0.3` |
| Windows x64 | 3.11–3.14 | 全部通过 | DirectML `1.24.4` + mineru-magika `1.0.3` |
| Linux x64 | 3.10–3.14 | 全部通过 | 仅 CPU ORT + mineru-magika `1.0.3` |
| Apple Silicon / macOS 14 | 3.10–3.14 | 全部通过 | 仅 CPU ORT + mineru-magika `1.0.3` |
| Intel Mac / macOS 14 | 3.10–3.13 | 求解通过；纯 wheel 检查失败 | 当前选出的 cryptography `50.0.2` 没有 Intel Mac wheel，源码构建未验收 |
| Intel Mac / macOS 14 | 3.14 | 求解失败 | ORT `1.23.2` 无 CPython 3.14 wheel，较新 ORT 无 Intel Mac wheel |

上述 Intel Mac 限制单独记录，未改变用户要求保留的其他平台依赖约束。
140 项 MinerU 平台 / Python / extra 标记测试及 DocVortex 安装元数据测试均确认 ORT 条件互斥。

并行检查还暴露了 Windows 下元数据缓存被另一进程读取时无法替换的竞争；检查脚本已改为复用已发布的完整缓存，
随后 Windows 和 Linux 的矩阵重跑通过。

## PyPI 发布边界

截至本轮检查，PyPI 的 DocVortex 仍为旧元数据的 `0.5.16`；兼容提交的源码版本号也仍是 `0.5.16`。
因此本轮通过的是**固定兼容源码的完整依赖验收**。
额外以普通 PyPI 依赖求解作对照，仍选出旧 `magika==1.0.3`、CPU `onnxruntime==1.31.0`
及 `onnxruntime-directml==1.24.4`，该组合明确判为不合格。

兼容版发布到 PyPI 后，需要提高 MinerU 的 DocVortex 最低版本，再验收不带源码覆盖的 PyPI 安装。
在此之前，按 [安装说明](onnx-device.md#安装) 显式指定固定提交。

原始构建产物、安装前后清单、求解输出、测试日志及推理结果保存在
`output/onnx-dependency-acceptance-20261018/`，未提交大型文档输出。
