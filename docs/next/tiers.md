# 解析档位与选择入口

事实来源：`mineru/types.py` 的显式选择顺序、`mineru/parser/tier.py` 的 effort 映射、
`mineru/filetypes.py` 的文件类型边界，以及各入口的调用代码。

| tier | effort | 执行用途 |
|---|---|---|
| flash | flash | 原生解析或 Flash OCR |
| basic | medium | 小模型推理 |
| standard | high | 小模型与 VLM |
| advanced | xhigh | 复用 Standard 模型，增加推理投入 |

基础包已经提供 ONNX 与 llama.cpp；Basic 不要求额外安装 Torch。
非 PDF 原生格式使用 DocVortex，与 PDF 推理档位不是同一种实现；不承诺 Flash 对原生格式的质量总是最低。

## 各入口的默认选择

| 场景 | 当前规则 |
|---|---|
| 直接 Python `parse` / `MinerUParser` | 默认 standard，不接受 `tier=None/auto`；原生格式实际输出 flash/txt |
| 本地 `mineru-kit parse` | PDF/图片默认 standard；原生单文件未指定 tier 时使用 flash，显式质量档位报错；批量中的原生文件归一为 flash |
| Tasks、HTTP SDK、file-parse | PDF/图片按 standard → basic → flash → advanced；仅有 Advanced 时默认选中它；原生文件固定 flash |
| Gradio 初始档位 | 与 Tasks 使用同一选择函数；外部 API 模式依据 `/v1/tiers` |
| Doclib 主动质量解析 | PDF/图片按 standard → basic，缺少可用默认档位则报错；显式档位和规则另行校验 |
| Doclib 已缓存结果读取 | 未指定 tier 时选可读的 advanced → standard → basic；不自动新建解析 |
| Doclib parsing rule 未指定 tier | standard → basic → flash；与主动质量解析不同 |

不要把 HTTP SDK 的 `None` 语义套用到直接 Python parser。Tasks 的 `null` 表示省略，file-parse multipart 应省略字段。
显式 PDF/图片档位不受服务支持时返回错误。原生格式、纯文本读取和缓存读取的具体入口边界见
[SDK 档位](sdk/tiers-errors.md)、[Tasks](https://opendatalab.github.io/MinerU/zh/usage/tasks/) 与 [CLI](cli/README.md)。

## 服务启动上限

`api-server --tier` 与 WebUI 的 `--tier` 控制最高开放档位并包含较低档位。
standard 不包含 advanced；Python `create_app(tiers=...)` 则保留精确集合。
启动集合与每次请求默认档位分开，`auto` 不是解析请求的第五个档位。

自动推荐采用有效小模型后端实际选择的 GPU／CPU：GPU 开放全部四档；CPU 可用逻辑核至少 12
开放到 Standard，4–11 核到 Basic，不足 4 核到 Flash。推荐不以显存容量作门槛。
Torch 路径不调用 ONNX 校准；ONNX 使用已有设备选择、缓存与适用的性能校准。
显式上限跳过自动推荐，依赖预检和模型准备仍各自生效。

`hardware detect` 共享推荐逻辑，`hardware info` 只读诊断。细节见
[硬件命令](cli/mineru-kit-hardware.md)、[ONNX 设备策略](onnx-device.md)。

## 数据、模型与隐私

结果记录实际 `extensions.mineru.tier/parse_mode`，生产者版本在 metadata；不能记录伪质量档位。
Standard/Advanced 共用部署模型，Advanced 没有独立下载档位。
小模型后端与 VLM 引擎独立配置，远程 VLM 也不自动扩大服务开放档位。
Doclib 的 remote 授权、直接 API SDK 的目标地址和远程 VLM 配置分别生效，不相互替代。
