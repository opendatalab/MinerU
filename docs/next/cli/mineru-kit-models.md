# mineru-kit models

状态: Draft
读者: 服务部署者、批处理开发者、解析内核开发者
范围: `mineru-kit models` 的模型下载、查看和校验
非目标: parse-server 生命周期；模型清理与回收
来源: ADR-0025

## 1. 定位

`mineru-kit models` 是 `mineru-kit` 的模型管理命令组。

第一阶段目标只有三个:

- 按模型 tier 或模型仓库下载模型
- 查看当前模型配置和 readiness
- 校验模型仓库关键路径

第一阶段不引入更重的模型管理能力，例如删除、清理、GC 或手工目录登记。

## 2. 配置文件

模型配置使用 `config.yaml`:

- 默认路径: `${MINERU_HOME:-~/.mineru}/config.yaml`
- 可由环境变量 `MINERU_CONFIG` 指定其它路径

模型相关配置:

```yaml
model:
  base_dir: ${MINERU_HOME:-~/.mineru}/models
  source: auto
  small_backend: auto
  vlm:
    engine: auto
```

`model.small_backend` 支持 `auto`、`onnx`、`torch`；`model.vlm.engine` 支持
`auto`、`llama-cpp`、`vllm`、`lmdeploy`、`mlx`。两者分别使用
`MINERU_MODEL_SMALL_BACKEND` 和 `MINERU_MODEL_VLM_ENGINE` 覆盖。
ARM Mac 自动选择 Torch MPS + llama；Linux/Windows 自动选择可用加速器与已安装引擎，否则回退 ONNX + llama。

`model.base_dir` 下使用以下稳定目录，不按安装 extra 划分：

- `MinerU-4_models_torch`
- `MinerU-4_models_onnx`
- `MinerU2.5-Pro-2605-1.2B`
- `MinerU2.5-Pro-2605-1.2B-GGUF`

basic 需要所选小模型资源；standard 另需所选 VLM 权重，llama 使用 GGUF，其余引擎使用原始权重。
远程 VLM 不要求本地 VLM 权重。显式 `--vlm-engine` 可在远程配置环境中准备本地模型。

`model.source` 支持:

- `auto`
- `huggingface`
- `modelscope`
- `local`

`MINERU_MODEL_SOURCE` 会覆盖 `model.source`。当环境变量覆盖为 `auto` 时，不会把自动探测结果写回配置文件。

## 3. 子命令

### 3.1 `mineru-kit models download`

下载指定模型仓库，或下载某个模型 tier 需要的模型仓库。

```bash
mineru-kit models download <repo> [flags]
mineru-kit models download --tier <basic|standard> [flags]
```

参数:

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--tier` | - | `basic \| standard` | - | 按模型 tier 下载所需模型 |
| `--small-backend` | - | `auto \| onnx \| torch` | 配置值 | 独立选择小模型后端，传入 repo 时忽略 |
| `--vlm-engine` | - | `auto \| llama-cpp \| vllm \| lmdeploy \| mlx` | 配置值 | 独立选择本地 VLM 引擎，传入 repo 时忽略 |
| `--source` | `-s` | `auto \| huggingface \| modelscope` | 配置值 | 本次下载源 |
| `--verbose` | `-v` | bool | false | 输出详细路径 |

规则:

- repo 位置参数与 `--tier` 互斥
- 不带 repo 且不带 `--tier` 会报错
- `--tier` 只接受 `basic` 和 `standard`；Flash 原生文本解析不需要推理模型，但 Flash OCR 仍使用小模型资源；Advanced 复用 Standard
- `--source` 不支持 `local`
- 如果当前配置为 `model.source: local`，显式 download 会临时按 `auto` 解析远端源，不改写配置
- 下载目标固定为 `config.model.base_dir` 下的 repo local dir
- ModelScope 使用 `modelscope>=1.39.1`；先把默认分支解析为固定 commit，再按同一 commit 查询清单和下载，避免分支更新导致清单与文件版本不一致

支持的 repo 名:

- `MinerU-4_models_torch`
- `MinerU-4_models_onnx`
- `MinerU2.5-Pro-2605-1.2B`
- `MinerU2.5-Pro-2605-1.2B-GGUF`

模型 tier 到 repo 的映射按两个后端组合：basic 使用小模型资源包；standard 追加实际引擎的 VLM 权重。
例如 Torch + llama 需要 `MinerU-4_models_torch` 和 `MinerU2.5-Pro-2605-1.2B-GGUF`。

模型命令没有 flash 下载档位；离线运行 Flash OCR 可准备所选后端的 Basic 小模型资源。Advanced 使用 Standard 模型集。

示例:

```bash
mineru-kit models download --tier basic
mineru-kit models download --tier standard --source huggingface
mineru-kit models download MinerU-4_models_torch
mineru-kit models download MinerU2.5-Pro-2605-1.2B --source huggingface
```

### 3.2 `mineru-kit models show`

显示当前模型配置与基本状态。

```bash
mineru-kit models show
mineru-kit models show --small-backend <auto|onnx|torch> --vlm-engine <auto|llama-cpp|vllm|lmdeploy|mlx>
```

两个后端选项仅覆盖当前命令，不修改配置；自动模式按平台、依赖和设备分别选择。

输出内容:

- 当前实际使用的 `config.yaml` 路径及是否存在
- `MINERU_MODEL_SOURCE`
- `model.base_dir`、`model.source`、`model.small_backend`、`model.vlm.engine` 及各自来源
- 实际生效的小模型后端与 VLM 引擎
- 每个 repo 的 local dir 和 readiness
- Basic 和 Standard 模型 tier 在当前后端组合下需要的 repo 集合

第一阶段不支持 `--json`。

### 3.3 `mineru-kit models verify`

校验模型仓库关键路径。

```bash
mineru-kit models verify
mineru-kit models verify <repo>
mineru-kit models verify --tier <basic|standard>
mineru-kit models verify --small-backend <auto|onnx|torch> --vlm-engine <auto|llama-cpp|vllm|lmdeploy|mlx>
```

参数:

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--tier` | - | `basic \| standard` | - | 按模型 tier 校验所需模型 |
| `--small-backend` | - | `auto \| onnx \| torch` | 配置值 | 独立选择小模型后端，传入 repo 时忽略 |
| `--vlm-engine` | - | `auto \| llama-cpp \| vllm \| lmdeploy \| mlx` | 配置值 | 独立选择本地 VLM 引擎，传入 repo 时忽略 |

规则:

- 默认校验当前后端组合下的全部 repo
- repo 位置参数与 `--tier` 互斥
- `--tier` 只接受 `basic` 和 `standard`；按 `--tier` 校验时使用独立后端选项解析出的资源组合
- 不是单纯目录存在性检查，还会检查 registry 中声明的关键路径
- 整仓和目录资源会读取 `.mineru_complete` 文件清单，检查预期文件是否存在、是否为有效文件，以及大小是否一致；正常校验无需联网，不读取大权重计算 hash
- 单文件下载把预期大小保存到仓库根目录的 `.mineru_files.json`，后续复用同样检查缺失、截断和失效链接；这份局部记录不会把父目录或整仓标为已完成
- 完整目录重新下载后，由目录清单接管该范围的旧文件记录；文件记录损坏时需执行 `mineru-kit models download <repo>` 重建，不能只确认一个文件就丢弃其他文件的校验约束
- 兼容旧版本的零字节完成标记：检查本地非空权重、registry 声明的必需路径以及权重索引引用的分片，通过后可直接离线复用
- 首次使用旧模型时后台尝试升级标记，包括 `model.source: local`：按缓存中的旧版本查询远端文件清单，存在性和大小全部匹配后原子发布新版清单；只查询元数据，不下载、更新或删除模型文件，也不改写来源配置
- 无网、源不可达、旧版本无法确定、核验不匹配或目录不可写时保留旧标记，不阻塞当前使用；失败后冷却 5 分钟，下次使用可重试。`models show/verify` 保持只读，不启动迁移
- 后续实际下载或显式执行 `mineru-kit models download <repo>` 时，由 provider 确认请求范围并生成新版清单；旧标记的兼容检查不具备新版清单的逐文件大小校验能力
- 文件缺失、被截断或完成清单无效时，校验失败并列出未通过检查的路径

示例:

```bash
mineru-kit models verify
mineru-kit models verify PDF-Extract-Kit-1.0
mineru-kit models verify --tier standard
```

## 4. 相关文档

- [ADR-0019](../decisions/0019-mineru-kit-models-command.md)
- [ADR-0025](../decisions/0025-model-download-local-dir.md)
- [mineru-kit](mineru-kit.md)
- [mineru-kit parse](mineru-kit-parse.md)
