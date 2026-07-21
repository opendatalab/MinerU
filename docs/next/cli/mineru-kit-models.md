# mineru-kit models

状态: Draft
读者: 服务部署者、批处理开发者、解析内核开发者
范围: `mineru-kit models` 的模型下载、查看和校验
非目标: parse-server 生命周期；模型清理与回收
来源: ADR-0025

## 1. 定位

`mineru-kit models` 是 `mineru-kit` 的模型管理命令组。

第一阶段目标只有三个:

- 按 tier 或模型仓库下载模型
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
```

`model.base_dir` 是所有 MinerU 模型仓库的根目录。当前模型仓库会落在:

- `{model.base_dir}/PDF-Extract-Kit-1.0`
- `{model.base_dir}/MinerU2.5-Pro-2605-1.2B`

`model.source` 支持:

- `auto`
- `huggingface`
- `modelscope`
- `local`

`MINERU_MODEL_SOURCE` 会覆盖 `model.source`。当环境变量覆盖为 `auto` 时，不会把自动探测结果写回配置文件。

## 3. 子命令

### 3.1 `mineru-kit models download`

下载指定模型仓库，或下载某个 tier 需要的模型仓库。

```bash
mineru-kit models download <repo> [flags]
mineru-kit models download --tier <flash|basic|standard|advanced> [flags]
```

参数:

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--tier` | - | `flash \| basic \| standard \| advanced` | - | 按 tier 下载所需模型 |
| `--source` | `-s` | `auto \| huggingface \| modelscope` | 配置值 | 本次下载源 |
| `--verbose` | `-v` | bool | false | 输出详细路径 |

规则:

- repo 位置参数与 `--tier` 互斥
- 不带 repo 且不带 `--tier` 会报错
- `--tier flash` 不需要模型，返回成功
- `--source` 不支持 `local`
- 如果当前配置为 `model.source: local`，显式 download 会临时按 `auto` 解析远端源，不改写配置
- 下载目标固定为 `config.model.base_dir` 下的 repo local dir

支持的 repo 名:

- `PDF-Extract-Kit-1.0`
- `MinerU2.5-Pro-2605-1.2B`

Tier 到 repo 的映射:

- `flash`: 不需要模型
- `basic`: `PDF-Extract-Kit-1.0`
- `standard`: `PDF-Extract-Kit-1.0` + `MinerU2.5-Pro-2605-1.2B`
- `advanced`: `PDF-Extract-Kit-1.0` + `MinerU2.5-Pro-2605-1.2B`

示例:

```bash
mineru-kit models download --tier basic
mineru-kit models download --tier standard --source modelscope
mineru-kit models download PDF-Extract-Kit-1.0
mineru-kit models download MinerU2.5-Pro-2605-1.2B --source huggingface
```

### 3.2 `mineru-kit models show`

显示当前模型配置与基本状态。

```bash
mineru-kit models show
```

输出内容:

- 当前实际使用的 `config.yaml` 路径
- `MINERU_MODEL_SOURCE`
- `model.base_dir` 和来源
- `model.source` 和来源
- 每个 repo 的 local dir 和 readiness
- 每个 tier 需要的 repo 集合

第一阶段不支持 `--json`。

### 3.3 `mineru-kit models verify`

校验模型仓库关键路径。

```bash
mineru-kit models verify
mineru-kit models verify <repo>
mineru-kit models verify --tier <flash|basic|standard|advanced>
```

规则:

- 默认校验全部 repo
- repo 位置参数与 `--tier` 互斥
- `--tier flash` 直接成功
- 不是单纯目录存在性检查，还会检查 registry 中声明的关键路径
- 第一阶段不做 hash 级完整性校验

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
