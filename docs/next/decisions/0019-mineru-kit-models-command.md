# ADR-0019: MinerU Kit Models Command

状态: Accepted
日期: 2026-06-17
相关文档:
- ../cli/mineru-kit.md
- ../cli/mineru-kit-models.md
- ../../../NEXT-CLI.md

## 背景

旧 CLI 提供 `mineru-models-download`，用于下载 Pipeline 或 VLM 模型，并在下载完成后更新 `mineru.json` 中的 `models-dir` 配置。

当前实现存在几个问题：

- 入口是单个顶级命令，不利于后续扩展查看、校验等配套能力。
- 参数是交互式可选输入，不适合作为稳定脚本接口。
- 下载、配置和状态查看没有形成一组清晰的命令面。
- 项目正在形成 `mineru-kit` 作为专家工具集合，需要把旧模型下载入口纳入统一命令体系。

同时，本阶段不打算把模型配置迁移到新的全局配置体系，仍继续使用旧 `mineru.json` 文件。

## 决策

`mineru-kit` 引入 `models` 命令组，第一阶段只提供三个子命令：

```bash
mineru-kit models download
mineru-kit models show
mineru-kit models verify
```

第一阶段不引入 `set-dir`、`remove`、`clean`、`gc` 等更重的模型管理命令。

### 1. 配置落点

第一阶段继续使用旧配置文件：

- 默认：`~/mineru.json`
- 可通过 `MINERU_TOOLS_CONFIG_JSON` 指定其它路径

仍沿用现有字段：

- `models-dir.pipeline`
- `models-dir.vlm`

第一阶段不迁移到新的 `mineru/config.py` / `mineru.yaml` 体系。

### 2. `mineru-kit models download`

该命令替代旧 `mineru-models-download`。

建议形态：

```bash
mineru-kit models download <pipeline|vlm|all> [flags]
```

参数：

- `bundle` 使用位置参数表达，取值为 `pipeline`、`vlm`、`all`
- `--source/-s`: `huggingface | modelscope`
- `--verbose/-v`

规则：

- 不再支持交互式 prompt，必须显式给出 `bundle`
- `--source` 默认值为 `huggingface`
- `bundle` 没有默认值，必须显式给出
- 下载完成后默认更新 `mineru.json`
- 不支持 `--no-config`
- 不支持 `--config`
- 不支持显式设置模型下载目录
- 保持当前下载行为：由 Hugging Face / ModelScope 下载器自行决定缓存落点，MinerU 只记录最终模型目录
- 当 `MINERU_MODEL_SOURCE=local` 时，下载命令仍临时切换到真实远端源执行下载

### 3. `mineru-kit models show`

该命令用于查看当前模型配置与基本状态。

建议输出：

- 当前实际使用的配置文件路径
- `models-dir.pipeline`
- `models-dir.vlm`
- 当前 `MINERU_MODEL_SOURCE` 环境值
- 配置路径是否存在

第一阶段不支持 `--json`。

### 4. `mineru-kit models verify`

该命令用于轻量校验模型配置与关键路径。

建议形态：

```bash
mineru-kit models verify [pipeline|vlm|all]
```

规则：

- 默认校验全部已配置 bundle
- 也可显式指定 `pipeline`、`vlm` 或 `all`
- 做轻量校验，但不只是目录存在性检查
- 需要检查关键子路径是否存在

第一阶段不支持 `--json`。

### 5. 当前 bundle 定义

第一阶段保持当前项目已有 bundle 划分，不重新拆分对象粒度：

- `pipeline`
- `vlm`
- `all = pipeline + vlm`

其中：

- `vlm` 对应单个 VLM 仓
- `pipeline` 对应一组版面、OCR、公式、表格相关模型路径

命令层不把这些内部模型拆成更多顶级下载对象。

## 替代方案

### 方案 A：保留单个 `download-models` 顶级命令

优点：

- 迁移成本最低

缺点：

- 不利于形成完整模型管理命令面
- 后续 `show` / `verify` 只能继续散落为新的顶级命令

未采用。

### 方案 B：保留更完整的命令组，包括 `set-dir`

候选命令面：

```bash
mineru-kit models download
mineru-kit models show
mineru-kit models set-dir
mineru-kit models verify
```

优点：

- 对离线部署和手工模型目录登记更完整

缺点：

- 第一阶段命令面偏重
- `set-dir` 更像旧配置文件管理能力，不是最核心的下载闭环

第一阶段未采用；后续如离线部署需求明确，可再引入。

### 方案 C：为 `download` 提供默认 bundle

候选默认值包括 `pipeline` 或 `all`。

未采用的原因：

- 模型体积较大，隐式默认下载代价高
- `pipeline` 与 `all` 都会在部分用户场景下造成明显浪费
- 对部署命令来说，显式选择比隐藏成本更重要

因此 `bundle` 作为必选位置参数。

## 影响

### 对 CLI

- `mineru-kit` 将覆盖旧 `mineru-models-download` 的核心能力
- `models` 命令组成为 `mineru-kit` 第一阶段正式子命令之一

### 对配置

- 第一阶段继续依赖旧 `mineru.json`
- 不引入新的模型配置持久化格式

### 对实现

- 旧 `models_download.py` 可作为 `download` 的主要实现参考
- 需要补充 `show` 与 `verify` 两个命令
- `verify` 需要整理 `pipeline` 与 `vlm` 的关键路径清单

### 对用户体验

- 下载行为从交互式改为显式参数，脚本友好
- 不再因为默认 bundle 导致隐式大体积下载
- 仍保留当前“下载后自动登记配置”的易用性

## 后续动作

1. 为 `mineru-kit models` 增加正式 CLI 文档。
2. 在 `mineru-kit` 总览与 `NEXT-CLI.md` 中补充该命令组。
3. 后续如需要支持离线部署目录登记，再单独讨论是否引入 `models set-dir`。
