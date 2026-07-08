# mineru-kit models

状态: Draft
读者: 服务部署者、批处理开发者、解析内核开发者
范围: `mineru-kit models` 的模型下载、查看和校验
非目标: parse-server 生命周期；新配置体系迁移；模型清理与回收
来源: 由根目录旧 CLI 底稿迁移整理而来

## 1. 定位

`mineru-kit models` 是 `mineru-kit` 的模型管理命令组。

第一阶段目标只有三个：

- 下载模型
- 查看当前模型配置
- 轻量校验模型配置和关键路径

第一阶段不引入更重的模型管理能力，例如删除、清理、GC 或手工目录登记。

## 2. 配置文件

第一阶段继续使用旧配置文件体系：

- 默认：`~/mineru.json`
- 可由环境变量 `MINERU_TOOLS_CONFIG_JSON` 指定其它路径

命令组默认读写其中的：

- `models-dir.pipeline`
- `models-dir.vlm`

当前不切换到新的全局配置体系。

## 3. 子命令

### 3.1 `mineru-kit models download`

下载指定模型 bundle，并在完成后更新配置文件。

```bash
mineru-kit models download <pipeline|vlm|all> [flags]
```

参数：

| Flag | 简写 | 类型 | 默认 | 说明 |
|------|------|------|------|------|
| `--source` | `-s` | `auto \| huggingface \| modelscope` | `auto` | 下载源 |
| `--verbose` | `-v` | bool | false | 输出详细日志 |

规则：

- `bundle` 使用位置参数，必须显式给出
- `--source auto` 会先探测 Hugging Face 是否可访问，并在本次下载中使用解析出的真实远端源
- 不支持交互式 prompt
- 下载完成后默认更新配置文件
- 不支持 `--no-config`
- 不支持 `--config`
- 不支持显式设置模型下载目录
- 保持当前下载行为：由 Hugging Face / ModelScope 下载器自身决定缓存落点，MinerU 只记录最终模型目录

`bundle` 的含义：

- `pipeline`: 下载 Pipeline 解析所需模型集合
- `vlm`: 下载 VLM 解析所需模型仓
- `all`: 同时下载 `pipeline` 与 `vlm`

示例：

```bash
mineru-kit models download pipeline
mineru-kit models download vlm --source modelscope
mineru-kit models download all --source auto
mineru-kit models download all
```

### 3.2 `mineru-kit models show`

显示当前模型配置与基本状态。

```bash
mineru-kit models show
```

建议输出内容：

- 当前实际使用的配置文件路径
- `model-source`
- `models-dir.pipeline`
- `models-dir.vlm`
- 当前 `MINERU_MODEL_SOURCE` 环境值
- 上述目录是否存在

第一阶段不支持 `--json`。

### 3.3 `mineru-kit models verify`

轻量校验模型配置与关键路径。

```bash
mineru-kit models verify [pipeline|vlm|all]
```

规则：

- 默认校验全部已配置 bundle
- 也可显式指定 `pipeline`、`vlm` 或 `all`
- 不是单纯目录存在性检查，还会检查关键子路径是否存在
- 第一阶段不做 hash 级完整性校验

示例：

```bash
mineru-kit models verify
mineru-kit models verify pipeline
mineru-kit models verify vlm
```

## 4. 当前 bundle 定义

当前项目的模型 bundle 划分如下：

- `pipeline`
- `vlm`
- `all = pipeline + vlm`

其中：

- `vlm` 对应单个 VLM 模型仓
- `pipeline` 对应一组版面、OCR、公式和表格相关模型路径

命令层不把这些内部模型再拆成更多顶级下载对象。

## 5. 非目标

第一阶段暂不支持：

- 模型目录显式自定义
- `--json`
- 模型删除 / 清理 / GC
- 通过命令设置多个模型目录
- 切换到新的全局配置文件体系

## 6. 相关文档

- [ADR-0019](../decisions/0019-mineru-kit-models-command.md)
- [mineru-kit](mineru-kit.md)
- [mineru-kit parse](mineru-kit-parse.md)
