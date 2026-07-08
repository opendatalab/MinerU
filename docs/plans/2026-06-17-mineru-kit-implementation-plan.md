# mineru/kit 命令行实施计划

**日期**: 2026-06-17  
**状态**: 待实施  
**范围**: 在 `mineru/kit` 目录下落地新 `mineru-kit` 命令行骨架与第一阶段四类命令：`models`、`parse`、`api-server`、`vlm-server`。  
**非目标**:
- 不删除 `mineru/cli_old` 中的旧实现
- 不在本计划内处理 `router` 与 `gradio` 的新命令面
- 不在本计划内实现新的模型配置文件体系
- 不在本计划内统一重写所有旧执行逻辑，仅在新命令面下按需复用旧实现

## 1. 背景

当前仓库已经完成 CLI 结构重排：

- `mineru/cli`：新的 `mineru` CLI
- `mineru/cli_old`：旧 CLI 兼容层

同时，`mineru-kit` 的命令设计已通过一系列 ADR 收敛：

- `models`
- `parse`
- `api-server`
- `vlm-server`

但这些命令尚未在代码中形成正式入口。下一步需要在 `mineru/kit` 中实现一个完整、可安装、可测试的 `mineru-kit` CLI。

## 2. 目标

本计划的目标是：

1. 新建 `mineru/kit` 包，作为 `mineru-kit` 的正式代码入口。
2. 在 `pyproject.toml` 中增加 `mineru-kit` 脚本入口。
3. 以 Typer 为基础，落地 `mineru-kit` 的第一阶段命令树。
4. 优先保证命令面、参数校验与文档契约一致。
5. 在第一阶段允许新命令复用 `mineru/cli_old` 中的旧实现逻辑，但不复用旧命令面。

## 3. 目标目录结构

第一阶段采用如下结构：

```text
mineru/
  kit/
    __init__.py
    main.py
    common.py
    errors.py
    output.py

    commands/
      __init__.py
      models.py
      parse.py
      api_server.py
      vlm_server.py
```

说明：

- `main.py`：`mineru-kit` 顶级 Typer app
- `commands/`：四个正式命令面
- `common.py`：共享路径和参数校验工具
- `errors.py`：命令层错误封装
- `output.py`：统一文本输出和 JSON 输出

第一阶段不再继续拆更多子模块，避免过早抽象。

## 4. 顶级入口

需要在 `pyproject.toml` 中新增：

```toml
mineru-kit = "mineru.kit.main:app"
```

现有旧入口全部保留：

- `mineru-models-download`
- `mineru-api`
- `mineru-router`
- `mineru-gradio`
- `mineru-vllm-server`
- `mineru-lmdeploy-server`
- `mineru-openai-server`

原因：

- 第一阶段以新增正式入口为主
- 不在本计划内移除兼容入口

## 5. 命令树

第一阶段仅注册四个顶级命令：

1. `models`
2. `parse`
3. `api-server`
4. `vlm-server`

其中：

- `models` 应作为 Typer 子应用，内部包含 `download`、`show`、`verify`
- `parse`、`api-server`、`vlm-server` 作为顶级单命令

## 6. 分阶段实施

### 6.1 阶段一：骨架搭建

任务：

1. 创建 `mineru/kit` 包和 `commands/` 子目录。
2. 创建 `mineru/kit/main.py`，组装顶级 Typer app。
3. 注册四个正式命令：
   - `models`
   - `parse`
   - `api-server`
   - `vlm-server`
4. 在 `pyproject.toml` 中增加 `mineru-kit` 脚本入口。

完成标准：

- `python -m mineru.kit.main --help` 可用
- `mineru-kit --help` 可展示四个顶级命令

### 6.2 阶段二：实现 `models`

命令范围：

- `mineru-kit models download`
- `mineru-kit models show`
- `mineru-kit models verify`

实施原则：

- 第一阶段继续使用旧 `mineru.json`
- 不引入新的模型配置文件体系
- `download` 去掉交互式 prompt，改为显式位置参数
- 保留旧下载逻辑中的可复用函数，但不复用旧 click 命令对象

建议做法：

1. 从 `mineru/cli_old/models_download.py` 中提取或复用：
   - `download_pipeline_models`
   - `download_vlm_models`
   - `temporary_model_source`
   - `get_effective_download_model_source`
2. 在 `models.py` 中重新定义 Typer 参数面：
   - `download <pipeline|vlm|all>`
   - `--source/-s`
   - `--verbose/-v`
3. 新写 `show`：
   - 展示当前配置文件路径
   - 展示 `models-dir.pipeline`
   - 展示 `models-dir.vlm`
   - 展示路径存在性
4. 新写 `verify`：
   - 检查配置项
   - 检查目录存在
   - 检查关键子路径存在

完成标准：

- `models download/show/verify` 参数与文档一致
- `download` 默认更新 `mineru.json`
- `show` 与 `verify` 无交互式输入

### 6.3 阶段三：实现 `api-server`

目标：

- 以新命令面包装正式 `api-server` 能力
- 参数面与 ADR-0017 对齐

实施原则：

- 命令层做参数收口和校验
- 底层尽量调用现有 `mineru/parser/api_server.py`
- 不再依赖 `mineru/cli_old/fast_api.py` 作为正式命令实现入口，除非短期仅作为桥接

重点事项：

- `--tier` 默认 `high`
- `--tier` / `--backend` 冲突时提前报错
- 不暴露 `--reload`
- 保留：
  - `--upload-dir`
  - `--url-timeout`
  - `--max-wait`

完成标准：

- `mineru-kit api-server --help` 与文档一致
- 能调用正式 `mineru.parser.api_server` 启动服务

### 6.4 阶段四：实现 `vlm-server`

目标：

- 以新命令面包装本地 VLM 服务
- 参数契约与 ADR-0018 对齐

实施原则：

- 命令层只稳定 `--engine`
- 其它参数原样透传到底层引擎服务
- 第一阶段以统一正式入口为主，不重写底层 serving 逻辑

重点事项：

- 支持：
  - `auto`
  - `vllm`
  - `lmdeploy`
  - `sglang`
  - `mlx`
- `/v1/responses` 不作为稳定命令层承诺

完成标准：

- `mineru-kit vlm-server --help` 与文档一致
- 能根据 `--engine` 跳转或透传到底层实现

### 6.5 阶段五：实现 `parse`

目标：

- 落地 `mineru-kit parse` 的新命令面
- 命令契约与 ADR-0016 对齐

实施原则：

- 命令层重写
- 可复用 `mineru/cli_old/client.py` 的部分执行逻辑
- 不复用旧 click 参数层
- 输出命名规则、路径校验、冲突策略在新 `kit` 层实现

重点事项：

- 输入仅支持：
  - 单文件
  - 多文件
  - 目录
- 不支持：
  - stdin
  - URL 文档输入
  - recursive
- `--output` 必填
- 多文件或目录输入时，若 `--output` 为单文件路径，在参数校验阶段直接报错
- local / remote 模式行为与文档一致

完成标准：

- 参数层完全符合文档
- 输出命名规则符合 ADR
- 主链路可跑通

## 7. 推荐实施顺序

建议按以下顺序推进：

1. 骨架与脚本入口
2. `models`
3. `api-server`
4. `vlm-server`
5. `parse`

理由：

- `models` 依赖最少，最适合作为第一批落地点
- `api-server` 与 `vlm-server` 主要是命令层封装
- `parse` 依赖最多，放最后风险最低

## 8. 测试策略

第一阶段测试重点放在命令面和参数校验，不立即追求完整端到端覆盖。

建议新增：

```text
tests/unittest/test_kit_models.py
tests/unittest/test_kit_parse.py
tests/unittest/test_kit_api_server.py
tests/unittest/test_kit_vlm_server.py
```

优先覆盖：

- `models` 三个命令的参数与基本行为
- `parse` 的路径规则、local/remote 互斥与输出校验
- `api-server` 的 tier/backend 规则
- `vlm-server` 的 engine 选择与透传

## 9. 风险与注意事项

1. 第一阶段复用旧实现时，必须避免把旧 click 参数层直接搬入新命令。
2. `mineru/cli_old` 仍保留，短期内要清楚区分：
   - 新命令面
   - 旧执行逻辑
3. `models` 继续使用旧配置文件，会让新旧体系短期并存；这是本阶段的接受现实，不应在本计划中顺手扩展为配置迁移。
4. `parse` 的实现要重点防止：
   - 输出命名规则与文档不一致
   - 多文件冲突策略退化回旧隐式行为

## 10. 完成标志

以下条件同时满足时，本计划可视为完成：

1. `mineru-kit` 入口已存在并可安装运行。
2. `models`、`parse`、`api-server`、`vlm-server` 全部可通过 `--help` 展示正式参数面。
3. 第一阶段四类命令均有对应单测。
4. CLI 文档、ADR 与实际参数面一致。
