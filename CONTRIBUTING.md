# 贡献指南

本指南面向两类读者：人类贡献者与编码 agent。编码规范（import 约定、类型注解、编程原则、Middle JSON Schema 2.0 架构等）以仓库根目录 [CLAUDE.md](CLAUDE.md) 为准，本文不重复，只补充工作流与验证命令。两者冲突时以 CLAUDE.md 为准。

## 开发环境

使用 **uv** 管理虚拟环境与依赖，步骤见 [CLAUDE.md](CLAUDE.md) 的「开发环境」一节。补充两点：

- 跑测试需要 test extra：`uv pip install -e ".[test]"`。
- 需要本地模型推理（`basic`/`standard`/`advanced` tier 的人工验证）时再安装 `.[torch]`；纯静态改动（文档、类型、纯逻辑重构）不强制。

运行 Python 一律用 `.venv/bin/python`，在仓库根目录执行。

## 验证循环

每次代码改动后，按以下顺序执行验证。所有命令可直接复制。

### 1. 格式化与 lint（ruff）

```bash
# 格式化自己改动的文件（行宽 128，配置在 pyproject.toml）
.venv/bin/ruff format <改动文件或目录>

# lint 自己改动的文件
.venv/bin/ruff check <改动文件或目录>
```

当前门槛是**自己的改动零新增违规**：next HEAD 上全仓 `ruff check` 尚有大量历史违规（主要是 `ANN` 系列），全仓跑 check 不作为合并门槛。圈定文件用：

```bash
.venv/bin/ruff format $(git diff --name-only --diff-filter=ACMR | grep '\.py$')
.venv/bin/ruff check $(git diff --name-only --diff-filter=ACMR | grep '\.py$')
```

### 2. 单元测试

```bash
# 跑单个测试文件（快速循环）
.venv/bin/python -m pytest tests/unittest/test_errors.py -q -o addopts=""

# 跑与改动模块相关的多个文件
.venv/bin/python -m pytest tests/unittest/test_pdf_document.py tests/unittest/test_pdf_render.py -q -o addopts=""

# 全量（见下方"已知基线"）
.venv/bin/python -m pytest tests/unittest -q -o addopts=""
```

说明：

- pyproject.toml 的 pytest `addopts` 默认带 `-s --cov=mineru --cov-report html`；快速迭代时用 `-o addopts=""` 关掉 coverage（coverage 报告反而显著拖慢单文件循环），最终提交前再跑一次带 coverage 的确认。
- 测试 fixtures 在 `tests/fixtures/` 与 `tests/unittest/pdfs/`，优先复用，不要为单个用例新造大文件。

### 3. 全量测试的已知基线

截至 next HEAD（317428b8），未安装 `.[torch]` 时的全量结果为约 5 分 40 秒、3649 passed / 37 failed / 11 skipped。已知的既有问题（均与你的改动无关，提交前对比失败集合是否新增即可）：

- 3 个文件无法收集：`test_mfr_latex_utils.py`、`test_pp_doclayoutv2_postprocess.py`（依赖 torch）、`test_render_html_table.py`（依赖未声明的 `markdown` 包）。排除命令：

  ```bash
  .venv/bin/python -m pytest tests/unittest -q -o addopts="" \
    --ignore=tests/unittest/test_mfr_latex_utils.py \
    --ignore=tests/unittest/test_pp_doclayoutv2_postprocess.py \
    --ignore=tests/unittest/test_render_html_table.py
  ```

- `tests/unittest/test_kit_commands.py` 整体失败：`mineru/cli_old/api_request.py:8` 仍在 import 已被删除的 `mineru.utils.backend_options`。
- 其余失败集中在 tier preflight、model runtime 依赖检查等用例（`test_parser_api_contract.py`、`test_doclib_app_startup.py`、`test_client_side_output.py`、`test_llm_aided_postprocess.py` 等），根因是缺 torch extra。

安装 `.[torch]` 后上述大部分失败会消失；全绿基线以 `[torch]` 环境为准。

### 4. 解析质量改动的人工验证

凡涉及解析结果本身（layout、内容抽取、渲染、Middle JSON 结构）的改动，单测之外必须做一次最小人工验证：

```bash
# 构造或选一个小输入（仓库自带一个 PDF fixture；Office/HTML/CSV 等各留一份小样本）
tests/unittest/pdfs/test.pdf

# 快速冒烟：flash tier 无需下载模型
mineru parse tests/unittest/pdfs/test.pdf --tier flash -o ./out/flash

# 若改动涉及 layout/表格/公式等模型路径，下载模型后逐 tier 验证
mineru-kit models        # 查看与下载模型（见 ADR-0025）
mineru parse tests/unittest/pdfs/test.pdf --tier standard -o ./out/standard
mineru parse tests/unittest/pdfs/test.pdf --tier advanced -o ./out/advanced
```

人工检查点：

- `./out/<tier>/` 下的 Markdown 是否保留了改动预期的结构（标题层级、表格、公式、阅读顺序）。
- 若改动触及 Middle JSON 结构：`.venv/bin/python -m pytest tests/unittest/test_middle_json_validator.py -q -o addopts=""` 必须通过，并目检一份 JSON 输出的 `pages`/`blocks`/InlineSpan 是否符合 CLAUDE.md 的 schema 2.0 约定。
- 若改动触及 CLI 输出契约：跑 `tests/unittest/test_cli_runtime_contract.py`、`test_cli_command_contract.py`，并实际执行一次 `mineru parse --json` 确认 stdout 结构未破坏（契约见 ADR-0023）。

## 提交与 PR 流程

### 分支

- 从 `next` 切出功能分支，命名 `<type>/<短横线描述>`，如 `docs/contributing-guide`、`fix/epub-spine-nav`。
- 不 force-push、不删除他人分支。

### Commit message

遵循仓库现有风格（Conventional Commits）：`feat:` / `fix:` / `refactor:` / `docs:` / `test:`，一行主题，必要时空行后补充正文。

关联 issue 时禁止使用会自动关闭 issue 的关键词（`fixes #n`、`closes #n`、`resolves #n`），使用 `Refs #n`、`Related to #n` 或 `Issue: #n`——详见 CLAUDE.md「GitHub Issue / PR 处理规范」。

### PR 描述

必须包含三部分：

1. **动机**：解决什么问题，关联哪个 issue。
2. **方案**：关键取舍一句话说清；涉及对外行为（CLI/API/SDK/Middle JSON 兼容性）的，先在 issue 中确认过方案再提交。
3. **验证证据**：实际执行过的命令与结果（如 `ruff check` 通过的文件范围、pytest 子集的 passed/failed 数字、人工验证的 tier 与现象）。没跑过的不要写。

### 评审

- PR 需至少一名维护者/评审 agent 批准后合并。
- 评审意见逐条回应；不夹带与 PR 范围无关的重构。
- 纯文档 PR 同样走 PR 流程，便于评审追溯。

## ADR 工作流

`docs/next/decisions/` 记录关键设计决策，规则详见 [docs/next/decisions/README.md](docs/next/decisions/README.md)。要点：

- **何时新增 ADR**：决策影响 API、CLI、SDK 或 middle_json 的长期兼容性；影响模块边界、数据模型、进程边界或存储模型；存在多个合理替代方案且维护者需要知道取舍理由；或决策回滚成本高。日常 bug 修复、参数微调不写 ADR。
- **编号**：四位递增（0001-0033 已用），文件名 `<编号>-<短横线主题>.md`，如 `0034-doclib-retention-policy.md`。
- **状态**：新建时 `Proposed`，评审通过后改 `Accepted`；被取代时标 `Superseded` 并指向新 ADR；同时更新 README.md 的决策表。
- **与专题文档的关系**：ADR 记录"为什么这么定"和替代方案；`docs/next/` 各专题文档（architecture、cli、api、middle-json 等）记录"定成了什么"。决策落定后，把结论同步进对应专题文档正文，ADR 不替代专题文档。
