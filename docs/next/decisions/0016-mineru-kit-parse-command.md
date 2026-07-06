# ADR-0016: MinerU Kit Parse Command

状态: Accepted
日期: 2026-06-17
相关文档:
- ../cli/mineru-kit.md
- ../cli/mineru-kit-parse.md
- ../tiers.md

## 背景

`mineru-kit parse` 是 `mineru-kit` 的主体命令。它需要与 `mineru parse`、`mineru-kit api-server` 和 parser SDK 保持边界清楚：

- `mineru parse` 面向用户和 Agent，强调 doclib、缓存和渐进式阅读。
- `mineru-kit parse` 面向批处理和解析内核开发者，强调无状态、显式参数和文件产物。
- `mineru-kit api-server` 已经确定了 tier、backend、`language`、`ocr-mode`、`effort` 和 `disable-*` 参数体系，`mineru-kit parse` 应尽量与之对齐。

在此基础上，需要明确 `mineru-kit parse` 的输入范围、local/remote 规则、tier/backend 语义和输出命名规则。

## 决策

### 1. 命令定位

`mineru-kit parse` 是无状态批处理解析命令：

- 不使用 `doclib.db`
- 不复用 doclib 缓存
- 不承担 Agent 默认阅读体验
- 主要输出文件产物

命令形态：

```bash
mineru-kit parse <input...> -o <output> [flags]
```

`--output/-o` 必填。

### 2. 输入范围

当前支持：

- 单文件
- 多文件
- 目录

当前不支持：

- `--recursive`
- URL 作为输入文档源
- stdin 文件内容
- stdin 路径列表
- 文件列表输入

目录输入只展开一层，不递归。

### 3. 基础参数

`mineru-kit parse` 复用 `mineru parse` 的以下基础参数名和短参数：

- `--pages`, `-p`
- `--format`, `-f`
- `--output`, `-o`
- `--verbose`, `-v`

其中：

- `--pages` 支持如 `1~5` 或 `all`
- `--format` 当前只支持 `markdown`、`middle_json`、`zip`
- 默认 `--format` 为 `markdown`

### 4. local 模式

不传 `--remote`、`--remote-url` 时进入 local 模式。

支持：

- `--tier`
- `--backend`

规则：

1. 可以只传 `--tier`
2. 可以只传 `--backend`
3. 可以同时传 `--tier` 和 `--backend`
4. 同时传且二者不兼容时，直接报错
5. 默认 tier 选择策略与 `mineru-kit api-server` 一致
6. 默认不会落到 `flash`
7. `flash` 只能显式指定：
   - `--tier flash`
   - `--backend flash`

### 5. remote 模式

通过以下参数进入 remote 模式：

- `--remote`
- `--remote-url <url>`
- `--api-key <key>`

规则：

1. `--remote` 连接 `mineru.net` 官方解析服务
2. `--remote-url` 连接指定解析服务
3. `--remote` 与 `--remote-url` 互斥
4. remote 模式下允许传 `--tier`
5. remote 模式下禁止传 `--backend`
6. remote 模式未传 `--tier` 时，使用目标服务提供的最高 tier
7. remote 模式传了 `--tier` 时：
   - 服务提供该 tier，则按该 tier 解析
   - 服务不提供该 tier，则报错

### 6. 与 api-server 对齐的解析参数

`mineru-kit parse` 在 local 模式下与 `mineru-kit api-server` 对齐，使用：

- `--language`
- `--ocr-mode`
- `--effort`
- `--disable-image-analysis`

不再以新设计暴露旧式命名，如 `--lang`、`--method`、`--image-analysis`。

### 7. 输出规则

`--output` 必填，且自动创建父目录。

单文件时，`--output` 可以是文件路径，也可以是目录路径。

多文件或目录输入时，`--output` 只能是目录路径；如果用户传入单文件路径，应在参数校验阶段直接报错。

当 `--output` 是目录路径时，落盘命名规则为：

- `markdown` -> `<原文件名去扩展>.md`
- `middle_json` -> `<原文件名去扩展>.json`
- `zip` -> `<原文件名去扩展>.zip`

多文件时，命名方式与单文件目录模式一致。

### 8. 同名冲突

如果一个批次内多个输入在输出目录上发生同名冲突：

- 直接报错
- 整个批次都不解析

当前不支持：

- rename
- path mirror
- 其他自动消歧策略

因此当前不需要 `--on-collision`。

## 替代方案

### 1. 支持 stdin、路径列表和 URL 输入

没有采用。当前目标是先把 `mineru-kit parse` 收口为清楚的文件/目录批处理命令，减少输入模型分叉。

### 2. 默认允许落到 `flash`

没有采用。默认 tier 与 `api-server` 保持一致，默认不能静默降到 `flash`。

### 3. 保留多种自动冲突处理策略

没有采用。当前阶段直接报错更简单，也更容易让批处理行为可预期。

## 影响

- `mineru-kit parse` 与 `mineru parse` 的职责边界更清楚。
- `mineru-kit parse` 与 `mineru-kit api-server` 的参数命名更统一。
- 输出路径和文件命名规则固定后，批处理脚本更容易稳定依赖。
- 旧 CLI 中的目录递归、stdin、路径列表和自动冲突处理能力，如未来需要恢复，应作为后续增量设计，而不是默认能力。

## 后续动作

- 更新 `docs/next/cli/mineru-kit.md`
- 更新 `docs/next/cli/mineru-kit-parse.md`
- 更新 `NEXT-CLI.md`
- 更新 `docs/next/open-questions.md`
- 后续实现 `mineru-kit` 顶层脚本与 `mineru-kit parse` 参数解析
