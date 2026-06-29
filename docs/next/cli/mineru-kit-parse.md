# mineru-kit parse

状态: Draft
读者: 批处理开发者、解析内核开发者
范围: `mineru-kit parse` 的输入、输出、local/remote 规则和命名约定
非目标: 本地文档库缓存；Agent 默认渐进式阅读体验；stdin 和路径列表输入
底稿: `../../../NEXT-CLI.md`

## 1. 定位

`mineru-kit parse` 是无状态解析命令。它面向批处理和开发者工作流，接受文件和目录输入，输出到指定文件或目录。

与 `mineru parse` 不同，它不维护本地文档库，也不复用 `doclib.db` 或 doclib 解析缓存。

## 2. Usage

```bash
mineru-kit parse <input...> -o <output> [flags]
```

`-o` 应作为批处理输出的核心参数。默认输出到文件系统，而不是面向 Agent 的 STDOUT 阅读。

## 3. 输入

当前支持的输入形态：

| 输入 | 说明 |
|------|------|
| 单文件 | 解析一个文件 |
| 多文件 | 一次提交多个文件 |
| 目录 | 展开目录中的可解析文件 |

当前不支持：

- `--recursive`
- URL 作为输入文档源
- stdin 文件内容
- stdin 路径列表
- 文件列表输入

目录输入只展开一层，不递归。

## 4. 输出

当前输出格式：

- markdown
- middle_json
- zip

默认格式是 `markdown`。

`--output` 必填，并自动创建父目录。

### 单文件输出

单文件时，`--output` 可以是文件路径，也可以是目录路径。

当 `--output` 是目录路径时，命名规则为：

- `markdown` -> `<原文件名去扩展>.md`
- `middle_json` -> `<原文件名去扩展>.json`
- `zip` -> `<原文件名去扩展>.zip`

### 多文件输出

多文件或目录输入时，`--output` 只能是目录路径。

如果 `--output` 是单文件路径，应在参数校验阶段直接报错。

多文件命名规则与单文件目录模式一致。

### 冲突处理

如果一个批次中多个输入在输出目录里产生同名目标：

- 直接报错
- 整个批次都不解析

示例：

```text
a/report.pdf
b/report.pdf
```

上例中两个输入都会映射到 `out/report.md`，因此应直接报错。

## 5. 执行模式

`mineru-kit parse` 支持 local 和 remote 两类执行路径。

| 模式 | 说明 |
|------|------|
| local | 使用本地解析能力 |
| remote | 调用 `mineru.net/api` 或指定 API |

remote 模式必须由用户显式选择，不得静默上传文档。

本地模式也保持无状态：即使同一文件已经被 `mineru` 或 doclib 解析过，`mineru-kit parse` 也不会查询或复用 doclib 缓存。

### local 模式

local 模式支持：

- `--tier`
- `--backend`
- `--effort`

规则：

1. 可以只传 `--tier`
2. 可以只传 `--backend`
3. 可以同时传 `--tier` 和 `--backend`
4. 如果二者不兼容，直接报错
5. 默认 tier 选择与 `mineru-kit api-server` 一致
6. 默认不会落到 `flash`
7. `flash` 只能显式指定：
   - `--tier flash`
   - `--backend flash`

### remote 模式

remote 模式通过以下参数进入：

- `--remote`
- `--remote-url <url>`
- `--api-key <key>`

规则：

1. `--remote` 与 `--remote-url` 互斥
2. remote 模式允许传 `--tier`
3. remote 模式禁止传 `--backend`
4. remote 模式未传 `--tier` 时，使用目标服务提供的最高 tier
5. remote 模式传了 `--tier` 时：
   - 服务提供该 tier，则按该 tier 解析
   - 服务不提供该 tier，则报错

## 6. 参数分组

| 分组 | 示例 |
|------|------|
| 文档参数 | pages、language、ocr-mode |
| local 模式 | tier、backend、effort、disable-* |
| remote 模式 | remote、remote-url、api-key |
| 输出控制 | format、output |

local 模式的 `language`、`ocr-mode`、`effort`、`disable-table`、`disable-formula`、`disable-image-analysis` 与 `mineru-kit api-server` 保持一致。

完整决策背景见 [ADR-0016](../decisions/0016-mineru-kit-parse-command.md)。
