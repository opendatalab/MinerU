# mineru-kit parse

状态: Draft
读者: 批处理开发者、解析内核开发者
范围: `mineru-kit parse` 的输入、输出、执行模式、并发、冲突处理和示例
非目标: 本地文档库缓存；Agent 默认渐进式阅读体验
底稿: `../../../NEXT-CLI.md`

## 1. 定位

`mineru-kit parse` 是无状态解析命令。它面向批处理和开发者工作流，可以接受文件、目录、URL、文件列表和 stdin，输出到指定目录或文件。

与 `mineru parse` 不同，它不维护本地文档库，也不复用 `mineru.db` 或 doclib 解析缓存。

## 2. Usage

```bash
mineru-kit parse <input...> -o <output> [flags]
```

`-o` 应作为批处理输出的核心参数。默认输出到文件系统，而不是面向 Agent 的 STDOUT 阅读。

## 3. 输入

支持的输入形态：

| 输入 | 说明 |
|------|------|
| 单文件 | 解析一个文件 |
| 多文件 | 一次提交多个文件 |
| 目录 | 展开目录中的可解析文件 |
| 递归目录 | 递归扫描子目录 |
| 文件列表 | 从列表读取输入 |
| stdin | 从管道读取文件内容或路径列表 |
| URL | 本地或远端模式下的 URL 输入 |

目录展开、隐藏文件、符号链接、重复文件和不可读文件的策略应显式可控。

## 4. 输出

输出格式可以是单一格式或多格式组合：

- markdown
- text
- middle_json
- content_list
- html
- images
- zip

批处理输出必须处理同名文件冲突。

### 冲突处理

| 策略 | 行为 |
|------|------|
| `error` | 默认，发现冲突时报错 |
| `rename` | 自动加前缀或后缀消歧 |
| `path` | 镜像输入目录结构 |

示例：

```text
a/report.pdf
b/report.pdf
```

`rename` 可输出：

```text
out/a_report.md
out/b_report.md
```

`path` 可输出：

```text
out/a/report.md
out/b/report.md
```

## 5. 执行模式

`mineru-kit parse` 支持本地和远端两类执行路径。

| 模式 | 说明 |
|------|------|
| local | 使用本地解析能力 |
| remote | 调用 `mineru.net/api` 或指定 API |
| sync | 等待结果完成 |
| async | 提交任务后返回 task id 或 batch id |

远端模式必须由用户显式选择，不得静默上传文档。

本地模式也保持无状态：即使同一文件已经被 `mineru` 或 doclib 解析过，`mineru-kit parse` 也不会查询或复用 doclib 缓存。

## 6. 参数分组

| 分组 | 示例 |
|------|------|
| 文档参数 | pages、language、ocr |
| 本地模式 | backend、method、device |
| 远端模式 | api-key、base-url |
| 超时与并发 | timeout、concurrency、retry |
| 输出控制 | format、output、on-collision |

解析 tier 语义见 [解析 Tier](../tiers.md)。如果 `mineru-kit parse` 支持不传 tier 的默认选择，也必须遵守默认选择不解析为 `flash` 的规则。

## 未决问题

批量远端任务是否默认异步、stdin 语义和多格式输出目录 schema，集中维护在 [开放问题清单](../open-questions.md)。
