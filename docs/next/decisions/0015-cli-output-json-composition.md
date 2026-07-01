# ADR-0015: CLI Output 与 JSON 组合语义

状态: Accepted
日期: 2026-06-17
相关文档: ../cli/mineru.md, ../cli/mineru-parse.md, ../cli/mineru-read.md, ../cli/mineru-e2e-test-cases.md, 0014-mineru-read-command.md

## 背景

`mineru parse` 和 `mineru read` 都支持 `--output` 与 `--json`:

```bash
mineru parse <path> --output out.md --json
mineru read <locator> --output out.md --json
```

这两个选项处在不同语义层:

- `--output` 控制内容产物写到哪里。
- `--json` 控制 CLI stdout 的响应格式是否为机器可读 JSON。

如果二者同时指定时仍输出 `Written to ...` 这类人类文本，会破坏 JSON 模式的自动化契约。反过来，如果简单禁止组合，调用方在写文件后还需要额外调用 `show`、`read` 或 `list` 才能拿到 `sha256`、`short_id`、`tier`、`request_scope` 等结构化信息，降低 Agent 和脚本使用效率。

同时，`--output + --json` 不能引入新的顶层 JSON shape。`parse --json` 和 `read --json` 已各自有稳定的 JSON 输出结构；是否指定 `--output` 只能在该稳定结构上增加写入结果字段，不能改写其顶层组织方式。

因此需要明确 `--output` 与 `--json` 同时指定时的长期 CLI 行为。

## 决策

`--output` 与 `--json` 可以同时指定。

组合语义:

```text
--output 控制内容产物落盘。
--json 控制 stdout 输出命令结果 envelope。
```

当命令成功时:

- 内容写入 `--output` 指定路径。
- stdout 只输出 JSON。
- stdout 不输出 `Written to ...`、Rich 表格、Markdown 内容或其它人类提示文本。
- JSON 顶层 shape 保持与不带 `--output` 时一致。
- JSON 中必须包含写入结果和足够的文档定位信息。

当命令失败时:

- exit code != 0。
- 如果命令带 `--json`，stdout 应输出 JSON error。
- 不输出半截 JSON 混合人类文本。
- 不产生空的成功输出文件；如果部分文件已写入，应由实现尽量清理，或在 JSON error 中明确说明。

## 成功响应

### `mineru parse --output PATH --json`

成功时 stdout 继续输出稳定的 `parse --json` envelope，而不是文档全文。`--output` 只是在该 envelope 上增加写入结果字段:

```json
{
  "parse": {
    "sha256": "...",
    "tier": "flash",
    "page_range": "1",
    "status": "done"
  },
  "content": null,
  "output": {
    "status": "written",
    "path": "/abs/path/out.md"
  }
}
```

字段要求:

- 顶层继续使用稳定的 `parse` / `content` 结构。
- `parse`: 使用 `mineru parse --json` 已定义的稳定解析摘要结构。
- `content`: 当 `--output` 成功写出文件时可以为 `null`；CLI 不要求在写文件成功的同时再把全文内容重复放回 stdout JSON。
- `output`: 顶层对象，至少包含 `status` 和 `path` 两个字段。
- `output.status`: 当前固定为 `written`。
- `output.path`: 实际写入路径，建议为绝对路径或规范化绝对路径。

### `mineru read --output PATH --json`

成功时 stdout 继续输出稳定的 `read --json` 结构；`--output` 只是在该结构上增加写入结果字段。为避免 stdout JSON 同时携带完整正文和写出副作用结果，当 `--output` 生效时，`content` 字段应置为 `null`:

```json
{
  "sha256": "...",
  "short_id": "...",
  "tier": "flash",
  "format": "markdown",
  "content": null,
  "request_scope": {
    "locator": "doc:abc1234/tier:flash/page:1",
    "context": 0,
    "limit": 30000
  },
  "content_ranges": [],
  "output": {
    "status": "written",
    "path": "/abs/path/out.md"
  }
}
```

对于 image 输出:

```json
{
  "sha256": "...",
  "short_id": "...",
  "tier": "flash",
  "format": "image",
  "content": null,
  "request_scope": {
    "locator": "doc:abc1234/tier:flash/page:1",
    "context": 0,
    "limit": 30000
  },
  "asset": {
    "mime_type": "image/png",
    "size_bytes": 12345,
    "width": 1000,
    "height": 1414
  },
  "output": {
    "status": "written",
    "path": "/abs/path/page.png"
  }
}
```

`asset.path` 不应暴露服务端临时文件路径作为主要结果；`output` 是用户指定并可使用的产物路径。如需调试，可额外提供 `source_asset` 字段，但不能要求调用方依赖它。

规则:

- `read --json` 的顶层 shape 不因为 `--output` 改变。
- 对于 text / markdown 写出，CLI 继续返回 `DocContentResponse` 风格结构，但把 `content` 置为 `null`，并额外增加顶层 `output` 对象。
- 对于 image 写出，CLI 也继续返回 `DocContentResponse` 风格结构，并把 `content` 置为 `null`；`output` 是用户可用的最终路径；`asset.path` 如保留，应明确只是服务端临时来源，不是调用方应依赖的主路径。
- `output.status` 当前固定为 `written`；后续如需区分 `copied`、`overwritten` 等状态，应在不改变顶层 shape 的前提下扩展。

## `--output -`

`--output -` 表示输出到 stdout，等价于不指定 `--output`。该规则只适用于文本类输出；`read --format image --output -` 不支持，因为图片二进制不写 stdout。

规则:

- `mineru parse ... --output -` 输出内容到 stdout。
- `mineru read ... --output -` 在 markdown/text 输出时输出内容到 stdout。
- `mineru read ... --format image --output -` 返回 `image_output_extension_unsupported`。
- `--output - --json` 时，stdout 仍输出 JSON 响应，不写文件，也不新增 `output` 字段。
- 不创建名为 `-` 的文件。

## 输出路径

`--output PATH` 允许自动创建不存在的父目录。

规则:

- 如果父目录不存在，实现应创建父目录。
- 如果父目录无法创建或无写权限，命令失败。
- 已存在的输出文件允许覆盖，除非后续新增显式 `--no-clobber` 之类选项。
- `PATH` 中的 `~` 应由 CLI 主动 `expanduser`。

## JSON 错误契约

所有支持 `--json` 的命令都应遵守统一错误输出:

```json
{
  "error": {
    "type": "invalid_request_error",
    "code": "file_not_found",
    "message": "...",
    "param": "path"
  }
}
```

这条规则不仅适用于 `parse` 和 `read`，也适用于 `watch`、`show`、`config`、`cleanup` 等带 `--json` 的命令。

Typer 在参数解析阶段产生的错误可以保持框架默认输出；但进入命令实现后的业务错误，在 `--json` 下必须输出 JSON error。

## 替代方案

### 方案 A: 禁止 `--output` 与 `--json` 同时指定

不采用。

该方案实现简单，但会让自动化调用变笨。调用方写文件后仍需要额外查询结构化结果，不利于 Agent 和脚本编排。

### 方案 B: `--json` 时忽略 `--output`

不采用。

这会让用户传入的 `--output` 静默失效，风险高，且 `parse` 与 `read` 很容易出现不一致。

### 方案 C: `--output` 时忽略 `--json`

不采用。

这会破坏 `--json` 的核心承诺: stdout 可被机器直接解析。

### 方案 D: 同时输出 JSON 和人类文本

不采用。

混合输出会导致 JSON parser 失败，是最差的自动化体验。

## 影响

- `mineru parse --output PATH --json` 需要保持稳定的 `parse --json` 顶层结构，并增加 `output` 字段；不能输出 `Written to ...`。
- `mineru read --output PATH --json` 需要保持稳定的 `read --json` 顶层结构，并增加 `output` 字段；不能静默忽略 `--output`。
- `mineru read --format image --output PATH` 应只接受 `.png`、`.jpg`、`.jpeg`、`.webp`，并让 server 生成匹配编码的 asset 后再 copy。
- `parse` 和 `read` 的 help 文案应说明:
  - `--output` 可自动创建父目录。
  - 与 `--json` 同时指定时，stdout 输出写入结果 JSON。
- E2E 测试应断言:
  - `--output + --json` stdout 可直接 JSON parse。
  - 输出文件确实存在。
  - stdout 不包含 `Written to`。
- CLI path 处理应对 `--output` 和输入 path 执行 `expanduser`。

## 后续动作

1. 更新 `mineru parse` 和 `mineru read` 实现，统一 `--output + --json` 响应 envelope。
2. 更新 CLI help 文案。
3. 更新 E2E 测试用例，将不存在父目录的 `--output` 预期改为自动创建并成功。
4. 为 `parse/read --output --json` 添加单元测试和 E2E 复核。
5. 推进所有支持 `--json` 的命令统一业务错误 JSON 输出。
