# mineru CLI 端到端测试用例

状态: Draft
读者: CLI 测试执行 Agent、QA、核心开发者
范围: 只测试本地安装的 `mineru` 命令
非目标: 测试 `mineru-kit`、直接访问数据库、调用内部 Python API、验证具体解析模型质量

## 1. 执行规则

执行 Agent 只允许调用本地安装的 `mineru` 命令。

严禁调用:

- `mineru-kit`
- `python`
- `sqlite`
- `curl`
- 项目内部模块
- 任何直接修改 MinerU 数据库或缓存目录的命令

每条用例必须记录:

- case id
- command
- exit code
- stdout
- stderr
- pass / fail / blocked
- 失败原因

## 2. 前置条件

测试环境需提前准备:

- 本地命令 `mineru` 已安装并在 `PATH` 中。
- 环境变量 `MINERU_E2E_FIXTURE_DIR` 指向测试数据目录。
- 测试数据目录至少包含:
  - `$MINERU_E2E_FIXTURE_DIR/sample.pdf`
  - `$MINERU_E2E_FIXTURE_DIR/sample-copy.pdf`，内容与 `sample.pdf` 相同，路径不同。
  - `$MINERU_E2E_FIXTURE_DIR/watch-dir/watch-doc.pdf`
  - `$MINERU_E2E_FIXTURE_DIR/empty-dir/`，空目录。
  - `$MINERU_E2E_FIXTURE_DIR/unsupported.bin`，不支持的普通二进制或文本文件。
  - `$MINERU_E2E_FIXTURE_DIR/sample doc.pdf`，文件名包含空格，内容可与 `sample.pdf` 相同。
  - `$MINERU_E2E_FIXTURE_DIR/中文样例.pdf`，文件名包含中文，内容可与 `sample.pdf` 相同。
  - `$MINERU_E2E_FIXTURE_DIR/corrupted.pdf`，损坏 PDF。
  - `$MINERU_E2E_FIXTURE_DIR/empty.pdf`，空文件或无法解析出页面的 PDF。
  - `$MINERU_E2E_FIXTURE_DIR/sample.docx`、`sample.pptx`、`sample.xlsx`，Office 样例文件；当前全量 E2E 必须覆盖。
  - `$MINERU_E2E_FIXTURE_DIR/sample.md`、`sample.txt`、`sample.csv`，文本类样例文件；当前全量 E2E 必须覆盖。
  - `$MINERU_E2E_FIXTURE_DIR/sample.jpeg`，图片样例文件；若当前安装不支持图片输入，可按预期失败分支判定。
  - `$MINERU_E2E_FIXTURE_DIR/symlink-sample.pdf`，指向 `sample.pdf` 的符号链接；若平台不支持 symlink，可标记相关用例 BLOCKED。
  - `$MINERU_E2E_FIXTURE_DIR/no-read.pdf`，权限不可读文件；若平台无法稳定制造权限场景，可标记相关用例 BLOCKED。
  - `$MINERU_E2E_FIXTURE_DIR/output-dir/`，用于输出文件测试的目录。
- 测试环境允许启动本地 doclib server。
- 测试环境必须安装当前质量解析依赖 extra，以覆盖本地 quality parse-server；默认 tier 相关用例应验证本地 quality tier 可用，不再按缺少本地 quality tier 的预期失败分支判定。
- 除 PARSE-013A1 外，若 remote parse-server 不可用，`--remote` 相关用例按预期失败分支判定；若可用，必须验证 remote/via/privacy 等字段。
- PARSE-013A1 是 remote high 硬性测试，remote parse-server 不可用或不支持 high 均记录为失败。

### 2.1 测试 HOME 与隔离配置

测试 HOME 使用:

```bash
~/mineru-e2e-test
```

进入测试目录后，只需设置 `MINERU_HOME`，即可让默认配置文件（`$MINERU_HOME/config.yaml`）、DB、日志、endpoint discovery 文件、UDS socket（启用 UDS 时）和数据目录都落在测试目录中:

```bash
cd ~/mineru-e2e-test
export MINERU_HOME=`pwd`
```

### 2.2 安装方法

在测试 HOME 中创建独立虚拟环境并安装当前仓库:

```bash
cd ~/mineru-e2e-test
rm -rf .venv
uv venv .venv
source .venv/bin/activate
cd ~/MinerU-Repo
uv pip install ".[high]"
cd ~/mineru-e2e-test
mineru --help
mineru-kit --help
```

安装完成后，`mineru` 和 `mineru-kit` 都应由 `pyproject.toml` 的脚本入口直接提供，不需要配置 shell alias。

执行 Agent 后续仍只调用 `mineru ...` 命令，不测试 `mineru-kit`，也不直接调用 Python 模块、内部 API 或数据库。

### 2.3 Fixture 生成方法

fixture 准备阶段允许使用 shell 文件操作创建隔离测试数据。`mineru` 命令限制只适用于正式 case 执行阶段。

PDF 样例使用仓库内覆盖元素更完整的 demo 文件:

```bash
demo/pdfs/all_elements.pdf
```

图片样例使用仓库内 demo 文件:

```bash
demo/images/photo_20250417_111916.jpeg
```

推荐生成步骤:

```bash
cd ~/mineru-e2e-test
export MINERU_HOME=`pwd`
export MINERU_E2E_FIXTURE_DIR="$MINERU_HOME/fixtures"
rm -rf "$MINERU_E2E_FIXTURE_DIR"
mkdir -p "$MINERU_E2E_FIXTURE_DIR/watch-dir" "$MINERU_E2E_FIXTURE_DIR/empty-dir" "$MINERU_E2E_FIXTURE_DIR/output-dir" "$MINERU_HOME/removable-watch"

PDF_SOURCE="$HOME/MinerU-Repo/demo/pdfs/all_elements.pdf"

cp "$PDF_SOURCE" "$MINERU_E2E_FIXTURE_DIR/sample.pdf"
cp "$MINERU_E2E_FIXTURE_DIR/sample.pdf" "$MINERU_HOME/sample.pdf"
cp "$MINERU_E2E_FIXTURE_DIR/sample.pdf" "$MINERU_E2E_FIXTURE_DIR/sample-copy.pdf"
cp "$MINERU_E2E_FIXTURE_DIR/sample.pdf" "$MINERU_E2E_FIXTURE_DIR/watch-dir/watch-doc.pdf"
cp "$MINERU_E2E_FIXTURE_DIR/sample.pdf" "$MINERU_E2E_FIXTURE_DIR/sample doc.pdf"
cp "$MINERU_E2E_FIXTURE_DIR/sample.pdf" "$MINERU_E2E_FIXTURE_DIR/中文样例.pdf"

cp "$HOME/MinerU-Repo/demo/office_docs/docx_01.docx" "$MINERU_E2E_FIXTURE_DIR/sample.docx"
cp "$HOME/MinerU-Repo/demo/office_docs/pptx_01.pptx" "$MINERU_E2E_FIXTURE_DIR/sample.pptx"
cp "$HOME/MinerU-Repo/demo/office_docs/xlsx_01.xlsx" "$MINERU_E2E_FIXTURE_DIR/sample.xlsx"

printf '# MinerU E2E Markdown\n\nThis file verifies markdown input.\n' > "$MINERU_E2E_FIXTURE_DIR/sample.md"
printf 'MinerU E2E text fixture\nsecond line\n' > "$MINERU_E2E_FIXTURE_DIR/sample.txt"
printf 'name,value\nalpha,1\nbeta,2\n' > "$MINERU_E2E_FIXTURE_DIR/sample.csv"

printf 'not a supported document' > "$MINERU_E2E_FIXTURE_DIR/unsupported.bin"
printf 'not a real pdf' > "$MINERU_E2E_FIXTURE_DIR/corrupted.pdf"
: > "$MINERU_E2E_FIXTURE_DIR/empty.pdf"
cp "$MINERU_E2E_FIXTURE_DIR/sample.pdf" "$MINERU_E2E_FIXTURE_DIR/no-read.pdf"
chmod 000 "$MINERU_E2E_FIXTURE_DIR/no-read.pdf"
ln -sf "$MINERU_E2E_FIXTURE_DIR/sample.pdf" "$MINERU_E2E_FIXTURE_DIR/symlink-sample.pdf"

IMAGE_SOURCE="$HOME/MinerU-Repo/demo/images/photo_20250417_111916.jpeg"
cp "$IMAGE_SOURCE" "$MINERU_E2E_FIXTURE_DIR/sample.jpeg"
```

执行结束后应恢复不可读文件权限，避免清理失败:

```bash
chmod 644 "$MINERU_E2E_FIXTURE_DIR/no-read.pdf" 2>/dev/null || true
```

## 3. 顶层命令与帮助

### CLI-001 顶层 help

命令:

```bash
mineru --help
```

预期:

- exit code = 0
- 输出包含 `parse`、`read`、`scan`、`watch`、`search`、`find`、`list`、`show`、`server`、`config`、`invalidate`、`forget`、`cleanup`
- 输出包含 `telemetry`
- 输出不包含 `mineru-kit`

### CLI-002 子命令 help

命令:

```bash
mineru parse --help
mineru read --help
mineru scan --help
mineru watch --help
mineru search --help
mineru find --help
mineru list --help
mineru show --help
mineru telemetry --help
mineru server --help
mineru config --help
mineru invalidate --help
mineru forget --help
mineru cleanup --help
```

预期:

- 每条 exit code = 0
- 每条输出包含 usage 或 options 类帮助信息

### CLI-002A output/help 契约

命令:

```bash
mineru parse --help
mineru read --help
```

预期:

- 两条 exit code = 0
- `--output` help 文案说明可写入文件
- `--output` help 文案或相邻说明应体现父目录可自动创建
- `--json` help 文案或相邻说明应体现 JSON 输出
- parse/read help 不得暗示 `--output` 与 `--json` 互斥

### CLI-003 不存在旧命令

命令:

```bash
mineru info --help
```

预期:

- exit code != 0

## 4. Server 生命周期

### SERVER-001 查询初始状态

命令:

```bash
mineru server status
```

预期:

- exit code = 0
- 输出为 server 运行信息或 `Server is not running.`

### SERVER-001A 查询初始 JSON 状态

命令:

```bash
mineru server status --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- 如果 server 未运行，JSON 为稳定状态结果，至少包含 `running: false`
- server 未运行不应输出 JSON error
- stdout 不包含 `Server is not running.` 人类文本

### SERVER-002 启动 server

命令:

```bash
mineru server start
```

预期:

- exit code = 0
- 如果已运行，输出包含 `already running`
- 如果未运行，输出包含 `started`

### SERVER-003 查询运行状态

命令:

```bash
mineru server status
```

预期:

- exit code = 0
- 输出包含 server 状态信息
- 不提示连接失败

### SERVER-004 JSON 状态

命令:

```bash
mineru server status --json
```

预期:

- server 运行时 exit code = 0
- 输出为 JSON 或等价结构化 server 状态
- 至少包含 `running`、`socket_path`、`tcp`、`data_dir`、队列或文档库统计中的若干字段
- `tcp` 字段至少包含 `enabled`、`host`、`port`

### SERVER-005 重复启动 server

前置: SERVER-002 已成功，server 正在运行。

命令:

```bash
mineru server start
```

预期:

- exit code = 0
- 输出包含 `already running`
- 不启动第二个不可控 server 进程
- 后续 `mineru server status` 仍可正常返回

### SERVER-006 restart server

命令:

```bash
mineru server restart
mineru server status --json
```

预期:

- 两条 exit code = 0
- restart 输出包含 stopped/started、started 或等价生命周期信息
- status 输出为 JSON
- JSON 表示 server 正在运行
- `socket_path`、`data_dir`、`sqlite_path`、`log_path` 仍落在测试 HOME 配置下

### SERVER-007 stop 后依赖 server 的命令报错可读

命令:

```bash
mineru server stop
mineru list files --limit 1
mineru server start
```

预期:

- stop exit code = 0
- list files exit code != 0
- list files 输出包含 `Cannot connect`、`server`、`Run 'mineru server start' first` 或等价可操作错误
- list files 不包含 Python traceback
- 最后一条 start exit code = 0，用于恢复后续测试环境

### SERVER-009 环境变量路径生效

命令:

```bash
mineru server status --json
```

预期:

- exit code = 0
- 输出为 JSON
- `data_dir` 默认位于 `$MINERU_HOME/doclib`；若显式覆盖 `doclib.data_dir`，则返回该覆盖值
- `socket_path` 字段等价于 `MINERU_DOCLIB_UDS_PATH`，不得为空；即使当前 transport 是 TCP-only，也可以表示配置路径
- TCP 相关字段应位于 `tcp.enabled`、`tcp.host`、`tcp.port`
- 日志相关字段或启动失败日志路径应落在 `MINERU_DOCLIB_LOG_DIR`，或显式配置的 `MINERU_DOCLIB_LOG_APP_PATH`

### SERVER-013 endpoint discovery 文件写入

前置: server 已启动。

命令:

```bash
cat "$MINERU_HOME/doclib.endpoint.json"
mineru server status --json
```

预期:

- endpoint 文件存在，且 stdout 可解析为 JSON
- endpoint JSON 至少包含 `version`、`pid`、`transports`
- `transports` 非空，每个 transport 均包含 `type`
- 默认 UDS 可用环境下，`transports` 至少包含 `{"type": "uds", "path": ...}`，且 path 位于测试 HOME
- 如果当前环境使用 TCP transport，则对应项包含 `{"type": "tcp", "base_url": "http://127.0.0.1:<port>"}`
- `mineru server status --json` 能通过 discovery 正常连接 server

### SERVER-014 TCP-only fallback

前置:

- 使用一个新的独立 `MINERU_HOME`，避免与主测试 server 冲突。
- 执行前保存主测试 HOME，例如 `MAIN_MINERU_HOME="$MINERU_HOME"`；执行结束后恢复 `MINERU_HOME="$MAIN_MINERU_HOME"`。
- 设置:

```bash
MAIN_MINERU_HOME="$MINERU_HOME"
export MINERU_HOME="$HOME/mineru-e2e-test-tcp"
export MINERU_DOCLIB_UDS_ENABLED=false
export MINERU_DOCLIB_TCP_ENABLED=true
export MINERU_DOCLIB_TCP_PORT=0
```

命令:

```bash
mineru server start
mineru server status --json
cat "$MINERU_HOME/doclib.endpoint.json"
mineru server stop
export MINERU_HOME="$MAIN_MINERU_HOME"
unset MINERU_DOCLIB_UDS_ENABLED MINERU_DOCLIB_TCP_ENABLED MINERU_DOCLIB_TCP_PORT
```

预期:

- start exit code = 0
- status JSON 中 `running=true`
- status JSON 中 `tcp.enabled=true`
- status JSON 中 `tcp.host=127.0.0.1`
- status JSON 中 `tcp.port` 是大于 0 的实际端口，不应为 0
- endpoint JSON 的 `transports` 只需包含 TCP transport，`base_url` 端口必须等于 status JSON 的 `tcp.port`
- stop exit code = 0
- stop 后 `$MINERU_HOME/doclib.endpoint.json` 被清理，或后续 `mineru server status --json` 返回 `running=false`

### SERVER-015 endpoint stale 清理

前置:

- 使用独立 `MINERU_HOME`。
- 执行前保存主测试 HOME，例如 `MAIN_MINERU_HOME="$MINERU_HOME"`；执行结束后恢复 `MINERU_HOME="$MAIN_MINERU_HOME"`。
- 手工写入一个 stale `$MINERU_HOME/doclib.endpoint.json`，内容指向不存在的本地 TCP 端口。

命令:

```bash
MAIN_MINERU_HOME="$MINERU_HOME"
export MINERU_HOME="$HOME/mineru-e2e-test-stale-endpoint"
mkdir -p "$MINERU_HOME"
cat > "$MINERU_HOME/doclib.endpoint.json" <<'JSON'
{"version":1,"pid":999999,"transports":[{"type":"tcp","base_url":"http://127.0.0.1:9"}]}
JSON
mineru server status --json
mineru server stop
export MINERU_HOME="$MAIN_MINERU_HOME"
```

预期:

- status exit code = 0
- status JSON 返回 `running=false`，不因 stale endpoint 输出 Python traceback
- stop exit code = 0
- stop 输出 `Server is not running` 或等价信息
- stop 后 stale endpoint 文件被 best-effort 清理

## 4A. Telemetry 命令

### TELEMETRY-001 telemetry help

命令:

```bash
mineru telemetry --help
mineru telemetry status --help
mineru telemetry enable --help
mineru telemetry disable --help
mineru telemetry preview --help
mineru telemetry flush --help
```

预期:

- 每条 exit code = 0
- 输出包含 usage 或 options 类帮助信息
- `mineru telemetry --help` 输出包含 `status`、`enable`、`disable`、`preview`、`flush`
- 不触发 telemetry 首次选择 prompt
- 不包含 Python traceback

### TELEMETRY-002 status JSON 结构

前置: server 正在运行。

命令:

```bash
mineru telemetry status --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- JSON 包含 `state`、`installation_id`、`pending_periods`、`pending_metrics`
- `state` 取值为 `unset`、`enabled` 或 `disabled`
- `installation_id` 为非空字符串
- `pending_periods` 和 `pending_metrics` 为非负整数
- telemetry status 命令自身不触发首次选择 prompt
- 不包含 Python traceback

### TELEMETRY-003 preview JSON 结构

前置: server 正在运行。

命令:

```bash
mineru telemetry preview
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- JSON 为下一次外部 telemetry HTTP 请求 body 的预览，不额外包 CLI metadata
- JSON 包含 `api_version`、`schema_version`、`batch_id`、`installation_id`、`period_start`、`period_end`、`context`、`metrics`
- `installation_id` 与 TELEMETRY-002 的 `installation_id` 一致
- `metrics` 为数组；没有待上报数据时允许为空数组
- preview 不触发 flush，不要求外网可用
- 不包含 Python traceback

### TELEMETRY-004 enable / disable 与 installation_id 稳定性

前置: server 正在运行。

命令:

```bash
mineru telemetry status --json
mineru telemetry enable --json
mineru telemetry status --json
mineru telemetry disable --json
mineru telemetry status --json
```

预期:

- 每条 exit code = 0
- 每条 stdout 为可直接解析的 JSON
- enable 返回后，后续 status 的 `state` 为 `enabled`
- disable 返回后，后续 status 的 `state` 为 `disabled`
- 全流程中 `installation_id` 保持不变
- enable/disable 命令自身不触发首次选择 prompt
- 不包含 Python traceback

### TELEMETRY-005 disabled 时不新增聚合

前置: server 正在运行。

命令:

```bash
mineru telemetry disable --json
mineru search "mineru-e2e-query-that-should-not-exist" --limit 1 --json
mineru telemetry preview
```

预期:

- 三条 exit code = 0
- search stdout 为可直接解析的 JSON，允许结果为空
- preview stdout 为可直接解析的 JSON
- preview JSON 中 `metrics` 为空数组，或至少不包含本次 search 产生的 `search.request.count`、`search.finished.count`、`search.duration_bucket.count`、`search.results_bucket.count`
- 不包含 Python traceback

### TELEMETRY-006 unset 或 disabled 时 flush 不外发

前置: server 正在运行；如果当前状态不是 `disabled`，先执行 `mineru telemetry disable --json`。

命令:

```bash
mineru telemetry flush --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- JSON 中 `action` = `flush`
- JSON 中 `executed` = `false`
- JSON 中 `reason` = `telemetry_not_enabled`
- 不要求外网可用
- 不包含 Python traceback

### TELEMETRY-007 enabled 时 preview 可观察业务聚合

前置: server 正在运行。

命令:

```bash
mineru telemetry enable --json
mineru search "mineru-e2e-query-that-should-not-exist" --limit 1 --json
mineru telemetry preview
```

预期:

- 三条 exit code = 0
- search stdout 为可直接解析的 JSON，允许结果为空
- preview stdout 为可直接解析的 JSON
- preview JSON 中 `metrics` 包含 `search.request.count`
- preview JSON 中应包含 `search.finished.count` 和 `search.duration_bucket.count`
- 如果 search 正常返回，`search.finished.count` 的 status 维度为 `succeeded`
- `metrics` 中不得包含 query 原文 `mineru-e2e-query-that-should-not-exist`
- 不包含 Python traceback

### TELEMETRY-008 enabled flush 行为

前置: 已执行 TELEMETRY-007，且 preview 中有待上报 metrics。

命令:

```bash
mineru telemetry flush --json
mineru telemetry preview
```

预期分支:

- 如果外部 telemetry endpoint 可访问且接收成功:
  - flush exit code = 0
  - flush JSON 中 `action` = `flush`
  - flush JSON 中 `flush_result.status` 为 `success` 或 `partial_success`
  - 后续 preview 中已成功发送 period 的 metrics 被清除或减少
- 如果外部 telemetry endpoint 不可访问、超时或返回 5xx:
  - flush exit code = 0
  - flush JSON 中 `flush_result.status` 为 `failed`、`partial_success` 或等价重试状态
  - 后续 preview 可保留待上报 metrics
- 两个分支都不应输出外部 endpoint response body
- 两个分支都不应包含 Python traceback

### TELEMETRY-009 首次交互式 prompt

前置:

- 使用全新的 `MINERU_HOME`，server 已启动，且 `mineru telemetry status --json` 返回 `state=unset`。
- 本用例必须在真实交互式终端执行；如果执行 Agent 只能非交互执行命令，可标记 BLOCKED。

命令:

```bash
mineru search "mineru-e2e-query-that-should-not-exist" --limit 1
```

操作与预期:

- 命令进入业务执行前提示是否开启匿名聚合 telemetry
- prompt 文案应说明收集的是匿名、本地聚合的使用与诊断数据
- prompt 文案应说明不会收集文档内容、提取文本/图片、文件名、文件路径、URL、query、prompt、snippet、traceback、exception message、hostname、用户名、账号 ID、API Key 或精确 CPU/GPU 型号
- prompt 文案应说明 Enter / `y` / `yes` 表示开启，`n` / `no` 表示关闭
- prompt 文案应说明后续可通过 `mineru telemetry enable` / `mineru telemetry disable` 修改，并可通过 `mineru telemetry preview` 查看将要上报的 payload
- prompt 文案不得包含当前命令的文件路径、query、文档内容或其它业务输入
- 输入 Enter、`y` 或 `yes` 后，命令继续执行，后续 `mineru telemetry status --json` 返回 `state=enabled`
- 在另一轮全新 `MINERU_HOME` 中输入 `n` 或 `no` 后，命令继续执行，后续 `mineru telemetry status --json` 返回 `state=disabled`
- `mineru telemetry ...`、`mineru server ...`、`mineru config ...` 命令自身不触发 prompt
- `--help`、`--json`、CI、Agent caller、非 TTY、server 不可访问时不触发 prompt
- 不包含 Python traceback

### TELEMETRY-010 preview 隐私边界

前置: server 正在运行，telemetry 已启用。

命令:

```bash
mineru telemetry enable --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --json
mineru search "mineru-e2e-query-that-should-not-exist" --limit 1 --json
mineru telemetry preview
```

预期:

- 前三条业务命令 exit code = 0；如果 parse 因环境缺少 flash 能力失败，可记录失败分支，但仍继续检查 preview 隐私边界
- preview exit code = 0，stdout 为可直接解析的 JSON
- preview JSON 不包含:
  - `$MINERU_E2E_FIXTURE_DIR`
  - `sample.pdf`
  - `mineru-e2e-query-that-should-not-exist`
  - Markdown 正文、snippet、页面文本、traceback
  - API key、secret、配置对象原文
- preview JSON 中允许出现 metric name、低基数字段、bucket、status、tier、source、caller、installation_id 和粗粒度环境 context

## 5. Config 命令

### CONFIG-001 查看配置

命令:

```bash
mineru config show
```

预期:

- exit code = 0
- 输出包含 `Config` 或配置 key

### CONFIG-002 JSON 查看配置

命令:

```bash
mineru config show --json
```

预期:

- exit code = 0
- 输出为 JSON
- 包含 config 或 sources 信息

### CONFIG-003 设置和读取配置

命令:

```bash
mineru config set parse_server.local.managed_tier high
mineru config get parse_server.local.managed_tier
```

预期:

- 两条 exit code = 0
- get 输出包含 `parse_server.local.managed_tier`
- get 输出包含 `high`

### CONFIG-003A JSON 读取配置

命令:

```bash
mineru config set parse_server.local.managed_tier high
mineru config get parse_server.local.managed_tier --json
```

预期:

- 两条 exit code = 0
- set 使用普通文本输出；本用例不要求 `config set` 支持 `--json`
- get stdout 为可直接解析的 JSON
- get JSON 包含 `parse_server.local.managed_tier`
- get JSON 中 value 为 `high`，并且 source 应体现 override 或等价覆盖来源

### CONFIG-004 unset 配置

命令:

```bash
mineru config unset parse_server.local.managed_tier
mineru config get parse_server.local.managed_tier
```

预期:

- 两条 exit code = 0
- unset 输出包含 `removed` 或 `unchanged`
- get 仍能返回有效配置值，默认值应为 `high` 或等价默认 tier

### CONFIG-004A unset 后 JSON 读取配置

命令:

```bash
mineru config set parse_server.local.managed_tier high
mineru config unset parse_server.local.managed_tier
mineru config get parse_server.local.managed_tier --json
```

预期:

- 三条 exit code = 0
- unset 使用普通文本输出；本用例不要求 `config unset` 支持 `--json`
- get 输出为可直接解析的 JSON
- unset 后 get 仍能返回有效配置值，默认值应为 `high` 或等价默认 tier

### CONFIG-005 exclude-rules

命令:

```bash
mineru config exclude-rules add "*/tmp-mineru-e2e-ignore/*" --priority 10
mineru config exclude-rules list
```

预期:

- 两条 exit code = 0
- list 输出包含该 pattern 或规则 id

### CONFIG-005A exclude-rules JSON 和 remove

命令:

```bash
mineru config exclude-rules add "*/tmp-mineru-e2e-remove/*" --priority 11 --json
mineru config exclude-rules list --json
mineru config exclude-rules remove <rule_id>
mineru config exclude-rules list --json
```

执行说明:

- `<rule_id>` 从第一条 add JSON 输出或第二条 list JSON 输出中提取。

预期:

- 四条 exit code = 0
- add/list JSON 均可直接解析
- remove 前 list 中包含 pattern `*/tmp-mineru-e2e-remove/*`
- remove 输出包含 `removed`、`Exclude rule` 或等价成功信息
- remove 后 list 中不再包含该 rule id

### CONFIG-006 旧 exclude 命令不可用

命令:

```bash
mineru config exclude --help
```

预期:

- exit code != 0

### CONFIG-007 parsing-rules

命令:

```bash
mineru config parsing-rules add "*/mineru-e2e/*" --tier flash --pages all --name e2e-flash-rule
mineru config parsing-rules list
```

预期:

- 两条 exit code = 0
- list 输出包含 pattern
- list 输出包含 `flash`
- list 输出包含 `all`

### CONFIG-008 parsing-rules JSON 和 remove

命令:

```bash
mineru config parsing-rules add "*/mineru-e2e-remove/*" --tier flash --pages 1~1 --name e2e-remove-rule --json
mineru config parsing-rules list --json
mineru config parsing-rules remove <rule_id>
mineru config parsing-rules list --json
```

执行说明:

- `<rule_id>` 从第一条 add JSON 输出或第二条 list JSON 输出中提取。

预期:

- 四条 exit code = 0
- add/list JSON 均可直接解析
- remove 前 list 中包含 pattern、`flash`、`1~1`
- remove 输出包含 `removed`、`Parsing rule` 或等价成功信息
- remove 后 list 中不再包含该 rule id

### CONFIG-009 不存在的配置 key

命令:

```bash
mineru config get not.a.real.key
mineru config get not.a.real.key --json
```

预期:

- 两条命令应有一致、可解释的行为
- 如果 exit code = 0，输出必须明确表示 key 不存在、值为空或来源为默认缺失
- 如果 exit code != 0，输出必须包含 not found、unknown key、invalid key 或等价错误
- `--json` 失败时 stdout 必须为可直接解析的 JSON error，且包含 `error.code` 和 `error.message`
- 不包含 Python traceback

## 6. Watch 命令

### WATCH-001 添加 watch

命令:

```bash
mineru watch add "$MINERU_E2E_FIXTURE_DIR/watch-dir" --label e2e-watch
```

预期:

- exit code = 0
- 输出包含 `Watch added` 或 id

### WATCH-001A 重复添加 watch

前置: WATCH-001 已成功。

命令:

```bash
mineru watch add "$MINERU_E2E_FIXTURE_DIR/watch-dir" --label e2e-watch
```

预期:

- 命令应有明确行为:
  - 若允许幂等，exit code = 0，输出包含 existing、already、Watch added 或等价信息
  - 若不允许重复，exit code != 0，输出包含 duplicate、already exists 或等价错误
- 不产生不可解释的重复 watch 记录
- 不包含 Python traceback

### WATCH-002 列出 watch

命令:

```bash
mineru watch list
```

预期:

- exit code = 0
- 输出包含 `$MINERU_E2E_FIXTURE_DIR/watch-dir` 或 `e2e-watch`

### WATCH-003 JSON 列出 watch

命令:

```bash
mineru watch list --json
```

预期:

- exit code = 0
- 输出为 JSON
- 包含 watches 数组或 watch 信息

### WATCH-004 watch rescan

命令:

```bash
mineru watch rescan "$MINERU_E2E_FIXTURE_DIR/watch-dir" --wait 30
```

预期:

- exit code = 0
- 输出包含 `Scan`
- 状态为 done、running 或 pending 均可接受
- 不出现 `watch_not_found`

### WATCH-004A 使用 watch id rescan

前置: WATCH-002 或 WATCH-003 可获得 watch id。

命令模板:

```bash
mineru watch rescan <watch_id> --wait 30
mineru watch rescan <watch_id> --no-wait --json
```

预期:

- 两条 exit code = 0
- 第一条输出包含 `Scan`
- 第二条输出为可直接解析的 JSON
- JSON 包含 scan id、status、path 或 watch_id 中的部分字段

### WATCH-005 删除 watch

命令:

```bash
mineru watch remove "$MINERU_E2E_FIXTURE_DIR/watch-dir"
```

预期:

- exit code = 0
- 输出包含 `Watch removed`

### WATCH-006 删除不存在的 watch

命令:

```bash
mineru watch remove "$MINERU_E2E_FIXTURE_DIR/not-a-watch-dir"
```

预期:

- exit code != 0
- 输出包含 `watch_not_found`、not found 或等价错误
- 不包含 Python traceback

### WATCH-007 添加不存在目录

命令:

```bash
mineru watch add "$MINERU_E2E_FIXTURE_DIR/not-exist-dir" --label e2e-missing-watch
```

预期:

- exit code != 0
- 输出包含 not found、unreachable、invalid path 或等价错误
- 不包含 Python traceback

## 7. Scan 命令

### SCAN-001 扫描单文件

命令:

```bash
mineru scan "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --wait 30
```

预期:

- exit code = 0
- 输出包含 `Scan`
- 输出包含 seen、refreshed、new、changed、deleted 中的统计字段

### SCAN-002 扫描目录 no-wait

命令:

```bash
mineru scan "$MINERU_E2E_FIXTURE_DIR" --no-wait --json
```

预期:

- exit code = 0
- 输出为 JSON
- 包含 scan id、status、path 或 kind 字段
- kind 应为 manual

### SCAN-003 扫描不存在路径

命令:

```bash
mineru scan "$MINERU_E2E_FIXTURE_DIR/not-exist"
mineru scan "$MINERU_E2E_FIXTURE_DIR/not-exist" --json
```

预期:

- 两条 exit code != 0
- 输出包含 not found、unreachable、invalid path 或等价错误
- 不包含 Python traceback
- JSON 模式不得输出半截 JSON 混合 traceback

### SCAN-004 扫描空目录

前置: 测试数据目录预先包含空目录 `$MINERU_E2E_FIXTURE_DIR/empty-dir/`。

命令:

```bash
mineru scan "$MINERU_E2E_FIXTURE_DIR/empty-dir" --wait 30
mineru scan "$MINERU_E2E_FIXTURE_DIR/empty-dir" --wait 30 --json
```

预期:

- 两条 exit code = 0
- JSON 输出可直接解析
- 统计字段中 seen/new/changed/deleted 等计数为 0 或符合空目录语义
- 不包含 Python traceback

### SCAN-005 扫描不支持文件类型

前置: 测试数据目录预先包含不支持文件 `$MINERU_E2E_FIXTURE_DIR/unsupported.bin`。

命令:

```bash
mineru scan "$MINERU_E2E_FIXTURE_DIR/unsupported.bin" --wait 30
mineru scan "$MINERU_E2E_FIXTURE_DIR/unsupported.bin" --wait 30 --json
```

预期:

- 两条 exit code = 0 或返回明确的 unsupported 错误
- 如果 exit code = 0，输出统计中应体现 unsupported 或 excluded 计数
- 如果 exit code != 0，输出包含 unsupported、invalid type 或等价错误
- 不包含 Python traceback

## 8. Parse 命令

### PARSE-001 文件不存在

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/not-exist.pdf"
```

预期:

- exit code != 0
- 输出包含 `File not found`

### PARSE-002 显式 flash parse

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60
```

预期:

- exit code = 0
- 输出不包含 Python traceback
- 输出为可读 Markdown、文本内容或 parse 状态提示

### PARSE-003 flash parse JSON

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --json
```

预期:

- exit code = 0
- 输出为 JSON
- 顶层包含 `parse` 与 `content`
- `parse.sha256` 或等价 doc 标识存在
- `parse.tier = flash`
- `parse.status = done`
- `content` 不为 null
- `content.request_scope`、`content.content_ranges` 或 `content.next_request` 中至少一部分字段存在

### PARSE-004 cache hit 行为

重复执行:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --json
```

预期:

- exit code = 0
- 不重新失败
- 若 JSON 中包含 cache_hit，应为 true 或体现已复用结果
- tier 仍为 flash

### PARSE-005 no-wait 行为

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --no-wait --json
```

预期:

- exit code = 0
- 输出为 JSON
- 顶层包含 `parse` 与 `content`
- `parse.status` 为 `pending`、`parsing` 或 `done`
- 若 `parse.status` 不是 `done`，则 `content = null`
- 若命中缓存直接完成，则 `content` 可直接返回
- 不长时间阻塞

### PARSE-006 force 行为

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --force --wait 60 --json
```

预期:

- exit code = 0
- 输出为 JSON
- 不删除旧缓存导致后续普通 parse 失败

### PARSE-006A force 默认输出不打印过程 status

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --force --wait 60
```

预期:

- exit code = 0
- 输出为解析内容、Markdown 或最终可读结果
- 默认模式输出不包含 `Parse status:`
- 默认模式输出不包含 `Parse queued`
- 不包含 Python traceback

### PARSE-006B verbose 输出允许过程 status

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --force --wait 60 --verbose
```

预期:

- exit code = 0
- 输出允许包含 `Parse queued`
- 输出允许包含 `Parse status:`
- 最终仍输出解析内容、Markdown 或最终可读结果
- 不包含 Python traceback

### PARSE-007 默认 tier 行为

命令:

```bash
mineru config set parse_server.local.mode managed
mineru config set parse_server.local.managed_tier high
mineru server status --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --pages 1~1 --wait 20 --json
```

预期:

- 三条命令均 exit code = 0
- parse 前 `server status --json` 最终应体现 `parse_server.local.healthy=true`，`supported_tiers` 包含 `high`
- stdout 为可直接解析的 JSON
- 实际 tier 为 high
- 实际 tier 不为 flash
- 不允许静默返回 flash 内容
- 不包含 Python traceback

### PARSE-007A PDF local medium tier

命令:

```bash
mineru config set parse_server.local.mode managed
mineru config set parse_server.local.managed_tier medium
mineru server status --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier medium --pages 1~1 --force --wait 120 --json
```

预期:

- 四条命令均 exit code = 0
- parse 前 `server status --json` 最终应体现 `parse_server.local.healthy=true`，`supported_tiers` 包含 `medium`
- stdout 为可直接解析的 JSON
- `parse.tier = medium`
- `parse.status = done`
- `content` 不为 null
- JSON 不体现 remote/via remote，或明确体现 local transport
- 不包含 Python traceback

### PARSE-007B PDF local high tier

命令:

```bash
mineru config set parse_server.local.mode managed
mineru config set parse_server.local.managed_tier high
mineru server status --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier high --pages 1~1 --force --wait 180 --json
```

预期:

- 四条命令均 exit code = 0
- parse 前 `server status --json` 最终应体现 `parse_server.local.healthy=true`，`supported_tiers` 包含 `high`
- stdout 为可直接解析的 JSON
- `parse.tier = high`
- `parse.status = done`
- `content` 不为 null
- JSON 不体现 remote/via remote，或明确体现 local transport
- 不包含 Python traceback

### PARSE-008 输出到文件

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --output "$MINERU_E2E_FIXTURE_DIR/parse-output.md"
```

预期:

- exit code = 0
- 输出包含 `Written to` 或等价成功信息
- 不在 stdout 大量打印完整文档内容
- 后续可通过 `mineru read` 或重新 `mineru parse` 验证解析缓存仍可用
- 输出文件路径应为命令指定路径

### PARSE-009 JSON 与 output 组合

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --output "$MINERU_E2E_FIXTURE_DIR/parse-output-json.md" --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- JSON 顶层保持 `parse --json` envelope，包含 `parse` 和 `content`
- 当文件写出成功时，`content` 可以为 null
- JSON 顶层包含 `output`
- `output.status` = `written`
- `output.path` 为实际写入路径，建议为绝对路径或规范化绝对路径
- 输出文件存在，路径为命令指定路径
- stdout 不允许包含 `Written to`、Rich 表格、Markdown 正文或其它人类提示文本
- 不包含 Python traceback

### PARSE-010 limit 截断与 next_request

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --limit 20 --json
```

预期:

- exit code = 0
- 输出为可直接解析的 JSON
- `content` 不为 null
- 如果内容超过 limit，`content` 中应体现 truncated、next_request、cursor、after 或等价续读信息
- 如果内容未超过 limit，`content` 中应明确表示未截断或没有 next_request
- 不包含 JSON 外的 status/info 文本

### PARSE-011 after 续读

前置: PARSE-010 返回 next_request.after 或等价 cursor。

命令模板:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --after <after_cursor> --limit 30000 --json
```

预期:

- exit code = 0
- 输出为可直接解析的 JSON
- `content` 不为 null
- `content.request_scope`、`content.content_ranges` 或等价字段体现 after/cursor 已生效
- 不返回与第一页起始完全相同的截断片段，除非服务端明确说明 cursor 已到末尾

### PARSE-012 no-marker

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --limit 20 --no-marker
```

预期:

- exit code = 0
- 输出不包含 `<!-- Next:`
- 输出不包含 Python traceback

### PARSE-013 remote parse 分支

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --remote --wait 20 --json
```

预期分支:

- 如果 remote parse-server 可用:
  - exit code = 0
  - 输出为可直接解析的 JSON
  - JSON 体现 remote/via/privacy/tier 中的部分字段
- 如果 remote parse-server 不可用:
  - exit code != 0
  - 输出包含 remote、parse-server、unavailable、no_engine 或等价可操作错误
  - stdout 必须为可直接解析的 JSON error，且包含 `error.code` 和 `error.message`
  - 不包含 Python traceback

### PARSE-013A remote 默认 tier

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --remote --pages 1~1 --wait 60 --json
```

预期分支:

- 如果 remote parse-server 可用:
  - exit code = 0
  - 输出为可直接解析的 JSON
  - tier 应为 remote 支持的 quality tier；如果 remote 暴露 `high`，默认应为 `high`
  - JSON 体现 remote/via/privacy 中的部分字段
  - 不允许在未声明 fallback 的情况下静默返回本地 flash 内容
- 如果 remote parse-server 不可用:
  - exit code != 0
  - `--json` 输出必须为可直接解析的 JSON error
  - error code/message 包含 remote、parse-server、unavailable、no_engine、quality_tier_unavailable 或等价可操作信息
  - 不包含 Python traceback

### PARSE-013A1 PDF remote high tier

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier high --remote --pages 1~1 --force --wait 180 --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- `parse.tier = high`
- JSON 体现 remote/via/privacy 中的部分字段
- 不允许静默 fallback 到 local flash、local medium、local high 或 remote medium
- 不包含 Python traceback

### PARSE-013B remote no-wait

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --remote --pages 1~1 --no-wait --json
```

预期分支:

- 如果 remote parse-server 可用:
  - exit code = 0
  - 输出为可直接解析的 JSON
  - 输出为已完成内容、pending/parsing 任务状态或明确缓存结果
  - 不长时间阻塞
- 如果 remote parse-server 不可用:
  - exit code != 0
  - 输出包含 remote、parse-server、unavailable 或等价错误
  - stdout 必须为可直接解析的 JSON error，且包含 `error.code` 和 `error.message`
  - 不包含 Python traceback

### PARSE-013C remote force

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --remote --pages 1~1 --force --wait 60 --json
```

预期分支:

- 如果 remote parse-server 可用:
  - exit code = 0
  - 输出为可直接解析的 JSON
  - 不复用旧的 done batch 作为唯一依据，JSON 中应体现新 parse 或 force 后结果
  - 后续普通 remote parse 不因 force 损坏缓存
- 如果 remote parse-server 不可用:
  - exit code != 0
  - 输出包含 remote、parse-server、unavailable 或等价错误
  - stdout 必须为可直接解析的 JSON error，且包含 `error.code` 和 `error.message`
  - 不包含 Python traceback

### PARSE-013D remote output

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --remote --pages 1~1 --wait 60 --output "$MINERU_E2E_FIXTURE_DIR/output-dir/remote-output.md"
```

预期分支:

- 如果 remote parse-server 可用:
  - exit code = 0
  - 输出包含 `Written to` 或等价成功信息
  - stdout 不大量打印完整文档内容
  - 输出路径为命令指定路径
- 如果 remote parse-server 不可用:
  - exit code != 0
  - 输出包含 remote、parse-server、unavailable 或等价错误
  - 不创建空的成功输出文件
  - 不包含 Python traceback

### PARSE-013E remote 与 local cache 隔离

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --remote --pages 1~1 --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --json
```

预期:

- 第一条和第三条 local flash 命令 exit code = 0，tier = flash
- 第二条按 PARSE-013A 的 remote 可用/不可用分支判定
- remote 成功或失败均不得污染 local flash 缓存
- 第三条不应因为第二条 remote 失败而失败

### PARSE-013F remote 配置规则

命令:

```bash
mineru config parsing-rules add "*/sample.pdf" --remote --pages 1~1 --name e2e-remote-rule --json
mineru watch add "$MINERU_E2E_FIXTURE_DIR" --label e2e-remote-rule-watch --json
mineru watch rescan "$MINERU_E2E_FIXTURE_DIR" --wait 60 --json
mineru list parses --limit 20 --json
mineru watch remove "$MINERU_E2E_FIXTURE_DIR" --json
mineru config parsing-rules remove <rule_id>
```

执行说明:

- `<rule_id>` 从第一条 add JSON 中提取。
- watch id 从 `watch add` JSON 中提取；如果 watch 已存在，可复用已有 watch 或先 remove 后重建。

预期分支:

- add/watch/rescan/list/remove exit code = 0，JSON 可解析
- parsing-rule 的 `remote` 字段用于规则命中后的自动解析策略，尤其是 watch 自动触发/后台解析；不要求用户主动 `mineru parse <path>` 在未传 `--remote` 时也按该规则上传远端
- 如果 remote parse-server 可用，规则命中的自动解析任务应体现 remote/privacy/via/tier 中的部分字段，或在 `list parses` 中可观察到对应 remote 解析任务
- 如果 remote parse-server 不可用，规则命中的自动解析任务应记录明确 remote/parse-server 不可用错误；命令带 `--json` 时 stdout 必须为可直接解析的 JSON
- remove 后规则不再影响后续 watch 自动解析

### PARSE-014 非法 format

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --format html
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --format text
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --format json
```

预期:

- 三条 exit code != 0
- 输出包含 invalid choice、not one of、unsupported format 或等价错误
- 不启动解析任务
- 不包含 Python traceback

### PARSE-015 JSON 输出纯净性

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --force --wait 60 --json
```

预期:

- exit code = 0
- stdout 可被 JSON parser 直接解析
- stdout 第一段非空内容必须是 `{` 或 `[`
- stdout 不包含 `Parse status:`
- stdout 不包含 `Parse queued`
- stdout 不包含 `Written to`
- stderr 不包含非错误噪声

## 9. Read 命令

前置: PARSE-003 或 PARSE-004 至少成功一次，并从输出、`show file` 或 `list docs` 获得 doc short id。

如果无法获得 short id，READ 子套件标记为 BLOCKED。

### READ-001 读取 page locator

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --limit 30000
```

预期:

- exit code = 0
- 输出为 Markdown、文本或可读内容
- 不包含 traceback

### READ-002 read JSON

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --json
```

预期:

- exit code = 0
- 输出为 JSON
- JSON 包含 request_scope
- request_scope.locator 包含输入 locator 或规范化 locator

### READ-003 read context

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --context 1 --limit 1000
```

预期:

- exit code = 0
- 不报 `context_not_applicable`，除非 locator 粒度确实不支持 context

### READ-004 invalid locator

命令:

```bash
mineru read "doc:not-a-real-doc/tier:flash/page:1"
```

预期:

- exit code != 0
- 输出包含 not found、invalid 或 locator 相关错误
- 不包含 traceback

### READ-005 输出到文件

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --output "$MINERU_E2E_FIXTURE_DIR/read-output.md"
```

预期:

- exit code = 0
- 输出包含 `Written to` 或等价成功信息
- 不在 stdout 大量打印完整文档内容
- 输出文件路径应为命令指定路径
- 不包含 traceback

### READ-006 JSON 与 output 组合

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --output "$MINERU_E2E_FIXTURE_DIR/read-output-json.md" --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- JSON 顶层保持 `read --json` 的 DocContentResponse 风格结构
- JSON 包含 sha256、short_id、tier、format、request_scope 或等价字段
- 当文件写出成功时，`content` 应为 null
- JSON 顶层包含 `output`
- `output.status` = `written`
- `output.path` 为实际写入路径，建议为绝对路径或规范化绝对路径
- 输出文件存在，路径为命令指定路径
- stdout 不允许包含 `Written to`、Rich 表格、Markdown 正文或其它人类提示文本
- 不包含 Python traceback

### READ-007 image 格式

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --format image --json
```

预期分支:

- 如果该 locator 支持 image 输出:
  - exit code = 0
  - 输出为可直接解析的 JSON
  - JSON 包含 asset、mime_type、path 或等价字段
- 如果该 locator 不支持 image 输出:
  - exit code != 0
  - 输出包含 unsupported、not available、no image 或等价错误
- 不包含 Python traceback

### READ-008 image 输出到文件

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output "$MINERU_E2E_FIXTURE_DIR/read-page.png"
```

预期分支:

- 如果该 locator 支持 image 输出:
  - exit code = 0
  - 输出包含 `Written to` 或等价成功信息
  - 输出路径为命令指定路径
  - 输出文件后缀只允许 `.png`、`.jpg`、`.jpeg`、`.webp`
  - 输出文件真实编码必须与后缀匹配
- 如果该 locator 不支持 image 输出:
  - exit code != 0
  - 输出包含 unsupported、not available、no image 或等价错误
- 不包含 Python traceback

### READ-008A image JSON 与 output 组合

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output "$MINERU_E2E_FIXTURE_DIR/read-page-json.png" --json
```

预期分支:

- 如果该 locator 支持 image 输出:
  - exit code = 0
  - stdout 为可直接解析的 JSON
  - JSON 顶层保持 `read --json` 的 DocContentResponse 风格结构
  - JSON 包含 `asset` 或等价图片元数据
  - `content` 应为 null
  - JSON 顶层包含 `output`
  - `output.status` = `written`
  - `output.path` 为命令指定输出路径的实际写入路径
  - 不应要求调用方依赖服务端临时 asset path
- 如果该 locator 不支持 image 输出:
  - exit code != 0
  - stdout 必须为可直接解析的 JSON error，且包含 `error.code` 和 `error.message`
- 不包含 Python traceback

### READ-009 limit 截断与 next_request

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --limit 20 --json
```

预期:

- exit code = 0
- 输出为可直接解析的 JSON
- 如果内容超过 limit，JSON 应体现 truncated、next_request、cursor、locator 或等价续读信息
- 如果内容未超过 limit，JSON 应明确表示未截断或没有 next_request

### READ-010 使用 next_request 续读

前置: READ-009 返回 next_request.locator。

命令模板:

```bash
mineru read "<next_request.locator>" --limit 30000 --json
```

预期:

- exit code = 0
- 输出为可直接解析的 JSON
- request_scope.locator 体现输入 locator 或规范化 locator
- 不返回与上一段完全相同的起始截断片段，除非服务端明确说明 cursor 已到末尾

### READ-011 no-marker

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --limit 20 --no-marker
```

预期:

- exit code = 0
- 输出不包含 `<!-- Next:`
- 不包含 Python traceback

### READ-012 非法 format

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --format html
mineru read "doc:<short_id>/tier:flash/page:1" --format text
mineru read "doc:<short_id>/tier:flash/page:1" --format json
```

预期:

- 三条 exit code != 0
- 输出包含 invalid choice、not one of、unsupported format 或等价错误
- 不包含 Python traceback

### READ-013 JSON 输出纯净性

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --json
```

预期:

- exit code = 0
- stdout 可被 JSON parser 直接解析
- stdout 第一段非空内容必须是 `{` 或 `[`
- stdout 不包含 `Written to`
- stdout 不包含 Rich 表格、status/info 文本或 Markdown 内容前缀

## 10. Search 与 Find

### SEARCH-001 find by filename

命令:

```bash
mineru find "sample"
```

预期:

- exit code = 0
- 输出包含 `sample.pdf` 或明确 `No results found`
- 若前面 scan/parse 成功，优先期望找到 `sample.pdf`

### SEARCH-002 find JSON

命令:

```bash
mineru find "sample" --json
```

预期:

- exit code = 0
- 输出为 JSON
- 包含 results、total 或等价字段

### SEARCH-003 search 内容

命令:

```bash
mineru search "sample" --limit 10
```

预期:

- exit code = 0
- 输出为搜索结果或 `No results found`
- 不包含 traceback

### SEARCH-004 search tier filter

命令:

```bash
mineru search "sample" --tier flash --limit 10 --json
```

预期:

- exit code = 0
- 输出为 JSON
- 若有结果，结果 tier 应为 flash

### SEARCH-005 search min-tier

命令:

```bash
mineru search "sample" --min-tier flash --limit 10 --json
```

预期:

- exit code = 0
- 输出为 JSON

### SEARCH-006 find ext filter

命令:

```bash
mineru find "sample" --ext pdf --json
mineru find "sample" --ext docx --json
```

预期:

- 两条 exit code = 0
- 两条输出均为可直接解析的 JSON
- pdf 过滤下如果有结果，结果 ext 应为 pdf
- docx 过滤下如果有结果，结果 ext 应为 docx
- 不包含 Python traceback

### SEARCH-007 find limit 边界

命令:

```bash
mineru find "sample" --limit 1 --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- limit 生效，单次返回结果数量不超过 1
- `find` 当前只承诺文件名查询、扩展名过滤、limit 和 JSON 输出；offset 契约单独由 SEARCH-013 覆盖

### SEARCH-008 search type filter

命令:

```bash
mineru search "sample" --type pdf --limit 10 --json
mineru search "sample" --type docx --limit 10 --json
```

预期:

- 两条 exit code = 0
- 两条输出均为可直接解析的 JSON
- 如果有结果，结果文件类型应符合对应 type filter
- 不包含 Python traceback

### SEARCH-009 search offset

命令:

```bash
mineru search "sample" --limit 1 --offset 0 --json
mineru search "sample" --limit 1 --offset 1 --json
```

预期:

- 两条 exit code = 0
- 两条输出均为可直接解析的 JSON
- limit 生效，单次返回结果数量不超过 1
- offset 不导致崩溃

### SEARCH-010 search 无结果

命令:

```bash
mineru search "mineru-e2e-query-that-should-not-exist" --limit 10
mineru search "mineru-e2e-query-that-should-not-exist" --limit 10 --json
```

预期:

- 两条 exit code = 0
- 普通输出包含 `No results found` 或等价空结果提示
- JSON 输出可直接解析
- JSON 中 total 为 0 或 results 为空

### SEARCH-011 find 无结果

命令:

```bash
mineru find "mineru-e2e-filename-that-should-not-exist"
mineru find "mineru-e2e-filename-that-should-not-exist" --json
```

预期:

- 两条 exit code = 0
- 普通输出包含 `No results found` 或等价空结果提示
- JSON 输出可直接解析
- JSON 中 total 为 0 或 results 为空

### SEARCH-012 JSON 输出纯净性

命令:

```bash
mineru find "sample" --json
mineru search "sample" --limit 10 --json
```

预期:

- 两条 exit code = 0
- stdout 可被 JSON parser 直接解析
- stdout 第一段非空内容必须是 `{` 或 `[`
- stdout 不包含 Rich 表格、`No results found` 文本或非 JSON 前缀

## 11. List 命令

### LIST-001 list files

命令:

```bash
mineru list files --limit 20
mineru list files --limit 20 --json
```

预期:

- 两条 exit code = 0
- JSON 输出可解析
- 若前面 scan/parse 成功，应能看到 `sample.pdf` 或相关 file record

### LIST-001A list files filters

命令:

```bash
mineru list files --status active --limit 20 --json
mineru list files --ext pdf --limit 20 --json
mineru list files --limit 1 --offset 1 --json
```

预期:

- 三条 exit code = 0
- 三条输出均为可直接解析的 JSON
- status filter 下如果有结果，结果 status 应为 active
- ext filter 下如果有结果，结果 ext 应为 pdf
- limit/offset 不导致崩溃，返回数量不超过 limit

### LIST-002 list docs

命令:

```bash
mineru list docs --limit 20
mineru list docs --limit 20 --json
```

预期:

- 两条 exit code = 0
- JSON 输出可解析
- 若前面 parse 成功，应能看到对应 doc

### LIST-002A list docs offset

命令:

```bash
mineru list docs --limit 1 --offset 0 --json
mineru list docs --limit 1 --offset 1 --json
```

预期:

- 两条 exit code = 0
- 两条输出均为可直接解析的 JSON
- limit 生效，单次返回结果数量不超过 1
- offset 不导致崩溃

### LIST-003 list parses

命令:

```bash
mineru list parses --tier flash --limit 20
mineru list parses --tier flash --limit 20 --json
```

预期:

- 两条 exit code = 0
- JSON 输出可解析
- 若前面 parse 成功，应有 flash parse 记录

### LIST-003A list parses filters

命令:

```bash
mineru list parses --status done --limit 20 --json
mineru list parses --tier flash --status done --limit 20 --json
mineru list parses --limit 1 --offset 1 --json
```

预期:

- 三条 exit code = 0
- 三条输出均为可直接解析的 JSON
- status filter 下如果有结果，结果 status 应为 done
- tier/status 组合 filter 下如果有结果，结果 tier 应为 flash 且 status 为 done
- limit/offset 不导致崩溃，返回数量不超过 limit

### LIST-004 list scans

命令:

```bash
mineru list scans --limit 20
mineru list scans --limit 20 --json
```

预期:

- 两条 exit code = 0
- JSON 输出可解析
- 若前面 scan/watch rescan 成功，应有 scan 记录

### LIST-004A list scans filters

命令:

```bash
mineru list scans --status done --limit 20 --json
mineru list scans --kind manual --limit 20 --json
mineru list scans --limit 1 --offset 1 --json
```

预期:

- 三条 exit code = 0
- 三条输出均为可直接解析的 JSON
- status filter 下如果有结果，结果 status 应为 done
- kind filter 下如果有结果，结果 kind 应为 manual
- limit/offset 不导致崩溃，返回数量不超过 limit

### LIST-005 watch-id filters

前置: WATCH-003 可获得 watch id。

命令模板:

```bash
mineru list files --watch-id <watch_id> --limit 20 --json
mineru list scans --watch-id <watch_id> --limit 20 --json
```

预期:

- 两条 exit code = 0
- 两条输出均为可直接解析的 JSON
- 如果有结果，结果应关联该 watch id 或该 watch path

### LIST-006 JSON 输出纯净性

命令:

```bash
mineru list files --limit 1 --json
mineru list docs --limit 1 --json
mineru list parses --limit 1 --json
mineru list scans --limit 1 --json
```

预期:

- 四条 exit code = 0
- stdout 可被 JSON parser 直接解析
- stdout 第一段非空内容必须是 `{` 或 `[`
- stdout 不包含 Rich 表格、`No ... found` 文本或非 JSON 前缀

## 12. Show 命令

### SHOW-001 show file

命令:

```bash
mineru show file "$MINERU_E2E_FIXTURE_DIR/sample.pdf"
mineru show file "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --json
```

预期:

- 两条 exit code = 0
- JSON 输出可解析
- 输出包含 filename、size_bytes、page_count 或 parse tiers 中的部分字段

### SHOW-001A show file 不支持 sha256

前置: 从 `mineru show file "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --json` 或 `mineru list files --json` 中获得 sha256。

命令模板:

```bash
mineru show file <sha256>
mineru show file <sha256> --json
```

预期分支:

- 两条 exit code != 0
- `show file` 只支持 file path，不支持 sha256
- 普通输出包含 path、not found、invalid 或等价错误
- `--json` 输出为可直接解析的 JSON error
- 不包含 Python traceback
- 使用 sha256 查询文档应走 `mineru show doc <sha256>`

### SHOW-002 show doc

从 `mineru list docs --json` 中取一个 sha256。

命令模板:

```bash
mineru show doc <sha256>
mineru show doc <sha256> --json
```

预期:

- 两条 exit code = 0
- JSON 输出可解析
- 输出包含 sha256
- show doc 展示关联 files 或 doc metadata

### SHOW-003 show parse

从 `mineru list parses --json` 中取一个 parse id。

命令模板:

```bash
mineru show parse <parse_id>
mineru show parse <parse_id> --json
```

预期:

- 两条 exit code = 0
- 输出包含 parse status、tier、page_range

### SHOW-004 show scan

从 `mineru list scans --json` 中取一个 scan id。

命令模板:

```bash
mineru show scan <scan_id>
mineru show scan <scan_id> --json
```

预期:

- 两条 exit code = 0
- 输出包含 scan status 和统计字段

### SHOW-005 show 不存在资源

命令:

```bash
mineru show parse 999999999
mineru show scan 999999999
mineru show doc 0000000000000000000000000000000000000000000000000000000000000000
mineru show file "$MINERU_E2E_FIXTURE_DIR/not-exist.pdf"
```

预期:

- 四条 exit code != 0
- 输出包含 not found、invalid 或等价错误
- 不包含 Python traceback

### SHOW-006 JSON 输出纯净性

命令模板:

```bash
mineru show file "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --json
mineru show doc <sha256> --json
mineru show parse <parse_id> --json
mineru show scan <scan_id> --json
```

预期:

- 四条 exit code = 0
- stdout 可被 JSON parser 直接解析
- stdout 第一段非空内容必须是 `{` 或 `[`
- stdout 不包含 Rich 表格或非 JSON 前缀

## 13. Invalidate

### INVALIDATE-001 invalidate flash

命令:

```bash
mineru invalidate "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash
```

预期:

- exit code = 0
- 输出包含 `Invalidated` 或 `No done batches found`
- 不删除源文件

### INVALIDATE-001A invalidate all tiers

命令:

```bash
mineru invalidate "$MINERU_E2E_FIXTURE_DIR/sample.pdf"
```

预期:

- exit code = 0
- 输出包含 `Invalidated`、`No done batches found` 或等价结果
- 不删除源文件
- 后续可重新 parse

### INVALIDATE-001B invalidate JSON 支持性

命令:

```bash
mineru invalidate "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --json
```

预期:

- 命令必须有明确行为:
  - 如果支持 JSON，exit code = 0，stdout 为可直接解析的 JSON
  - 如果不支持 JSON，exit code != 0，输出包含 no such option、unknown option 或等价错误
- 不包含 Python traceback

### INVALIDATE-001C invalidate 不存在文件

命令:

```bash
mineru invalidate "$MINERU_E2E_FIXTURE_DIR/not-exist.pdf" --tier flash
```

预期:

- exit code != 0
- 输出包含 file not found、not found 或等价错误
- 不包含 Python traceback

### INVALIDATE-002 invalidate 后重新 parse

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --json
```

预期:

- exit code = 0
- 输出为 JSON
- tier = flash
- 不因 invalidate 后旧状态损坏而失败

## 14. Forget

### FORGET-001 默认 dry-run

命令:

```bash
mineru forget "$MINERU_E2E_FIXTURE_DIR/sample-copy.pdf"
```

预期:

- exit code = 0
- 输出包含 `Would forget`
- 不删除源文件
- 后续 `mineru show file "$MINERU_E2E_FIXTURE_DIR/sample-copy.pdf"` 不因 dry-run 改变状态

### FORGET-002 实际 forget

命令:

```bash
mineru forget "$MINERU_E2E_FIXTURE_DIR/sample-copy.pdf" --no-dry-run
```

预期:

- exit code = 0
- 输出包含 `Forgot`
- 不删除源文件
- 如果后续重新 scan/parse，该路径可被重新发现

### FORGET-003 forget JSON

命令:

```bash
mineru forget "$MINERU_E2E_FIXTURE_DIR/sample-copy.pdf" --json
```

预期:

- exit code = 0
- 输出为 JSON
- 包含 forgotten_files、matched_as、warnings 或等价字段

### FORGET-004 forget 目录 dry-run

命令:

```bash
mineru forget "$MINERU_E2E_FIXTURE_DIR/watch-dir"
```

预期:

- exit code = 0
- 输出包含 `Would forget`
- 不删除源目录或目录内文件
- 后续 `mineru watch list` 或 `mineru scan` 不因 dry-run 破坏状态

### FORGET-005 forget 目录 execute

命令:

```bash
mineru forget "$MINERU_E2E_FIXTURE_DIR/watch-dir" --no-dry-run
```

预期:

- exit code = 0
- 输出包含 `Forgot`
- 不删除源目录或目录内文件
- 后续重新 scan/watch rescan 可重新发现该目录下文件

### FORGET-006 forget 不存在路径

命令:

```bash
mineru forget "$MINERU_E2E_FIXTURE_DIR/not-exist.pdf"
mineru forget "$MINERU_E2E_FIXTURE_DIR/not-exist.pdf" --json
```

预期:

- 两条命令必须有明确行为
- 普通命令如果 exit code = 0，输出必须明确 forgotten_files 为 0 或有 warning
- 普通命令如果 exit code != 0，输出必须包含 not found 或等价错误
- JSON 命令如果 exit code = 0，stdout 必须为可直接解析的 JSON，且明确 forgotten_files 为 0 或有 warning
- JSON 命令如果 exit code != 0，stdout 必须为可直接解析的 JSON error，且包含 `error.code` 和 `error.message`
- 不包含 Python traceback

### FORGET-007 JSON 输出纯净性

命令:

```bash
mineru forget "$MINERU_E2E_FIXTURE_DIR/sample-copy.pdf" --json
```

预期:

- exit code = 0
- stdout 可被 JSON parser 直接解析
- stdout 第一段非空内容必须是 `{` 或 `[`
- stdout 不包含 `Would forget`、`Forgot` 或非 JSON 前缀

## 15. Cleanup

### CLEANUP-001 cleanup deleted-files dry-run

命令:

```bash
mineru cleanup deleted-files
mineru cleanup deleted-files --json
```

预期:

- 两条 exit code = 0
- 默认 dry-run
- JSON 输出可解析

### CLEANUP-002 cleanup deleted-files execute

命令:

```bash
mineru cleanup deleted-files --no-dry-run
```

预期:

- exit code = 0
- 输出包含 `Removed`

### CLEANUP-003 cleanup orphan-docs dry-run

命令:

```bash
mineru cleanup orphan-docs
mineru cleanup orphan-docs --json
```

预期:

- 两条 exit code = 0
- 默认 dry-run
- JSON 输出可解析

### CLEANUP-004 cleanup temp

命令:

```bash
mineru cleanup temp --older-than 7
mineru cleanup temp --older-than 7 --json
```

预期:

- 两条 exit code = 0
- JSON 输出可解析
- 输出包含 removed temp file count 或等价字段

### CLEANUP-005 cleanup deleted-files execute JSON

命令:

```bash
mineru cleanup deleted-files --no-dry-run --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- JSON 包含 removed/deleted/dry_run 或等价字段
- 不包含 Python traceback

### CLEANUP-006 cleanup orphan-docs execute

命令:

```bash
mineru cleanup orphan-docs --no-dry-run
mineru cleanup orphan-docs --no-dry-run --json
```

预期:

- 普通命令 exit code = 0
- 普通输出包含 `Removed`
- JSON 命令 exit code = 0
- JSON 命令 stdout 为可直接解析的 JSON
- JSON 包含 removed/orphan/dry_run 或等价字段
- 不包含 Python traceback

### CLEANUP-007 cleanup temp 边界参数

命令:

```bash
mineru cleanup temp --older-than 0 --json
mineru cleanup temp --older-than -1
```

预期:

- `--older-than 0 --json` exit code = 0，输出为可直接解析的 JSON
- `--older-than -1` exit code != 0
- `--older-than -1` 输出包含 invalid、must be non-negative 或等价错误
- 不包含 Python traceback

### CLEANUP-008 JSON 输出纯净性

命令:

```bash
mineru cleanup deleted-files --json
mineru cleanup orphan-docs --json
mineru cleanup temp --older-than 7 --json
```

预期:

- 三条 exit code = 0
- stdout 可被 JSON parser 直接解析
- stdout 第一段非空内容必须是 `{` 或 `[`
- stdout 不包含 `Would remove`、`Removed` 或非 JSON 前缀

## 16. 扩展与边界场景

本章节补充跨命令契约、路径边界、并发、文件类型和更深层行为验证。除明确说明外，执行 Agent 仍只调用 `mineru` 命令。

### JSONERR-001 read JSON 错误输出纯净性

命令:

```bash
mineru read "doc:not-a-real-doc/tier:flash/page:1" --json
```

预期:

- exit code != 0
- stdout 必须为可直接解析的 JSON error
- JSON 包含 `error.type`、`error.code`、`error.message`
- stdout 不得混合 `Error:`、Rich 文本、traceback 或半截 JSON
- stderr 不得混入额外的人类解释文本

### JSONERR-002 watch JSON 错误输出纯净性

命令:

```bash
mineru watch add "$MINERU_E2E_FIXTURE_DIR/not-exist-dir" --label e2e-missing-watch --json
mineru watch remove "$MINERU_E2E_FIXTURE_DIR/not-a-watch-dir" --json
mineru watch rescan "$MINERU_E2E_FIXTURE_DIR/not-a-watch-dir" --json
```

预期:

- 三条 exit code != 0
- 三条 stdout 均必须为可直接解析的 JSON error
- JSON 包含 `error.type`、`error.code`、`error.message`
- stdout 不得混合 `Error:`、Rich 文本、traceback 或半截 JSON
- stderr 不得混入额外的人类解释文本

### JSONERR-003 show/config/cleanup JSON 错误输出纯净性

命令:

```bash
mineru show parse 999999999 --json
mineru config get not.a.real.key --json
mineru cleanup temp --older-than -1 --json
```

预期:

- 三条 exit code != 0
- 三条 stdout 均必须为可直接解析的 JSON error
- JSON 包含 `error.type`、`error.code`、`error.message`
- stdout 不得混合 `Error:`、Rich 文本、traceback 或半截 JSON
- stderr 不得混入额外的人类解释文本

说明:

- 本用例只覆盖进入命令实现后的业务错误。
- Typer/Click 参数解析阶段错误不在 JSON 错误统一契约范围内。

### JSONERR-004 server 不可连接时 JSON 错误输出纯净性

命令:

```bash
mineru server stop
mineru scan "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --json
mineru list files --limit 1 --json
mineru search "sample" --limit 1 --json
mineru find "sample" --limit 1 --json
mineru server start
```

预期:

- stop exit code = 0
- scan/list/search/find 四条命令 exit code != 0
- scan/list/search/find 四条 stdout 均必须为可直接解析的 JSON error
- JSON 包含 `error.type`、`error.code`、`error.message`
- stdout 不得混合 `Error:`、Rich 文本、traceback 或半截 JSON
- `mineru server status --json` 的未运行状态不适用本错误契约，见 JSONERR-005
- start exit code = 0，用于恢复后续测试环境

### JSONERR-005 server status JSON 未运行状态

命令:

```bash
mineru server stop
mineru server status --json
mineru server start
```

预期:

- stop exit code = 0
- status exit code = 0
- status stdout 为可直接解析的 JSON
- status JSON 至少包含 `running: false`
- status stdout 不包含 `Server is not running.` 或 JSON error
- start exit code = 0，用于恢复后续测试环境

### WATCH-008 watch add JSON

命令:

```bash
mineru watch add "$MINERU_E2E_FIXTURE_DIR/watch-dir" --label e2e-watch-json --json
mineru watch list --json
mineru watch remove "$MINERU_E2E_FIXTURE_DIR/watch-dir" --json
```

预期:

- 三条 exit code = 0
- 三条 stdout 均为可直接解析的 JSON
- add JSON 包含 id、path、label、status 或等价字段
- list JSON 能看到该 watch
- remove JSON 包含 watch_id/removed 或等价字段

### WATCH-009 removable watch

命令:

```bash
mineru watch add "$MINERU_E2E_FIXTURE_DIR/watch-dir" --label e2e-removable --removable --json
mineru watch list --json
mineru watch remove "$MINERU_E2E_FIXTURE_DIR/watch-dir"
```

预期:

- 三条 exit code = 0
- add/list JSON 可直接解析
- watch 记录中 removable 应为 true
- 普通 list 输出中应能体现 removable 或 JSON 中明确体现 removable

### WATCH-010 removable watch unreachable 行为

前置: 需要一个可移除或可临时变为不可访问的 watch fixture；如果当前平台无法稳定制造，标记为 BLOCKED。

命令模板:

```bash
mineru watch add <removable_watch_path> --removable --json
mineru watch rescan <watch_id> --wait 30 --json
mineru watch list --json
```

预期:

- 如果路径可访问，rescan exit code = 0，watch status 为 active
- 如果路径变为不可访问，命令应返回明确 unreachable 状态、scan error 或 watch status=unreachable
- 不包含 Python traceback

### LIST-007 list docs file-type filter

命令:

```bash
mineru list docs --file-type pdf --limit 20 --json
mineru list docs --file-type docx --limit 20 --json
```

预期:

- 两条 exit code = 0
- 两条 stdout 均为可直接解析的 JSON
- 如果有结果，结果 file_type 应符合过滤条件
- 不包含 Python traceback

### SEARCH-013 find offset 契约

命令:

```bash
mineru find "sample" --limit 1 --offset 1 --json
```

预期:

- 如果 CLI 设计支持 find offset，exit code = 0，stdout 为可直接解析的 JSON，返回数量不超过 1
- 如果 CLI 设计不支持 find offset，exit code != 0，输出包含 no such option、unknown option 或等价错误
- 不包含 Python traceback

### PARSE-016 page range 边界

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages all --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 2~1 --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 999~1000 --wait 60 --json
```

预期:

- `--pages all` exit code = 0，stdout 为可直接解析的 JSON
- 反向 page range exit code != 0 或返回明确 page_range_invalid/error JSON
- 超出页码 exit code != 0 或返回明确 page_range_invalid/not found/empty range 错误
- 不包含 Python traceback

### READ-014 locator 粒度边界

前置: 从成功 parse/read JSON 中获得 block/char locator 或 next_request locator。

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1/block:1" --json
mineru read "doc:<short_id>/tier:flash/page:1/block:1/char:1" --json
mineru read "doc:<short_id>/tier:flash/page:999" --json
mineru read "doc:<short_id>/tier:flash/page:1/block:999" --json
```

预期:

- 合法 block/char locator exit code = 0，stdout 为可直接解析的 JSON
- 超出 page/block locator exit code != 0，stdout 必须为可直接解析的 JSON error
- request_scope.locator 体现输入 locator 或规范化 locator
- 不包含 Python traceback

### READ-015 跨页 context

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --context 2 --limit 30000 --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- request_scope.context = 2 或等价字段体现 context 生效
- 如果文档页数不足，命令应优雅返回可用范围，不报 traceback

### FILETYPE-001 文档类输入文件

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.docx" --tier flash --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pptx" --tier flash --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.xlsx" --tier flash --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.md" --tier flash --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.txt" --tier flash --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.csv" --tier flash --wait 60 --json
```

预期:

- 六条命令均 exit code = 0
- stdout 均为可直接解析的 JSON
- JSON 中 content 或 doc metadata 合理存在
- 不包含 Python traceback

### FILETYPE-001A 图片输入

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.jpeg" --tier flash --wait 60 --json
```

预期分支:

- 如果当前安装支持该文件类型，exit code = 0，stdout 为可直接解析的 JSON，content 或 doc metadata 合理存在
- 如果当前安装不支持该文件类型，exit code != 0，输出包含 unsupported、file_type_unsupported、no_engine 或等价错误
- 不包含 Python traceback

### FILETYPE-002 损坏、空、不可读文件

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/corrupted.pdf" --tier flash --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/empty.pdf" --tier flash --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/no-read.pdf" --tier flash --wait 60 --json
```

预期:

- 三条 exit code != 0，除非当前实现对空文件有明确可接受语义
- 输出包含 file_corrupted、parse_empty、file_permission_denied、file_not_found、unsupported 或等价错误
- 失败命令带 `--json` 时，stdout 必须为可直接解析的 JSON error
- 如果 `no-read.pdf` 权限场景无法稳定制造，可标记该子项 BLOCKED
- 不包含 Python traceback

### PATH-001 路径字符边界

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample doc.pdf" --tier flash --pages 1~1 --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/中文样例.pdf" --tier flash --pages 1~1 --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/symlink-sample.pdf" --tier flash --pages 1~1 --wait 60 --json
```

预期:

- 文件名含空格和中文的两条命令 exit code = 0，stdout 为可直接解析的 JSON
- symlink 命令如果平台支持 symlink，exit code = 0；如果不支持或 fixture 不存在，标记 BLOCKED
- JSON 中 source/path/request_scope 不应乱码
- 不包含 Python traceback

### PATH-002 相对路径和 home 路径

命令:

```bash
mineru parse "sample.pdf" --tier flash --pages 1~1 --wait 60 --json
mineru show file "sample.pdf" --json
mineru parse "~/mineru-e2e-test/sample.pdf" --tier flash --pages 1~1 --wait 60 --json
```

预期:

- 在测试 HOME 下执行时，三条 exit code = 0
- stdout 均为可直接解析的 JSON
- show/parse 应解析到同一个实际文件或等价 canonical path
- CLI 必须主动对输入 path 中的 `~` 执行 `expanduser`
- 不能把带引号的 `"~/mineru-e2e-test/sample.pdf"` 当作当前目录下的字面量相对路径
- 不包含 Python traceback

### OUTPUT-001 output 边界

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --output -
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --output "$MINERU_E2E_FIXTURE_DIR/output-dir/existing.md"
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --output "$MINERU_E2E_FIXTURE_DIR/not-exist-dir/out.md"
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --output "~/mineru-e2e-test/output-dir/home-output.md"
```

预期:

- `--output -` 应等价于 stdout 输出，exit code = 0，不创建名为 `-` 的文件
- 输出到已有文件应有明确覆盖或拒绝策略；若覆盖，exit code = 0 且输出包含 `Written to`
- 输出到不存在父目录应自动创建父目录并成功，exit code = 0
- 不存在父目录下的目标文件必须存在，且不为空
- 输出路径中的 `~` 必须由 CLI 主动 `expanduser`，不能创建字面量 `~` 目录
- 不包含 Python traceback

### OUTPUT-001A output 与 JSON 边界

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --wait 60 --output - --json
mineru read "doc:<short_id>/tier:flash/page:1" --output - --json
```

预期:

- 两条 exit code = 0
- stdout 均为可直接解析的 JSON
- `--output - --json` 等价于不指定 `--output`
- JSON 中不新增 `output` 字段
- 不创建名为 `-` 的文件
- stdout 不包含 `Written to`、Rich 表格、Markdown 正文或其它人类提示文本

补充分支:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output - --json
```

预期:

- exit code != 0
- stdout 为 JSON error envelope
- `error.code` = `image_output_extension_unsupported`
- `error.param` = `output`
- 不创建名为 `-` 的文件

### OUTPUT-002 read image output 后缀边界

命令模板:

```bash
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output "$MINERU_E2E_FIXTURE_DIR/output-dir/page.png"
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output "$MINERU_E2E_FIXTURE_DIR/output-dir/page.jpg"
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output "$MINERU_E2E_FIXTURE_DIR/output-dir/page.jpeg"
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output "$MINERU_E2E_FIXTURE_DIR/output-dir/page.webp"
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output "$MINERU_E2E_FIXTURE_DIR/output-dir/page"
mineru read "doc:<short_id>/tier:flash/page:1" --format image --output "$MINERU_E2E_FIXTURE_DIR/output-dir/page-as-md.md"
```

预期:

- `.png`、`.jpg`、`.jpeg`、`.webp` 均 exit code = 0
- 输出文件真实编码必须分别匹配 PNG、JPEG、JPEG、WebP
- 无后缀和 `.md` 均 exit code != 0
- JSON 模式下错误为 JSON error envelope，`error.code` = `image_output_extension_unsupported`
- 不包含 Python traceback

### CONCURRENCY-001 并发 parse force

执行方式:

- 启动两个独立 shell，同时执行同一条命令。
- 执行 Agent 仍只调用 `mineru` 命令，不直接访问数据库。

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --force --wait 60 --json
mineru parse "$MINERU_E2E_FIXTURE_DIR/sample.pdf" --tier flash --pages 1~1 --force --wait 60 --json
```

预期:

- 两个进程均不出现 Python traceback
- 至少一个命令成功；另一个可成功、等待已有任务、或返回明确 lock/queue/busy 错误
- 不允许数据库损坏、后续普通 parse 失败或永久 pending

### CONCURRENCY-002 并发 scan 同一目录

执行方式:

- 启动两个独立 shell，同时执行同一条命令。

命令:

```bash
mineru scan "$MINERU_E2E_FIXTURE_DIR" --wait 30 --json
mineru scan "$MINERU_E2E_FIXTURE_DIR" --wait 30 --json
```

预期:

- 两个进程均不出现 Python traceback
- 两条命令 exit code = 0 或其中一条返回明确 busy/lock/queue 错误
- 后续 `mineru list scans --json` 可正常返回

### CONCURRENCY-003 并发 server start

执行方式:

- 确保 server 已停止。
- 启动两个独立 shell，同时执行 `mineru server start`。

命令:

```bash
mineru server start
mineru server start
```

预期:

- 至少一个命令 exit code = 0
- 另一个命令 exit code = 0 且输出 already running，或 exit code != 0 且输出可读的 socket/busy/started by another process 错误
- 最终 `mineru server status --json` 表示只有一个可用 server
- 不包含 Python traceback

### CONFIG-010 配置持久化

命令:

```bash
mineru config set parse_server.local.managed_tier high
mineru server restart
mineru config get parse_server.local.managed_tier --json
mineru config unset parse_server.local.managed_tier
```

预期:

- 四条 exit code = 0
- restart 后 get JSON 仍体现 `high`，并且 source 应体现 override 或等价覆盖来源
- unset 后恢复默认配置来源
- 不包含 Python traceback

### CONFIG-011 rule hit_count 与 scan 影响

命令:

```bash
mineru config exclude-rules add "*/unsupported.bin" --priority 100 --json
mineru scan "$MINERU_E2E_FIXTURE_DIR" --wait 30 --json
mineru config exclude-rules list --json
mineru config exclude-rules remove <rule_id>
```

预期:

- 四条 exit code = 0
- scan JSON 中 excluded 或 unsupported 计数符合 exclude 语义
- list JSON 中该 rule 的 hit_count 应增加，或当前实现明确不统计 scan hit_count
- remove 后规则不再出现在 list 中

### CLEANUP-009 cleanup deleted-files 真实效果

前置: 通过 `mineru forget "$MINERU_E2E_FIXTURE_DIR/sample-copy.pdf" --no-dry-run` 或其它 CLI 操作制造可清理记录。

命令:

```bash
mineru cleanup deleted-files --no-dry-run --json
mineru list files --json
```

预期:

- 两条 exit code = 0
- cleanup JSON 可直接解析
- list files 中不再包含已清理的 deleted 记录，或 cleanup JSON 明确 deleted_files=0
- 不包含 Python traceback

### CLEANUP-010 cleanup orphan-docs 真实效果

前置: 通过 forget 所有关联文件制造 orphan doc；如果无法只通过 CLI 稳定制造，标记 BLOCKED。

命令:

```bash
mineru cleanup orphan-docs --no-dry-run --json
mineru list docs --json
```

预期:

- 两条 exit code = 0
- cleanup JSON 可直接解析
- orphan doc 不再出现在 list docs 中，或 cleanup JSON 明确 orphan_docs=0
- parsed 输出目录不应再被 read/parse 错误引用
- 不包含 Python traceback

### CLEANUP-011 cleanup temp 新旧文件边界

前置: 测试环境预置 doclib temp 目录中的过期文件和未过期文件；如果无法只通过 CLI 稳定制造，标记 BLOCKED。

命令:

```bash
mineru cleanup temp --older-than 7 --json
```

预期:

- exit code = 0
- stdout 为可直接解析的 JSON
- 过期 temp 文件被清理
- 未过期 temp 文件不被清理
- 不包含 Python traceback

### SERVER-012 error summary 与 recent logs

命令:

```bash
mineru parse "$MINERU_E2E_FIXTURE_DIR/corrupted.pdf" --tier flash --wait 20 --json
mineru server status --json
```

预期:

- 第一条按 FILETYPE-002 损坏 PDF 分支判定
- 第二条 exit code = 0，stdout 为可直接解析的 JSON
- status JSON 中 `app_logs` 应来自 `MINERU_DOCLIB_LOG_APP_PATH`，或由 `MINERU_DOCLIB_LOG_DIR` 派生的 app log，最多返回最后 25 条
- status JSON 中 `access_logs`、`stderr_logs`、`stdout_logs` 应分别来自对应日志文件，最多返回最后 10 条
- `recent_logs` 保留为兼容字段，内容应等同于最后 25 条 app log
- 如果错误被记录，error_summary 中应出现 parse/file 错误计数
- 如果当前实现不记录该错误，记录为 fail/issue，不作为 BLOCKED

## 17. 停止 server

### SERVER-010 停止 server

命令:

```bash
mineru server stop
mineru server status
```

预期:

- stop exit code = 0
- status exit code = 0
- status 输出 `Server is not running` 或等价信息

### SERVER-011 启动失败时保留子进程 stderr

执行顺序:

- 本用例必须作为整套 E2E 的最后一个测试执行。
- 本用例会故意制造 server 启动失败，执行后无需恢复测试 server。

执行方式:

- 人工制造一个不会成功建立任何 local transport 的 server 启动环境。可选方式:
  - 设置 `MINERU_DOCLIB_UDS_ENABLED=false` 且 `MINERU_DOCLIB_TCP_ENABLED=false`。
  - 或显式设置 `MINERU_DOCLIB_UDS_ENABLED=true`，并把 `MINERU_DOCLIB_UDS_PATH` 指向一个父目录不存在且不可创建的位置。
- 只允许通过环境变量制造失败，不直接修改数据库或内部状态。

命令:

```bash
mineru server start
```

预期:

- exit code != 0
- 输出包含 `See log:` 或等价日志路径提示
- 日志路径应为 `MINERU_DOCLIB_LOG_APP_PATH` / `MINERU_DOCLIB_LOG_STDERR_PATH` 指向的文件，或由测试 HOME 下 `logs/` 派生的 mineru log
- 日志文件中能看到 server 子进程 stderr 或异常信息
- stdout/stderr 不只给出空泛的启动失败

## 18. 补充覆盖执行套件

本章节用于补足阶段性执行中容易漏跑的场景。执行 Agent 在完成核心 smoke 与新契约回归后，必须按以下批次继续执行，除非对应前置条件无法满足并明确标记 BLOCKED。

### COVERAGE-001 help 与入口补充

必跑 case:

- CLI-001 顶层 help
- CLI-002 子命令 help
- CLI-003 不存在旧命令

预期:

- 所有 help 类命令输出稳定、可读
- 顶层 help 不出现 `mineru-kit`
- 旧命令入口保持不可用

### COVERAGE-002 server 生命周期补充

必跑 case:

- SERVER-003 查询运行状态
- SERVER-005 重复启动 server
- SERVER-006 restart server
- SERVER-007 stop 后依赖 server 的命令报错可读
- SERVER-009 环境变量路径生效
- SERVER-013 endpoint discovery 文件写入
- SERVER-014 TCP-only fallback
- SERVER-015 endpoint stale 清理
- SERVER-012 error summary 与 recent logs
- SERVER-010 停止 server

执行要求:

- SERVER-011 必须仍作为整套测试最后一个 case 执行，不放入本批次中间。
- stop/restart 或独立 `MINERU_HOME` transport 测试后必须恢复主测试 server，避免影响后续批次。

### COVERAGE-003 config 普通文本与错误分支补充

必跑 case:

- CONFIG-001 查看配置
- CONFIG-003 设置和读取配置
- CONFIG-004 unset 配置
- CONFIG-005 exclude-rules
- CONFIG-006 旧 exclude 命令不可用
- CONFIG-007 parsing-rules
- CONFIG-009 不存在的配置 key

预期:

- 普通文本输出可读且不包含 traceback
- JSON 分支遵守 JSON error 契约
- add/list/remove 后配置状态一致

### COVERAGE-004 watch 普通文本、重复与 removable 补充

必跑 case:

- WATCH-001 添加 watch
- WATCH-001A 重复添加 watch
- WATCH-002 列出 watch
- WATCH-004 watch rescan
- WATCH-006 删除不存在的 watch
- WATCH-007 添加不存在目录
- WATCH-009 removable watch
- WATCH-010 removable watch unreachable 行为

执行要求:

- WATCH-010 如果平台无法稳定制造路径不可访问，可标记 BLOCKED，但必须说明原因。
- watch add/remove 需要保持清理，避免污染后续 list/watch-id 用例。

### COVERAGE-005 scan 边界补充

必跑 case:

- SCAN-003 扫描不存在路径
- SCAN-004 扫描空目录
- SCAN-005 扫描不支持文件类型

预期:

- 错误分支清晰，不包含 traceback
- JSON 模式下业务错误输出 JSON error
- 空目录和不支持类型的行为符合用例分支说明

### COVERAGE-006 parse 边界、续读、cache 与 remote 组合补充

必跑 case:

- PARSE-001 文件不存在
- PARSE-002 显式 flash parse
- PARSE-004 cache hit 行为
- PARSE-005 no-wait 行为
- PARSE-006 force 行为
- PARSE-006A force 默认输出不打印过程 status
- PARSE-006B verbose 输出允许过程 status
- PARSE-007 默认 tier 行为
- PARSE-007A PDF local medium tier
- PARSE-007B PDF local high tier
- PARSE-008 输出到文件
- PARSE-010 limit 截断与 next_request
- PARSE-011 after 续读
- PARSE-012 no-marker
- PARSE-013 remote parse 分支
- PARSE-013A remote 默认 tier
- PARSE-013A1 PDF remote high tier
- PARSE-013B remote no-wait
- PARSE-013C remote force
- PARSE-013D remote output
- PARSE-013E remote 与 local cache 隔离
- PARSE-013F remote 配置规则
- PARSE-015 JSON 输出纯净性
- PARSE-016 page range 边界

执行要求:

- remote 不可用时，相关命令必须返回 JSON error 或普通可读错误，不能 traceback。
- remote 可用时，必须验证 remote/via/privacy/tier 中至少部分字段。
- PARSE-013A1 是 remote high 硬性测试；remote 不可用或不支持 high 均记录为失败，不能静默 fallback 到 local 或其它 tier。
- force/cache/no-wait 用例必须记录 parse id/status 是否符合预期。

### COVERAGE-007 read 边界、续读、image 与 context 补充

必跑 case:

- READ-001 读取 page locator
- READ-003 read context
- READ-004 invalid locator
- READ-005 输出到文件
- READ-007 image 格式
- READ-008 image 输出到文件
- READ-009 limit 截断与 next_request
- READ-010 使用 next_request 续读
- READ-011 no-marker
- READ-012 非法 format
- READ-013 JSON 输出纯净性
- READ-014 locator 粒度边界
- READ-015 跨页 context

执行要求:

- 如果无法从解析结果获得 block/char locator，READ-014 中对应子项可标记 BLOCKED，但 page 越界子项仍必须执行。
- image 不支持时必须按 JSON error 或普通可读错误判定，不得 traceback。

### COVERAGE-008 search/list/show 过滤、分页与不存在资源补充

必跑 case:

- SEARCH-002 find JSON
- SEARCH-004 search tier filter
- SEARCH-005 search min-tier
- SEARCH-006 find ext filter
- SEARCH-007 find limit 和 offset 边界
- SEARCH-008 search type filter
- SEARCH-009 search offset
- SEARCH-010 search 无结果
- SEARCH-011 find 无结果
- SEARCH-012 JSON 输出纯净性
- SEARCH-013 find offset 契约
- LIST-001A list files filters
- LIST-002A list docs offset
- LIST-003A list parses filters
- LIST-004A list scans filters
- LIST-005 watch-id filters
- LIST-006 JSON 输出纯净性
- LIST-007 list docs file-type filter
- SHOW-001A show file 不支持 sha256
- SHOW-005 show 不存在资源
- SHOW-006 JSON 输出纯净性

执行要求:

- `show doc` 必须使用完整 sha256，不使用 short_id。
- `show file` 必须使用 file path；sha256 查询必须使用 `show doc`。
- 所有 JSON 输出必须可直接 `json.loads`。

### COVERAGE-009 invalidate 与 forget 补充

必跑 case:

- INVALIDATE-001 invalidate flash
- INVALIDATE-001A invalidate all tiers
- INVALIDATE-001B invalidate JSON 支持性
- INVALIDATE-001C invalidate 不存在文件
- INVALIDATE-002 invalidate 后重新 parse
- FORGET-001 默认 dry-run
- FORGET-002 实际 forget
- FORGET-003 forget JSON
- FORGET-004 forget 目录 dry-run
- FORGET-005 forget 目录 execute
- FORGET-006 forget 不存在路径
- FORGET-007 JSON 输出纯净性

执行要求:

- invalidate/forget 不得删除源文件。
- 实际 forget 后必须验证重新 scan/parse 可重新发现。

### COVERAGE-010 cleanup 真实副作用与边界补充

必跑 case:

- CLEANUP-002 cleanup deleted-files execute
- CLEANUP-005 cleanup deleted-files execute JSON
- CLEANUP-006 cleanup orphan-docs execute
- CLEANUP-007 cleanup temp 边界参数
- CLEANUP-008 JSON 输出纯净性
- CLEANUP-009 cleanup deleted-files 真实效果
- CLEANUP-010 cleanup orphan-docs 真实效果
- CLEANUP-011 cleanup temp 新旧文件边界

执行要求:

- `--no-dry-run` 用例执行前必须先制造可清理数据。
- 清理后必须通过 list/show/read/parse 复核真实效果。

### COVERAGE-011 文件类型、路径与 output 边界补充

必跑 case:

- FILETYPE-001 文档类输入文件
- FILETYPE-001A 图片输入
- FILETYPE-002 损坏、空、不可读文件中尚未执行的 empty/no-read 子项
- PATH-001 路径字符边界
- PATH-002 相对路径和 home 路径中相对路径、show file 同源子项
- OUTPUT-002 read image output 后缀边界

执行要求:

- docx/pptx/xlsx/md/txt/csv 输入为全量 E2E 必测项，不支持时记录为失败。
- image 输入如果当前安装不支持，按预期失败分支判定。
- symlink 或 no-read 权限场景无法稳定制造时，相关子项可 BLOCKED，但必须说明平台限制。

### COVERAGE-012 并发补充

必跑 case:

- CONCURRENCY-001 并发 parse force
- CONCURRENCY-002 并发 scan 同一目录
- CONCURRENCY-003 并发 server start

执行要求:

- 并发命令只能调用 `mineru`，不得直接操作数据库。
- 允许一个命令返回 busy/lock/queue 类明确错误，但不得出现 traceback 或损坏后续 server 状态。

### COVERAGE-013 配置持久化与规则影响补充

必跑 case:

- CONFIG-010 配置持久化
- CONFIG-011 rule hit_count 与 scan 影响

预期:

- server restart 后配置仍可读取
- rule hit_count 或等价统计在 scan 后体现规则被使用

### COVERAGE-014 telemetry 补充

必跑 case:

- TELEMETRY-001 telemetry help
- TELEMETRY-002 status JSON 结构
- TELEMETRY-003 preview JSON 结构
- TELEMETRY-004 enable / disable 与 installation_id 稳定性
- TELEMETRY-005 disabled 时不新增聚合
- TELEMETRY-006 unset 或 disabled 时 flush 不外发
- TELEMETRY-007 enabled 时 preview 可观察业务聚合
- TELEMETRY-008 enabled flush 行为
- TELEMETRY-010 preview 隐私边界

条件必跑 case:

- TELEMETRY-009 首次交互式 prompt：仅在执行 Agent 能使用真实交互式终端时必跑；否则标记 BLOCKED，并说明执行环境不支持交互输入。

执行要求:

- telemetry E2E 只能通过 `mineru telemetry ...` 和其它 `mineru` 业务命令观察结果，不得直接读取 SQLite 或调用内部 API。
- 外部 telemetry endpoint 不可用时，flush 用例按失败/保留待上报数据分支判定，不应因此整体失败。
- 所有 preview 隐私边界检查必须确认不包含路径、文件名、query、snippet、正文、traceback 或 API key。

## 19. 最终报告格式

执行 Agent 最终必须按以下格式汇总:

```text
Summary:
- total: N
- passed: N
- failed: N
- blocked: N

Failures:
1. [case id] command
   expected:
   actual exit code:
   actual stdout:
   actual stderr:
   analysis:

Blocked:
1. [case id] reason

Environment:
- mineru version: <如果 help/status 输出能看到则填写，否则 unknown>
- fixture dir: ...
- quality tier available: <available quality tiers>
- remote high available: yes
- pdf fixture source: ...
```
