# 代码结构优化任务状态

状态：**本轮结构优化及合入收尾全部完成；PR #5665 已合入 dev，合入前后跨平台 CI、文档和 CLA 验证通过，保留一项非阻断后续项**。记录日期：2026-10-25。

## 工作区与改动边界

- 仓库：`/Volumes/2TB/projects/20240809magic_pdf/Magic-PDF`
- 分支：`codex/architecture-lifecycle-refactor`；HEAD：`968ee71a6bfc802f92345f51f6554e0845bf606d`，基于 `ec6c23cff`。
- 前八批、集中复审 5 文件修复及单文件测试等待预算调整均已推送至 upstream 同名分支；各批验收 wheel 已保存，未进行合并或 PyPI 发布。
- 用户原有改动必须保留：`tests/unittest/test_pdf_text_styles.py`（当前相对 HEAD 为 49 行新增、7 行删除）和未跟踪的 `examples/`。本任务没有修改它们。
- 新增生产模块和测试均已纳入提交。本状态文件为本地检查点；`examples/` 和用户原有测试改动未混入提交。
- Python 环境：`/Users/myhloli/projects/20240809magic_pdf/Magic-PDF/.venv4`。命令在本仓库根目录执行。

## 第一批：已完成并验证

- SQLite 连接级参数在每次 `_connect()` 时应用；WAL 仅在初始化时设置。
- 配置导入不再读取文件或校验环境；使用 `get_config()` 首次装载进程默认配置，`load_config()` 独立装载。
- 删除旧模块级 `config` 实例，生产代码、脚本和测试已迁移；依赖 `MINERU_HOME` 的默认路径改为构造时解析。
- Doclib 应用保存配置副本，日志默认级别取自该应用配置。
- 非 Torch 回归累计 4013 项通过（包含 6 项修正后重跑），3 项跳过。

## 第二批：已完成并验证

- `mineru/tasks/` 分离 contracts、errors、execution、service、http；本地适配位于 `parser/task_backend.py`。
- 删除旧 `parser/task_api.py`，Router 不再依赖本地 API Server。
- SDK/file_parse 共享文件身份、上传描述和解析选项，保留各自响应接受边界和恢复语义。
- 新增 `doclib/services/content_read_svc.py`，承担定位、渐进 Markdown、图片读取与导出。
- `doclib/core/parse_artifacts.py` 统一缓存路径、读写、页选择和 coverage。
- 历史协议读取入口固定为 `doclib/core/middle_json.py:read_cached_middle_json()`。
- 16 个协议模型 JSON Schema、本地 API 与 Router 完整 OpenAPI 和改动前一致。
- 完整非 Torch 回归：4044 passed、3 skipped、108 deselected。

## 第三批：已完成并通过最终本地验收

### 单文件执行

- `TaskExecution` 为有类型的单文件快照，`TaskBackend.submit()` 接收一个文件和可选上传。
- 本地使用 `ExecutionStore`、`_ExecutionRecord`、`_ExecutionRequest`、`_run_execution`；旧多文件 Job 模型、循环和索引映射已移除。
- 父 Task 负责批次进度和 `partial` 聚合；公开结果 ID 与内部源文件 ID 分离。
- 并发、准备限流、取消排空、结果保留期及公开 usage 字段语义保持。
- 本地 API 与 Router 的完整 OpenAPI 再次与第三批前基线一致。

### 模型生命周期

- `model/runtime/ownership.py` 提供应用、文档和初始化租约，以及统一 `release_runtime_owner()`。
- Hybrid/Atom 通过 `local_cache.py` 记录应用和模型依赖，支持共享、CLI 固定缓存及初始化失败回滚。
- VLM `RuntimeHandle` 在创建时绑定明确的关闭动作，移除动态私有句柄和关闭方法猜测。
- 关闭等待已接纳文档和初始化；失效应用的迟到结果不进入缓存。
- VLM 构造、正常关闭及失败清理在缓存锁外执行；同键初始化/关闭有等待屏障。
- `closing.py` 的关闭回调上下文拒绝阻塞式重入，覆盖同步、异步线程及直接句柄关闭。
- `RuntimeHandle.aclose()` 被重复取消后仍排空客户端和引擎清理，再传播最早观察到的异常。
- MLX 串行等待可取消，已开始的同步推理先排空再归还锁。
- 小模型只承诺租约正确和宿主引用释放，没有宣称同键原生析构完全串行或 RSS 峰值改善。

### Gradio 编排

- `conversion_service.py` 接收显式依赖并返回具名转换结果。
- `conversion_ui.py` 生成既有组件更新和浏览器回执。
- 公开 4 入参、16 项输出和浏览器 15 项协议不变；保留会话隔离、迟到响应过滤和取消排空。
- 会话不持有 `ParseResult`；有弱引用回归确认转换完成后可回收。

### 第三批已完成的验证

- Gradio 相关 Python：431 passed、3 skipped；7 个 Node 回归脚本通过。
- 模型扩展聚焦曾通过 212 项；最终关闭竞争修正后，模型聚焦 70 项通过。
- 新增 `test_runtime_shutdown_regressions.py` 的 10 项有超时保护回归全部通过；连同 ownership/native async/API 集成，共 48 passed。
- `test_api_model_ownership.py` 使用真实 Tasks API 和微型 CPU ONNX 会话，验证两个应用共享、单个退出不误释放、最后退出后回收，以及新应用重新构造。无需下载模型。
- 单文件执行经过独立源码复审；身份、取消、保留期、用量及 Router 续租未发现剩余阻断。
- 最后静态检查：全仓 E9/F63/F7/F82 通过；累计 124 个改动 Python 文件没有新增 Ruff 诊断，67 条为对照 HEAD 的既有诊断；`git diff --check` 通过。
- 此阶段未运行 GPU 大模型；后续已补充 llama.cpp/Metal 实机验收，详见交付记录。LMDeploy/vLLM/MLX 仍限于源码契约及替身验证。

## 最终验收（恢复任务后）

用户此前要求暂停时，第三批整套回归在 898 passed、108 deselected 处被 KeyboardInterrupt 停止。
恢复后重新执行完整命令，最终结果为：

- **4080 passed、3 skipped、108 deselected，exit 0**，耗时 203.31 秒。
- 继续排除用户正在修改的 `tests/unittest/test_pdf_text_styles.py`；108 项为本轮非 Torch 筛选未选中的测试。
- 当前本地 API、Router 完整 OpenAPI 与第三批前基线分别完全一致，各 11 条路径。
- 全仓 E9/F63/F7/F82 通过；124 个改动 Python 文件对照 HEAD 没有新增 Ruff 诊断，67 条为既有诊断。
- 16 个新增生产 Python 文件的函数、方法均有中文职责注释；`git diff --check` 通过。
- 本轮观察到 63 条警告，包括已确认的 Gradio 事件循环回收基线问题，详见下节。
- 三批约定的实现与本地回归没有剩余任务。后续提交、安装包、跨平台和实机验收结果见下节。

最终验收命令：

```bash
/Users/myhloli/projects/20240809magic_pdf/Magic-PDF/.venv4/bin/python -m pytest \
  -q -o addopts='' --strict-markers -m 'not torch' \
  --ignore=tests/unittest/test_pdf_text_styles.py tests/unittest
```

后续若继续修改，应针对新改动选择回归范围。提交时保留用户原有改动，并确认新增未跟踪模块和测试均被正确纳入。

## 交付收尾与实机验收

### 提交与 CI

- 提交：`afda7189de1debaf6482a1f430571d6ffeda267e`，标题为 `refactor: clarify task, document and model lifecycle boundaries`。
- 分支：`upstream/codex/architecture-lifecycle-refactor`。
- [跨平台 CI](https://github.com/opendatalab/MinerU/actions/runs/40171000355)：**16 个任务全部成功**，检查的 head SHA 与上述提交一致。
- Linux Python 3.11–3.14：每组 4090 passed、6 skipped（JUnit 共 4096 项）。
- Windows/macOS Python 3.11–3.14：每组 123 项公开合同测试通过。
- Ubuntu Torch CPU：106 项通过；本地 CI helper 测试另有 24 项通过。
- 用户修改中的 PDF 样式测试未提交；额外提取其 HEAD 原始版本，在安装环境验证 10 项全部通过。

### wheel 与完整隔离安装

- 工作盘当时仅剩约 55 MiB，构建快照和环境放在内置盘临时目录，未清理用户工作盘文件。
- wheel：`mineru-4.0.12-py3-none-any.whl`，3,774,304 字节。
- SHA-256：`14bb9b9746123b7aa4f9fc290a505fe5d2a4ec0387ec99d0d19acf4006ea2dd2`。
- 658 个包内文件逐字节匹配已提交源码；包含所有新增生产模块，不包含旧 `parser/task_api.py`。
- 最终在完全隔离的 Python 3.14.4 环境安装所有依赖；126 个包通过 `uv pip check`，没有依赖开发环境的 site-packages。
- DocVortex 使用 PyPI 二进制 wheel 0.6.1，没有源码构建。最终主要依赖包括 Torch 2.14.1、Gradio 6.30.0、ONNX Runtime 1.31.0、mineru-llama-cpp 0.2.0。
- 安装后验证：两页真实 PDF 原生解析、九种渲染格式、结果包保存、Tasks 创建/上传/解析/下载/取消、Gradio 4 入参/16 输出、四个 CLI 的 `--help` 均通过。
- 已核对实际导入路径来自安装环境，并验证原始语义树未被渲染修改。

### Metal 实机模型生命周期

- 后端：`llama-cpp-engine`，原生日志确认 `MTL0 (Apple M4), backend=MTL`。
- 使用本地 MinerU2.5-Pro-2605-1.2B Q8_0 GGUF 与 mmproj，未下载权重；这项验收针对 VLM 资源层，不代表完整 OCR/Hybrid 模型集验收。
- 两个 `RuntimeOwner` 共享同一模型实例；释放第一个后，第二个仍能正常提取图中文字。
- 已开始的两页同步推理被取消后排空，再传播取消；在干净环境中等待约 12.72 秒，随后同一实例继续成功推理。
- 最后持有者退出后缓存条目归零、原生引擎已关闭；新持有者能重新加载、推理并正常关闭。
- 进程 Metal 驱动分配量：推理后约 1.50 GiB，最后持有者关闭后 737,280 字节（约 0.70 MiB）；重建关闭后保持该值。
- RSS 从推理后约 1.59 GiB 降到约 428 MiB，仍包含进程及框架缓存；未宣称 RSS 完全恢复到启动值。

### 本地交付产物

- 目录：`/Users/myhloli/Downloads/MinerU-architecture-afda7189-0d9iu402`。
- `delivery.json`：提交、wheel、完整依赖版本、CI 矩阵及复现命令。
- `wheel-provenance.json`、`wheel-smoke.json`、`gpu-lifecycle.json`：产物身份、安装验收和实机指标。
- `ci-reports/`、`ci-torch/`、`ci-test-summary.json`：已下载的 JUnit 与覆盖率报告。
- 同目录保存 wheel、验证脚本和原生日志，可独立核对。

Gradio 构建期循环清理已在第四批完成，Router 状态类型化已在第五批完成实现。

## 第四批：Gradio 构建期循环归属

- 提交：`24661719f92b29506ed9bd3a2f9440127f2e9606`，`fix: manage Gradio setup event loop lifetime`，只包含 4 个任务文件。
- `lifecycle.with_setup_loop()` 为同步构建和 MinerU 启动入口提供一个自有的临时循环，结束时排空并关闭；异步宿主已有运行循环时直接复用。
- 显式 `loop_factory` 不覆盖或关闭宿主尚未运行的当前循环；不使用 `Runner.run()`，保留 Gradio 阻塞等待中首次 Ctrl+C 的行为。
- 同时覆盖 `build_gradio_app()` 和 `launch()` 再次创建 Gradio App 的路径；没有修改、猴子补丁或屏蔽 Gradio 警告。
- 原始同步构建探针为 10 个循环全部未关闭；修复后为 1 个循环已关闭。测试还覆盖异常时清理、后台任务排空、工作线程重复构建和异步宿主继续调度。
- 真实监听端口验证两个独立会话上传、排队、原生 HTML 转换、关闭、重新构建和首次 Ctrl+C。公开 4 输入/16 输出和同会话取消语义保持原有测试覆盖。
- 新增 7 项回归，已加入 Windows/macOS CI；本地 Gradio 相关回归为 **438 passed、3 skipped**。
- 完整非 Torch 回归：**4087 passed、3 skipped、108 deselected、62 warnings**，206.25 秒。继续排除用户修改中的 PDF 样式测试；本次没有再出现事件循环析构告警。
- Gradio 6.21.0 与配套 client 2.6.0 的 7 项回归通过；6.30.0 的真实上传/队列/中断验证通过。开发环境仍为 6.27.0，未修改依赖下限。
- wheel 已在全新 Python 3.14 环境完整安装，126 个依赖通过 `uv pip check`；`-I` 验证实际导入安装包，并通过循环清理、真实队列和首次 Ctrl+C 验收。
- wheel 共 659 个包内文件逐字节匹配上述提交；SHA-256 为 `2d851ca5ed331f999d698dfa7895011a4fe116557c4e5e6119be5a076f4cc32c`。
- 新模块与测试 Ruff/格式、全仓 E9/F63/F7/F82、CI helper 24 项与 `git diff --check` 均通过。
- 本地验收目录：`/private/tmp/mineru-gradio-lifecycle-delivery-844p9mld`；持久化报告、wheel 和复现脚本：`/Users/myhloli/Downloads/MinerU-gradio-lifecycle-24661719-2o68czuu`。
- [本轮跨平台 CI](https://github.com/opendatalab/MinerU/actions/runs/40173191691)：**16 个任务全部成功**，head SHA 确认为本轮提交。
- 已下载并核对全部 13 份 JUnit：Linux Python 3.11–3.14 每组 **4097 passed、6 skipped**；Windows/macOS Python 3.11–3.14 每组 **130 passed**（包含新增生命周期回归）；Torch CPU **106 passed**。Python 3.12 覆盖率 **63.49%**，高于要求的 58%。
- 用户原有 PDF 样式补丁逐字节不变，三个 examples 文件 SHA-256 均与前三批验收记录一致。

## 第五批：Router 路由、租约和预占状态类型化

- 提交：`9f7f66412776482667ee2b0e3367a5f66a87b05b`，`refactor: type router route and result lease state`，15 个任务文件，用户原有改动未混入。
- 新增 `route_state.py`：`ExecutionRouteState` 明确持有执行快照、下游文件身份、认证头和预占；`ResultLease` 明确持有结果保护、墙上时间期限和单调时钟续租顺序。
- 文件与执行使用独立的 `FileRoute` / `ExecutionRoute`。资源种类为只读属性；执行注册必须同时提供完整执行状态，查询与枚举按资源种类返回具体类型。
- 删除旧 `ResourceRoute.metadata` 和通用 `register()`，所有仓库内生产调用与测试迁移到明确字段及 `register_file()` / `register_execution()`，没有保留兼容转发。
- 上游 HTTP payload 作为独立响应缓存保留原始扩展；内部租约、固定计数和认证不存放其中。文件元数据刷新保持租约和传输固定状态。
- 预占对象是活动执行的归属依据，取消、终态和 worker 丢失统一幂等释放；移除 `active_counted`、代理层兜底计数以及无调用方的手工开始/结束计数方法。
- 保留公开 Task/File ID、逐文件状态、worker generation 校验、结果保留和租约时长语义。Router/API 完整 OpenAPI（各 11 条路径）与全部 17 个协议模型 JSON Schema 均与第五批前基线完全一致。
- 新增 6 项边界回归：重复与迟到预占释放、响应刷新和扩展隔离、注册冲突不覆盖旧归属、跨 worker 续租超时及强制释放清理。已加入 Windows/macOS CI。
- 聚焦回归：改动前 **202 passed**，最终 **208 passed**；状态类型与注册表两个源文件通过定向 mypy（`--follow-imports=skip --ignore-missing-imports`），改动文件 Ruff/格式及全仓 E9/F63/F7/F82 通过，CI helper **24 passed**。
- 本地完整非 Torch：**4093 passed、3 skipped、108 deselected、62 warnings**，213.86 秒；继续排除用户修改中的 PDF 样式测试。
- wheel 在全新 Python 3.14 环境安装，126 个依赖通过 `uv pip check`。`-I` 验证导入安装包；真实启动两个 Flash API 子进程，经 HTTP 验证负载分配、ZIP 内容、元数据刷新、hash 缓存复用并产生新结果 ID、取消、私有认证清理和下游租约释放；两个子进程均以 0 正常退出。没有下载模型或运行推理模型。
- wheel 660 个包文件逐字节匹配上述提交，SHA-256：`769a56606a287acb05d083424226b6ff1a71c36a782e0986d839a7f736f8ef25`。
- [第五批跨平台 CI](https://github.com/opendatalab/MinerU/actions/runs/40175872292)：**16 个任务全部成功**，head SHA 确认为本轮提交。已下载并核对全部 13 份 JUnit 与覆盖率报告。
- Linux Python 3.11–3.14 每组 **4103 passed、6 skipped**；Windows/macOS Python 3.11–3.14 每组 **136 passed**；Torch CPU **106 passed**。Python 3.12 覆盖率 **63.58%**，高于要求的 58%。
- 本地验收：`/private/tmp/mineru-router-state-9_p7f9tj`；持久化 wheel、报告和复现脚本：`/Users/myhloli/Downloads/MinerU-router-state-9f7f6641-ff9pbwhw`。
- 用户原有 PDF 样式补丁与三个 examples 文件已按原始补丁及 SHA-256 核对保持不变。

## 第六批：Doclib 文件刷新与入库职责拆分

- 实现提交：`f18db4a3f7a76c9efe6f5a46d4648d7983ceb153`，`refactor: separate doclib file ingestion from parsing`，29 个任务文件；后续夹具修正为 `d129c237fb8bdfe6aa9c69ad135cf995806a2030`，只改一个测试文件。
- `IngestService` 独立拥有文件刷新、状态分类、hash 去重、元数据、文件名/文本 FTS、短标识分配和初始解析入队；`ParseService` 从 1610 行减至 1063 行，显式注入入库服务后保留解析请求和执行职责。
- 扫描、watch、后台入库、HTTP 文件接口和搜索探测直接使用应用的同一 `IngestService`；解析工作池仍只使用 `ParseService`。两个应用的数据库、FTS、规则和入库对象保持独立。
- 不保留旧 `ParseService.refresh_file/ensure_ingested/ingest_file` 或旧私有模块转发；仓库内构造和调用全部迁移。既有 `services` 明确导出门面继续从新归属模块导出相应符号。
- 公用默认页范围与后台规则档位放在 `services/parse_policy.py`，FTS 首尾截断归属 `core/fts.py`。队列 SQL、锁超时、默认初始解析策略及历史缓存兼容边界保持。
- 10 个迁移方法经 AST 核对：忽略中文职责注释、规范模块名和局部可空类型标注后，SQL 与控制流完全一致。
- 真实数据库前后回放：4 个文档、5 个文件、2 项原生解析与 2 份缓存；文件状态、去重、队列、FTS、缓存命中响应、指标事件及完整缓存 JSON 均相同。仅排除数据库墙上时间字段与缓存文件名的写入时间戳；原始回放 JSON 与数据库保留在本地验收目录。
- 原有聚焦回归 **748 passed**；新增 4 项服务独立性、应用实例隔离和架构边界回归通过；新入库/策略模块定向 mypy 通过（`--follow-imports=skip --ignore-missing-imports`）。
- 完整非 Torch 回归：**4097 passed、3 skipped、108 deselected、62 warnings**，223.78 秒；继续排除用户修改中的 PDF 样式测试。
- Doclib 完整 OpenAPI **34 条路径**与 **79 个协议模型 JSON Schema**均与第六批前基线完全一致。改动 Python 文件没有新增 Ruff 诊断（42 条为已有诊断），新函数均有中文说明，全仓 E9/F63/F7/F82、格式及 `git diff --check` 通过；CI helper **24 passed**。
- wheel 已在全新 Python 3.14 环境完整安装，126 个依赖通过 `uv pip check`。`-I` 验证导入安装包；真实后台扫描/入库/解析、HTTP 元数据和 Markdown 读取、缓存命中、FTS 搜索、watch 修改重入库与删除均通过；应用退出后后台任务全部结束。
- wheel **662 个包文件**逐字节匹配上述提交；SHA-256：`8f142a6b10e6609b4b8d5c9d3eb32e74a6016d5c32fc4c6ee2adc9e8cbb32d75`。
- [首轮 CI](https://github.com/opendatalab/MinerU/actions/runs/40178458709) 的 Linux 3.11 在 `test_file_parse.py::test_formats_are_returned_unchanged[formats9]` 失败：上游附件和预期 ZIP 分别使用当前时间构造，跨两秒边界后时间戳字节不同。
- `d129c237` 固定该 ZIP 夹具时间戳，保留严格附件字节断言；`test_file_parse.py` **56 passed**，15 种格式组合在受控跨时间边界验证中均保持字节一致。wheel 662 个生产文件再次核对匹配该最终提交，未重新改动生产实现。
- [第六批最终跨平台 CI](https://github.com/opendatalab/MinerU/actions/runs/40179228519)：**16 个任务全部成功**，head SHA 为 `d129c237fb8bdfe6aa9c69ad135cf995806a2030`。首轮失败 JUnit 与最终结果分开保存在验收目录。
- 已下载核对最终 13 份 JUnit 与覆盖率报告：Linux Python 3.11–3.14 每组 **4107 passed、6 skipped**；Windows/macOS Python 3.11–3.14 每组 **143 passed**；Torch CPU **106 passed**。Python 3.12 覆盖率 **63.62%**，高于要求的 58%。
- 本地验收：`/private/tmp/mineru-doclib-ingest-awlirx1s`；持久化报告、wheel 与复现脚本：`/Users/myhloli/Downloads/MinerU-doclib-ingest-f18db4a3-yc6hbssm`。
- 用户原有 PDF 样式补丁及三个 examples 文件均已核对保持不变。

## 第七批：Doclib 解析队列与执行职责拆分

- 提交：`680d56eef7d8e8333349a8c7b7ee15696e637b1d`，`refactor: separate doclib parse queue and execution`，32 个任务文件。
- 删除旧 `ParseService` / `parse_svc.py`，没有保留委托壳或私有路径别名。仓库内生产调用、测试组装及当前架构/工作流/不变量/代码地图文档全部迁移。
- `ParseQueueService`（450 行）负责请求校验、缓存覆盖、活动任务复用和提权、队列抢占、记录查询及失效后索引重建；`ParseExecutionService`（601 行）消费已抢占的行，负责本地/API 解析、产物写出、元数据/索引和终态更新。
- 二者平级共享数据库、FTS 和数据目录。HTTP 层只依赖队列服务；工作池显式接收队列及执行服务，保持原有轮询、抢占 SQL、工作池异常兜底和停止行为。
- 共享失败类型归属 `parse_errors.py`；默认档位及纯页范围策略归属 `parse_policy.py`。API 返回的内容页号和布局几何仍一起校验并原子重映射。
- 20 个迁移方法经 AST 核对，除职责说明和规范策略模块名外，SQL 和控制流完全相同。
- 真实数据库前后回放：4 文档、5 文件、3 次原生解析和 3 份缓存；状态、队列、强制重解析、活动任务复用、缓存失效、FTS、缓存命中响应、指标事件及完整缓存 JSON 一致，仅排除数据库墙上时间与缓存文件名写入时间戳。
- 原有聚焦回归 **752 passed**；新增 5 项队列独立性、并发唯一抢占/优先级/过期锁、工作池异常解锁与架构依赖回归通过。队列/执行/策略/错误四个源文件定向 mypy 通过（`--follow-imports=skip --ignore-missing-imports`）。
- 完整非 Torch：**4102 passed、3 skipped、108 deselected、62 warnings**，203.73 秒；继续排除用户修改中的 PDF 样式测试。CI helper **24 passed**，全仓 E9/F63/F7/F82、改动格式及 `git diff --check` 通过；无新增 Ruff 诊断（41 条为既有诊断）。新函数均有中文职责说明。
- Doclib 完整 OpenAPI **34 条路径**和 **79 个协议模型 JSON Schema**均与第七批前基线一致。
- wheel 在全新 Python 3.14 环境完整安装，126 个依赖通过 `uv pip check`；`-I` 验证实际导入安装包且旧模块不存在。真实后台队列/执行/扫描/入库、HTTP 内容读取、强制重解析、失效后重建、缓存、FTS 和 watch 修改/删除全部通过；后台任务退出完成。
- wheel **664 个包文件**逐字节匹配上述提交；SHA-256：`7de7a5d6c0a60399e65b0b867378f2fb18603da05d28e7d42f2c793a1dee56fa`。
- [第七批跨平台 CI](https://github.com/opendatalab/MinerU/actions/runs/40181892861)：**16 个任务全部成功**，head SHA 已核对为上述提交；最终 13 份 JUnit 与覆盖率报告已下载核对。
- Linux Python 3.11–3.14 每组 **4112 passed、6 skipped**；Windows/macOS Python 3.11–3.14 每组 **148 passed**；Torch CPU **106 passed**。Python 3.12 覆盖率 **63.68%**，高于要求的 58%。
- 本地验收：`/private/tmp/mineru-doclib-parse-xssyghst`；持久化报告、wheel 和验证脚本：`/Users/myhloli/Downloads/MinerU-doclib-parse-680d56ee-t8q0j_4t`。
- 用户原有 PDF 样式补丁及三个 examples 文件均已按原始补丁和 SHA-256 核对保持不变。

## 第八批：Doclib 配置与健康状态的实例归属

- 提交：`95f97619904c410eaf08601065b3c584a9fdf375`，`refactor: scope doclib configuration and health to each app`，29 个任务文件。
- 新增轻量 `parse_server_state.py` 统一定义探测结果、目标探测状态及应用健康记录；删除模块级 `_parse_server_health` 和 `get_health()`。每个 `AppState` 构造独立状态，入库、策略、队列、执行、HTTP 与探测器显式共享同一实例。
- `ParseServerHealthCheck` 显式接收健康对象、启动超时、托管配置和日志配置，不在构造或运行中读取进程默认值；应用关闭和探测器停止只处理各自持有的进程。原有探测、退避、generation、重启及停止控制流经 AST 归一核对保持一致。
- 内容读取、完整 Markdown 导出和失效后的 FTS 重建使用宿主传入的公式定界符快照；`DoclibServer` 显式接收宿主配置和健康对象，数据目录与日志路径回落到该配置。
- 保留应用/CLI 默认构造、endpoint 解析和独立日志工具的默认配置入口。内部 `_tail_log` 只读取明确传入的路径，状态接口负责提供本实例的正确日志路径；没有新增运行时兼容别名。
- 单实例真实数据库/原生解析回放与第八批前一致：状态、队列、强制重解析、任务复用、失效、FTS、完整缓存 JSON、回执及指标均相同，仍只排除墙上时间字段和缓存文件名写入时间戳。
- 新增 4 项实例隔离和架构回归。两个实际应用针对相同公式缓存按各自配置读取、导出和重建 FTS；原始配置构造后被外部修改不影响应用。健康/档位读取相互独立，关闭一方不影响另一方；探测器停止只清理自己的托管进程。
- 测试禁止宿主路径读取进程默认配置，所有实例仍正常工作。原有 Doclib/格式聚焦 **760 passed**；最终聚焦 **791 passed**（含 27 项远端用量和新增 4 项），新状态/策略模块定向 mypy 通过（`--follow-imports=skip --ignore-missing-imports`）。
- 完整非 Torch：**4106 passed、3 skipped、108 deselected、62 warnings**，205.57 秒；继续排除用户修改中的 PDF 样式测试。CI helper **24 passed**，全仓 E9/F63/F7/F82、格式和 `git diff --check` 通过；没有新增 Ruff 诊断（91 条为涉及文件的既有诊断）。
- Doclib 完整 OpenAPI **34 条路径**与 **79 个协议模型 JSON Schema**均与第八批前基线相同。
- wheel 在全新 Python 3.14 环境完整安装，126 个依赖通过 `uv pip check`；`-I` 验证包路径。实际后台解析、强制重解析、失效重建、内容读取及 watch 验收通过；安装环境内的双实例公式/索引/健康隔离也通过。
- wheel **665 个包文件**逐字节匹配上述提交，SHA-256：`f3d02814755e72e2b75a24ed153e2c9327a9804cae692d68bf5c4d373af411ce`。
- [第八批跨平台 CI](https://github.com/opendatalab/MinerU/actions/runs/40184436309)：**16 个任务全部成功**，head SHA 已核对为上述提交；最终 13 份 JUnit 与覆盖率报告已下载核对。
- Linux Python 3.11–3.14 每组 **4116 passed、6 skipped**；Windows/macOS Python 3.11–3.14 每组 **152 passed**；Torch CPU **106 passed**。Python 3.12 覆盖率 **63.69%**，高于要求的 58%。
- 本地验收：`/private/tmp/mineru-doclib-instance-xg4kliwr`；持久化报告、wheel 与复现脚本：`/Users/myhloli/Downloads/MinerU-doclib-instance-95f97619-09_8qv_x`。
- 用户原有 PDF 样式补丁与三个 examples 文件按原始补丁及 SHA-256 核对保持不变。

## 第九轮：集中复审与 Doclib 生命周期修复

- 实现提交：`73424fa72d231edffac74a15e19d583ffa6c5e6e`，`fix: complete doclib cleanup across lifespan failures`，仅 5 个任务文件；后续 `968ee71a` 只调整一个真实 PDF 功能测试的等待预算。
- 集中核对真实入口、导入关系、资源关闭链、公开边界及文档：共享 Tasks、Doclib core、队列/执行、Router 与本地 API、模型 runtime 的关键分层检查通过；旧私有 `parser/task_api.py`、`services/parse_svc.py` 均不存在。
- 349 个 Python 文件的静态导入清点和 9 个独立冷导入通过。公开入口不加载 Torch/Transformers/ONNX Runtime，导入没有创建默认 home 目录。静态清点包含惰性与 TYPE_CHECKING 边，不将包导出门面的回环当成缺陷。
- 实证确认生命周期缺口：启动创建托管进程后失败会跳过清理；退出时在途健康探测能再次启动进程；单个停止错误或外部取消能截断后续清理。新增 5 个受控用例在旧代码全部失败，修复后全部通过（包含重复取消、主错误优先保留）。
- 生命周期现在将启动纳入保护区；先取消并等待后台编排循环，再停止明确拥有的组件，继续处理错误后的其它资源。托管进程、数据库及发现信息最终清理；保留最早资源错误，启动/请求主错误不被清理错误覆盖。
- 清理以独立任务执行并屏蔽外部取消，重复取消后先排空清理再传播取消；没有更改解析算法、任务策略、CLI/API 协议或同步 CPU 工作的终止边界。
- 架构、资源归属与生命周期聚焦 **112 passed**；完整非 Torch **4111 passed、3 skipped、108 deselected、62 warnings**，210.30 秒。继续排除用户修改中的 PDF 样式测试。CI helper **24 passed**，本轮改动 Ruff/格式、全仓 E9/F63/F7/F82 和 `git diff --check` 通过。
- Doclib 完整 OpenAPI **34 条路径**及 **79 个协议模型 JSON Schema**与复审前完全相同。
- 隔离安装的 126 个依赖通过 `uv pip check`；实际后台解析/强制重解析/失效重建/watch 和 5 个安装态 CLI 帮助入口通过。额外创建真实 Python 子进程，确认启动失败由应用关闭它；重复取消退出后数据库仍完成清理。该进程不加载模型，使用正常终止信号退出，不将其描述为模型推理验收。
- wheel **665 个包文件**逐字节匹配该提交；SHA-256：`2e5f7110decb32bbdbd865157a424b0462ce0f23dcb3c5b022842efecee56dac`。
- 早期 `implementation-plan.md` 任务卡片仍引用已移除的组合服务，已加当前归属迁移说明，保留历史卡片内容。未发现值得继续大范围拆分的边界问题；`core/db.py` 导入时读取只读初始化 SQL 资源保留为低优先级可选项，本轮没有据此扩大修复范围。
- [首轮复审 CI](https://github.com/opendatalab/MinerU/actions/runs/40187166784) 的 Linux 3.12 在一个真实 PDF 功能测试超出 5 秒等待预算；日志显示覆盖率插桩下三页推理已用 4.405871 秒，任务还需打包产物，其余格式及结果断言没有报错。
- `968ee71a` 仅将该用例等待预算显式设为 20 秒，与其它真实端到端用例一致，保留全部实际页范围、逐文件终态及错误断言；本地覆盖率插桩重跑该文件 **73 passed**。生产代码及 wheel 不变，665 个生产文件再次核对匹配最终提交。
- [复审最终 CI](https://github.com/opendatalab/MinerU/actions/runs/40188252418)：**16 个任务全部成功**，head SHA 已核对为 `968ee71a6bfc802f92345f51f6554e0845bf606d`；首轮失败记录与最终报告分开归档。
- 最终 13 份 JUnit 与覆盖率报告已下载核对：Linux Python 3.11–3.14 每组 **4121 passed、6 skipped**；Windows/macOS Python 3.11–3.14 每组 **157 passed**；Torch CPU **106 passed**。Python 3.12 覆盖率 **63.70%**，高于要求的 58%。
- 本地验收：`/private/tmp/mineru-architecture-audit-8cicpvqf`；持久化报告、wheel 和复现脚本：`/Users/myhloli/Downloads/MinerU-architecture-audit-73424fa7-06zqur5w`。
- 用户原有 PDF 样式补丁及三个 examples 文件按原始补丁与 SHA-256 核对保持不变。

## 合入前最终审阅：已完成并收尾

- 审阅 HEAD：`968ee71a6bfc802f92345f51f6554e0845bf606d`；已刷新目标 `upstream/dev` 至 `986416e4e4eb17fd63e5580408e9a9afc8480ec1`。
- 相对目标共 176 个改动文件；范围包含优化起点已有的 `ec6c23cff` Gradio/SDK 取消增强提交，未将它混记为结构拆分的新行为。
- 合并模拟无冲突，生成树与 HEAD 完全相同；没有执行实际合并。没有发现需要阻止本轮合入的新问题。
- 本次针对性回归 **138 passed、2 warnings**，18.07 秒；全仓 E9/F63/F7/F82、分支补丁检查、新增生产函数中文说明核查通过。
- 直接对比最新目标源码：API/Router 完整 OpenAPI 各 **11 条路径**、Doclib **34 条路径**与 **79 个模型 Schema**均完全一致。最终 CI 再次查询确认 **16 个任务成功**，head SHA 与审阅提交相同；生产代码没有新增改动，既有完整回归及隔离安装包验收继续适用。
- 非阻断后续项 **P3（30/100）**：显式宿主使用自定义公式分隔符时，初次 FTS 索引仍通过 `ParseResult.markdown()` 读取进程默认配置，重建索引则使用宿主配置。真实数据库探针已确认；默认配置一致，公式正文未丢失。后续扩展嵌入式宿主配置隔离时，再将公式配置显式传入执行侧索引渲染，本轮不扩大范围。
- 正式报告与证据：`/Users/myhloli/Downloads/MinerU-final-review-968ee71a-vb8q3e21/review.md`；包括目标对比、AST 差异、真实 FTS 探针、回归日志和审阅结论 JSON。
- 用户原有 PDF 样式补丁及三个 examples 文件已再次按补丁与 SHA-256 核对保持不变。

## 合入流程：已完成

- [PR #5665](https://github.com/opendatalab/MinerU/pull/5665)：`codex/architecture-lifecycle-refactor` → `dev`，已创建正式 PR 并附加到当前任务。
- PR head：`968ee71a6bfc802f92345f51f6554e0845bf606d`；目标：`986416e4e4eb17fd63e5580408e9a9afc8480ec1`。
- PR 跨平台 CI：[40190457124](https://github.com/opendatalab/MinerU/actions/runs/40190457124) 的 **16 个任务全部成功**，13 份 JUnit 已下载核对，覆盖率 **63.70%**；文档构建与 CLA 也通过。
- 已于 `2026-10-25T12:00:58Z` 正常合并，提交 `c8028a89ff0de12ddd7c9d31827215e4599f3c3b`。GitHub PR 的 MERGED 状态及 upstream/dev 指向均已核对，合并树与已审阅 HEAD 完全一致。
- 合入后跨平台 CI：[40191282151](https://github.com/opendatalab/MinerU/actions/runs/40191282151) 的 **16 个任务全部成功**；[文档构建](https://github.com/opendatalab/MinerU/actions/runs/40191282208) 通过。两者 head SHA 均核对为合并提交。
- 合入后 13 份 JUnit 与覆盖率报告已下载核对：Linux Python 3.11–3.14 每组 **4121 passed、6 skipped**，Windows/macOS 每组 **157 passed**，Torch CPU **106 passed**，覆盖率 **63.71%**。
- 合入记录：`/Users/myhloli/Downloads/MinerU-merge-5665-z32pkixt`。

## 收尾状态

- 计划内结构优化、集中复审、合入前审阅及 PR 合入与验收均已完成，本轮任务结束。
- 低优先级项已记录，后续按实际需求独立处理；PR 已合入 dev，合入前后检查全部通过；发布和部署尚未执行。

## 已确认的基线问题与辅助证据

- 历史基线：Gradio 6.27.0 构建队列时可能遗留事件循环，并在退出/GC 时出现 `BaseEventLoop.__del__: Invalid file descriptor: -1`。第四批已在 MinerU 的构建和启动入口修复。
- 使用第三批前原始 `app.py` 的同组 54 个测试也复现该警告；不导入 MinerU 的 `Blocks + queue` 探针同样观察到循环未关闭，说明它不是第三批转换服务引入。
- 第三批 OpenAPI 基线临时目录：`/var/folders/tm/fmb8lj6d3sq28z7x0cgjmcrc0000gn/T/mineru-single-execution-jfur9eyr`。
- Gradio 诊断临时目录：`/var/folders/tm/fmb8lj6d3sq28z7x0cgjmcrc0000gn/T/mineru-gradio-loop-diag-e8kxxwna`。
- 临时目录可能被系统清理；恢复依据优先使用本状态文件、当前源码和持久化回归测试。
