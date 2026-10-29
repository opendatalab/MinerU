# mineru-kit webui

页码规范与历史结果兼容说明见 [PDF 页码范围规范](../page-ranges.md)。

状态: Implemented

## PDF 预览与版本要求

Gradio 界面要求 `gradio>=6.21,<7`，使用 6.21 修复后的纯前端事件链式触发，不再需要安装 `gradio-pdf`。
PDF 预览使用独立 iframe 和随 MinerU 分发的 PDF.js 6.3.289，支持连续阅读、翻页、页码跳转、缩放和适应宽度。
核心模块、worker、CMap、标准字体及解码资源均从当前 Gradio 站点加载，预览无需外部 CDN 或 npm。

上传时显示原始 PDF；解析完成后显示布局 PDF，布局文件不可用时显示裁页后的原始 PDF。
每次切换文档回到第一页并适应宽度；浏览页码不改变解析页范围，翻页和缩放不调用 Python 解析接口。
需要密码或损坏的 PDF 会显示预览错误。文字搜索、编辑、打印和缩略图不在当前预览功能范围内。
静态资源和文件沿用 Gradio 的认证与文件路由，支持反向代理子路径；自定义挂载仍需传入自己的结果目录 `allowed_paths`。

Gradio 6.21 要求 `starlette>=1.0.1,<2` 和 `huggingface-hub>=1.16,<2`，与当前 MinerU 的 Transformers 5 依赖范围兼容。
安装时让依赖解析器同时处理 MinerU 和推理后端，无需为 WebUI 单独提高 Transformers、vLLM 或 LMDeploy 的下限。
UI 也可安装在独立环境，通过 `--api-url` 连接推理服务。

维护预览资源时执行 `python scripts/vendor_pdfjs.py`，脚本会校验固定上游压缩包并生成逐文件 SHA-256 清单。
构建后使用 `python scripts/verify_pdfjs_distribution.py dist/*.whl dist/*.tar.gz` 检查资源和许可证是否完整。
浏览器样本可用 `python tests/browser/gradio_pdf_fixtures.py output/playwright/pdfjs-fixtures` 生成。

`mineru-kit webui` 提供一个基于 MinerU V1 API 的文档解析界面。它不直接调用旧 `/file_parse` 或 `/tasks` 接口，也不使用 `doclib` 缓存。

## 启动

安装 Gradio 基础依赖：

```bash
pip install mineru
```

自动启动本地 V1 API server：

```bash
mineru-kit webui
# 与省略参数等价，自动推荐自托管服务上限
mineru-kit webui --tier auto
# 显式限制最高档位
mineru-kit webui --tier basic
```

连接已有 self-hosted 或 remote V1 API server：

```bash
mineru-kit webui \
  --api-url http://127.0.0.1:16580 \
  --api-key "$MINERU_API_KEY"
```

未指定 `--api-url` 时，Gradio 会启动一个 loopback `mineru-kit api-server`。本地 server 默认使用 `--tier auto`，由 API Server 按硬件和有效后端推荐 PDF/图片上限；可用 `--tier flash|basic|standard|advanced` 显式指定上限。原生文档始终支持并固定 flash。所有文件通过 Tasks 提交 hash 清单，缓存未命中时上传，同机也不使用路径免上传。

未指定 `--server-port` 时，界面由 Gradio 从 `7860` 开始寻找空闲端口；可通过 `GRADIO_SERVER_PORT` 设置起始端口、`GRADIO_NUM_PORTS` 设置尝试数量（默认 `100`）。显式指定 `--server-port 7861` 时只尝试该端口，覆盖环境变量；该端口已被占用则启动失败。界面最终访问地址以启动输出为准，与内部 API server 的端口不同。

## 主要参数

| 参数 | 说明 |
|------|------|
| `--api-url` | 已有 V1 API base URL；不传则自动启动本地 server。 |
| `--api-key` | Bearer API Key；省略时读取 `MINERU_API_KEY`。 |
| `--server-name` / `--server-port` | Gradio UI 的监听地址和端口；省略端口时自动寻找空闲端口。 |
| `--output-dir` | 本地 UI 产物根目录，默认 `./output`。 |
| `--max-pages` | 单次 PDF 解析的最多页数，必须为正整数；省略则不限制。不限制非 PDF。 |
| `--tier` | 自托管 API Server 的上限：`auto`（默认）、`flash`、`basic`、`standard`、`advanced`；指定 `--api-url` 时忽略。 |
| `--api-server-concurrency` | 自动启动 server 的最大并发任务数。 |
| `--api-server-preload-models` | 在自动启动阶段预加载模型。 |
| `--api-server-disable-image-analysis` | 自动启动 server 时关闭图片分析。 |
| `--enable-example` | 显示当前工作目录 `examples/` 中的示例文件。 |
| `--enable-api` | 暴露 Gradio 转换事件 API。 |

Gradio 只保留 `--tier`，不接受旧名 `--api-server-tier`。Gradio 不接受 `--api-server-no-flash`、`--no-flash`、`--api-server-no-advanced`、`--no-advanced`；传入会报错。独立 API 与 WebUI 的单值托管配置统一表示能力上限，并包含自身和全部较低档位。

指定外部 `--api-url` 时，Gradio 校验 `/v1/health` 和 `/v1/tiers` 后直接使用该服务。公开 tiers 只约束 PDF/图片；原生文档同样上传给远端，不因列表中缺少 flash 而启动本地补充服务或增加 PDF 档位。认证只用于所连接服务，连接或解析失败保持原错误。

## 解析流程

界面一次提交一个文件，支持 `filetypes.PARSEABLE_EXTENSIONS` 中的 PDF、图片、Office/ODF、RTF、HTML、MHTML/MHT、CSV/TSV、EPUB 和 OFD。

解析 tier 使用原生离散滑块选择，上方即时显示当前档位。滑块按 `flash → basic → standard → advanced` 排列，只包含所连接服务发布的 PDF/图片档位。界面与 Tasks 省略 tier 共用 `standard → basic → flash → advanced` 选择规则，仅服务只开放 advanced 时默认选 advanced；仅有一个档位时隐藏滑块，保留固定档位说明。

上传 Office/ODF、RTF、HTML、MHTML/MHT、CSV/TSV、EPUB、OFD 时，显示固定 `flash` 并隐藏滑块，不要求服务在 tiers 中发布 flash。界面记住本会话最后一次未锁定的档位，连续更换这类文件不会覆盖记忆；切回 PDF/图片或清除文件时恢复此前选择。PDF/图片之间切换继续保留当前档位。锁定与恢复同时更新档位标签、页码控件和转换按钮，无需向 Python 发送滑块拖动请求。

原生文档不校验隐藏滑块的位置，实际请求固定走原生解析路径；PDF/图片继续校验位置及服务公开的档位。

上传原始 PDF 时，tier 下方显示“强制 OCR”开关，所有档位（包括 Flash）均可使用。默认关闭，提交 `ocr_mode:"auto"` 自动判断；开启时提交 `ocr_mode:"ocr"`，忽略 PDF 文本层并进行 OCR。更换或清除文件会重置开关，同一文件切换 tier 保留选择。图片与其他格式隐藏开关，提交时固定使用 `auto`。

Gradio 转换事件新增 `force_ocr` 布尔参数，默认 `false`，位于 `raw_page_range` 之后。服务端处理转换事件时仍会检查原始文件类型，非 PDF 即使传入 `true` 也不会强制 OCR。

OCR 模式由每次 V1 解析任务决定；API Server 不再提供启动时的 OCR 配置，Gradio 的 `--api-server-ocr-mode` 及其 `--ocr-mode` 别名已移除，传入会报错。连接外部 V1 服务时，需要该服务支持任务级 `ocr_mode` 字段。

启用 Gradio 事件 API 时，转换事件的 `tier_position` 参数为从 `0` 开始的整数位置，替代原来的 tier 字符串。例如四档齐全时 `3` 对应 `advanced`；远端仅支持 `basic/standard` 时，`0` 对应 `basic`，`1` 对应 `standard`。位置始终按照上述顺序对实际可用档位编号，越界位置会报错。V1 API 的 `tier` 参数仍使用字符串。

仅在已上传原始 PDF 时显示页码双滑块，所有档位（包括 Flash）一致。轨道范围由 `pypdfium2` 读取的实际页数确定，为 `1～n`，两端对应包含首尾页的连续选区。其他文件格式隐藏控件，并始终全部解析。

默认选择全部页；配置 `mineru-kit webui --max-pages 20` 后，100 页 PDF 的初始选区为 `[1-20]`，轨道仍为 `1～100`。选区最多 20 页，允许缩小；只有超限时才联动另一端。例如将右端拖到 40 得到 `[21-40]`，再把左端拖到 15 得到 `[15-34]`，随后把左端拖到 20 得到 `[20-34]`。

两个滑块可以互相越过并实时交换起止角色，拖动过程中始终抓住同一个滑块，不需要松手。显示和提交的范围始终从小到大排列。例如从 `[20-35]` 把原左滑块拖到 40，得到 `[35-40]`；同一滑块继续到 60，按上限联动为 `[41-60]`；不松手退到 30，则得到 `[30-41]`。两端重合表示单页，此时保留原来的角色，直到严格越过才交换；键盘操作遵循相同规则。

更换或清除文件会重置选区与角色；同一 PDF 切换 tier 保留两个滑块的位置和角色，包括切到 Flash 后再切回。读取页数期间，PDF 暂不可提交；文件损坏或需要密码时显示错误，不降级为全部解析。

Gradio 转换事件保留 `raw_page_range` 字符串接口，仍接受 V1 `page_range` 语法，例如 `1-5,8,r1`。未指定时按当前上限选择前若干页，无上限则全部；显式选区（包括 `all`）超限会返回 `page_range_invalid`，不会截断。提交端按真实页数复核，直接调用事件也不能绕过上限。该限制不改变通用 Python、V1 API 或 Doclib 的选页规则。

Gradio 首先发现 `/v1/health` 和 `/v1/tiers`，然后通过 `MinerUApiParser` 提交 hash 清单到 `/v1/tasks`、按需上传、轮询父任务并下载逐文件默认 ZIP 结果。解析结果保存为严格 Middle JSON 2.0。

左下角使用与 3.4.5 一致的两列步骤卡片，展示准备请求、检查服务、提交任务、排队、解析中、下载结果、整理输出和完成。文件准备状态来自 V1 的 `waiting-file` 响应；排队状态来自本地任务槽或 V1 的 `pending` 响应，圆点每秒循环更新；不显示前方任务数。每个界面服务仍一次执行一个转换，等待中的会话也会立即显示排队动画。

收到 V1 的 `running` 后开始解析计时，由浏览器本地刷新；收到任务终态时冻结。API 路径的最终完成文案优先使用响应中已有的 `files[0].parse.duration_ms`，包含服务端解析和结果导出/打包，不包含排队、客户端下载和 WebUI 整理输出，因此即使轮询跳过 `running` 也能显示亚秒耗时。HF 等直接调用解析函数的适配层继续使用实际阶段通知之间的计时；缺少 API 耗时字段时也保留这种计时回退。完成数值按两位小数四舍五入，最低显示 `0.01 秒`。两种路径均不承诺该值等同于日志中的纯 `infer` 耗时。界面动画不增加 API 请求，任务轮询仍为每秒一次。

清除、更换文件、同会话重新提交或正常关闭页面会立即停止旧任务的界面状态流，并向该任务所属 V1 服务发送 `POST /v1/tasks/{task_id}/cancel`。每个任务最多发起一次取消，本地执行槽在取消清理结束后释放；已知服务端任务终结时不会再次取消，也不删除已完成产物。创建回执尚未返回时先等待同一次创建取得 ID（整体最多 120 秒），随后取消请求最多等待 5 秒；网络故障可能使远端取消无法确认，此时记录日志。页面关闭联动依赖服务端收到 Gradio unload 事件。HF 进程内适配层没有 HTTP Tasks，仍等待已进入同步 CPU/ZeroGPU 的工作安全结束后释放执行槽。

`MinerUApiParser.parse()` 和 `parse_async()` 提供可选的 `status_callback` 参数，接收公开的 `ApiTaskStatus`（`waiting-file`、`pending`、`running`、`completed`、`partial`、`failed`、`canceled`）。回调在任务提交响应及每次轮询响应后同步调用；回调异常只记录日志。这里的 `completed` 表示服务端任务完成，解析方法仍需下载并构造 `ParseResult` 后才返回。可选的 `duration_callback` 在最终成功响应之后、开始下载之前接收既有文件耗时（浮点毫秒）；缺失或非法耗时不触发回调，回调异常不影响解析结果。该值属于本次调用，不存放在共享 parser 实例或 `ParseResult` 中。

## 结果与下载

结果栏提供两个标签：

- Markdown：实际展示 HTML renderer 的完整输出，保留表格、公式和代码高亮。
- JSON：以只读、可复制的语法高亮视图展示本次保存的 `structured_content.json`，与 JSON 下载包中的内容一致。
  新转换开始、失败、更换文件和清除时同步清空；切换标签不会重新解析或渲染。

Gradio 转换事件在原有输出末尾追加 Structured Content JSON 文本，既有输出的位置保持不变。

下载按钮位于右侧结果栏的标签行最右端，图标右侧按界面语言显示“下载”或“Download”，鼠标悬停或键盘聚焦即可展开菜单。菜单包含 Markdown、JSON、HTML、DOCX、LaTeX、EPUB 和 PDF。下载文件由 Gradio 从本次保存的结果按需生成，不新增 API job 输出格式。Markdown 和 JSON 分别下载独立 ZIP，根目录放置同名 `.md` 或 `.json`，图片放在 `images/`，正文及 JSON 图片字段使用相对路径。JSON 内容使用 Structured Content，不包含 Middle JSON。LaTeX 下载包包含 `.tex` 与 `images/`，可交给 XeLaTeX 使用；Gradio 不自动执行 TeX 编译。预览和下载的单个 HTML 均使用当前 Gradio 服务的图片 HTTP 链接，不嵌入 base64；链接可用性取决于当前服务和任务文件。

图片名称使用原始零基页索引和所属父块的 `type/index`，例如 `images/page_1_image_3.jpg` 表示原第 2 页、索引为 3 的图片块。整表截图使用 `page_1_table_3.jpg`，表内图片按出现顺序使用 `page_1_table_image_3_1.png`、`page_1_table_image_3_2.png`。裁页不会重新编号，下载打包保持相同图片名称；代码和算法不生成截图。历史任务不迁移，重新转换后使用该命名。

每次选择格式后，按钮会显示“准备中”，文件就绪后自动开始下载，无需再次点击。再次下载时，HTML 根据当前访问地址重新生成，其他格式复用已有文件；生成失败时显示错误并允许重试。清除、更换文件或重新解析会使旧下载请求失效。

每次解析的文件保存在独立的 `output-dir/gradio/<run-id>/` 目录，包含源文件、Middle JSON、基础文本产物以及可选的 `origin.pdf`、`layout.pdf` 和图片资源。路径只向 Gradio 暴露在配置的 output root 内。

## 界面语言

MinerU 自定义文案跟随浏览器首选语言：中文（包括 `zh-CN`、`zh-TW` 等）统一显示简体中文，其他语言或无法识别时显示英文。Header、控件说明、页码提示、任务状态及下载提示使用同一规则；文件名、文档正文和服务端原始错误详情保持原文。Gradio 原生控件保留框架自带的国际化能力，上传提示、复制工具栏、页脚等可以按浏览器语言显示日语、法语等其他受支持语言。

强制 OCR 的说明为“忽略 PDF 文本层并进行 OCR”；英文为“Ignore the PDF text layer and perform OCR”。

## 预览与兼容边界

Office 类文件（DOC/DOCX、PPT/PPTX、XLS/XLSX、RTF、ODT/ODS/ODP）上传后立即显示“Office 在线预览”提示卡和在线预览框。提示文案和布局沿用 3.4.5，文件链接以站点加文件名尾部的短地址显示；实际预览使用完整文件 URL。该文件需要能被 Microsoft 在线预览服务访问，解析不依赖在线预览是否成功。

Office 转换开始、失败及成功后都保留已挂载的源预览。“忽略”仅隐藏当前提示卡，“不再提示”保存浏览器偏好；转换完成不会重新加载预览框或恢复已忽略的提示。更换文件更新源预览，清除恢复空预览。

转换开始、排队、解析和整理输出期间保留当前预览及浏览位置；PDF 和图片成功生成结果后刷新预览。转换失败也保留当前预览；点击清除会恢复空预览，更换文件会显示新文件的源预览或对应格式提示。

PDF 和图片会生成与解析范围一致的 `origin.pdf`；布局预览使用 schema 2.0 顶层 block/bbox 生成语义 overlay。无法生成 overlay 时仍保留 origin PDF 预览。

`mineru-webui` 提供为 `mineru-kit webui` 的独立命令名，参数与行为完全相同；旧 HTTP 协议和旧 Gradio 专属参数不再提供。

Gradio 使用 Tasks 能力发现和逐文件 `upload` 描述，固定使用 PUT 上传；仅保存 Middle JSON 和素材，下载菜单从保存的 Middle JSON 按需导出。

## 嵌入宿主的文档操作

`mineru.kit.gradio` 公开导出 `GradioDocumentIO` 协议。调用 `build_gradio_app()` 时可通过关键字参数
`document_io` 注入宿主实现，统一接管 `pdf_page_metadata()`、`effective_page_range()` 和
`persist_parse_result()`。不传时沿用进程内的页数读取、选页校验和产物保存。

界面页数回调与转换服务共享同一实例，各应用之间互不影响。选页与持久化仍在工作线程中执行，
取消必须等待同步工作结束后再释放转换槽。实现的构造过程应只定义行为，不启动进程、打开 PDF 或加载模型。

HF Demo 使用该接口将三类操作委托 CPU sidecar，生产与预检共用界面构建入口，不再替换 Gradio 模块函数。
HF 适配代码要求支持此协议与 `document_io` 参数的 MinerU；跨越该接口升级边界时，应成对更新或回滚宿主与 MinerU。
