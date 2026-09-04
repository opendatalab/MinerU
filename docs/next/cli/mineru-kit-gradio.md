# mineru-kit gradio

页码规范与历史结果兼容说明见 [PDF 页码范围规范](../page-ranges.md)。

状态: Implemented

`mineru-kit gradio` 提供一个基于 MinerU V1 API 的文档解析界面。它不直接调用旧 `/file_parse` 或 `/tasks` 接口，也不使用 `doclib` 缓存。

## 启动

安装 Gradio 可选依赖：

```bash
pip install 'mineru[gradio]'
```

自动启动本地 V1 API server：

```bash
mineru-kit gradio
```

连接已有 self-hosted 或 remote V1 API server：

```bash
mineru-kit gradio \
  --api-url http://127.0.0.1:16580 \
  --api-key "$MINERU_API_KEY"
```

未指定 `--api-url` 时，Gradio 会启动一个 loopback `mineru-kit api-server`。本地 server 默认使用 Standard 能力上限、开启 Flash、关闭 `local` source，文件通过 V1 Uploads API 提交。

未指定 `--server-port` 时，界面由 Gradio 从 `7860` 开始寻找空闲端口；可通过 `GRADIO_SERVER_PORT` 设置起始端口、`GRADIO_NUM_PORTS` 设置尝试数量（默认 `100`）。显式指定 `--server-port 7861` 时只尝试该端口，覆盖环境变量；该端口已被占用则启动失败。界面最终访问地址以启动输出为准，与内部 API server 的端口不同。

## 主要参数

| 参数 | 说明 |
|------|------|
| `--api-url` | 已有 V1 API base URL；不传则自动启动本地 server。 |
| `--api-key` | Bearer API Key；省略时读取 `MINERU_API_KEY`。 |
| `--server-name` / `--server-port` | Gradio UI 的监听地址和端口；省略端口时自动寻找空闲端口。 |
| `--output-dir` | 本地 UI 产物根目录，默认 `./output`。 |
| `--max-pages` | 单次非 Flash PDF 解析的最多页数，必须为正整数；省略则不限制。不限制 Flash 或非 PDF。 |
| `--api-server-tier` | 自动启动 server 的能力档位：`flash`、`basic`、`standard`。 |
| `--api-server-concurrency` | 自动启动 server 的最大并发任务数。 |
| `--api-server-language` | 自动启动 server 的 OCR 语言提示。 |
| `--api-server-ocr-mode` | 自动启动 server 的 OCR 模式：`auto`、`txt`、`ocr`。 |
| `--api-server-preload-models` | 在自动启动阶段预加载模型。 |
| `--api-server-no-flash` | 自动启动 server 时关闭 Flash。 |
| `--api-server-disable-image-analysis` | 自动启动 server 时关闭图片分析。 |
| `--enable-example` | 显示当前工作目录 `examples/` 中的示例文件。 |
| `--enable-api` | 暴露 Gradio 转换事件 API。 |
| `--latex-delimiters-type` | Markdown 预览公式分隔符：`a`、`b` 或 `all`。 |

## 解析流程

界面一次提交一个文件，支持 `filetypes.PARSEABLE_EXTENSIONS` 中的 PDF、图片、Office/ODF、RTF、HTML、CSV/TSV、EPUB 和 OFD。

解析 tier 使用原生离散滑块选择，上方即时显示当前档位。滑块按 `flash → basic → standard → advanced` 排列，仅包含服务实际支持的档位。默认优先选择 `standard`，否则选择最高可用档位；仅有一个档位时禁用滑块。上传新文件或清除结果会保留当前选择。

启用 Gradio 事件 API 时，转换事件的 `tier_position` 参数为从 `0` 开始的整数位置，替代原来的 tier 字符串。例如四档齐全时 `3` 对应 `advanced`；仅支持 `basic/standard/advanced` 时 `2` 对应 `advanced`。位置始终按照上述顺序对实际可用档位编号，越界位置会报错。V1 API 的 `tier` 参数仍使用字符串，启动参数不变。

仅在已上传原始 PDF 且 tier 不是 `flash` 时显示页码双滑块。轨道范围由 `pypdfium2` 读取的实际页数确定，为 `1～n`，两端对应包含首尾页的连续选区。Flash 和其他文件格式隐藏控件，并始终全部解析。

默认选择全部页；配置 `mineru-kit gradio --max-pages 20` 后，100 页 PDF 的初始选区为 `[1-20]`，轨道仍为 `1～100`。选区最多 20 页，允许缩小；只有超限时才联动另一端。例如将右端拖到 40 得到 `[21-40]`，再把左端拖到 15 得到 `[15-34]`，随后把左端拖到 20 得到 `[20-34]`。

两个滑块可以互相越过并实时交换起止角色，拖动过程中始终抓住同一个滑块，不需要松手。显示和提交的范围始终从小到大排列。例如从 `[20-35]` 把原左滑块拖到 40，得到 `[35-40]`；同一滑块继续到 60，按上限联动为 `[41-60]`；不松手退到 30，则得到 `[30-41]`。两端重合表示单页，此时保留原来的角色，直到严格越过才交换；键盘操作遵循相同规则。

更换或清除文件会重置选区与角色；同一 PDF 切换 tier 保留两个滑块的位置和角色，包括切到 Flash 后再切回。读取页数期间，非 Flash PDF 暂不可提交；文件损坏或需要密码时显示错误，不降级为全部解析。

Gradio 转换事件保留 `raw_page_range` 字符串接口，仍接受 V1 `page_range` 语法，例如 `1-5,8,r1`。未指定时按当前上限选择前若干页，无上限则全部；显式选区（包括 `all`）超限会返回 `page_range_invalid`，不会截断。提交端按真实页数复核，直接调用事件也不能绕过上限。该限制不改变通用 Python、V1 API 或 Doclib 的选页规则。

Gradio 首先发现 `/v1/health` 和 `/v1/tiers`，然后通过 `MinerUApiParser` 上传文件、创建 `/v1/parse/jobs`、轮询任务并下载 ZIP 结果。解析结果保存为严格 Middle JSON 2.0。

## 结果与下载

页面只展示：

- Markdown 渲染；
- Markdown 源码；
- Structured Content 源码。

下载图标位于右侧结果栏的标签行最右端，鼠标悬停或键盘聚焦即可展开菜单。菜单包含 ZIP、HTML、DOCX、LaTeX bundle、EPUB 和 PDF。HTML、DOCX、EPUB、PDF 和 LaTeX 不作为 API job 输出请求，而是在用户点击菜单后从本次保存的 Middle JSON 按需渲染。LaTeX 下载包包含 `.tex` 与 `images/`，可交给 XeLaTeX 使用；Gradio 不自动执行 TeX 编译。

每次解析的文件保存在独立的 `output-dir/gradio/<run-id>/` 目录，包含源文件、Middle JSON、基础文本产物以及可选的 `origin.pdf`、`layout.pdf` 和图片资源。路径只向 Gradio 暴露在配置的 output root 内。

## 预览与兼容边界

PDF 和图片会生成与解析范围一致的 `origin.pdf`；布局预览使用 schema 2.0 顶层 block/bbox 生成语义 overlay。无法生成 overlay 时仍保留 origin PDF 预览。

`mineru-gradio` 保留为 `mineru-kit gradio` 的兼容命令名，参数与行为完全相同；旧 HTTP 协议和旧 Gradio 专属参数不再提供。
