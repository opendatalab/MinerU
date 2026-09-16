# Copyright (c) Opendatalab. All rights reserved.
"""CLI 双语词典:mineru 与 mineru-kit 的用户可见文案,英文原文 -> 简体中文。

key 为英文原文(gettext 约定),调用点漏译时自动回退英文。含 ``{name}``
占位符的词条,中文值必须保留同名占位符(test_cli_i18n.py 有占位符一致性守卫)。
不收录:click/typer 脚手架词、JSON 字段名、`<!-- Next: ... -->` 等协议串。
"""

from __future__ import annotations

ZH_MESSAGES: dict[str, str] = {
    # ------------------------------------------------------------------ 共享
    "JSON output": "JSON 输出",
    "Verbose output": "详细输出",
    "Max rows": "最大行数",
    "Result offset": "结果偏移量",
    "yes": "是",
    "no": "否",
    "Error:": "错误:",
    "Output path; creates parent directories": "输出路径;自动创建父目录",
    "Soft character limit for STDOUT content": "STDOUT 内容的软字符上限",
    "File not found: {path}": "文件不存在: {path}",
    "File or directory not found: {path}": "文件或目录不存在: {path}",
    "Show the version and exit.": "显示版本并退出。",
    "Print MinerU and Python versions.": "显示 MinerU 与 Python 版本。",
    "MinerU version: {version}": "MinerU 版本: {version}",
    "Python version: {version}": "Python 版本: {version}",
    "Written to {path}": "已写入 {path}",
    "No renderable content in requested pages.": "请求的页面中没有可渲染内容。",
    "Parse tier: flash, basic, standard, advanced": "解析档位: flash、basic、standard、advanced",
    "Scan failed: {code} {msg}": "扫描失败: {code} {msg}",
    # ------------------------------------------------------------- mineru 根
    "MinerU — your personal document center, built for agents": "MinerU — 你的个人文档中心,为 agent 而建",
    "Parse a document file.": "解析一个文档文件。",
    "Read parsed doclib content by locator.": "按定位符读取已解析的 doclib 内容。",
    "Search parsed document content.": "搜索已解析的文档内容。",
    "Search filenames only (not document content).": "仅搜索文件名(不搜索文档内容)。",
    "Show Remote API usage and limits.": "显示 Remote API 用量与限额。",
    "Mark done parse results as superseded.": "将已完成的解析结果标记为已作废。",
    # ----------------------------------------------------------------- parse
    "Path to the document file": "文档文件路径",
    "Parse tier: flash, basic, standard, advanced (default: server decides)": (
        "解析档位: flash、basic、standard、advanced(默认由服务端决定)"
    ),
    "PDF pages: '1-5,8,r3-r1' or 'all'; default: first 10 pages": "PDF 页码: '1-5,8,r3-r1' 或 'all';默认前 10 页",
    "Continue reading after a content cursor": "从内容游标之后继续读取",
    "Output format: markdown": "输出格式: markdown",
    "Force re-parse, ignore cache": "强制重新解析,忽略缓存",
    "Use remote parse-server": "使用远程 parse-server",
    "Max seconds to wait for parse to complete": "等待解析完成的最长秒数",
    "Don't wait — return immediately": "不等待,立即返回",
    "Omit document structure markers from output": "输出中省略文档结构标记",
    "Cache hit — returning cached result.": "命中缓存,返回缓存结果。",
    "Parse queued (tier={tier}). Waiting up to {wait}s...": "解析已排队(tier={tier})。最多等待 {wait} 秒…",
    "  Parse status: {status}": "  解析状态: {status}",
    "Parse failed.": "解析失败。",
    "Parse did not finish within {seconds} seconds.": "解析未在 {seconds} 秒内完成。",
    "Re-run the same command to continue waiting.": "重新运行同一命令可继续等待。",
    "No content returned from parse.": "解析未返回内容。",
    "Parse complete (tier={tier}) {tip}": "解析完成(tier={tier}) {tip}",
    "Parse still in progress{elapsed} (tier={tier}).": "解析仍在进行中{elapsed}(tier={tier})。",
    "after {seconds}s": "已等待 {seconds} 秒",
    "Parse ID": "解析任务 ID",
    "Parse IDs": "解析任务 ID",
    "{label}: {ids}": "{label}: {ids}",
    # ------------------------------------------------------------------ read
    "Doclib locator, e.g. doc:ab12cd3/tier:basic/page:4": "doclib 定位符,如 doc:ab12cd3/tier:basic/page:4",
    "Read N pages/blocks before and after the locator": "在定位符前后各读取 N 页/块",
    "Output format: markdown, image": "输出格式: markdown、image",
    "Omit continuation marker from output": "输出中省略续读标记",
    "No image asset returned.": "未返回图片素材。",
    "Image output requires a file path ending with .png, .jpg, .jpeg, or .webp; stdout is not supported.": (
        "图片输出需要以 .png、.jpg、.jpeg 或 .webp 结尾的文件路径,不支持 stdout。"
    ),
    "Image output path must end with .png, .jpg, .jpeg, or .webp.": "图片输出路径必须以 .png、.jpg、.jpeg 或 .webp 结尾。",
    # ------------------------------------------------------------------ scan
    "File or directory path to scan": "要扫描的文件或目录路径",
    "Max seconds to wait for scan completion": "等待扫描完成的最长秒数",
    "Return immediately after creating the scan": "创建扫描任务后立即返回",
    "Scan {id} {status}: seen={seen}, refreshed={refreshed}, new={new}, changed={changed}, deleted={deleted}, "
    "unreachable={unreachable}, excluded={excluded}, unsupported={unsupported}": (
        "扫描 {id} {status}: seen={seen}, refreshed={refreshed}, new={new}, changed={changed}, deleted={deleted}, "
        "unreachable={unreachable}, excluded={excluded}, unsupported={unsupported}"
    ),
    # ----------------------------------------------------------------- watch
    "Watch target management": "监视目标管理",
    "Add a directory to watch.": "添加要监视的目录。",
    "List watched directories.": "列出被监视的目录。",
    "Remove a watched directory.": "移除一个被监视的目录。",
    "Create a watch scan task for an existing watch target.": "为既有监视目标创建扫描任务。",
    "Directory path to watch": "要监视的目录路径",
    "Removable device": "可移动设备",
    "Label for this watch": "此监视的标签",
    "Watch id or exact watch root path to remove": "要移除的监视 id 或精确的监视根路径",
    "Watch id or exact watch root path": "监视 id 或精确的监视根路径",
    "Watch added: {path} (id={id})": "已添加监视: {path}(id={id})",
    "No watches configured.": "未配置任何监视。",
    "Watches": "监视列表",
    "ID": "ID",
    "Path": "路径",
    "Status": "状态",
    "Removable": "可移动",
    "Label": "标签",
    "Watch {watch_id} removed.": "监视 {watch_id} 已移除。",
    "Watch {watch_id} unchanged.": "监视 {watch_id} 无变化。",
    "Watch id {watch_id} not found.": "监视 id {watch_id} 不存在。",
    "Watch path {path} not found.": "监视路径 {path} 不存在。",
    # ---------------------------------------------------------------- search
    "Search query": "搜索词",
    "File type filter: {types}": "文件类型过滤: {types}",
    "Exact search index tier: flash, basic, standard, advanced": "精确的搜索索引档位: flash、basic、standard、advanced",
    "Minimum search index tier: flash, basic, standard, advanced": "最低搜索索引档位: flash、basic、standard、advanced",
    "Max results": "最大结果数",
    "Filename search query": "文件名搜索词",
    "File extension filter: {exts}": "文件扩展名过滤: {exts}",
    "No results found.": "未找到结果。",
    "Search results ({total} total)": "搜索结果(共 {total} 条)",
    "Document {short_id}": "文档 {short_id}",
    "Tier: {tier}": "档位: {tier}",
    "Files:": "文件:",
    "File no longer exists.": "文件已不存在。",
    # ----------------------------------------------------------------- usage
    "Remote API Usage": "Remote API 用量",
    "Remote URL: {url}": "Remote URL: {url}",
    "Access level: {level}": "访问级别: {level}",
    "Billing period: {period}": "计费周期: {period}",
    "Current": "当前",
    "Pages processed: {count}": "已处理页数: {count}",
    "Files processed: {count}": "已处理文件数: {count}",
    "Jobs created: {count}": "已创建任务数: {count}",
    "Limits": "限额",
    "Max pages per file: {count}": "单文件最大页数: {count}",
    "Max file size: {size}": "单文件最大体积: {size}",
    "Max files per job: {count}": "单任务最大文件数: {count}",
    "Max concurrent jobs: {count}": "最大并发任务数: {count}",
    "File retention: {retention}": "文件保留期: {retention}",
    "{start} - ongoing": "{start} - 进行中",
    "not specified": "未指定",
    "{days} day": "{days} 天",
    "{days} days": "{days} 天",
    # ------------------------------------------------------------------ list
    "List doclib resources": "列出 doclib 资源",
    "List parse tasks.": "列出解析任务。",
    "List scan tasks.": "列出扫描任务。",
    "List file path records.": "列出文件路径记录。",
    "List active docs.": "列出活跃文档。",
    "Parse status filter": "解析状态过滤",
    "Parse tier filter": "解析档位过滤",
    "Scan status filter": "扫描状态过滤",
    "Scan kind filter": "扫描类型过滤",
    "Watch id filter": "监视 id 过滤",
    "File status filter": "文件状态过滤",
    "File extension filter, e.g. pdf": "文件扩展名过滤,如 pdf",
    "Document file type filter, e.g. pdf": "文档文件类型过滤,如 pdf",
    "No parses found.": "未找到解析任务。",
    "Parses ({total} total)": "解析任务(共 {total} 条)",
    "Tier": "档位",
    "Pages": "页码",
    "Doc ID": "文档 ID",
    "No scans found.": "未找到扫描任务。",
    "Scans ({total} total)": "扫描任务(共 {total} 条)",
    "Kind": "类型",
    "Seen": "已发现",
    "Refreshed": "已刷新",
    "Errors": "错误数",
    "No files found.": "未找到文件。",
    "Files ({total} total)": "文件(共 {total} 条)",
    "Ext": "扩展名",
    "No docs found.": "未找到文档。",
    "Docs ({total} total)": "文档(共 {total} 条)",
    "Type": "类型",
    "Title": "标题",
    # ------------------------------------------------------------------ show
    "Show doclib resource details": "显示 doclib 资源详情",
    "Show one parse task.": "显示单个解析任务。",
    "Show one scan task.": "显示单个扫描任务。",
    "Show file, doc, and parse state for a local path.": "显示某个本地路径的文件、文档与解析状态。",
    "Show one doc by Doc ID or content hash.": "按 Doc ID 或内容哈希显示单个文档。",
    "Parse task id": "解析任务 id",
    "Scan task id": "扫描任务 id",
    "File path": "文件路径",
    "Document Doc ID or SHA-256": "文档 Doc ID 或 SHA-256",
    "Parse {id}: {status}": "解析 {id}: {status}",
    "Field": "字段",
    "Value": "值",
    "SHA-256": "SHA-256",
    "Privacy": "隐私",
    "Error": "错误",
    "Scan {id}: {status}": "扫描 {id}: {status}",
    "Metric": "指标",
    "New": "新增",
    "Changed": "已变更",
    "Deleted": "已删除",
    "Unreachable": "不可达",
    "Excluded": "已排除",
    "Unsupported": "不支持",
    "File not found in database.": "数据库中未找到该文件。",
    "File Info: {name}": "文件信息: {name}",
    "Size": "大小",
    "Page count": "页数",
    "Author": "作者",
    "Tiers": "档位",
    "Doc {short_id}": "文档 {short_id}",
    "Image based": "是否图片型",
    "Files": "文件",
    # -------------------------------------------------------- telemetry 命令
    "Telemetry management": "遥测管理",
    "Show telemetry status.": "显示遥测状态。",
    "Print the next telemetry request body without sending it.": "打印下一次遥测请求体,但不发送。",
    "Enable telemetry.": "启用遥测。",
    "Disable telemetry and clear pending local aggregates.": "禁用遥测并清除待发送的本地聚合数据。",
    "Flush pending telemetry now when telemetry is enabled (or unset during prerelease).": (
        "当遥测已启用(或预发布期未设置)时,立即上报待发送遥测。"
    ),
    "never": "从未",
    "state: {state}": "状态(state): {state}",
    "installation_id: {id}": "installation_id: {id}",
    "pending_periods: {n}": "待发送周期数: {n}",
    "pending_metrics: {n}": "待发送指标数: {n}",
    "last_flush_at: {value}": "上次上报时间: {value}",
    "telemetry flush: {reason}": "遥测已上报: {reason}",
    "telemetry {state}": "遥测 {state}",
    # ---------------------------------------------------------------- server
    "Server lifecycle management": "服务生命周期管理",
    "Start the mineru server in the background.": "在后台启动 mineru 服务。",
    "Stop the mineru server gracefully.": "优雅停止 mineru 服务。",
    "Restart the mineru server.": "重启 mineru 服务。",
    "Show server status.": "显示服务状态。",
    "Server is already running.": "服务已在运行。",
    "Another mineru server start is already in progress.": "另一个 mineru 服务启动操作正在进行中。",
    "Server is still starting (PID {pid}).": "服务仍在启动中(PID {pid})。",
    "Server failed to start within {seconds} seconds. See log: {log}; stdout: {stdout}; stderr: {stderr}": (
        "服务未在 {seconds} 秒内启动完成。日志: {log};stdout: {stdout};stderr: {stderr}"
    ),
    "Server failed to start: {error}. See log: {log}; stdout: {stdout}; stderr: {stderr}": (
        "服务启动失败: {error}。日志: {log};stdout: {stdout};stderr: {stderr}"
    ),
    "Server started (PID {pid}).": "服务已启动(PID {pid})。",
    "Server is not running.": "服务未在运行。",
    "Failed to request MinerU server shutdown: {error}": "请求 MinerU 服务关闭失败: {error}",
    "MinerU server did not stop within 15 seconds. The server was not restarted.": (
        "MinerU 服务未在 15 秒内停止。服务未被重启。"
    ),
    "Server stopped.": "服务已停止。",
    "Server is still starting.": "服务仍在启动中。",
    "MinerU Server": "MinerU 服务",
    "PID": "PID",
    "Uptime": "运行时长",
    "Home": "主目录",
    "Version": "版本",
    "Python": "Python",
    "Socket": "Socket",
    "Data dir": "数据目录",
    "SQLite": "SQLite",
    "SQLite size": "SQLite 体积",
    "Log": "日志",
    "TCP": "TCP",
    "Files tracked": "已跟踪文件",
    "Docs indexed": "已索引文档",
    "Active scans": "进行中的扫描",
    "Last scan": "上次扫描",
    "Parse queue": "解析队列",
    "Ingest queue": "入库队列",
    "(pending)": "(未分配)",
    "disabled": "已禁用",
    "Workers": "工作进程",
    "Component": "组件",
    "Running": "运行中",
    "Watch": "监视",
    "Scan": "扫描",
    "Ingest": "入库",
    "Parse": "解析",
    "Device monitor": "设备监视器",
    "Compaction": "压缩整理",
    "Health check": "健康检查",
    "Watch Stats": "监视统计",
    "Docs": "文档数",
    "Active": "活跃",
    "Pending ingest": "待入库",
    "Parses done/pending/parsing/failed": "解析 完成/排队/解析中/失败",
    "Error Summary": "错误汇总",
    "Scope": "范围",
    "Code": "错误码",
    "Count": "数量",
    "Recent Scans": "最近的扫描",
    "Source": "来源",
    "Error code": "错误码",
    "Parse Server": "Parse Server",
    "Target": "目标",
    "Healthy": "健康",
    "Endpoint": "端点",
    "Managed": "托管",
    "Restart": "重启",
    "Last probe": "上次探测",
    "Last ok": "上次成功",
    "Last fail": "上次失败",
    "Local": "本地",
    "Remote": "远程",
    "starting": "启动中",
    "Recent App Logs": "最近的应用日志",
    "Recent Access Logs": "最近的访问日志",
    "Recent Stderr Logs": "最近的 stderr 日志",
    "Recent Stdout Logs": "最近的 stdout 日志",
    "Recent Parse Server Stderr Logs": "最近的 Parse Server stderr 日志",
    "Recent Parse Server Stdout Logs": "最近的 Parse Server stdout 日志",
    "(empty)": "(空)",
    "{age}s ago": "{age} 秒前",
    "{age}m ago": "{age} 分钟前",
    "{age}h ago": "{age} 小时前",
    "{age}d ago": "{age} 天前",
    # ---------------------------------------------------------------- config
    "Configuration management": "配置管理",
    "Exclude rule management": "排除规则管理",
    "Parsing rule management": "解析规则管理",
    "Show effective configuration values.": "显示生效的配置值。",
    "Show one effective configuration value.": "显示单个生效配置值。",
    "Set a configuration override.": "设置一个配置覆盖。",
    "Remove a configuration override and fall back to the default.": "移除配置覆盖并回退到默认值。",
    "Configuration key": "配置键",
    "Configuration value": "配置值",
    "Config": "配置",
    "Key": "键",
    "default": "默认",
    "{key} = {value}  [{source}]": "{key} = {value}  [{source}]",
    "removed": "已移除",
    "unchanged": "无变化",
    "{key} = {value}  [{source}] ({action})": "{key} = {value}  [{source}]({action})",
    "Glob pattern to exclude": "要排除的 glob 模式",
    "Rule priority": "规则优先级",
    "Add an exclusion rule.": "添加一条排除规则。",
    "List exclusion rules.": "列出排除规则。",
    "Rule id to remove": "要移除的规则 id",
    "Remove an exclusion rule.": "移除一条排除规则。",
    "Exclude rule added: id={id}": "已添加排除规则: id={id}",
    "No exclude rules configured.": "未配置任何排除规则。",
    "Exclude Rules": "排除规则",
    "Pattern": "模式",
    "Priority": "优先级",
    "Exclude rule {rule_id} removed.": "排除规则 {rule_id} 已移除。",
    "Exclude rule {rule_id} unchanged.": "排除规则 {rule_id} 无变化。",
    "Glob pattern to match": "要匹配的 glob 模式",
    "PDF pages, e.g. all, 1-10 or r3-r1": "PDF 页码,如 all、1-10 或 r3-r1",
    "Allow remote parsing": "允许远程解析",
    "Rule name": "规则名称",
    "Add a parsing rule.": "添加一条解析规则。",
    "List parsing rules.": "列出解析规则。",
    "Remove a parsing rule.": "移除一条解析规则。",
    "Parsing rule added: id={id}": "已添加解析规则: id={id}",
    "No parsing rules configured.": "未配置任何解析规则。",
    "Parsing Rules": "解析规则",
    "Name": "名称",
    "Parsing rule {rule_id} removed.": "解析规则 {rule_id} 已移除。",
    "Parsing rule {rule_id} unchanged.": "解析规则 {rule_id} 无变化。",
    # --------------------------------------------------------------- cleanup
    "Clean up local doclib records and temp files.": "清理本地 doclib 记录与临时文件。",
    "Preview only": "仅预览",
    "Days threshold for temp cleanup": "临时文件清理的天数阈值",
    "Remove all file rows already marked as deleted.": "移除所有已标记为删除的文件记录。",
    "Remove docs that are no longer referenced by any file row.": "移除不再被任何文件记录引用的文档。",
    "Remove old process temp files.": "移除过期的进程临时文件。",
    "Would remove {count} deleted file record(s). Use --no-dry-run to proceed.": (
        "将移除 {count} 条已删除的文件记录。加 --no-dry-run 执行。"
    ),
    "Removed {count} deleted file record(s).": "已移除 {count} 条已删除的文件记录。",
    "Would remove {count} orphan doc(s). Use --no-dry-run to proceed.": "将移除 {count} 个孤儿文档。加 --no-dry-run 执行。",
    "Removed {count} orphan doc(s).": "已移除 {count} 个孤儿文档。",
    "Removed {count} temp file(s).": "已移除 {count} 个临时文件。",
    # ---------------------------------------------------------------- forget
    "File or directory path to forget from doclib": "要从 doclib 移除记录的文件或目录路径",
    "Would forget {count} file record(s) (matched_as={matched_as}). Use --no-dry-run to proceed.": (
        "将移除 {count} 条文件记录(matched_as={matched_as})。加 --no-dry-run 执行。"
    ),
    "Forgot {count} file record(s) (matched_as={matched_as}).": "已移除 {count} 条文件记录(matched_as={matched_as})。",
    # ------------------------------------------------------------ invalidate
    "Parse tier to invalidate (omit = all tiers)": "要作废的解析档位(省略 = 全部档位)",
    "Invalidated {count} batch(es) for {doc_id}{tier_label}. Use 'mineru parse' to re-parse.": (
        "已作废 {doc_id}{tier_label} 的 {count} 个批次。使用 'mineru parse' 重新解析。"
    ),
    "No done batches found for {doc_id}{tier_label}.": "未找到 {doc_id}{tier_label} 的已完成批次。",
    # ------------------------------------------------------ errors.py 字面量
    "Local mineru server is not running. Run 'mineru server start'.": (
        "本地 mineru 服务未运行。请先运行 'mineru server start'。"
    ),
    "MinerU server is busy. Retry the request.": "MinerU 服务繁忙。请重试请求。",
    # -------------------------------------------------------------- guidance
    "Configure a valid Official API Key to continue.": "请配置有效的官方 API Key 后继续。",
    "This Remote API feature requires an Official API Key.": "此 Remote API 功能需要官方 API Key。",
    "An Official API Key is optional and may provide registered rate limits.": (
        "官方 API Key 为可选项,配置后可获得注册用户的速率限制。"
    ),
    "An Official API Key is optional and enables registered access.": "官方 API Key 为可选项,配置后可启用注册访问。",
    "Manage or create an API Key:": "管理或创建 API Key:",
    "Set the API Key:": "设置 API Key:",
    # --------------------------------------------------- 遥测首次运行同意文案
    "Help improve MinerU by sending anonymous, locally aggregated usage and diagnostic data.\n"
    "\n"
    "Collected:\n"
    "    command names, MinerU version, OS, architecture, Python version, install channel,\n"
    "    coarse CPU/GPU categories, success/failure status, error categories, tiers,\n"
    "    and performance timing buckets.\n"
    "\n"
    "NOT collected:\n"
    "    document contents, extracted text/images, file names, file paths, raw URLs,\n"
    "    search queries, prompts, snippets, tracebacks, exception messages, hostnames,\n"
    "    usernames, account IDs, API keys, or exact CPU/GPU models.\n"
    "\n"
    "Press Enter or type Y to enable, or type N to disable.\n"
    "You can change this later with `mineru telemetry enable` or `mineru telemetry disable`.\n"
    "Preview what would be sent with `mineru telemetry preview`.": (
        "帮助改进 MinerU:发送匿名且本地聚合的使用与诊断数据。\n"
        "\n"
        "收集内容:\n"
        "    命令名、MinerU 版本、操作系统、架构、Python 版本、安装渠道、\n"
        "    粗粒度 CPU/GPU 类别、成功/失败状态、错误类别、档位,以及性能耗时分桶。\n"
        "\n"
        "不收集的内容:\n"
        "    文档内容、提取的文本/图片、文件名、文件路径、原始 URL、搜索词、提示词、\n"
        "    片段、堆栈、异常消息、主机名、用户名、账号 ID、API 密钥,或精确 CPU/GPU 型号。\n"
        "\n"
        "按回车或输入 Y 启用,输入 N 禁用。\n"
        "之后可通过 `mineru telemetry enable` 或 `mineru telemetry disable` 修改。\n"
        "使用 `mineru telemetry preview` 预览将发送的内容。"
    ),
    "Enable telemetry?": "启用遥测?",
    # ------------------------------------------------------------- kit 根
    "MinerU Kit — parsing and service tools": "MinerU Kit — 解析与服务工具",
    "Parse files or directories into markdown, middle JSON, or zip outputs.": (
        "将文件或目录解析为 markdown、middle JSON 或 zip 输出。"
    ),
    "Start the Gradio document parsing web UI backed by the MinerU V1 API.": "启动基于 MinerU V1 API 的 Gradio 文档解析界面。",
    "Forward explicit startup options and launch the self-hosted MinerU parsing API service.": (
        "转发显式启动参数,启动 self-hosted MinerU 解析 API 服务。"
    ),
    "Start the local VLM server with OpenAI-compatible chat completions.": (
        "启动本地 VLM 服务,提供 OpenAI 兼容的 chat completions。"
    ),
    "Start a standalone Router service exposing only the MinerU V1 API.": "启动只暴露 MinerU V1 API 的独立 Router 服务。",
    # ------------------------------------------------------------ kit parse
    "Input files or directories": "输入文件或目录",
    "Output path; required": "输出路径;必填",
    "PDF pages: '1-5,8,r3-r1' or 'all'; default: all pages": "PDF 页码: '1-5,8,r3-r1' 或 'all';默认全部页面",
    "Output format: markdown, middle_json, zip": "输出格式: markdown、middle_json、zip",
    "Expert backend override": "专家后端覆盖",
    "Use mineru.net official remote parse service": "使用 mineru.net 官方远程解析服务",
    "Use a custom remote parse service URL": "使用自定义远程解析服务 URL",
    "API key for remote parse service": "远程解析服务的 API key",
    "OCR mode: auto, txt, ocr": "OCR 模式: auto、txt、ocr",
    "Disable image analysis": "禁用图像分析",
    "At least one input path is required.": "至少需要一个输入路径。",
    "--backend is not allowed in remote mode.": "远程模式下不允许使用 --backend。",
    "When input is multiple files or directories, --output must be a directory path.": (
        "当输入为多个文件或目录时,--output 必须是目录路径。"
    ),
    "Failed to parse {path}: {error}": "解析 {path} 失败: {error}",
    "Parsed {count} input(s).": "已解析 {count} 个输入。",
    # ------------------------------------------------------------ kit webui
    "Gradio is a base dependency; repair the installation with `pip install 'mineru'`.": (
        "Gradio 是基础依赖;请用 `pip install 'mineru'` 修复安装。"
    ),
    "Gradio >=6.8,<7 is required; upgrade with `pip install --upgrade 'mineru'`.": (
        "需要 Gradio >=6.8,<7;请用 `pip install --upgrade 'mineru'` 升级。"
    ),
    "Unsupported API server tier '{tier}'. Supported tiers: {tiers}": "不支持的 API server 档位 '{tier}'。支持的档位: {tiers}",
    "{name} must be greater than zero": "{name} 必须大于零",
    "server_port must be between 1 and 65535": "server_port 必须在 1 到 65535 之间",
    "External MinerU V1 API base URL": "外部 MinerU V1 API 基础 URL",
    "Bearer API key; falls back to MINERU_API_KEY": "Bearer API key;未提供时回退到 MINERU_API_KEY",
    "Web UI bind host": "Web UI 绑定主机",
    "Web UI bind port; omitted: auto-select from 7860 or GRADIO_SERVER_PORT": (
        "Web UI 绑定端口;省略时从 7860 或 GRADIO_SERVER_PORT 自动选择"
    ),
    "Directory for Web UI artifacts": "Web UI 产物目录",
    "Maximum pages per non-Flash PDF conversion; omitted: unlimited": "非 Flash PDF 单次转换的最大页数;省略时不限制",
    "Show local examples": "显示本地示例",
    "Expose the Gradio event API": "开放 Gradio 事件 API",
    "LaTeX delimiters used by the Markdown preview": "Markdown 预览使用的 LaTeX 定界符",
    "Managed API server capability tier": "托管 API server 的能力档位",
    "Managed server job concurrency": "托管服务的任务并发数",
    "Managed server OCR language": "托管服务的 OCR 语言",
    "Disable managed server image analysis": "禁用托管服务的图像分析",
    "Preload managed server models": "预加载托管服务的模型",
    # ------------------------------------------------------- kit api-server
    "Unsupported server tier '{tier}'. Supported server tiers: {tiers}": "不支持的服务档位 '{tier}'。支持的档位: {tiers}",
    "Server host": "服务监听主机",
    "Server port": "服务监听端口",
    "Upload directory": "上传目录",
    "Server capability tier: flash, basic, or standard": "服务能力档位: flash、basic 或 standard",
    "Disable Flash tier advertisement and execution": "禁用 Flash 档位的广播与执行",
    "Disable Advanced tier advertisement and execution": "禁用 Advanced 档位的广播与执行",
    "Maximum concurrent parse jobs": "最大并发解析任务数",
    "Timeout for URL source downloads": "URL 来源下载超时",
    "Allow local source paths": "允许本地来源路径",
    "Maximum decoded bytes for inline sources": "内联来源的最大解码字节数",
    "Allow URL sources to use plain HTTP": "允许 URL 来源使用明文 HTTP",
    "Hybrid medium OCR language hint; accepted by other efforts for compatibility": (
        "Hybrid 中档 OCR 语言提示;为兼容性也接受其他档位传入"
    ),
    "Initialize VLM client and local Hybrid models at startup": "启动时初始化 VLM 客户端与本地 Hybrid 模型",
    "Optional fixed API key": "可选的固定 API key",
    "Remote VLM URL; empty value selects local VLM": "远程 VLM URL;留空选择本地 VLM",
    "Bearer key for the remote VLM server": "远程 VLM 服务的 Bearer key",
    "Remote VLM model name; empty value enables discovery": "远程 VLM 模型名;留空启用自动发现",
    "VLM HTTP timeout in seconds (default: 600)": "VLM HTTP 超时秒数(默认 600)",
    "VLM inference concurrency (default: 100)": "VLM 推理并发数(默认 100)",
    "API service log level: critical, error, warning, info, debug, trace; default: global log.level. "
    "Also filters the Loguru default model-log sink": (
        "API 服务日志级别: critical、error、warning、info、debug、trace;默认取全局 log.level。同时过滤 Loguru 默认模型日志 sink"
    ),
    # ------------------------------------------------------- kit vlm-server
    "VLM serving engine: auto, vllm, lmdeploy, mlx": "VLM 推理服务引擎: auto、vllm、lmdeploy、mlx",
    "Unsupported engine '{engine}'.": "不支持的引擎 '{engine}'。",
    "Using vLLM as the inference engine for VLM server.": "使用 vLLM 作为 VLM 服务的推理引擎。",
    "Using LMDeploy as the inference engine for VLM server.": "使用 LMDeploy 作为 VLM 服务的推理引擎。",
    "No automatic VLM server engine is installed. Install vLLM/LMDeploy or explicitly choose --engine mlx.": (
        "未安装可自动选择的 VLM 服务引擎。请安装 vLLM/LMDeploy,或显式指定 --engine mlx。"
    ),
    "vLLM is not installed. Please install vLLM or choose lmdeploy/mlx as the engine.": (
        "未安装 vLLM。请安装 vLLM,或选择 lmdeploy/mlx 作为引擎。"
    ),
    "LMDeploy is not installed. Please install LMDeploy or choose vllm/mlx as the engine.": (
        "未安装 LMDeploy。请安装 LMDeploy,或选择 vllm/mlx 作为引擎。"
    ),
    "MLX server is unavailable.": "MLX 服务不可用。",
    "MLX server requires Apple Silicon and macOS 14 or newer.": "MLX 服务需要 Apple Silicon 和 macOS 14 及以上。",
    "MLX server requires mlx-vlm>=0.7.0,<0.8.0; installed: {version}. Install 'mlx-vlm>=0.7.0,<0.8.0'.": (
        "MLX 服务需要 mlx-vlm>=0.7.0,<0.8.0;当前安装: {version}。请安装 'mlx-vlm>=0.7.0,<0.8.0'。"
    ),
    "mlx_vlm.server is unavailable. Install 'mlx-vlm>=0.7.0,<0.8.0'.": (
        "mlx_vlm.server 不可用。请安装 'mlx-vlm>=0.7.0,<0.8.0'。"
    ),
    "MLX-VLM is not installed. Install 'mlx-vlm>=0.7.0,<0.8.0'.": "未安装 MLX-VLM。请安装 'mlx-vlm>=0.7.0,<0.8.0'。",
    # ---------------------------------------------------------- kit models
    "Download, inspect, and verify local MinerU models.": "下载、查看与校验本地 MinerU 模型。",
    "Download an explicit repo or the tier resources required by the current backend combination.": (
        "下载显式仓库或当前后端组合所需的档位资源。"
    ),
    "Show configuration sources, effective backends, and tier resources.": "显示配置来源、有效后端与档位资源。",
    "Verify local resources for an explicit repo or the current backend combination.": (
        "校验显式仓库或当前后端组合所需的本地资源。"
    ),
    "Unsupported source '{source}'. Expected one of: {expected}.": "不支持的来源 '{source}'。可选值: {expected}。",
    "Pass either a model repo name or --tier, not both.": "请只传模型仓库名或 --tier 其中之一。",
    "Pass a model repo name or --tier.": "请传入模型仓库名或 --tier。",
    "ready": "就绪",
    "missing": "缺失",
    "Failed to download {repo}: {error}": "下载 {repo} 失败: {error}",
    "Downloaded models for {label}.": "已下载 {label} 的模型。",
    "tier {tier}": "档位 {tier}",
    "Config: {path}": "配置文件: {path}",
    "Config exists: {value}": "配置文件存在: {value}",
    "Effective small backend: {backend}": "生效的小模型后端: {backend}",
    "Effective VLM engine: {engine}": "生效的 VLM 引擎: {engine}",
    "Repos:": "仓库:",
    "Model tiers:": "模型档位:",
    "(none)": "(无)",
    "Model repo: MinerU-4_models_torch, MinerU-4_models_onnx, or a VLM repo": (
        "模型仓库: MinerU-4_models_torch、MinerU-4_models_onnx 或 VLM 仓库"
    ),
    "Model tier to prepare: basic or standard": "要准备的模型档位: basic 或 standard",
    "Small model backend: auto, onnx, torch. Ignored when REPO is given.": "小模型后端: auto、onnx、torch。指定 REPO 时忽略。",
    "Local VLM engine: auto, llama-cpp, vllm, lmdeploy, mlx. Ignored when REPO is given.": (
        "本地 VLM 引擎: auto、llama-cpp、vllm、lmdeploy、mlx。指定 REPO 时忽略。"
    ),
    "Model source: auto, huggingface, or modelscope": "模型来源: auto、huggingface 或 modelscope",
    "Optional model repo name": "可选的模型仓库名",
    "Optional model tier: basic or standard": "可选的模型档位: basic 或 standard",
    "{repo}: ok": "{repo}: 正常",
    "{repo}: missing key paths: {missing}": "{repo}: 缺少关键路径: {missing}",
    # --------------------------------------------------------- kit router
    "Unsupported worker tier '{tier}'. Supported tiers: {tiers}": "不支持的 worker 档位 '{tier}'。支持的档位: {tiers}",
    "Enable auto-reload": "启用自动重载",
    "Existing MinerU V1 API base URL; repeat to add upstreams": "既有的 MinerU V1 API 基础 URL;可重复传入以添加多个上游",
    "Local workers: auto, none, or GPU CSV": "本地 worker: auto、none 或 GPU 列表(CSV)",
    "Host for managed api-server workers": "托管 api-server worker 的主机",
    "Managed worker tier: flash, basic, standard": "托管 worker 档位: flash、basic、standard",
    "Concurrency per managed worker": "每个托管 worker 的并发数",
    "Preload models in managed workers": "在托管 worker 中预加载模型",
    # -------------------------------------------------------- kit common
    "No input files found.": "未找到输入文件。",
    "Directory input must be expanded before validation: {path}": "目录输入必须先展开再校验: {path}",
    "Unsupported file type: {path}": "不支持的文件类型: {path}",
    "Invalid input path.": "无效的输入路径。",
    "Output name collision: {existing} and {source} both map to {dest}": "输出名称冲突: {existing} 与 {source} 都映射到 {dest}",
    "--remote and --remote-url are mutually exclusive.": "--remote 与 --remote-url 互斥。",
    "Unsupported format: {format}": "不支持的格式: {format}",
}
