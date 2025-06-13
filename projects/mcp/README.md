# MinerU MCP-Server

## 1. 概述

这个项目提供了一个 **MinerU MCP 服务器** (`mineru-mcp`)，它基于 **FastMCP** 框架构建。其主要功能是作为 **MinerU API** 的接口，用于将文档转换为 Markdown格式。

该服务器通过 MCP 协议公开了以下主要工具：

1. `parse_documents`：统一接口，支持处理本地文件和URL，自动根据配置选择最合适的处理方式，并自动读取转换后的内容
2. `get_ocr_languages`：获取OCR支持的语言列表

这使得其他应用程序或 MCP 客户端能够轻松地集成 MinerU 的 文档 到 Markdown 转换功能。

## 2. 核心功能

* **文档提取**: 接收文档文件输入（单个或多个 URL、单个或多个本地路径，支持doc、ppt、pdf、图片多种格式），调用 MinerU API 进行内容提取和格式转换，最终生成 Markdown 文件。
* **批量处理**: 支持同时处理多个文档文件（通过提供由空格、逗号或换行符分隔的 URL 列表或本地文件路径列表）。
* **OCR 支持**: 可选启用 OCR 功能（默认不开启），以处理扫描版或图片型文档。
* **多语言支持**: 支持多种语言的识别，可以自动检测文档语言或手动指定。
* **自动化流程**: 自动处理与 MinerU API 的交互，包括任务提交、状态轮询、结果下载解压、结果文件读取。
* **本地解析**: 支持调用本地部署的mineru模型直接解析文档，不依赖远程 API，适用于隐私敏感场景或离线环境。
* **智能路径处理**: 自动识别URL和本地文件路径，根据USE_LOCAL_API配置选择最合适的处理方式。

## 3. 安装

在开始安装之前，请确保您的系统满足以下基本要求：
* Python >= 3.10

### 3.1 使用 pip 安装 (推荐)

如果你的包已发布到 PyPI 或其他 Python 包索引，可以直接使用 pip 安装：

```bash
pip install mineru-mcp==1.0.0
```

目前版本：1.0.0

这种方式适用于不需要修改源代码的普通用户。

### 3.2 从源码安装

如果你需要修改源代码或进行开发，可以从源码安装。

克隆仓库并进入项目目录：

```bash
git clone <repository-url> # 替换为你的仓库 URL
cd mineru-mcp
```

推荐使用 `uv` 或 `pip` 配合虚拟环境进行安装：

**使用 uv (推荐):**

```bash
# 安装 uv (如果尚未安装)
# pip install uv

# 创建并激活虚拟环境
uv venv

# Linux/macOS
source .venv/bin/activate 
# Windows
# .venv\\Scripts\\activate

# 安装依赖和项目
uv pip install -e .
```

**使用 pip:**

```bash
# 创建并激活虚拟环境
python -m venv .venv

# Linux/macOS
source .venv/bin/activate 
# Windows
# .venv\\Scripts\\activate

# 安装依赖和项目
pip install -e .
```

## 4. 环境变量配置

本项目支持通过环境变量进行配置。你可以选择直接设置系统环境变量，或者在项目根目录创建 `.env` 文件（参考 `.env.example` 模板）。

### 4.1 支持的环境变量

| 环境变量                  | 说明                                                            | 默认值                    |
| ------------------------- | --------------------------------------------------------------- | ------------------------- |
| `MINERU_API_BASE`       | MinerU 远程 API 的基础 URL                                      | `https://mineru.net`    |
| `MINERU_API_KEY`        | MinerU API 密钥，需要从[官网](https://mineru.net)申请              | -                         |
| `OUTPUT_DIR`            | 转换后文件的保存路径                                            | `./downloads`           |
| `USE_LOCAL_API`         | 是否使用本地 API 进行解析                                      | `false`                 |
| `LOCAL_MINERU_API_BASE` | 本地 API 的基础 URL（当 `USE_LOCAL_API=true` 时有效）         | `http://localhost:8080` |

### 4.2 远程 API 与本地 API

本项目支持两种 API 模式：

* **远程 API**：默认模式，通过 MinerU 官方提供的云服务进行文档解析。优点是无需本地部署复杂的模型和环境，但需要网络连接和 API 密钥。
* **本地 API**：在本地部署 MinerU 引擎进行文档解析，适用于对数据隐私有高要求或需要离线使用的场景。设置 `USE_LOCAL_API=true` 时生效。

### 4.3 获取 API 密钥

要获取 `MINERU_API_KEY`，请访问 [MinerU 官网](https://mineru.net) 注册账号并申请 API 密钥。

## 5. 使用方法

### 5.1 工具概览

本项目通过 MCP 协议提供以下工具：

1. **parse_documents**：统一接口，支持处理本地文件和URL，根据 `USE_LOCAL_API` 配置自动选择合适的处理方式，并自动读取转换后的文件内容
2. **get_ocr_languages**：获取 OCR 支持的语言列表

### 5.2 参数说明

#### 5.2.1 parse_documents

| 参数                | 类型    | 说明                                                                | 默认值   | 适用模式 |
| ------------------- | ------- | ------------------------------------------------------------------- | -------- | -------- |
| `file_sources`      | 字符串  | 文件路径或URL，多个可用逗号或换行符分隔 (支持pdf、ppt、pptx、doc、docx以及图片格式jpg、jpeg、png) | -        | 全部 |
| `enable_ocr`        | 布尔值  | 是否启用 OCR 功能                                                   | `false`  | 全部 |
| `language`          | 字符串  | 文档语言，默认"ch"中文，可选"en"英文等                            | `ch`     | 全部 |
| `page_ranges`       | 字符串 (可选) | 指定页码范围，格式为逗号分隔的字符串。例如："2,4-6"：表示选取第2页、第4页至第6页；"2--2"：表示从第2页一直选取到倒数第二页。（远程API）  | `None`   | 远程API |

> **注意**：
> - 当 `USE_LOCAL_API=true` 时，如果提供了URL，这些URL会被过滤掉，只处理本地文件路径
> - 当 `USE_LOCAL_API=false` 时，会同时处理URL和本地文件路径

#### 5.2.2 get_ocr_languages

无需参数

## 6. MCP 客户端集成

你可以在任何支持 MCP 协议的客户端中使用 MinerU MCP 服务器。

### 6.1 在 Claude 中使用

将 MinerU MCP 服务器配置为 Claude 的工具，即可在 Claude 中直接使用文档转 Markdown 功能。配置工具时详情请参考 MCP 工具配置文档。根据不同的安装和使用场景，你可以选择以下两种配置方式：

#### 6.1.1 源码运行方式

如果你是从源码安装并运行 MinerU MCP，可以使用以下配置。这种方式适合你需要修改源码或者进行开发调试的场景：

```json
{
  "mcpServers": {
    "mineru-mcp": {
      "command": "uv",
      "args": ["--directory", "/Users/adrianwang/Documents/minerU-mcp", "run", "-m", "mineru.cli"],
      "env": {
        "MINERU_API_BASE": "https://mineru.net",
        "MINERU_API_KEY": "ey...",
        "OUTPUT_DIR": "./downloads",
        "USE_LOCAL_API": "true",
        "LOCAL_MINERU_API_BASE": "http://localhost:8080"
      }
    }
  }
}
```

这种配置的特点：

- 使用 `uv` 命令
- 通过 `--directory` 参数指定源码所在目录
- 使用 `-m mineru.cli` 运行模块
- 适合开发调试和定制化需求

#### 6.1.2 安装包运行方式

如果你是通过 pip 或 uv 安装了 mineru-mcp 包，可以使用以下更简洁的配置。这种方式适合生产环境或日常使用：

```json
{
  "mcpServers": {
    "mineru-mcp": {
      "command": "uvx",
      "args": ["mineru-mcp"],
      "env": {
        "MINERU_API_BASE": "https://mineru.net",
        "MINERU_API_KEY": "ey...",
        "OUTPUT_DIR": "./downloads",
        "USE_LOCAL_API": "true",
        "LOCAL_MINERU_API_BASE": "http://localhost:8080"
      }
    }
  }
}
```

这种配置的特点：

- 使用 `uvx` 命令直接运行已安装的包
- 配置更加简洁
- 不需要指定源码目录
- 适合稳定的生产环境使用

### 6.2 在 FastMCP 客户端中使用


```python
from fastmcp import FastMCP

# 初始化 FastMCP 客户端
client = FastMCP(server_url="http://localhost:8001")

# 使用 parse_documents 工具处理单个文档
result = await client.tool_call(
    tool_name="parse_documents",
    params={"file_sources": "/path/to/document.pdf"}
)

# 混合处理URLs和本地文件
result = await client.tool_call(
    tool_name="parse_documents",
    params={"file_sources": "/path/to/file.pdf, https://example.com/document.pdf"}
)

# 启用OCR
result = await client.tool_call(
    tool_name="parse_documents",
    params={"file_sources": "/path/to/file.pdf", "enable_ocr": True}
)
```

### 6.3 直接运行服务

你可以通过设置环境变量并直接运行命令的方式启动 MinerU MCP 服务器，这种方式特别适合快速测试和开发环境。

#### 6.3.1 设置环境变量

首先，确保设置了必要的环境变量。你可以通过创建 `.env` 文件（参考 `.env.example`）或直接在命令行中设置：

```bash
# Linux/macOS
export MINERU_API_BASE="https://mineru.net"
export MINERU_API_KEY="your-api-key"
export OUTPUT_DIR="./downloads"
export USE_LOCAL_API="true"  # 可选，如果需要本地解析
export LOCAL_MINERU_API_BASE="http://localhost:8080"  # 可选，如果启用本地 API

# Windows
set MINERU_API_BASE=https://mineru.net
set MINERU_API_KEY=your-api-key
set OUTPUT_DIR=./downloads
set USE_LOCAL_API=true
set LOCAL_MINERU_API_BASE=http://localhost:8080
```

#### 6.3.2 启动服务

使用以下命令启动 MinerU MCP 服务器，支持多种传输模式：

**SSE 传输模式**：
```bash
uv run mineru-mcp --transport sse
```

**Streamable HTTP 传输模式**：
```bash
uv run mineru-mcp --transport streamable-http
```

或者，如果你使用全局安装：

```bash
mineru-mcp --transport sse
# 或
mineru-mcp --transport streamable-http
```

服务默认在 `http://localhost:8001` 启动，使用的传输协议取决于你指定的 `--transport` 参数。

> **注意**：不同传输模式使用不同的路由路径：
> - SSE 模式：`/sse`（例如：`http://localhost:8001/sse`）
> - Streamable HTTP 模式：`/mcp`（例如：`http://localhost:8001/mcp`）


## 7. Docker 部署

本项目支持使用 Docker 进行部署，使你能在任何支持 Docker 的环境中快速启动 MinerU MCP 服务器。

### 7.1 使用 Docker Compose

1. 确保你已经安装了 Docker 和 Docker Compose
2. 复制项目根目录中的 `.env.example` 文件为 `.env`，并根据你的需求修改环境变量
3. 运行以下命令启动服务：

```bash
docker-compose up -d
```

服务默认会在 `http://localhost:8001` 启动。

### 7.2 手动构建 Docker 镜像

如果需要手动构建 Docker 镜像，可以使用以下命令：

```bash
docker build -t mineru-mcp:latest .
```

然后启动容器：

```bash
docker run -p 8001:8001 --env-file .env mineru-mcp:latest
```

更多 Docker 相关信息，请参考 `DOCKER_README.md` 文件。

## 8. 常见问题

### 8.1 API 密钥问题

**问题**：无法连接 MinerU API 或返回 401 错误。
**解决方案**：检查你的 API 密钥是否正确设置。在 `.env` 文件中确保 `MINERU_API_KEY` 环境变量包含有效的密钥。

### 8.2 如何优雅退出服务

**问题**：如何正确地停止 MinerU MCP 服务？
**解决方案**：服务运行时，可以通过按 `Ctrl+C` 来优雅地退出。系统会自动处理正在进行的操作，并确保所有资源得到正确释放。如果一次 `Ctrl+C` 没有响应，可以再次按下 `Ctrl+C` 强制退出。

### 8.3 文件路径问题

**问题**：使用 `parse_documents` 工具处理本地文件时报找不到文件错误。
**解决方案**：请确保使用绝对路径，或者相对于服务器运行目录的正确相对路径。

### 8.4 MCP 服务调用超时问题

**问题**：调用 `parse_documents` 工具时出现 `Error calling tool 'parse_documents': MCP error -32001: Request timed out` 错误。
**解决方案**：这个问题常见于处理大型文档或网络不稳定的情况。在某些 MCP 客户端（如 Cursor）中，超时后可能导致无法再次调用 MCP 服务，需要重启客户端。最新版本的 Cursor 中可能会显示正在调用 MCP，但实际上没有真正调用成功。建议：
1. **等待官方修复**：这是Cursor客户端的已知问题，建议等待Cursor官方修复
2. **处理小文件**：尽量只处理少量小文件，避免处理大型文档导致超时
3. **分批处理**：将多个文件分成多次请求处理，每次只处理一两个文件
4. 增加超时时间设置（如果客户端支持）
5. 对于超时后无法再次调用的问题，需要重启 MCP 客户端
6. 如果反复出现超时，请检查网络连接或考虑使用本地 API 模式

