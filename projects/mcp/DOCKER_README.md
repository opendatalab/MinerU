# MinerU MCP-Server Docker 部署指南

## 1. 简介

本文档提供了使用 Docker 部署 MinerU MCP-Server 的详细指南。通过 Docker 部署，你可以在任何支持 Docker 的环境中快速启动 MinerU MCP 服务器，无需考虑复杂的环境配置和依赖管理。

Docker 部署的主要优势：

- **一致的运行环境**：确保在任何平台上都有相同的运行环境
- **简化部署流程**：一键启动，无需手动安装依赖
- **易于扩展和迁移**：便于在不同环境间迁移和扩展服务
- **资源隔离**：避免与宿主机其他服务产生冲突

## 2. 先决条件

在开始之前，请确保你的系统已安装以下软件：

- [Docker](https://www.docker.com/get-started) (19.03 或更高版本)
- [Docker Compose](https://docs.docker.com/compose/install/) (1.27.0 或更高版本)

你可以通过以下命令检查它们是否已正确安装：

```bash
docker --version
docker-compose --version
```

同时，你需要：

- 从 [MinerU 官网](https://mineru.net) 获取的 API 密钥（如果需要使用远程 API）
- 充足的硬盘空间，用于存储转换后的文件

## 3. 使用 Docker Compose 部署（推荐）

Docker Compose 提供了最简单的部署方式，特别适合快速开始使用或开发环境。

### 3.1 准备配置文件

1. 克隆仓库（如果尚未克隆）：

   ```bash
   git clone <repository-url>
   cd mineru-mcp
   ```

2. 创建环境变量文件：

   ```bash
   cp .env.example .env
   ```

3. 编辑 `.env` 文件，设置必要的环境变量：

   ```
   MINERU_API_BASE=https://mineru.net
   MINERU_API_KEY=你的API密钥
   OUTPUT_DIR=./downloads
   USE_LOCAL_API=false
   LOCAL_MINERU_API_BASE=http://localhost:8080
   ```

   如果你计划使用本地 API，请将 `USE_LOCAL_API` 设置为 `true`，并确保 `LOCAL_MINERU_API_BASE` 指向你的本地 API 服务地址。

### 3.2 启动服务

在项目根目录下运行：

```bash
docker-compose up -d
```

这将会：
- 构建 Docker 镜像（如果尚未构建）
- 创建并启动容器
- 在后台运行服务 (`-d` 参数)

服务将在 `http://localhost:8001` 上启动。你可以通过 MCP 客户端连接此地址。

### 3.3 查看日志

要查看服务日志，运行：

```bash
docker-compose logs -f
```

按 `Ctrl+C` 退出日志查看。

### 3.4 停止服务

要停止服务，运行：

```bash
docker-compose down
```

如果你想同时删除构建的镜像，可以使用：

```bash
docker-compose down --rmi local
```

## 4. 手动构建和运行 Docker 镜像

如果你需要更多的控制或自定义，你可以手动构建和运行 Docker 镜像。

### 4.1 构建镜像

在项目根目录下运行：

```bash
docker build -t mineru-mcp:latest .
```

这将根据 Dockerfile 构建一个名为 `mineru-mcp` 的 Docker 镜像，标签为 `latest`。

### 4.2 运行容器

使用环境变量文件运行容器：

```bash
docker run -p 8001:8001 --env-file .env mineru-mcp:latest
```

或者直接指定环境变量：

```bash
docker run -p 8001:8001 \
  -e MINERU_API_BASE=https://mineru.net \
  -e MINERU_API_KEY=你的API密钥 \
  -e OUTPUT_DIR=/app/downloads \
  -v $(pwd)/downloads:/app/downloads \
  mineru-mcp:latest
```

### 4.3 挂载卷

为了持久化存储转换后的文件，你应该挂载宿主机目录到容器的输出目录：

```bash
docker run -p 8001:8001 --env-file .env \
  -v $(pwd)/downloads:/app/downloads \
  mineru-mcp:latest
```

这将挂载当前工作目录下的 `downloads` 文件夹到容器内的 `/app/downloads` 目录。

## 5. 环境变量配置

Docker 环境中支持的环境变量与标准环境相同：

| 环境变量 | 说明 | 默认值 |
| ------------------------- | -------------------------------------------------------------- | ------------------------- |
| `MINERU_API_BASE` | MinerU 远程 API 的基础 URL | `https://mineru.net` |
| `MINERU_API_KEY` | MinerU API 密钥，需要从官网申请 | - |
| `OUTPUT_DIR` | 转换后文件的保存路径 | `/app/downloads` |
| `USE_LOCAL_API` | 是否使用本地 API 进行解析（仅适用于 `local_parse_pdf` 工具） | `false` |
| `LOCAL_MINERU_API_BASE` | 本地 API 的基础 URL（当 `USE_LOCAL_API=true` 时有效） | `http://localhost:8080` |

在 Docker 环境中，你可以：

- 通过 `--env-file` 指定环境变量文件
- 通过 `-e` 参数直接指定环境变量
- 在 `docker-compose.yml` 文件中的 `environment` 部分配置环境变量
