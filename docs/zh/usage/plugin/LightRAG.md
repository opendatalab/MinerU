# 在LightRAG中使用MinerU文件解析引擎

## LightRAG简介

[LightRAG](https://github.com/HKUDS/LightRAG)  是一个轻量级的知识图谱 RAG 框架，被视为 Microsoft GraphRAG 的高效替代方案。它采用双层架构来同时管理知识图谱（KG）和向量嵌入，完美填补了传统基于向量的 RAG 与基于图谱的 RAG 之间的技术鸿沟。LightRAG专为高扩展性而设计，有效地解决了大规模图谱索引和查询时计算开销大、响应缓慢以及增量更新成本高等问题；LightRAG在支持大规模数据集的同时，即使搭载 30B开源大语言模型（LLM），也能保持极高的RAG质量。

## 在LightRAG中使用MinerU

### 配置MinerU访问方式

LightRAG支持使用以下两种模式访问MinerU：

- `official`模式：使用MinerU云端的 API v4 服务。需要先到 [MinerU官网](https://mineru.net/) 注册账号并创建API-KEY。然后在LightRAG的 `.env` 文件中添加以下配置：

```bash
MINERU_API_MODE=official
MINERU_API_TOKEN=<your_token>
# MINERU_OFFICIAL_ENDPOINT=https://mineru.net   # 默认值，通常无需修改
```

* `local`模式：使用本地部署的MInerU服务。部署方式见后面的说明。本地MinerU服务启动后在LightRAG的 `.env` 文件中添加以下配置：

```bash
MINERU_API_MODE=local
MINERU_LOCAL_ENDPOINT=http://<your_mineru_local_server_ip>:8000
```

其余MinerU的详细配置请参考LightRAG GitHub仓库根目录环境变量示例文件 [env.example](https://github.com/HKUDS/LightRAG/blob/main/env.example) 中的 MinerU 小节。针对 `official` 和 `local` 两种模式，分别有不同的环境变量配置。需要仔细阅读示例文件中的说明。

### 使用MinerU文件解析引擎

LightRAG文档处理管线支持使用MinerU作为文件解析器。启用方法可以通过配置`LIGHTRAG_PARSER`环境变量来指定什么文件后缀使用MinerU来进行文件解析。具体示例如下：

```bash
# 仅使用Mineru解析PDF文件
LIGHTRAG_PARSER=pdf:mineru
# 使用Mineru解析所有其支持的文件格式
LIGHTRAG_PARSER=*:mineru
# 使用Mineru解析所有其支持的文件格式，开启多模态（图片、表格、公式）分析，使用段落语义分块策略
LIGHTRAG_PARSER=*:mineru-iteP
```

可以单独指定某个文件使用MinerU进行文件解析。方法是在文件名后缀前添加文件名处理提示（hint）。具体示例如下：

```
my-proposal.[mineru].docx
my-proposal.[mineru-ietP].docx
```

详细的文件处理配置请参阅: [FileProcessingPipeline-zh.md](https://github.com/HKUDS/LightRAG/blob/main/docs/FileProcessingPipeline-zh.md)

## 为LightRAG本地部署 MinerU

### 通过docker compose部署MinerU

使用 Github官方介绍的方式构建docker镜像和启动服务。把本项目 `/docker` 目录下的 Dockerfile 和 compose.yaml 拷贝到本地。然后执行以下命令构建 docker 镜像:

```bash
docker build --tag mineru:latest .
```

镜像构建好之后通过以下命令启动 API 服务（参数 `--profile api` 标识仅启动MinerU的 API 服务，服务默认监听 8000 端口）：

```bash
docker compose -f compose.yaml --profile api up -d
```

### 本地部署Miner的进阶配置

在基础部署之上，建议为本地 MinerU 额外开启两项 MinerU **服务端**功能。这两项都改的是 MinerU 容器侧配置（容器内 `mineru.json` 与官方 `compose.yaml`），不涉及 LightRAG 的 env 变量；其中标题层级修正还需要一个可用的 LLM API。

- **vLLM 启动预加载**：让容器启动时就把 VLM 模型加载进显存，避免首个解析请求承担模型加载延迟。
- **标题层级修正（`title_aided`）**：MinerU 借助一个外部 LLM 修正解析输出的标题层级，提升结构化产物质量。这对依赖标题结构的 [P（段落语义）分块策略](#25-文件处理选项)尤其有帮助；`P分块策略` 优先按标题分割，标题层级越准确，分块语义越好。

**步骤1：导出并修改 `mineru-lightrag.json`**

从官方镜像中把 `/root/mineru.json` 拷到宿主机当前目录的 `mineru-lightrag.json`（用固定容器名 `temp_mineru`，无需运行容器）：

```bash
docker create --name temp_mineru mineru:latest
docker cp temp_mineru:/root/mineru.json ./mineru-lightrag.json
docker rm temp_mineru
```

然后修改 `mineru-lightrag.json` 中的 `llm-aided-config.title_aided`：填入 `api_key`，并把 `enable` 改为 `true`：

```json
"llm-aided-config": {
    "title_aided": {
        "api_key": "your_api_key",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.5-plus",
        "enable_thinking": false,
        "enable": true
    }
}
```

> `api_key` / `base_url` / `model` 需替换为用户自己可用的 LLM 服务（示例使用阿里云 DashScope 的 OpenAI 兼容接口）。

**步骤2：修改官方 `compose.yaml` 的 `api` profile 服务（`mineru-api`）**

在 `mineru-api` 服务上做三处改动：`environment` 增加 `MINERU_TOOLS_CONFIG_JSON`（让 MinerU 读改过的配置而非镜像内置 `mineru.json`），`volumes` 把宿主机 `mineru-lightrag.json` 挂进容器，`command` 追加 `--enable-vlm-preload true` 开启 vLLM 预加载。改好后的完整 `mineru-api` profile 如下（以 `# <-- 新增` 标注三处增量）：

```yaml
  mineru-api:
    image: mineru:latest
    container_name: mineru-api
    restart: always
    profiles: ["api"]
    ports:
      - 8000:8000
    environment:
      MINERU_MODEL_SOURCE: local
      MINERU_TOOLS_CONFIG_JSON: /root/mineru-lightrag.json   # <-- Added
    volumes:
      - ./mineru-lightrag.json:/root/mineru-lightrag.json    # <-- Added
    entrypoint: mineru-api
    command:
      --host 0.0.0.0
      --port 8000
      --allow-public-http-client
      --gpu-memory-utilization 0.45                          # Reserved 10GB is fine, preventing OOM errors
      --enable-vlm-preload true                              # <-- Added
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
```

> 示范中请按实际显卡情况调整 `gpu-memory-utilization` ；`environment` / `volumes` / `command` 三处为本次新增项，其余保持官方原样。

**步骤3：重启生效**

改完后重新启动 API 服务让改动生效：

```bash
docker compose -f compose.yaml --profile api up -d
```



