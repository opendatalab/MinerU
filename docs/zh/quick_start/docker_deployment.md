# 使用docker部署Mineru

MinerU提供了便捷的docker部署方式，这有助于快速搭建环境并解决一些棘手的环境兼容问题。

## 使用 Dockerfile 构建镜像

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/Dockerfile
docker build -t mineru-sglang:latest -f Dockerfile .
```

> [!TIP]
> [Dockerfile](https://github.com/opendatalab/MinerU/blob/master/docker/china/Dockerfile)默认使用`lmsysorg/sglang:v0.4.9.post3-cu126`作为基础镜像，支持Turing/Ampere/Ada Lovelace/Hopper平台，
> 如您使用较新的`Blackwell`平台，请将基础镜像修改为`lmsysorg/sglang:v0.4.9.post3-cu128-b200` 再执行build操作。

## Docker说明

Mineru的docker使用了`lmsysorg/sglang`作为基础镜像，因此在docker中默认集成了`sglang`推理加速框架和必需的依赖环境。因此在满足条件的设备上，您可以直接使用`sglang`加速VLM模型推理。
> [!NOTE]
> 使用`sglang`加速VLM模型推理需要满足的条件是：
> 
> - 设备包含Turing及以后架构的显卡，且可用显存大于等于8G。
> - 物理机的显卡驱动应支持CUDA 12.6或更高版本，`Blackwell`平台应支持CUDA 12.8及更高版本，可通过`nvidia-smi`命令检查驱动版本。
> - docker中能够访问物理机的显卡设备。
>
> 如果您的设备不满足上述条件，您仍然可以使用MinerU的其他功能，但无法使用`sglang`加速VLM模型推理，即无法使用`vlm-sglang-engine`后端和启动`vlm-sglang-server`服务。

## 启动 Docker 容器

```bash
docker run --gpus all \
  --shm-size 32g \
  -p 30000:30000 -p 7860:7860 -p 8000:8000 \
  --ipc=host \
  -it mineru-sglang:latest \
  /bin/bash
```

执行该命令后，您将进入到Docker容器的交互式终端，并映射了一些端口用于可能会使用的服务，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuisglang-clientserver)。

## 通过 Docker Compose 直接启动服务

我们提供了[compose.yml](https://github.com/opendatalab/MinerU/blob/master/docker/compose.yaml)文件，您可以通过它来快速启动MinerU服务。

```bash
# 下载 compose.yaml 文件
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/compose.yaml
```
>[!NOTE]
>  
>- `compose.yaml`文件中包含了MinerU的多个服务配置，您可以根据需要选择启动特定的服务。
>- 不同的服务可能会有额外的参数配置，您可以在`compose.yaml`文件中查看并编辑。
>- 由于`sglang`推理加速框架预分配显存的特性，您可能无法在同一台机器上同时运行多个`sglang`服务，因此请确保在启动`vlm-sglang-server`服务或使用`vlm-sglang-engine`后端时，其他可能使用显存的服务已停止。

---

### 启动 sglang-server 服务
并通过`vlm-sglang-client`后端连接`sglang-server`
  ```bash
  docker compose -f compose.yaml --profile sglang-server up -d
  ```
  >[!TIP]
  >在另一个终端中通过sglang client连接sglang server（只需cpu与网络，不需要sglang环境）
  > ```bash
  > mineru -p <input_path> -o <output_path> -b vlm-sglang-client -u http://<server_ip>:30000
  > ```

---

### 启动 Web API 服务
  ```bash
  docker compose -f compose.yaml --profile api up -d
  ```
  >[!TIP]
  >在浏览器中访问 `http://<server_ip>:8000/docs` 查看API文档。

---

### 启动 Gradio WebUI 服务
  ```bash
  docker compose -f compose.yaml --profile gradio up -d
  ```
  >[!TIP]
  > 
  >- 在浏览器中访问 `http://<server_ip>:7860` 使用 Gradio WebUI。
  >- 访问 `http://<server_ip>:7860/?view=api` 使用 Gradio API。