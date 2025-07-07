# 本地部署

## 安装 MinerU

### 使用 pip 或 uv 安装

```bash
pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple
pip install uv -i https://mirrors.aliyun.com/pypi/simple
uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple 
```

### 源码安装

```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
uv pip install -e .[core] -i https://mirrors.aliyun.com/pypi/simple
```

> [!NOTE]
> Linux和macOS系统安装后自动支持cuda/mps加速，Windows用户如需使用cuda加速，
> 请前往 [Pytorch官网](https://pytorch.org/get-started/locally/) 选择合适的cuda版本安装pytorch。

### 安装完整版（支持 sglang 加速）（需确保设备有Turing及以后架构，8G显存及以上显卡）

如需使用 **sglang 加速 VLM 模型推理**，请选择合适的方式安装完整版本：

- 使用uv或pip安装
  ```bash
  uv pip install -U "mineru[all]" -i https://mirrors.aliyun.com/pypi/simple
  ```
- 从源码安装：
  ```bash
  uv pip install -e .[all] -i https://mirrors.aliyun.com/pypi/simple
  ```
  
> [!TIP]
> sglang安装过程中如发生异常，请参考[sglang官方文档](https://docs.sglang.ai/start/install.html)尝试解决或直接使用docker方式安装。

- 使用 Dockerfile 构建镜像：
  ```bash
  wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/Dockerfile
  docker build -t mineru-sglang:latest -f Dockerfile .
  ```
  启动 Docker 容器：
  ```bash
  docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    --ipc=host \
    mineru-sglang:latest \
    mineru-sglang-server --host 0.0.0.0 --port 30000
  ```
  或使用 Docker Compose 启动：
  ```bash
    wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/compose.yaml
    docker compose -f compose.yaml up -d
  ```
  
> [!TIP]
> Dockerfile默认使用`lmsysorg/sglang:v0.4.8.post1-cu126`作为基础镜像，支持Turing/Ampere/Ada Lovelace/Hopper平台，
> 如您使用较新的`Blackwell`平台，请将基础镜像修改为`lmsysorg/sglang:v0.4.8.post1-cu128-b200`。

### 安装client（用于在仅需 CPU 和网络连接的边缘设备上连接 sglang-server）

```bash
uv pip install -U mineru -i https://mirrors.aliyun.com/pypi/simple
mineru -p <input_path> -o <output_path> -b vlm-sglang-client -u http://<host_ip>:<port>
```

---