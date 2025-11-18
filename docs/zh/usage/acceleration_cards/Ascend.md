## 1. 测试平台
```
os: CTyunOS 22.06  
cpu: Kunpeng-920 (aarch64)  
npu: Ascend 910B2  
driver: 23.0.3 
docker: 20.10.12
```

## 2. 环境准备

Ascend加速卡支持使用`lmdeploy`或`vllm`进行VLM模型推理加速。请根据实际需求选择安装和使用其中之一:

### 2.1 使用 lmdeploy

#### 使用 Dockerfile 构建镜像

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/npu.Dockerfile
docker build --build-arg INFERENCE_ENGINE=lmdeploy -t mineru:lmdeploy-latest -f npu.Dockerfile .
```

### 2.2 使用 vllm

#### 使用 Dockerfile 构建镜像

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/npu.Dockerfile
# 下载 Dockerfile后，使用编辑器打开并注释掉第3行的基础镜像，打开第5行的基础镜像后再执行后续操作
docker build --build-arg INFERENCE_ENGINE=vllm -t mineru:vllm-latest -f npu.Dockerfile .
``` 

## 3. 启动 Docker 容器

```bash
docker run -u root --name mineru_docker --privileged=true \
    -p 30000:30000 -p 7860:7860 -p 8000:8000 \
    --ipc=host \
    --network=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /var/log/npu/:/usr/slog \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -itd mineru:lmdeploy-latest \
    /bin/bash
```

>[!TIP]
> 请根据实际情况选择使用`lmdeploy`或`vllm`版本的镜像，替换上述命令中的`mineru:lmdeploy-latest`为`mineru:vllm-latest`即可。

执行该命令后，您将进入到Docker容器的交互式终端，并映射了一些端口用于可能会使用的服务，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。