## 1. 测试平台
以下为本指南测试使用的平台信息，供参考：
```
os: Ubuntu 22.04   
cpu: INTEL x86_64
gpu: C500  
driver: 2.12.13
docker: 28.1.1
```

## 2. 环境准备

>[!NOTE]
>maca加速卡支持使用`vllm`或`lmdeploy`进行VLM模型推理加速。请根据实际需求选择安装和使用其中之一:

### 2.1 使用metax官方镜像作为基础镜像构建vllm环境镜像

1. 从metax官方仓库拉取基础镜像
    - 1.1 镜像获取地址：[https://developer.metax-tech.com/softnova/docker](https://developer.metax-tech.com/softnova/docker)  
    - 1.2 在镜像网站选择`AI`分类，软件包类型选择`vllm`，操作系统选择`ubuntu` 
    - 1.3 找到`vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64`镜像，复制拉取命令并在本地终端执行
2. 使用 Dockerfile 构建镜像 （vllm）
    ```bash
    wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/maca.Dockerfile
    docker build --network=host -t mineru:maca-vllm-latest -f maca.Dockerfile .
    ```

  
### 2.2 使用 Dockerfile 构建镜像 （lmdeploy）

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/maca.Dockerfile
# 将基础镜像从 vllm 切换为 lmdeploy
sed -i '3s/^/# /' maca.Dockerfile && sed -i '5s/^# //' maca.Dockerfile
docker build --network=host -t mineru:maca-lmdeploy-latest -f maca.Dockerfile .
```

## 3. 启动 Docker 容器

```bash
docker run --ipc host \
   --cap-add SYS_PTRACE \
   --privileged=true \
   --device=/dev/mem \
   --device=/dev/dri \
   --device=/dev/mxcd \
   --device=/dev/infiniband \
   --group-add video \
   --network=host \
   --shm-size '100gb' \
   --ulimit memlock=-1 \
   --security-opt seccomp=unconfined \
   --security-opt apparmor=unconfined \
   --name mineru_docker \
   -v /datapool:/datapool \
   -e MINERU_MODEL_SOURCE=local \
   -e MINERU_LMDEPLOY_DEVICE=maca \
   -it mineru:maca-vllm-latest \
   /bin/bash
```

>[!TIP]
> 请根据实际情况选择使用`vllm`或`lmdeploy`版本的镜像，如需使用lmdeploy，替换上述命令中的`mineru:maca-vllm-latest`为`mineru:maca-lmdeploy-latest`即可。

执行该命令后，您将进入到Docker容器的交互式终端，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。

## 4. 注意事项

不同环境下，MinerU对maca加速卡的支持情况如下表所示：

<table border="1">
  <thead>
    <tr>
      <th rowspan="2" colspan="2">使用场景</th>
      <th colspan="2">容器环境</th>
    </tr>
    <tr>
      <th>vllm</th>
      <th>lmdeploy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">命令行工具(mineru)</td>
      <td>pipeline</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-transformers</td>
      <td>🟡</td>
      <td>🟡</td>
    </tr>
    <tr>
      <td>vlm-&lt;engine_name&gt;-engine</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-http-client</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td rowspan="4">fastapi服务(mineru-api)</td>
      <td>pipeline</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-transformers</td>
      <td>🟡</td>
      <td>🟡</td>
    </tr>
    <tr>
      <td>vlm-&lt;engine_name&gt;-engine</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-http-client</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td rowspan="4">gradio界面(mineru-gradio)</td>
      <td>pipeline</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-transformers</td>
      <td>🟡</td>
      <td>🟡</td>
    </tr>
    <tr>
      <td>vlm-&lt;engine_name&gt;-engine</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-http-client</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td colspan="2">openai-server服务（mineru-openai-server）</td>
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td colspan="2">数据并行 (--data-parallel-size/--dp)</td>
      <td>🔴</td>
      <td>🔴</td>
    </tr>
  </tbody>
</table>
  
注：  
🟢: 支持，运行较稳定，精度与Nvidia GPU基本一致  
🟡: 支持但较不稳定，在某些场景下可能出现异常，或精度存在一定差异  
🔴: 不支持，无法运行，或精度存在较大差异  

>[!TIP]
>MACA加速卡指定可用加速卡的方式与NVIDIA GPU类似，请参考[使用指定GPU设备](https://opendatalab.github.io/MinerU/zh/usage/advanced_cli_parameters/#cuda_visible_devices)章节说明。