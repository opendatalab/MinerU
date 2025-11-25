## 1. 测试平台
以下为本指南测试使用的平台信息，供参考：
```
os: CTyunOS 22.06  
cpu: Kunpeng-920 (aarch64)  
npu: Ascend 910B2  
driver: 23.0.3 
docker: 20.10.12
```

## 2. 环境准备

>[!NOTE]
>Ascend加速卡支持使用`vllm`或`lmdeploy`进行VLM模型推理加速。请根据实际需求选择安装和使用其中之一:

### 2.1 使用 Dockerfile 构建镜像 （vllm）

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/npu.Dockerfile
docker build --network=host -t mineru:npu-vllm-latest -f npu.Dockerfile .
``` 

### 2.2 使用 Dockerfile 构建镜像 （lmdeploy）

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/npu.Dockerfile
# 将基础镜像从 vllm 切换为 lmdeploy
sed -i '3s/^/# /' npu.Dockerfile && sed -i '5s/^# //' npu.Dockerfile
docker build --network=host -t mineru:npu-lmdeploy-latest -f npu.Dockerfile .
```

## 3. 启动 Docker 容器

```bash
docker run -u root --name mineru_docker --privileged=true \
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
    -e MINERU_MODEL_SOURCE=local \
    -e MINERU_LMDEPLOY_DEVICE=ascend \
    -it mineru:npu-vllm-latest \
    /bin/bash
```

>[!TIP]
> 请根据实际情况选择使用`vllm`或`lmdeploy`版本的镜像，如需使用lmdeploy，替换上述命令中的`mineru:npu-vllm-latest`为`mineru:npu-lmdeploy-latest`即可。

执行该命令后，您将进入到Docker容器的交互式终端，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。

## 4. 注意事项

不同环境下，MinerU对Ascend加速卡的支持情况如下表所示：

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
      <td>🟢</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-&lt;engine_name&gt;-engine</td>
      <td>🔴</td>
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
      <td>🟢</td>
      <td>🟢</td>
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
      <td>🟢</td>
      <td>🟢</td>
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
      <td>🟢</td>
      <td>🔴</td>
    </tr>
  </tbody>
</table>

注：  
🟢: 支持，运行较稳定，精度与Nvidia GPU基本一致  
🟡: 支持但较不稳定，在某些场景下可能出现异常，或精度存在一定差异  
🔴: 不支持，无法运行，或精度存在较大差异  

>[!NOTE]
>由于npu卡的特殊性，单次服务启动后，可能会在运行过程中切换推理后端（backend）类型（pipeline/vlm）时出现异常，请尽量根据实际需求选择合适的推理后端进行使用。  
>如在服务中切换推理后端类型遇到报错或异常，请重新启动服务即可。

>[!TIP]
>NPU加速卡指定可用加速卡的方式与NVIDIA GPU类似，请参考[ASCEND_RT_VISIBLE_DEVICES](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/maintenref/envvar/envref_07_0028.html)