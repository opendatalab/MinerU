## 1. 测试平台
```
os: Ubuntu 22.04   
cpu: INTEL x86_64
ppu: ZW810E  
driver: 1.4.0
docker: 26.1.4
```

## 2. 环境准备

ppu加速卡支持使用`lmdeploy`或`vllm`进行VLM模型推理加速。请根据实际需求选择安装和使用其中之一:

### 2.1 使用 Dockerfile 构建镜像 （lmdeploy）

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/ppu.Dockerfile
docker build --network=host -t mineru:ppu-lmdeploy-latest -f ppu.Dockerfile .
```

### 2.2 使用 Dockerfile 构建镜像 （vllm）

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/ppu.Dockerfile
# 将基础镜像从lmdeploy切换为vllm
sed -i '2s/^/# /' ppu.Dockerfile && sed -i '4s/^# //' ppu.Dockerfile
docker build --network=host -t mineru:ppu-vllm-latest -f ppu.Dockerfile .
``` 

## 3. 启动 Docker 容器

```bash
docker run --privileged=true \
  --name mineru_docker \
  --device=/dev/alixpu \
  --device=/dev/alixpu_ctl \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=500g \
  -v /mnt:/mnt \
  -v /datapool:/datapool \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e MINERU_MODEL_SOURCE=local \
  -it mineru:ppu-lmdeploy-latest \
  /bin/bash
```

>[!TIP]
> 请根据实际情况选择使用`lmdeploy`或`vllm`版本的镜像，替换上述命令中的`mineru:ppu-lmdeploy-latest`为`mineru:ppu-vllm-latest`即可。

执行该命令后，您将进入到Docker容器的交互式终端，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。

## 4. 注意事项

不同环境下，MinerU对ppu加速卡的支持情况如下表所示：

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
      <td>🟡</td>
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
      <td>🔴</td>
      <td>🟡</td>
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
      <td>🔴</td>
      <td>🟡</td>
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
      <td>🟧</td>
      <td>🔴</td>
    </tr>
  </tbody>
</table>

注：  
🟢: 支持  
🟡: 支持但稍有不稳定，在少数某些场景下可能出现异常  
🟧: 支持但极度不稳定，在大多数场景下可能出现异常  
🔴: 不支持  
