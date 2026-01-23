## 1. 测试平台
以下为本指南测试使用的平台信息，供参考：
```
os: Ubuntu 22.04.4 LTS  
cpu: Intel x86-64
gcu: Enflame S60 
driver:  1.7.0.9
docker: 28.0.1
```

## 2. 环境准备

### 2.1 使用 Dockerfile 构建镜像

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/gcu.Dockerfile
docker build --network=host -t mineru:gcu-vllm-latest -f gcu.Dockerfile .
```


## 3. 启动 Docker 容器

```bash
docker run -u root --name mineru_docker \
      --network=host \
      --ipc=host \
      --privileged \
      -e MINERU_MODEL_SOURCE=modelscope \
      -it enflame:docker_images_topsrider_i3x_3.6.20260106_vllm0.11 \
      /bin/bash
```

执行该命令后，您将进入到Docker容器的交互式终端，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。


## 4. 注意事项

不同环境下，MinerU对Enflame加速卡的支持情况如下表所示：

<table border="1">
  <thead>
    <tr>
      <th rowspan="2" colspan="2">使用场景</th>
      <th colspan="2">容器环境</th>
    </tr>
    <tr>
      <th>vllm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">命令行工具(mineru)</td>
      <td>pipeline</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-auto-engine</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td rowspan="3">fastapi服务(mineru-api)</td>
      <td>pipeline</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-auto-engine</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td rowspan="3">gradio界面(mineru-gradio)</td>
      <td>pipeline</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-auto-engine</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>&lt;vlm/hybrid&gt;-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td colspan="2">openai-server服务（mineru-openai-server）</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td colspan="2">数据并行 (--data-parallel-size)</td>
      <td>🔴</td>
    </tr>
  </tbody>
</table>

注：  
🟢: 支持，运行较稳定，精度与Nvidia GPU基本一致  
🟡: 支持但较不稳定，在某些场景下可能出现异常，或精度存在一定差异  
🔴: 不支持，无法运行，或精度存在较大差异

>[!TIP]
>GCU加速卡指定可用加速卡的方式与AMD GPU类似，请参考[GPU isolation techniques](https://rocm.docs.amd.com/en/docs-6.2.4/conceptual/gpu-isolation.html)