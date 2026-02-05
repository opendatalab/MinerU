## 1. 测试平台
以下为本指南测试使用的平台信息，供参考：
```
os: Ubuntu 22.04.5 LTS  
cpu: AMD EPYC 9654 96-Core Processor
gpu: T100
driver: 3.0.0
docker: 28.0.4
```

## 2. 环境准备

>[!NOTE]
>Tecorigin加速卡支持使用`vllm`进行VLM模型推理加速。请根据实际需求选择安装和使用:


### 2.1 下载并加载镜像 （vllm）

```bash
wget http://wb.tecorigin.com:8082/repository/teco-customer-repo/Course/MinerU/mineru-vllm.tar

docker load -i mineru-vllm.tar
```

## 3. 启动 Docker 容器

```bash
docker run -dit --name mineru_docker \
    --privileged \
    --cap-add SYS_PTRACE \
    --cap-add SYS_ADMIN \
    --network=host \
    --shm-size=500G \
    mineru:sdaa-vllm-latest \
    /bin/bash
```

>[!TIP]
> 如需使用`vllm`环境,请执行以下操作：
> - 进入容器后，通过以下命令切换到conda环境：
>   ```bash
>   conda activate vllm_env_py310
>   ```
>
> - 切换成功后，您可以在命令行前看到`(vllm_env_py310)`的标识，这表示您已成功进入`vllm`的虚拟环境。

执行该命令后，您将进入到Docker容器的交互式终端，您可以直接在容器内运行MinerU相关命令来使用MinerU的功能。
您也可以直接通过替换`/bin/bash`为服务启动命令来启动MinerU服务，详细说明请参考[通过命令启动服务](https://opendatalab.github.io/MinerU/zh/usage/quick_usage/#apiwebuihttp-clientserver)。


## 4. 注意事项

不同环境下，MinerU对T100加速卡的支持情况如下表所示：

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
      <td colspan="2">张量并行 (--tensor-parallel-size)</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td colspan="2">数据并行 (--data-parallel-size)</td>
      <td>🟡</td>
    </tr>
  </tbody>
</table>

注：  
🟢: 支持，运行较稳定，精度与Nvidia GPU基本一致  
🟡: 支持但较不稳定，在某些场景下可能出现异常，或精度存在一定差异  
🔴: 不支持，无法运行，或精度存在较大差异

>[!TIP]
> - T100加速卡指定可用加速卡的方式与NVIDIA GPU类似，请参考[使用指定GPU设备](https://opendatalab.github.io/MinerU/zh/usage/advanced_cli_parameters/#cuda_visible_devices)章节说明,
>将环境变量`CUDA_VISIBLE_DEVICES`替换为`SDAA_VISIBLE_DEVICES`即可。 
> - 在太初平台可以通过`teco-smi -c`命令查看加速卡的使用情况，并根据需要指定空闲的加速卡ID以避免资源冲突。