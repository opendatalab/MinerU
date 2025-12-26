## 1. 瀚博半导体

![vastaitech](https://github.com/Vastai/VastModelZOO/blob/main/images/index/logo.png?raw=true)

- 官方网址：https://www.vastaitech.com
- 模型中心：https://github.com/Vastai/VastModelZOO


## 2. 测试平台

- 以下为本指南测试使用的平台信息，供参考
    ```
    os: Ubuntu-22.04.3-LTS-x86_64
    cpu: Hygon C86-4G
    gpu: VA16 / VA1L / VA10L
    torch: 2.8.0+cpu
    torch-vacc: 1.3.3.626
    vllm: 0.11.1.dev0+gb8b302cde.d20251030.cpu
    vllm-vacc: 0.11.0.626 
    driver: 00.25.12.02 d3_3_v2_9_a3_1 3ef7cf3 20251202
    docker: 28.1.1
    ```

## 3. 环境准备

- 获取Docker镜像
    ```bash
    sudo docker pull harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP1
    ```

- 启动Docker容器
    ```bash
    sudo docker run -it \
        --privileged=true \
        --shm-size=256g \
        --name vllm_service \
        --ipc=host \
        --network=host \
        harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP1 bash
    ```


>[!TIP]
> - 镜像内已包含`torch/vllm`等相关依赖
> - 和`NVIDIA`硬件下`CUDA_VISIBLE_DEVICES`类似；在`VastAI`硬件中可以使用`VACC_VISIBLE_DEVICES`指定`可见计算卡ID`，如`-e VACC_VISIBLE_DEVICES=0,1,2,3`
> - 需指定适当的`--shm-size`虚拟内存

## 4. MinerU功能

>[!NOTE]
> - `VastAI`加速卡仅支持使用`vlm-vllm-engine`和`vlm-http-client`形式进行`VLM`模型推理加速

- 进入容器
    ```bash
    sudo docker exec -it vllm_service bash
    ```

- 安装MinerU

   - 参考官方文档安装：[README_zh-CN.md#安装-mineru](https://github.com/opendatalab/MinerU/blob/master/README_zh-CN.md#安装-mineru)

        ```bash
        # https://mirrors.163.com/pypi/simple/
        # https://mirrors.aliyun.com/pypi/simple/
        # https://pypi.mirrors.ustc.edu.cn/simple/
        # https://pypi.tuna.tsinghua.edu.cn/simple/
        # https://mirror.baidu.com/pypi/simple

        # 通过源码安装MinerU
        git clone https://github.com/opendatalab/MinerU.git
        git checkout eed479eb56bba93ee99c1a8c255d509bd2f837e5
        pip install -e .[core] -i https://mirrors.aliyun.com/pypi/simple

        # 使用pip安装MinerU
        pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple
        ```

- 使用MinerU

    - 模型准备，参考官方介绍：[model_source.md](https://github.com/opendatalab/MinerU/blob/master/docs/zh/usage/model_source.md)

    - 方式一：`vlm-vllm-engine`

        ```bash
        export MINERU_MODEL_SOURCE=modelscope

        # step1, 以`vlm-vllm-engine`方式启动MinerU解析任务
        mineru -p /path/to/demo/pdfs/demo1.pdf \
        -o ./output \
        -b vlm-vllm-engine \
        --http-timeout 1200 \
        --tensor-parallel-size 2 \
        --enforce_eager \
        --trust-remote-code \
        --max-model-len 16384
        ```

    - 方式二：`vlm-http-client`

        ```bash
        # step1, 启动vLLM API server
        vllm serve /root/.cache/modelscope/hub/models/OpenDataLab/MinerU2.5-2509-1.2B \
        --tensor-parallel-size 2 \
        --trust-remote-code \
        --enforce_eager \
        --port 8090 \
        --max-model-len 16384 \
        --served-model-name MinerU2.5-2509-1.2B

        # step2，以`vlm-http-client`方式启动MinerU解析任务
        mineru -p /path/to/demo/pdfs/demo1.pdf \
        -o ./output \
        -b vlm-http-client \
        -u http://127.0.0.1:8090 \
        --http-timeout 1200
        ```


>[!NOTE]
> - 截至`2025/12/23`，`VastAI`已支持`MinerU`至最新版本`2.6.8`，`master分支eed479eb`
> - 注意在执行任意与`vllm`相关命令需追加`--enforce_eager`参数


## 5. 注意事项

`VastAI`加速卡对`MinerU`的支持情况如下表所示：


<table border="1">
  <thead>
    <tr>
      <th rowspan="2" colspan="2">使用场景</th>
      <th>支持情况</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">命令行工具(mineru)</td>
      <td>pipeline</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-transformers</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-vllm-engine</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-lmdeploy-client</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td rowspan="5">fastapi服务(mineru-api)</td>
      <td>pipeline</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-transformers</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-vllm-engine</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-lmdeploy-engine</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td rowspan="5">gradio界面(mineru-gradio)</td>
      <td>pipeline</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-transformers</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-vllm-engine</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>vlm-lmdeploy-engine</td>
      <td>🔴</td>
    </tr>
    <tr>
      <td>vlm-http-client</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td colspan="2">openai-server服务（mineru-openai-server）</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td colspan="2">Tensor并行 (--tensor-parallel-size/--tp)</td>
      <td>🟢 仅支持tp1/tp2</td>
    </tr>
    <tr>
      <td colspan="2">数据并行 (--data-parallel-size/--dp)</td>
      <td>🔴</td>
    </tr>
  </tbody>
</table>


注：  
🟢: 支持，运行较稳定，精度与Nvidia GPU基本一致  
🟡: 支持但较不稳定，在某些场景下可能出现异常，或精度存在一定差异  
🔴: 不支持，无法运行，或精度存在较大差异
