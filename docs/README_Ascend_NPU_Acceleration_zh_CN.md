# Ascend NPU 加速

## 简介

本文档介绍如何在 Ascend NPU 上使用 MinerU。本文档内容已在`华为Atlas 800T A2`服务器上测试通过。
```
CPU：鲲鹏 920 aarch64 2.6GHz
NPU：Ascend 910B 64GB
OS：openEuler 22.03 (LTS-SP3)
```
由于适配 Ascend NPU 的环境较为复杂，建议使用 Docker 容器运行 MinerU。

通过docker运行MinerU前需确保物理机已安装支持CANN 8.0.RC2的驱动和固件。


## 构建镜像
请保持网络状况良好，并执行以下代码构建镜像。    
```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/ascend_npu/Dockerfile -O Dockerfile
docker build -t mineru_npu:latest .
```
如果构建过程中未发生报错则说明镜像构建成功。


## 运行容器

```bash
docker run --rm -it -u root --privileged=true \
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
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    mineru_npu:latest \
    /bin/bash -c "echo 'source /opt/mineru_venv/bin/activate' >> ~/.bashrc && exec bash"

magic-pdf --help
```


## 已知问题

- paddleocr使用内嵌onnx模型，仅在默认语言配置下能以较快速度对中英文进行识别
- 自定义lang参数时，paddleocr速度会存在明显下降情况
- layout模型使用layoutlmv3时会发生间歇性崩溃，建议使用默认配置的doclayout_yolo模型
- 表格解析仅适配了rapid_table模型，其他模型可能会无法使用