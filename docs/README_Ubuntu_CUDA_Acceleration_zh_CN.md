# Ubuntu 22.04 LTS

## 1. 检测是否已安装nvidia驱动

```bash
nvidia-smi
```

如果看到类似如下的信息，说明已经安装了nvidia驱动，可以跳过步骤2

> [!NOTE]
> `CUDA Version` 显示的版本号应 >= 12.4，如显示的版本号小于12.4，请升级驱动

```plaintext
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.07             Driver Version: 572.83         CUDA Version: 12.8   |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                     TCC/WDDM  | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3060 Ti   WDDM  | 00000000:01:00.0  On |                  N/A |
|  0%   51C    P8              12W / 200W |   1489MiB /  8192MiB |      5%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

## 2. 安装驱动

如没有驱动，则通过如下命令

```bash
sudo apt-get update
sudo apt-get install nvidia-driver-570-server
```

安装专有驱动，安装完成后，重启电脑

```bash
reboot
```

## 3. 安装anacoda

如果已安装conda，可以跳过本步骤

```bash
wget -U NoSuchBrowser/1.0 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
```

最后一步输入yes，关闭终端重新打开

## 4. 使用conda 创建环境

```bash
conda create -n mineru 'python=3.10' -y
conda activate mineru
```

## 5. 安装应用

```bash
pip install -U magic-pdf[full] -i https://mirrors.aliyun.com/pypi/simple
```

> [!TIP]
> 下载完成后，您可以通过以下命令检查`magic-pdf`的版本：
>
> ```bash
> magic-pdf --version
> ```


## 6. 下载模型


详细参考 [如何下载模型文件](how_to_download_models_zh_cn.md)

## 7. 了解配置文件存放的位置

完成[6.下载模型](#6-下载模型)步骤后，脚本会自动生成用户目录下的magic-pdf.json文件，并自动配置默认模型路径。
您可在【用户目录】下找到magic-pdf.json文件。

> [!TIP]
> linux用户目录为 "/home/用户名"

## 8. 第一次运行

从仓库中下载样本文件，并测试

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/demo/pdfs/small_ocr.pdf
magic-pdf -p small_ocr.pdf -o ./output
```

## 9. 测试CUDA加速

如果您的显卡显存大于等于 **6GB** ，可以进行以下流程，测试CUDA解析加速效果

**1.修改【用户目录】中配置文件magic-pdf.json中"device-mode"的值**

```json
{
  "device-mode":"cuda"
}
```

**2.运行以下命令测试cuda加速效果**

```bash
magic-pdf -p small_ocr.pdf -o ./output
```
> [!TIP]
> CUDA加速是否生效可以根据log中输出的各个阶段cost耗时来简单判断，通常情况下，使用cuda加速会比cpu更快。
