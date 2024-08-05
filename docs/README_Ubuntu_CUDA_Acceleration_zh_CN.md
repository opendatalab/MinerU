# Ubuntu 22.04 LTS

## 1. 更新apt 
```bash
sudo apt-get update
```
## 2. 检测是否已安装nvidia驱动
```bash
nvidia-smi 
```
如果看到类似如下的信息，说明已经安装了nvidia驱动，可以跳过步骤3
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 537.34                 Driver Version: 537.34       CUDA Version: 12.2     |
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
## 3. 安装驱动
如没有驱动，则通过如下命令
```bash
sudo apt-get install nvidia-driver-545
```
安装专有驱动，安装完成后，重启电脑
```bash
reboot
```
## 4. 安装anacoda
如果已安装conda，可以跳过本步骤
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash Anaconda3-2024.06-1-Linux-x86_64.sh
```
最后一步输入yes，关闭终端重新打开
## 5. 使用conda 创建环境
需指定python版本为3.10
```bash
conda create -n MinerU python=3.10
conda activate MinerU
```
## 6. 安装应用
```bash
pip install magic-pdf[full]==0.6.2b1 detectron2 --extra-index-url https://wheels.myhloli.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```
> ❗️下载完成后，务必通过以下命令确认magic-pdf的版本是否正确
> 
> ```bash
> magic-pdf --version
>```
> 如果版本号小于0.6.2，请到issue中向我们反馈

## 7. 下载模型
详细参考 [如何下载模型文件](how_to_download_models_zh_cn.md)  
下载后请将models目录移动到空间较大的ssd磁盘目录  
> ❗️模型下载后请务必检查模型文件是否下载完整
> 
> 请检查目录下的模型文件大小与网页上描述是否一致，如果可以的话，最好通过sha256校验模型是否下载完整
> 
## 8. 第一次运行前的配置
在仓库根目录可以获得 [magic-pdf.template.json](../magic-pdf.template.json) 配置模版文件
> ❗️务必执行以下命令将配置文件拷贝到【用户目录】下，否则程序将无法运行
>  
> linux用户目录为 "/home/用户名"
```bash
wget https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json
cp magic-pdf.template.json ~/magic-pdf.json
```

在用户目录中找到magic-pdf.json文件并配置"models-dir"为[7. 下载模型](#7-下载模型)中下载的模型权重文件所在目录
> ❗️务必正确配置模型权重文件所在目录的【绝对路径】，否则会因为找不到模型文件而导致程序无法运行
> 
```json
{
  "models-dir": "/tmp/models"
}
```

## 9. 第一次运行
从仓库中下载样本文件，并测试
```bash
wget https://github.com/opendatalab/MinerU/raw/master/demo/small_ocr.pdf
magic-pdf pdf-command --pdf small_ocr.pdf
```
## 10. 测试CUDA加速
如果您的显卡显存大于等于8G，可以进行以下流程，测试CUDA解析加速效果

**1.修改【用户目录】中配置文件magic-pdf.json中"device-mode"的值**
```json
{
  "device-mode":"cuda"
}
```
**2.运行以下命令测试cuda加速效果**
```bash
magic-pdf pdf-command --pdf small_ocr.pdf
```

## 11. 为ocr开启cuda加速
> ❗️以下操作需显卡显存大于等于16G才可进行，否则会因为显存不足导致程序崩溃或运行速度下降

**1.下载paddlepaddle-gpu, 安装完成后会自动开启ocr加速**
```bash
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```
**2.运行以下命令测试ocr加速效果**
```bash
magic-pdf pdf-command --pdf small_ocr.pdf
```
