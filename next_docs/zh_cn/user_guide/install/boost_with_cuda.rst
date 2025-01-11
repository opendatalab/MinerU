使用 CUDA 加速
================

如果您的设备支持 CUDA 并符合主线环境的 GPU 要求，您可以使用 GPU 加速。请选择适合您系统的指南：

-  :ref:`ubuntu_22_04_lts_section`
-  :ref:`windows_10_or_11_section`
-  使用 Docker 快速部署
 
.. admonition:: Important
    :class: tip

    Docker 需要至少 16GB 显存的 GPU，并且所有加速功能默认启用。
   
    在运行此 Docker 容器之前，您可以使用以下命令检查您的设备是否支持 Docker 上的 CUDA 加速。

    .. code-block:: sh

      docker run --rm --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

.. code:: sh

    wget https://github.com/opendatalab/MinerU/raw/master/Dockerfile
    docker build -t mineru:latest .
    docker run --rm -it --gpus=all mineru:latest /bin/bash
    magic-pdf --help


.. _ubuntu_22_04_lts_section:

Ubuntu 22.04 LTS
----------------
1. 检测是否已安装 nvidia 驱动
---------------------------

.. code:: bash

   nvidia-smi

如果看到类似如下的信息，说明已经安装了 nvidia 驱动，可以跳过步骤2

.. admonition:: Important
    :class: tip

    ``CUDA Version`` 显示的版本号应 >=12.1，如显示的版本号小于12.1，请升级驱动

.. code:: text

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

2. 安装驱动
-----------

如没有驱动，则通过如下命令

.. code:: bash

   sudo apt-get update
   sudo apt-get install nvidia-driver-545

安装专有驱动，安装完成后，重启电脑

.. code:: bash

   reboot

3. 安装 anacoda
--------------

如果已安装 conda，可以跳过本步骤

.. code:: bash

   wget -U NoSuchBrowser/1.0 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
   bash Anaconda3-2024.06-1-Linux-x86_64.sh

最后一步输入yes，关闭终端重新打开

4. 使用 conda 创建环境
---------------------

需指定 python 版本为3.10

.. code:: bash

   conda create -n MinerU python=3.10
   conda activate MinerU

5. 安装应用
-----------

.. code:: bash

   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple

.. admonition:: Important
    :class: tip

    下载完成后，务必通过以下命令确认magic-pdf的版本是否正确

    .. code:: bash

       magic-pdf --version

    如果版本号小于0.7.0，请到issue中向我们反馈

6. 下载模型
-----------

详细参考 :doc:`download_model_weight_files`

7. 了解配置文件存放的位置
-------------------------

完成\ `6.下载模型 <#6-下载模型>`__\ 步骤后，脚本会自动生成用户目录下的magic-pdf.json文件，并自动配置默认模型路径。您可在【用户目录】下找到magic-pdf.json文件。

.. admonition:: Tip
    :class: tip

    linux用户目录为 “/home/用户名”

8. 第一次运行
-------------

从仓库中下载样本文件，并测试

.. code:: bash

   wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/demo/small_ocr.pdf
   magic-pdf -p small_ocr.pdf -o ./output

9. 测试CUDA加速
---------------

如果您的显卡显存大于等于 **8GB**
，可以进行以下流程，测试CUDA解析加速效果

**1.修改【用户目录】中配置文件 magic-pdf.json 中”device-mode”的值**

.. code:: json

   {
     "device-mode":"cuda"
   }

**2.运行以下命令测试 cuda 加速效果**

.. code:: bash

   magic-pdf -p small_ocr.pdf -o ./output


.. admonition:: Tip
    :class: tip

    CUDA 加速是否生效可以根据 log 中输出的各个阶段 cost 耗时来简单判断，通常情况下， ``layout detection cost`` 和 ``mfr time`` 应提速10倍以上。

10. 为 ocr 开启 cuda 加速
---------------------

**1.下载paddlepaddle-gpu, 安装完成后会自动开启ocr加速**

.. code:: bash

   python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

**2.运行以下命令测试ocr加速效果**

.. code:: bash

   magic-pdf -p small_ocr.pdf -o ./output

.. admonition:: Tip
    :class: tip

    CUDA 加速是否生效可以根据 log 中输出的各个阶段 cost 耗时来简单判断，通常情况下， ``ocr cost`` 应提速10倍以上。


.. _windows_10_or_11_section:

Windows 10/11
--------------

1. 安装 cuda 和 cuDNN
------------------

需要安装的版本 CUDA 11.8 + cuDNN 8.7.0

-  CUDA 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive
-  cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x https://developer.nvidia.com/rdp/cudnn-archive

2. 安装 anaconda
---------------

如果已安装 conda，可以跳过本步骤

下载链接：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

3. 使用 conda 创建环境
---------------------

需指定python版本为3.10

.. code:: bash

   conda create -n MinerU python=3.10
   conda activate MinerU

4. 安装应用
-----------

.. code:: bash

   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple

.. admonition:: Important
    :class: tip

    下载完成后，务必通过以下命令确认magic-pdf的版本是否正确

    .. code:: bash

      magic-pdf --version

    如果版本号小于0.7.0，请到issue中向我们反馈

5. 下载模型
-----------

详细参考 :doc:`download_model_weight_files`

6. 了解配置文件存放的位置
-------------------------

完成\ `5.下载模型 <#5-下载模型>`__\ 步骤后，脚本会自动生成用户目录下的magic-pdf.json文件，并自动配置默认模型路径。您可在【用户目录】下找到 magic-pdf.json 文件。

.. admonition:: Tip
    :class: tip

    windows 用户目录为 “C:/Users/用户名”

7. 第一次运行
-------------

从仓库中下载样本文件，并测试

.. code:: powershell

    wget https://github.com/opendatalab/MinerU/raw/master/demo/small_ocr.pdf -O small_ocr.pdf
    magic-pdf -p small_ocr.pdf -o ./output

8. 测试 CUDA 加速
---------------

如果您的显卡显存大于等于 **8GB**，可以进行以下流程，测试 CUDA 解析加速效果

**1.覆盖安装支持cuda的torch和torchvision**

.. code:: bash

   pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

.. admonition:: Important
    :class: tip

    务必在命令中指定以下版本

    .. code:: bash

      torch==2.3.1 torchvision==0.18.1

    这是我们支持的最高版本，如果不指定版本会自动安装更高版本导致程序无法运行

**2.修改【用户目录】中配置文件magic-pdf.json中”device-mode”的值**

.. code:: json

   {
     "device-mode":"cuda"
   }

**3.运行以下命令测试cuda加速效果**

.. code:: bash

   magic-pdf -p small_ocr.pdf -o ./output

.. admonition:: Tip
    :class: tip

    CUDA 加速是否生效可以根据 log 中输出的各个阶段的耗时来简单判断，通常情况下， ``layout detection time`` 和 ``mfr time`` 应提速10倍以上。

9. 为 ocr 开启 cuda 加速
--------------------

**1.下载paddlepaddle-gpu, 安装完成后会自动开启ocr加速**

.. code:: bash

   pip install paddlepaddle-gpu==2.6.1

**2.运行以下命令测试ocr加速效果**

.. code:: bash

   magic-pdf -p small_ocr.pdf -o ./output

.. admonition:: Tip
    :class: tip

    CUDA 加速是否生效可以根据 log 中输出的各个阶段 cost 耗时来简单判断，通常情况下， ``ocr time`` 应提速10倍以上。
