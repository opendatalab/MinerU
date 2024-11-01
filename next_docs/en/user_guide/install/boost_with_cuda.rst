
Boost With Cuda 
================


If your device supports CUDA and meets the GPU requirements of the
mainline environment, you can use GPU acceleration. Please select the
appropriate guide based on your system:

-  :ref:`ubuntu_22_04_lts_section`
-  :ref:`windows_10_or_11_section`

-  Quick Deployment with Docker > Docker requires a GPU with at least
   16GB of VRAM, and all acceleration features are enabled by default.

.. note:: 

   Before running this Docker, you can use the following command to
   check if your device supports CUDA acceleration on Docker. 

   bash  docker run --rm --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

.. code:: sh

   wget https://github.com/opendatalab/MinerU/raw/master/Dockerfile
   docker build -t mineru:latest .
   docker run --rm -it --gpus=all mineru:latest /bin/bash
   magic-pdf --help

.. _ubuntu_22_04_lts_section:

Ubuntu 22.04 LTS
-----------------

1. Check if NVIDIA Drivers Are Installed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   nvidia-smi

If you see information similar to the following, it means that the
NVIDIA drivers are already installed, and you can skip Step 2.

Notice:``CUDA Version`` should be >= 12.1, If the displayed version
number is less than 12.1, please upgrade the driver.

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

2. Install the Driver
~~~~~~~~~~~~~~~~~~~~~

If no driver is installed, use the following command:

.. code:: sh

   sudo apt-get update
   sudo apt-get install nvidia-driver-545

Install the proprietary driver and restart your computer after
installation.

.. code:: sh

   reboot

3. Install Anaconda
~~~~~~~~~~~~~~~~~~~

If Anaconda is already installed, skip this step.

.. code:: sh

   wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
   bash Anaconda3-2024.06-1-Linux-x86_64.sh

In the final step, enter ``yes``, close the terminal, and reopen it.

4. Create an Environment Using Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specify Python version 3.10.

.. code:: sh

   conda create -n MinerU python=3.10
   conda activate MinerU

5. Install Applications
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com

❗ After installation, make sure to check the version of ``magic-pdf``
using the following command:

.. code:: sh

   magic-pdf --version

If the version number is less than 0.7.0, please report the issue.

6. Download Models
~~~~~~~~~~~~~~~~~~

Refer to detailed instructions on :doc:`download_model_weight_files`

7. Understand the Location of the Configuration File
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After completing the `6. Download Models <#6-download-models>`__ step,
the script will automatically generate a ``magic-pdf.json`` file in the
user directory and configure the default model path. You can find the
``magic-pdf.json`` file in your user directory.

   The user directory for Linux is “/home/username”.

8. First Run
~~~~~~~~~~~~

Download a sample file from the repository and test it.

.. code:: sh

   wget https://github.com/opendatalab/MinerU/raw/master/demo/small_ocr.pdf
   magic-pdf -p small_ocr.pdf

9. Test CUDA Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~

If your graphics card has at least **8GB** of VRAM, follow these steps
to test CUDA acceleration:

   ❗ Due to the extremely limited nature of 8GB VRAM for running this
   application, you need to close all other programs using VRAM to
   ensure that 8GB of VRAM is available when running this application.

1. Modify the value of ``"device-mode"`` in the ``magic-pdf.json``
   configuration file located in your home directory.

   .. code:: json

      {
        "device-mode": "cuda"
      }

2. Test CUDA acceleration with the following command:

   .. code:: sh

      magic-pdf -p small_ocr.pdf

10. Enable CUDA Acceleration for OCR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download ``paddlepaddle-gpu``. Installation will automatically enable
   OCR acceleration.

   .. code:: sh

      python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

2. Test OCR acceleration with the following command:

   .. code:: sh

      magic-pdf -p small_ocr.pdf

.. _windows_10_or_11_section:

Windows 10/11
--------------

1. Install CUDA and cuDNN
~~~~~~~~~~~~~~~~~~~~~~~~~

Required versions: CUDA 11.8 + cuDNN 8.7.0

-  CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
-  cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x:
   https://developer.nvidia.com/rdp/cudnn-archive

2. Install Anaconda
~~~~~~~~~~~~~~~~~~~

If Anaconda is already installed, you can skip this step.

Download link: https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

3. Create an Environment Using Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python version must be 3.10.

::

   conda create -n MinerU python=3.10
   conda activate MinerU

4. Install Applications
~~~~~~~~~~~~~~~~~~~~~~~

::

   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com

..

   ❗️After installation, verify the version of ``magic-pdf``:

   .. code:: bash

      magic-pdf --version

   If the version number is less than 0.7.0, please report it in the
   issues section.

5. Download Models
~~~~~~~~~~~~~~~~~~

Refer to detailed instructions on :doc:`download_model_weight_files`

6. Understand the Location of the Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After completing the `5. Download Models <#5-download-models>`__ step,
the script will automatically generate a ``magic-pdf.json`` file in the
user directory and configure the default model path. You can find the
``magic-pdf.json`` file in your 【user directory】 .

   The user directory for Windows is “C:/Users/username”.

7. First Run
~~~~~~~~~~~~

Download a sample file from the repository and test it.

.. code:: powershell

     wget https://github.com/opendatalab/MinerU/raw/master/demo/small_ocr.pdf -O small_ocr.pdf
     magic-pdf -p small_ocr.pdf

8. Test CUDA Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~

If your graphics card has at least 8GB of VRAM, follow these steps to
test CUDA-accelerated parsing performance.

   ❗ Due to the extremely limited nature of 8GB VRAM for running this
   application, you need to close all other programs using VRAM to
   ensure that 8GB of VRAM is available when running this application.

1. **Overwrite the installation of torch and torchvision** supporting
   CUDA.

   ::

      pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

   ..

      ❗️Ensure the following versions are specified in the command:

      ::

         torch==2.3.1 torchvision==0.18.1

      These are the highest versions we support. Installing higher
      versions without specifying them will cause the program to fail.

2. **Modify the value of ``"device-mode"``** in the ``magic-pdf.json``
   configuration file located in your user directory.

   .. code:: json

      {
        "device-mode": "cuda"
      }

3. **Run the following command to test CUDA acceleration**:

   ::

      magic-pdf -p small_ocr.pdf

9. Enable CUDA Acceleration for OCR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Download paddlepaddle-gpu**, which will automatically enable OCR
   acceleration upon installation.

   ::

      pip install paddlepaddle-gpu==2.6.1

2. **Run the following command to test OCR acceleration**:

   ::

      magic-pdf -p small_ocr.pdf

