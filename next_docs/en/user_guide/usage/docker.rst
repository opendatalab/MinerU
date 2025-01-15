

Docker 
=======

.. admonition:: Important
   :class: tip

   Docker requires a GPU with at least 16GB of VRAM, and all acceleration features are enabled by default.

   Before running this Docker, you can use the following command to check if your device supports CUDA acceleration on Docker. 

   .. code-block:: bash

      bash  docker run --rm --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi


.. code:: sh

   wget https://github.com/opendatalab/MinerU/raw/master/Dockerfile
   docker build -t mineru:latest .
   docker run --rm -it --gpus=all mineru:latest /bin/bash
   magic-pdf --help

