# Frequently Asked Questions

If your question is not listed, try using [DeepWiki](https://deepwiki.com/opendatalab/MinerU)'s AI assistant for common issues.

For unresolved problems, join our [Discord](https://discord.gg/Tdedn9GTXq) or [WeChat](https://mineru.net/community-portal/?aliasId=3c430f94) community for support.

??? question "Why is Windows inference slow, or engine installation failing?"

    ### Windows CUDA acceleration {#windows-cuda-acceleration}

    The base package can use ONNX / CPU and llama.cpp. For Torch small models and LMDeploy, install `"mineru[full]>=4.0,<5"` in the owning environment, then check Torch CUDA support, GPU drivers, and the Python range of engine wheels.

    4.0 uses `lmdeploy>=0.17.0,<0.18` and Transformers 5. Do not reuse old LMDeploy 0.11.x wheel recipes or bypass dependency checks. See [extension modules](../quick_start/extension_modules.md) and [tiers](../usage/tiers.md).

??? question "What changed in commands, APIs, and configuration?"

    The 4.0 WebUI uses `mineru-kit webui` / `mineru-webui`, parsing APIs use `/v1/*`, and configuration uses `config.yaml`. Old `/file_parse`, `/tasks`, and `model.stack` cannot be reused directly. See [migration](../reference/migration_4.md).

??? question "Encountered the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` in Ubuntu 22.04 on WSL2"

    ### Encountered the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` in Ubuntu 22.04 on WSL2 {#wsl2-ubuntu2204-libgl}

    The `libgl` library is missing in Ubuntu 22.04 on WSL2. You can install the `libgl` library with the following command to resolve the issue:
    
    ```bash
    sudo apt-get install libgl1-mesa-glx
    ```
    
    Reference: [#388](https://github.com/opendatalab/MinerU/issues/388)


??? question "Missing text information in parsing results when installing and using on Linux systems."

    ### Missing text information in parsing results when installing and using on Linux systems. {#linux-missing-text-cjk-fonts}

    MinerU uses `pypdfium2` instead of `pymupdf` as the PDF page rendering engine in versions >=2.0 to resolve AGPLv3 license issues. On some Linux distributions, due to missing CJK fonts, some text may be lost during the process of rendering PDFs to images.
    To solve this problem, you can install the noto font package with the following commands, which are effective on Ubuntu/Debian systems:
    ```bash
    sudo apt update
    sudo apt install fonts-noto-core
    sudo apt install fonts-noto-cjk
    fc-cache -fv
    ```
    You can also directly use our [Docker deployment](../quick_start/docker_deployment.md) method to build the image, which includes the above font packages by default.
    
    Reference: [#2915](https://github.com/opendatalab/MinerU/issues/2915)
