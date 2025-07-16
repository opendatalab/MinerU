# Frequently Asked Questions

If your question is not listed, try using [DeepWiki](https://deepwiki.com/opendatalab/MinerU)'s AI assistant for common issues.

For unresolved problems, join our [Discord](https://discord.gg/Tdedn9GTXq) or [WeChat](http://mineru.space/s/V85Yl) community for support.

??? question "Encountered the error `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` in Ubuntu 22.04 on WSL2"

    The `libgl` library is missing in Ubuntu 22.04 on WSL2. You can install the `libgl` library with the following command to resolve the issue:
    
    ```bash
    sudo apt-get install libgl1-mesa-glx
    ```
    
    Reference: [#388](https://github.com/opendatalab/MinerU/issues/388)


??? question "Error when installing MinerU on CentOS 7 or Ubuntu 18: `ERROR: Failed building wheel for simsimd`"

    The new version of albumentations (1.4.21) introduces a dependency on simsimd. Since the pre-built package of simsimd for Linux requires a glibc version greater than or equal to 2.28, this causes installation issues on some Linux distributions released before 2019. You can resolve this issue by using the following command:
    ```
    conda create -n mineru python=3.11 -y
    conda activate mineru
    pip install -U "mineru[pipeline_old_linux]"
    ```
    
    Reference: [#1004](https://github.com/opendatalab/MinerU/issues/1004)


??? question "Missing text information in parsing results when installing and using on Linux systems."

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
