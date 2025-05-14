# Windows 10/11

### 1. Install CUDA and cuDNN

You need to install a CUDA version that is compatible with torch's requirements. For details, please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/).

- CUDA 11.8 https://developer.nvidia.com/cuda-11-8-0-download-archive
- CUDA 12.4 https://developer.nvidia.com/cuda-12-4-0-download-archive
- CUDA 12.6 https://developer.nvidia.com/cuda-12-6-0-download-archive
- CUDA 12.8 https://developer.nvidia.com/cuda-12-8-0-download-archive

### 2. Install Anaconda

If Anaconda is already installed, you can skip this step.

Download link: https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

### 3. Create an Environment Using Conda

```bash
conda create -n mineru 'python=3.12' -y
conda activate mineru
```

### 4. Install Applications

```
pip install -U magic-pdf[full]
```

> [!IMPORTANT]
> After installation, you can check the version of `magic-pdf` using the following command:
>
> ```bash
> magic-pdf --version
> ```


### 5. Download Models

Refer to detailed instructions on [how to download model files](how_to_download_models_en.md).

### 6. Understand the Location of the Configuration File

After completing the [5. Download Models](#5-download-models) step, the script will automatically generate a `magic-pdf.json` file in the user directory and configure the default model path.
You can find the `magic-pdf.json` file in your 【user directory】 .

> [!TIP]
> The user directory for Windows is "C:/Users/username".

### 7. First Run

Download a sample file from the repository and test it.

```powershell
  wget https://github.com/opendatalab/MinerU/raw/master/demo/pdfs/small_ocr.pdf -O small_ocr.pdf
  magic-pdf -p small_ocr.pdf -o ./output
```

### 8. Test CUDA Acceleration

If your graphics card has at least 6GB of VRAM, follow these steps to test CUDA-accelerated parsing performance.

1. **Overwrite the installation of torch and torchvision** supporting CUDA.(Please select the appropriate index-url based on your CUDA version. For more details, refer to the [PyTorch official website](https://pytorch.org/get-started/locally/).)

   ```
   pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Modify the value of `"device-mode"`** in the `magic-pdf.json` configuration file located in your user directory.

   ```json
   {
     "device-mode": "cuda"
   }
   ```


3. **Run the following command to test CUDA acceleration**:

   ```
   magic-pdf -p small_ocr.pdf -o ./output
   ```