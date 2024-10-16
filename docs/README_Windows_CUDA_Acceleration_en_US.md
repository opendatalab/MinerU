# Windows 10/11

### 1. Install CUDA and cuDNN
Required versions: CUDA 11.8 + cuDNN 8.7.0
   - CUDA 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
   - cuDNN v8.7.0 (November 28th, 2022), for CUDA 11.x: https://developer.nvidia.com/rdp/cudnn-archive
   
### 2. Install Anaconda
   If Anaconda is already installed, you can skip this step.
   
Download link: https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Windows-x86_64.exe

### 3. Create an Environment Using Conda
   Python version must be 3.10.
   ```
   conda create -n MinerU python=3.10
   conda activate MinerU
   ```

### 4. Install Applications
   ```
   pip install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
   ```
   >❗️After installation, verify the version of `magic-pdf`:
   >  ```bash
   >  magic-pdf --version
   >  ```
   > If the version number is less than 0.7.0, please report it in the issues section.
   
### 5. Download Models
   Refer to detailed instructions on [how to download model files](how_to_download_models_en.md).

### 6. Understand the Location of the Configuration File

After completing the [5. Download Models](#5-download-models) step, the script will automatically generate a `magic-pdf.json` file in the user directory and configure the default model path.
You can find the `magic-pdf.json` file in your 【user directory】 .
> The user directory for Windows is "C:/Users/username".

### 7. First Run
   Download a sample file from the repository and test it.
   ```powershell
     (New-Object System.Net.WebClient).DownloadFile('https://github.com/opendatalab/MinerU/raw/master/demo/small_ocr.pdf', 'small_ocr.pdf')
     magic-pdf -p small_ocr.pdf
   ```

### 8. Test CUDA Acceleration
   If your graphics card has at least 8GB of VRAM, follow these steps to test CUDA-accelerated parsing performance.
   1. **Overwrite the installation of torch and torchvision** supporting CUDA.
      ```
      pip install --force-reinstall torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
      ```
      >❗️Ensure the following versions are specified in the command:
      >```
      > torch==2.3.1 torchvision==0.18.1
      >```
      >These are the highest versions we support. Installing higher versions without specifying them will cause the program to fail.
   2. **Modify the value of `"device-mode"`** in the `magic-pdf.json` configuration file located in your user directory.
     
      ```json
      {
        "device-mode": "cuda"
      }
      ```
   3. **Run the following command to test CUDA acceleration**:

      ```
      magic-pdf -p small_ocr.pdf
      ```

### 9. Enable CUDA Acceleration for OCR
   >❗️This operation requires at least 16GB of VRAM on your graphics card, otherwise it will cause the program to crash or slow down.
   1. **Download paddlepaddle-gpu**, which will automatically enable OCR acceleration upon installation.
      ```
      pip install paddlepaddle-gpu==2.6.1
      ```
   2. **Run the following command to test OCR acceleration**:
      ```
      magic-pdf -p small_ocr.pdf
      ```
