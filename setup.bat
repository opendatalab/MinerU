@echo off
setlocal

REM Check if Conda is installed
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Conda is not installed. Installing Miniconda...
    REM Download and install Miniconda silently
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe' -OutFile 'Miniconda3-latest-Windows-x86_64.exe'"
    Miniconda3-latest-Windows-x86_64.exe /S /D=%USERPROFILE%\Miniconda3
    REM Add Miniconda to the PATH
    setx PATH "%USERPROFILE%\Miniconda3;%USERPROFILE%\Miniconda3\Scripts;%USERPROFILE%\Miniconda3\Library\bin;%PATH%"
    REM Initialize Conda
    "%USERPROFILE%\Miniconda3\Scripts\conda.exe" init
    REM Activate base environment
    "%USERPROFILE%\Miniconda3\Scripts\conda.exe" activate base
)

REM Ensure Conda is available
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Conda installation failed. Please install Conda manually.
    exit /b 1
)

REM Step 1: Create and activate a Conda environment
echo Creating Conda environment 'MinerU' with Python 3.10...
conda create -n MinerU python=3.10 -y
conda activate MinerU

REM Step 2: Install MinerU dependencies
echo Installing MinerU dependencies...
pip install magic-pdf[full]==0.7.0b1 --extra-index-url https://wheels.myhloli.com
if %ERRORLEVEL% neq 0 (
    echo Failed to install dependencies. Exiting.
    exit /b 1
)

REM Step 3: Download model weight files
echo Downloading model weight files...
REM Replace with the actual command or link to download the model weights
REM Example placeholder URL, replace with the actual download URL
powershell -Command "Invoke-WebRequest -Uri 'https://example.com/path/to/model/files.zip' -OutFile 'model_files.zip'"
if %ERRORLEVEL% neq 0 (
    echo Failed to download model files. Exiting.
    exit /b 1
)
REM Extract model files
powershell -Command "Expand-Archive -Path 'model_files.zip' -DestinationPath '%USERPROFILE%\models'"
if %ERRORLEVEL% neq 0 (
    echo Failed to extract model files. Exiting.
    exit /b 1
)

REM Step 4: Copy and configure the template file
echo Configuring template file...
copy magic-pdf.template.json %USERPROFILE%\magic-pdf.json
if %ERRORLEVEL% neq 0 (
    echo Failed to copy template file. Exiting.
    exit /b 1
)

REM Modify the JSON configuration file
set "MODELS_DIR=%USERPROFILE%\models"
powershell -Command "(Get-Content %USERPROFILE%\magic-pdf.json) -replace '\"models-dir\": \"\"', '\"models-dir\": \"%MODELS_DIR%\"' | Set-Content %USERPROFILE%\magic-pdf.json"

REM Step 5: Run a quick demo to verify installation
echo Running a quick demo to verify installation...
magic-pdf --help
if %ERRORLEVEL% neq 0 (
    echo MinerU setup failed. Please check the installation steps.
    exit /b 1
)

echo MinerU setup completed successfully!
pause
endlocal
