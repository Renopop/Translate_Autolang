@echo off
chcp 65001 >nul

echo ============================================
echo    Translate Autolang - Installation
echo    Optimise pour GPU NVIDIA (RTX 4090+)
echo ============================================
echo.

:: Verifier Python
echo [1/4] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH
    echo Telechargez Python 3.10+ depuis https://python.org
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo       Python %PYVER% detecte

:: Verifier CUDA
echo.
echo [2/4] Verification de CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [ATTENTION] nvidia-smi non trouve - Mode CPU sera utilise
    echo Pour GPU: Installez les drivers NVIDIA et CUDA Toolkit
    set CUDA_AVAILABLE=0
) else (
    echo       CUDA disponible
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    set CUDA_AVAILABLE=1
)

:: Mise a jour pip
echo.
echo [3/4] Installation des dependances...
python -m pip install --upgrade pip wheel setuptools >nul 2>&1

:: Installer PyTorch avec CUDA
echo       Installation de PyTorch...
if "%CUDA_AVAILABLE%"=="1" (
    echo       [GPU] Installation PyTorch avec CUDA 12.1...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo       [CPU] Installation PyTorch CPU...
    pip install torch torchvision torchaudio
)

:: Installer les autres dependances
echo       Installation des autres dependances...
pip install -r requirements.txt

:: Verifier bitsandbytes
echo.
echo [4/4] Verification de bitsandbytes...
python -c "import bitsandbytes" >nul 2>&1
if errorlevel 1 (
    echo       [INFO] bitsandbytes non disponible - quantization desactivee
    echo       La quantization INT8/INT4 ne sera pas disponible
) else (
    echo       bitsandbytes installe correctement
)

:: Test final
echo.
echo ============================================
echo    Verification de l'installation
echo ============================================
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ============================================
echo    Installation terminee !
echo ============================================
echo.
echo Pour lancer l'application:
echo    1. Double-cliquez sur launch.bat
echo    OU
echo    2. streamlit run app.py
echo.
pause
