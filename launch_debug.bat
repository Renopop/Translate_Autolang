@echo off
chcp 65001 >nul

echo ============================================
echo    Translate Autolang - Mode Debug
echo ============================================
echo.

echo [DEBUG] Informations systeme:
echo ============================================
python --version
echo.

echo [DEBUG] PyTorch et CUDA:
echo ============================================
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"N/A\"}')"
echo.

echo [DEBUG] GPU Info:
echo ============================================
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory // (1024**3)} GB)') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('Aucun GPU detecte')"
echo.

echo [DEBUG] Batch size auto-detecte:
echo ============================================
python -c "from translator_core import get_optimal_batch_size, get_gpu_tier, DEFAULT_BATCH_SIZE; print(f'GPU Tier: {get_gpu_tier()}'); print(f'Batch size optimal: {get_optimal_batch_size()}'); print(f'DEFAULT_BATCH_SIZE: {DEFAULT_BATCH_SIZE}')"
echo.

echo [DEBUG] Test bitsandbytes (quantization):
echo ============================================
python -c "import bitsandbytes as bnb; print(f'bitsandbytes version: {bnb.__version__}'); print('Quantization INT8/INT4 disponible')" 2>nul || echo "bitsandbytes NON disponible - quantization desactivee"
echo.

echo [DEBUG] Modules installes:
echo ============================================
python -c "import transformers, streamlit, pandas, langdetect; print(f'transformers: {transformers.__version__}'); print(f'streamlit: {streamlit.__version__}'); print(f'pandas: {pandas.__version__}')"
echo.

echo ============================================
echo    Lancement en mode debug (logs actives)
echo ============================================
echo.

:: Activer les logs verbeux
set VERBOSE_LOGGING=1

:: Lancer avec logs
streamlit run app.py --server.headless true --logger.level debug

pause
