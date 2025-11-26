@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================
echo    Translate Autolang - Lancement
echo ============================================
echo.

:: Verifier que venv existe
if not exist "venv\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel non trouve!
    echo Veuillez d'abord executer install.bat
    pause
    exit /b 1
)

:: Activer venv
call venv\Scripts\activate.bat

:: Afficher info GPU
echo [INFO] Detection du GPU...
python -c "import torch; gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; vram = torch.cuda.get_device_properties(0).total_memory // (1024**3) if torch.cuda.is_available() else 0; print(f'       GPU: {gpu_name}'); print(f'       VRAM: {vram} GB') if vram else None" 2>nul

:: Afficher batch size recommande
echo.
echo [INFO] Configuration auto-adaptive...
python -c "from translator_core import get_optimal_batch_size, get_gpu_tier; bs=get_optimal_batch_size(); tier=get_gpu_tier(); print(f'       Tier GPU: {tier}'); print(f'       Batch size recommande: {bs}')" 2>nul

echo.
echo ============================================
echo    Demarrage de l'application...
echo ============================================
echo.
echo L'application va s'ouvrir dans votre navigateur.
echo Pour arreter: Ctrl+C dans cette fenetre
echo.

:: Lancer Streamlit
streamlit run app.py --server.headless true --browser.gatherUsageStats false

:: Si Streamlit se ferme
echo.
echo Application fermee.
pause
