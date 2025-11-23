# 🪟 Installation sur Windows

Guide d'installation pas à pas pour Windows 10/11.

---

## 📋 Prérequis

1. **Python 3.10 ou 3.11** (PAS 3.12/3.13 - incompatibilités avec certaines dépendances)
   - Télécharger : https://www.python.org/downloads/
   - ⚠️ Cocher "Add Python to PATH" lors de l'installation

2. **Drivers NVIDIA** à jour (pour GPU NVIDIA)
   - Télécharger : https://www.nvidia.com/Download/index.aspx
   - Version recommandée : 535+ (pour CUDA 12.x)

3. **CUDA Toolkit 12.1** (optionnel mais recommandé pour GPU)
   - Télécharger : https://developer.nvidia.com/cuda-downloads
   - Sélectionnez Windows → x86_64 → Version

---

## 🚀 Installation rapide

### Option 1: Script automatique (recommandé)

```bash
# 1. Ouvrir PowerShell ou CMD dans le dossier du projet
cd C:\Users\renau\PycharmProjects\Translate_Autolang

# 2. Vérifier Python
python --version
# Devrait afficher: Python 3.10.x ou 3.11.x

# 3. Lancer le script d'installation
python install_dependencies.py
```

Le script va :
- ✅ Vérifier toutes les dépendances
- ✅ Proposer d'installer ce qui manque
- ✅ Vous guider étape par étape

### Option 2: Installation manuelle

```bash
# 1. Installer toutes les dépendances
pip install -r requirements.txt

# 2. Vérifier l'installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## ⚠️ Problèmes courants sur Windows

### Erreur: "ModuleNotFoundError: No module named 'pandas'"

**Cause**: Dépendances non installées

**Solution**:
```bash
pip install -r requirements.txt
```

### Erreur: "NVML Shared Library Not Found" (pynvml)

**Cause**: Drivers NVIDIA non trouvés ou pynvml mal installé

**Solution**:
```bash
# Option 1: Réinstaller les drivers NVIDIA
# Télécharger depuis: https://www.nvidia.com/Download/index.aspx

# Option 2: Ignorer (le monitoring GPU sera désactivé mais l'app fonctionne)
# L'application détecte automatiquement l'absence de pynvml

# Option 3: Réinstaller nvidia-ml-py3
pip uninstall nvidia-ml-py3 -y
pip install nvidia-ml-py3
```

### Erreur: "Microsoft Visual C++ 14.0 is required"

**Cause**: Compilation de certains packages nécessite Visual C++

**Solution**:
```bash
# Télécharger et installer Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# OU installer une version précompilée de bitsandbytes
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

### Erreur: PyTorch CUDA non disponible

**Vérifier**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Devrait afficher: True
```

**Si False**, réinstaller PyTorch avec CUDA:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Erreur: "KeyError: 'translator_core'"

**Cause**: Import échoue à cause de dépendances manquantes

**Solution**:
```bash
# Tester l'import manuellement
python -c "import translator_core"

# Si erreur, installer les dépendances manquantes
pip install -r requirements.txt
```

---

## 🧪 Vérification de l'installation

```bash
# 1. Test complet des dépendances
python install_dependencies.py

# 2. Test PyTorch + CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 3. Test transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 4. Test Streamlit
streamlit hello
```

Si tous ces tests passent, vous êtes prêt ! 🎉

---

## 🚀 Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse:
```
http://localhost:8501
```

---

## 🐍 Versions Python recommandées

| Version Python | Statut | Notes |
|---------------|--------|-------|
| **3.10.x** | ✅ Recommandé | Meilleure compatibilité |
| **3.11.x** | ✅ Recommandé | Bon support |
| 3.12.x | ⚠️ Partiel | Certains packages incompatibles |
| 3.13.x | ❌ Non supporté | Trop récent, incompatibilités |

Si vous avez Python 3.12+, installez Python 3.11 :
- Télécharger : https://www.python.org/downloads/release/python-3119/
- Sélectionner "Windows installer (64-bit)"

---

## 💾 Espace disque requis

- **Modèles** : 2-5 GB par modèle (téléchargés automatiquement)
  - NLLB-200 600M : ~2.5 GB
  - NLLB-200 1.3B : ~5 GB
  - M2M100 1.2B : ~4.8 GB

- **Dépendances** : ~5 GB (PyTorch, Transformers, etc.)

**Total** : ~10-15 GB minimum

---

## 🎯 Configuration optimale pour RTX 4090

Dans l'interface Streamlit :
```
Modèle: Quality (NLLB-200 1.3B)
Quantization: none
Batch size: 128 ou 256
Preset: Quality+
```

Avec ces paramètres :
- VRAM utilisée : ~7-10 GB (sur 24 GB disponibles)
- Vitesse : Très rapide
- Qualité : Maximale

---

## 📞 Support

Si vous rencontrez toujours des problèmes :

1. **Vérifier les logs** : Les erreurs s'affichent dans le terminal
2. **Tester avec le script de diagnostic** : `python test_quantization.py`
3. **Partager les logs** : Copiez l'erreur complète depuis le terminal

### Informations utiles à fournir :
```bash
python --version
pip list | findstr "torch transformers pandas streamlit"
nvidia-smi
```
