# 🔧 Troubleshooting RTX 4090

## Problème: La quantization ne fonctionne pas sur RTX 4090

### Diagnostic automatique

Exécutez le script de diagnostic :

```bash
python test_quantization.py
```

Ce script va tester :
1. PyTorch et CUDA
2. Bitsandbytes
3. Transformers
4. Chargement d'un modèle quantizé

---

## Solutions communes

### Solution 1: Réinstaller bitsandbytes (recommandé)

La RTX 4090 nécessite une version spécifique de bitsandbytes compilée pour CUDA 12.x :

```bash
# Désinstaller l'ancienne version
pip uninstall bitsandbytes -y

# Réinstaller avec la bonne version CUDA
pip install bitsandbytes>=0.41.0
```

### Solution 2: Vérifier la version CUDA de PyTorch

La RTX 4090 fonctionne mieux avec CUDA 12.1+ :

```bash
# Vérifier la version actuelle
python -c "import torch; print(torch.version.cuda)"

# Si < 12.1, réinstaller PyTorch
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121
```

### Solution 3: Utiliser sans quantization (temporaire)

Avec 24 GB de VRAM, vous n'avez PAS besoin de quantization ! Utilisez `quantization: none` dans l'interface.

**Estimations VRAM sans quantization (BF16) :**
- NLLB-200 600M : ~2.5 GB ✅
- NLLB-200 1.3B : ~5.0 GB ✅
- M2M100 1.2B : ~4.8 GB ✅

Vous avez largement la place !

### Solution 4: Problème de compatibilité bitsandbytes-CUDA

Si bitsandbytes ne se compile pas correctement :

```bash
# Option 1: Utiliser une version pré-compilée
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl

# Option 2 (Linux): Compiler depuis les sources
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes
CUDA_VERSION=121 make cuda12x
python setup.py install
```

### Solution 5: Problème WSL2

Si vous utilisez WSL2 sur Windows :

1. Installez CUDA Toolkit dans WSL2 (pas seulement sur Windows)
2. Vérifiez que nvidia-smi fonctionne dans WSL2

```bash
nvidia-smi  # Doit afficher votre RTX 4090
```

---

## Erreurs courantes et solutions

### Erreur: "CUDA Setup failed"

```
RuntimeError: CUDA Setup failed despite GPU being available.
```

**Solution:**
```bash
pip uninstall bitsandbytes -y
pip install bitsandbytes --no-cache-dir
```

### Erreur: "libbitsandbytes_cpu.so not found"

**Solution (Linux):**
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python -c "import bitsandbytes; print(bitsandbytes.__path__[0])")
```

**Solution (Windows):**
Installez la version Windows de bitsandbytes (voir Solution 4)

### Erreur: "Unsupported GPU type"

Cette erreur ne devrait PAS apparaître avec une RTX 4090 (Ada Lovelace, compute capability 8.9).

Si elle apparaît :
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# Devrait afficher: (8, 9)
```

---

## Configuration optimale pour RTX 4090

### Recommandations

1. **Sans quantization (recommandé)** : Vous avez 24 GB, pas besoin de quantization
   - Qualité maximale
   - Vitesse maximale
   - Aucun problème de compatibilité

2. **Avec INT8 (si vous voulez l'utiliser)** :
   - Économise de la VRAM pour des modèles plus gros
   - Qualité quasi identique
   - Nécessite bitsandbytes fonctionnel

3. **Batch size optimal** :
   - Sans quantization : 128-256
   - Avec INT8 : 256-512
   - Avec INT4 : 512-1024

### Paramètres recommandés dans l'interface

```
Modèle: Quality (NLLB-200 1.3B)
Quantization: none
Batch size: 256
Preset: Quality+
```

Avec ces paramètres, vous utilisez ~5-6 GB sur vos 24 GB disponibles.

---

## Vérification de l'installation

```bash
# 1. Vérifier PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# 2. Vérifier GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. Vérifier VRAM
python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB')"

# 4. Vérifier bitsandbytes
python -c "import bitsandbytes as bnb; print(f'bitsandbytes: {bnb.__version__}')"

# 5. Test complet
python test_quantization.py
```

---

## Besoin d'aide ?

Si aucune de ces solutions ne fonctionne, partagez la sortie de :

```bash
python test_quantization.py 2>&1 | tee diagnostic.log
```

Et incluez :
- Système d'exploitation (Windows/Linux/WSL2)
- Version Python
- Output de `nvidia-smi`
- Le message d'erreur complet
