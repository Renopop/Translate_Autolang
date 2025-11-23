# 🌐 Traducteur Excel & Word Multilingue

Application de traduction multilingue pour fichiers Excel et Word utilisant les modèles de traduction neuronale NLLB-200 et M2M100.

## ✨ Fonctionnalités

- **Interface moderne Streamlit** : Interface web intuitive et moderne
- **Support Excel & Word** : Traduit les fichiers .xlsx, .xls et .docx
- **Préservation de la mise en forme** : Conserve le formatage des documents Word (gras, italique, couleurs, alignement)
- **Support multi-langues** : Plus de 20 langues supportées
- **Modèles de haute qualité** : NLLB-200 (600M, 1.3B) et M2M100 (1.2B)
- **Optimisation GPU** : Support CUDA avec BF16 et SDPA/FlashAttention
- **Gestion intelligente de la mémoire** : Backoff automatique en cas d'OOM
- **Mode hors-ligne** : Possibilité d'utiliser des modèles pré-téléchargés
- **Détection automatique de langue** : Détecte automatiquement la langue source
- **Découpage intelligent** : Segmentation token-aware pour de meilleures traductions

## 🚀 Installation

### Installation rapide

```bash
# 1. Vérifier les dépendances et installer automatiquement
python install_dependencies.py

# 2. Ou installer manuellement
pip install -r requirements.txt
```

### Guides d'installation détaillés

- **Windows** : Voir [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md) pour un guide complet
- **Problèmes RTX 4090** : Voir [TROUBLESHOOTING_RTX4090.md](TROUBLESHOOTING_RTX4090.md)
- **Test quantization** : Lancer `python test_quantization.py` pour diagnostiquer

### Prérequis

- **Python 3.10 ou 3.11** (recommandé, éviter 3.12+)
- (Optionnel) GPU NVIDIA avec CUDA pour de meilleures performances
- (Optionnel) Drivers NVIDIA à jour pour monitoring GPU

### Pour l'utilisation GPU (recommandé)

```bash
# PyTorch avec CUDA 12.1 (recommandé pour RTX 40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Ou CUDA 11.8 (pour GPU plus anciens)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Utilisation

### Interface Streamlit (Recommandé)

Lancez l'application web :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

#### Configuration :

1. **Modèle** : Choisissez entre Fast, Quality ou Very High
   - Fast (600M) : Rapide, qualité correcte
   - Quality (1.3B) : Bon équilibre qualité/vitesse
   - Very High (1.2B) : Meilleure qualité, plus lent

2. **Langue cible** : Sélectionnez la langue de traduction

3. **Preset** : Choisissez le compromis qualité/vitesse
   - Speed : Plus rapide (num_beams=1)
   - Balanced : Équilibré (num_beams=3)
   - Quality+ : Meilleure qualité (num_beams=5)

4. **Batch size** : Ajustez selon votre GPU (16-1024)

5. **Mode hors-ligne** : Activez si vous avez déjà téléchargé les modèles

6. **Dossier de cache** : Spécifiez où stocker/lire les modèles

#### Traduction :

1. Uploadez votre fichier Excel (.xlsx, .xls) ou Word (.docx)
2. Configurez les paramètres dans la barre latérale
3. Cliquez sur "🚀 Lancer la traduction"
4. Téléchargez le fichier traduit

### 📄 Support des documents Word (.docx)

L'application préserve la mise en forme des documents Word :

- ✅ **Styles de texte** : Gras, italique, souligné
- ✅ **Polices** : Nom, taille, couleur
- ✅ **Alignement** : Gauche, centre, droite, justifié
- ✅ **Tableaux** : Structure et contenu
- ✅ **Paragraphes** : Espacement et structure
- ✅ **Listes** : Puces et numérotation

**Note** : Les images ne sont pas traduites mais sont préservées dans le document.

## 🎯 Langues supportées

Français, English, Español, Deutsch, Italiano, Português, Nederlands, Polski, Svenska, Norsk, Dansk, Suomi, Čeština, Slovenčina, Slovenščina, Română, Български, Русский, Українська, Ελληνικά, Türkçe, العربية, עברית, हिन्दी, 中文, 日本語, 한국어

## 🏗️ Architecture

### Fichiers principaux

- **app.py** : Interface Streamlit moderne (Excel & Word)
- **translator_core.py** : Logique métier de traduction (ExcelTranslator & DocxTranslator)
- **docx_handler.py** : Gestion des documents Word avec préservation de la mise en forme
- **requirements.txt** : Liste des dépendances
- **README.md** : Documentation complète

### Optimisations

- **Séparation UI/Logique** : Code modulaire et réutilisable
- **Gestion mémoire** : Purge automatique VRAM, backoff OOM
- **Détection GPU** : Auto-tune des paramètres selon le GPU
- **Cache modèles** : Les modèles spécialistes sont mis en cache
- **Batch dynamique** : Ajustement automatique selon la VRAM disponible
- **Préservation formatage** : Métadonnées de style pour documents Word

## ⚙️ Configuration GPU

L'application détecte automatiquement votre GPU et optimise les paramètres :

- **≤ 8 GiB VRAM** : Modèle Fast, batch réduit, purges fréquentes
- **8-12 GiB VRAM** : Modèle Fast, preset Balanced
- **12-24 GiB VRAM** : Modèle Quality, preset Quality+
- **> 24 GiB VRAM** : Modèle Very High, gros batches

## 🔧 Mode hors-ligne

Pour utiliser le mode hors-ligne :

1. Téléchargez d'abord les modèles en mode en ligne
2. Les modèles sont stockés dans le dossier de cache spécifié
3. Activez le mode hors-ligne dans l'interface
4. Les modèles seront chargés depuis le cache local

Structure du cache :
```
cache_dir/
├── facebook/
│   ├── nllb-200-distilled-600M/
│   ├── nllb-200-1.3B/
│   └── m2m100_1.2B/
└── Helsinki-NLP/
    ├── opus-mt-ru-en/
    └── ...
```

## 📊 Performances

### GPU recommandé

- **Minimum** : NVIDIA GTX 1060 (6 GB)
- **Recommandé** : NVIDIA RTX 3060 (12 GB) ou supérieur
- **Optimal** : NVIDIA RTX 4090 (24 GB) ou A100

### Vitesse de traduction

- **GPU RTX 3060** : ~50-100 segments/seconde (modèle Fast)
- **GPU RTX 4090** : ~150-300 segments/seconde (modèle Fast)
- **CPU** : ~5-10 segments/seconde (beaucoup plus lent)

## 🐛 Dépannage

### Erreur "CUDA out of memory"

- Réduisez le batch size
- Utilisez le modèle "Fast"
- Activez le preset "Speed"
- Purgez la VRAM avec le bouton dédié

### Modèle introuvable en mode hors-ligne

- Vérifiez que le dossier de cache contient les modèles
- Assurez-vous que la structure est correcte (org/model)
- Téléchargez d'abord en mode en ligne

### Application lente

- Vérifiez que CUDA est disponible (voir debug info)
- Utilisez un GPU si possible
- Augmentez le batch size si vous avez de la VRAM
- Utilisez le preset "Speed"

## 📝 Changelog

### Version 2.1 (Support Word)
- 📄 **Support des documents Word (.docx)**
- 🎨 **Préservation de la mise en forme** (gras, italique, couleurs, polices, alignement)
- 📊 **Support des tableaux** dans les documents Word
- 🔧 Classe DocxTranslator dédiée
- 📦 Module docx_handler pour la gestion des métadonnées
- 📚 Documentation enrichie

### Version 2.0 (Streamlit)
- ✨ Interface Streamlit moderne et intuitive
- 🏗️ Refactoring complet du code
- 📦 Séparation UI/Logique métier
- 🎨 Design moderne avec CSS personnalisé
- 📊 Barre de progression en temps réel
- 💾 Téléchargement direct du fichier traduit
- 🔍 Informations de debug intégrées

### Version 1.0 (Tkinter)
- Interface Tkinter fonctionnelle
- Support multi-langues Excel
- Optimisations GPU/CUDA
- Mode hors-ligne

## 👤 Auteur

Renaud LOISON

## 📄 Licence

Ce projet est sous licence MIT.

## 🙏 Remerciements

- Modèles NLLB-200 par Meta AI
- Modèles M2M100 par Meta AI
- Modèles OPUS-MT par Helsinki-NLP
- Streamlit pour l'interface web
- Hugging Face Transformers
- python-docx pour la gestion des documents Word
