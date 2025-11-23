#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'installation et vérification des dépendances
Vérifie que tous les packages nécessaires sont installés
"""

import sys
import subprocess
import importlib.util

# Liste des dépendances critiques à vérifier
REQUIRED_PACKAGES = {
    'torch': 'torch>=2.0.0',
    'transformers': 'transformers>=4.30.0',
    'pandas': 'pandas>=1.5.0',
    'openpyxl': 'openpyxl>=3.0.0',
    'xlsxwriter': 'xlsxwriter>=3.0.0',
    'langdetect': 'langdetect>=1.0.9',
    'tqdm': 'tqdm>=4.65.0',
    'streamlit': 'streamlit>=1.28.0',
    'numpy': 'numpy>=1.24.0',
}

# Dépendances optionnelles
OPTIONAL_PACKAGES = {
    'psutil': 'psutil>=5.9.0',
    'pynvml': 'nvidia-ml-py3>=7.352.0',
    'bitsandbytes': 'bitsandbytes>=0.41.0',
    'accelerate': 'accelerate>=0.20.0',
    'docx': 'python-docx>=0.8.11',  # Note: import name differs from package name
}

def is_package_installed(package_name):
    """Vérifie si un package est installé"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_spec):
    """Installe un package avec pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("="*60)
    print("🔍 VÉRIFICATION DES DÉPENDANCES")
    print("="*60)

    missing_required = []
    missing_optional = []

    # Vérifier les dépendances critiques
    print("\n1️⃣ Dépendances critiques:")
    for package_name, package_spec in REQUIRED_PACKAGES.items():
        installed = is_package_installed(package_name)
        status = "✅" if installed else "❌"
        print(f"   {status} {package_name}")
        if not installed:
            missing_required.append(package_spec)

    # Vérifier les dépendances optionnelles
    print("\n2️⃣ Dépendances optionnelles:")
    for package_name, package_spec in OPTIONAL_PACKAGES.items():
        installed = is_package_installed(package_name)
        status = "✅" if installed else "⚠️"
        print(f"   {status} {package_name}")
        if not installed:
            missing_optional.append(package_spec)

    # Rapport
    print("\n" + "="*60)
    if not missing_required and not missing_optional:
        print("✅ TOUTES LES DÉPENDANCES SONT INSTALLÉES!")
        print("="*60)
        print("\n💡 Vous pouvez maintenant lancer l'application:")
        print("   streamlit run app.py")
        return 0

    # Installer les dépendances manquantes
    if missing_required:
        print(f"❌ {len(missing_required)} dépendances critiques manquantes")
        print("="*60)

        response = input("\n📥 Voulez-vous les installer automatiquement? (o/n): ")
        if response.lower() in ['o', 'y', 'yes', 'oui']:
            print("\n📥 Installation des dépendances critiques...")
            for package in missing_required:
                print(f"   Installing {package}...")
                if install_package(package):
                    print(f"   ✅ {package} installé")
                else:
                    print(f"   ❌ Échec installation {package}")
        else:
            print("\n💡 Installez manuellement avec:")
            print("   pip install -r requirements.txt")
            return 1

    if missing_optional:
        print(f"\n⚠️ {len(missing_optional)} dépendances optionnelles manquantes")
        print("   Ces packages améliorent les fonctionnalités mais ne sont pas critiques:")
        for package in missing_optional:
            print(f"   - {package}")

        response = input("\n📥 Voulez-vous les installer? (o/n): ")
        if response.lower() in ['o', 'y', 'yes', 'oui']:
            print("\n📥 Installation des dépendances optionnelles...")
            for package in missing_optional:
                print(f"   Installing {package}...")
                install_package(package)  # Ignore failures for optional packages

    print("\n" + "="*60)
    print("✅ INSTALLATION TERMINÉE")
    print("="*60)
    print("\n💡 Lancez l'application avec:")
    print("   streamlit run app.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
