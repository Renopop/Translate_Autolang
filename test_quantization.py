#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de diagnostic pour la quantization
Teste si bitsandbytes fonctionne correctement avec votre GPU
"""

import sys
import torch

print("="*60)
print("🔍 DIAGNOSTIC QUANTIZATION")
print("="*60)

# 1. Test PyTorch et CUDA
print("\n1️⃣ PyTorch et CUDA:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    print("   ❌ CUDA non disponible!")
    sys.exit(1)

# 2. Test bitsandbytes
print("\n2️⃣ Bitsandbytes:")
try:
    import bitsandbytes as bnb
    print(f"   ✅ bitsandbytes version: {bnb.__version__}")

    # Test CUDA ops
    try:
        # Test si les opérations CUDA sont disponibles
        test_tensor = torch.randn(10, 10).cuda()
        print(f"   ✅ CUDA ops disponibles")
    except Exception as e:
        print(f"   ⚠️ CUDA ops error: {e}")

except ImportError as e:
    print(f"   ❌ bitsandbytes non installé: {e}")
    print(f"   💡 Installez avec: pip install bitsandbytes")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Erreur: {e}")
    sys.exit(1)

# 3. Test transformers et BitsAndBytesConfig
print("\n3️⃣ Transformers:")
try:
    import transformers
    from transformers import BitsAndBytesConfig
    print(f"   ✅ transformers version: {transformers.__version__}")
    print(f"   ✅ BitsAndBytesConfig disponible")
except ImportError as e:
    print(f"   ❌ Erreur import: {e}")
    sys.exit(1)

# 4. Test de chargement avec quantization
print("\n4️⃣ Test de chargement quantizé:")
print("   Tentative de chargement d'un petit modèle en int8...")

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "facebook/nllb-200-distilled-600M"

    # Test INT8
    print(f"   📥 Chargement {model_name} en INT8...")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print(f"   ✅ Chargement INT8 réussi!")

    # Vérifier la VRAM utilisée
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print(f"   📊 VRAM allouée: {allocated:.2f} GB")
        print(f"   📊 VRAM réservée: {reserved:.2f} GB")

    # Nettoyage
    del model
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("✅ TOUS LES TESTS RÉUSSIS!")
    print("="*60)
    print("\n💡 La quantization devrait fonctionner correctement.")
    print("   Si vous avez toujours des problèmes, partagez les logs.")

except Exception as e:
    print(f"   ❌ Erreur lors du chargement: {e}")
    print(f"\n📋 Type d'erreur: {type(e).__name__}")
    import traceback
    print("\n📋 Traceback complet:")
    traceback.print_exc()

    print("\n" + "="*60)
    print("❌ ÉCHEC DU TEST")
    print("="*60)

    # Suggestions de solutions
    print("\n💡 Solutions possibles:")
    print("   1. Réinstallez bitsandbytes:")
    print("      pip uninstall bitsandbytes -y")
    print("      pip install bitsandbytes>=0.41.0")
    print()
    print("   2. Vérifiez la compatibilité CUDA:")
    print("      pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("   3. Si vous utilisez WSL2, installez CUDA Toolkit dans WSL")
    print()
    print("   4. Essayez sans quantization (none) pour vérifier que le reste fonctionne")

    sys.exit(1)
