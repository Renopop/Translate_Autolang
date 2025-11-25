# -*- coding: utf-8 -*-
"""
Core translation logic - Multilingue Excel Translator
Auteur : Renaud LOISON (optimisé et restructuré)
OPTIMIZED VERSION - RTX 4090 Performance Tuning
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:true,max_split_size_mb:512")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")  # Async CUDA
os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")  # cuDNN v8 optimizations

import re, gc, time, warnings, threading, subprocess, shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as hf_logging

# ============================================
#    PERFORMANCE CONFIGURATION - RTX 4090
# ============================================

# Enable TF32 for massive speedup on Ampere/Ada GPUs
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.fp32_precision = "high"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# Enable cuDNN autotuner for optimal convolution algorithms
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

if torch.cuda.is_available():
    try:
        # FlashAttention-2 and memory-efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        # Prefer faster CUDA ops
        torch.backends.cuda.preferred_linalg_library("cusolver")
    except Exception:
        pass

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning)

# Verbose logging control (set to False for max performance)
VERBOSE_LOGGING = False

# =========================
#    CONSTANTES & MODÈLES
# =========================

MODELS = {
    "Fast (NLLB-200 distilled 600M)": "facebook/nllb-200-distilled-600M",
    "Quality (NLLB-200 1.3B)": "facebook/nllb-200-1.3B",
    "Very High (M2M100 1.2B)": "facebook/m2m100_1.2B",
}

PAIR_SPECIALISTS = {
    ("rus_Cyrl", "eng_Latn"): "Helsinki-NLP/opus-mt-ru-en",
    ("ukr_Cyrl", "eng_Latn"): "Helsinki-NLP/opus-mt-uk-en",
    ("bul_Cyrl", "eng_Latn"): "Helsinki-NLP/opus-mt-bg-en",
    ("ara_Arab", "eng_Latn"): "Helsinki-NLP/opus-mt-ar-en",
    ("zho_Hans", "eng_Latn"): "Helsinki-NLP/opus-mt-zh-en",
    ("jpn_Jpan", "eng_Latn"): "Helsinki-NLP/opus-mt-ja-en",
    ("kor_Hang", "eng_Latn"): "Helsinki-NLP/opus-mt-ko-en",
    ("fra_Latn", "eng_Latn"): "Helsinki-NLP/opus-mt-fr-en",
    ("eng_Latn", "fra_Latn"): "Helsinki-NLP/opus-mt-en-fr",
}

LANG_CODES = {
    "Français (fr)": "fra_Latn", "English (en)": "eng_Latn", "Español (es)": "spa_Latn",
    "Deutsch (de)": "deu_Latn", "Italiano (it)": "ita_Latn", "Português (pt)": "por_Latn",
    "Nederlands (nl)": "nld_Latn", "Polski (pl)": "pol_Latn", "Svenska (sv)": "swe_Latn",
    "Norsk (no)": "nob_Latn", "Dansk (da)": "dan_Latn", "Suomi (fi)": "fin_Latn",
    "Čeština (cs)": "ces_Latn", "Slovenčina (sk)": "slk_Latn", "Slovenščina (sl)": "slv_Latn",
    "Română (ro)": "ron_Latn", "Български (bg)": "bul_Cyrl", "Русский (ru)": "rus_Cyrl",
    "Українська (uk)": "ukr_Cyrl", "Ελληνικά (el)": "ell_Grek", "Türkçe (tr)": "tur_Latn",
    "العربية (ar)": "ara_Arab", "עברית (he)": "heb_Hebr", "हिन्दी (hi)": "hin_Deva",
    "中文 (zh)": "zho_Hans", "日本語 (ja)": "jpn_Jpan", "한국어 (ko)": "kor_Hang",
}

ISO2_TO_NLLB = {
    "fr": "fra_Latn", "en": "eng_Latn", "es": "spa_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "pl": "pol_Latn",
    "sv": "swe_Latn", "no": "nob_Latn", "da": "dan_Latn", "fi": "fin_Latn",
    "cs": "ces_Latn", "sk": "slk_Latn", "sl": "slv_Latn", "ro": "ron_Latn",
    "bg": "bul_Cyrl", "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "el": "ell_Grek",
    "tr": "tur_Latn", "ar": "ara_Arab", "he": "heb_Hebr", "iw": "heb_Hebr",
    "hi": "hin_Deva", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang"
}

NLLB_TO_M2M = {
    "eng_Latn": "en", "fra_Latn": "fr", "spa_Latn": "es", "deu_Latn": "de",
    "ita_Latn": "it", "por_Latn": "pt", "nld_Latn": "nl", "pol_Latn": "pl",
    "swe_Latn": "sv", "nob_Latn": "no", "dan_Latn": "da", "fin_Latn": "fi",
    "ces_Latn": "cs", "slk_Latn": "sk", "slv_Latn": "sl", "ron_Latn": "ro",
    "bul_Cyrl": "bg", "rus_Cyrl": "ru", "ukr_Cyrl": "uk", "ell_Grek": "el",
    "tur_Latn": "tr", "ara_Arab": "ar", "heb_Hebr": "he", "hin_Deva": "hi",
    "zho_Hans": "zh", "jpn_Jpan": "ja", "kor_Hang": "ko"
}

# ============================================
#    AUTO-ADAPTIVE PARAMETERS (GPU-aware)
# ============================================
MAX_TOKENS_PER_CHUNK = 420
DYNAMIC_FACTOR_OUT = 1.25
MIN_NEW_TOKENS = 50
MIN_BATCH_SIZE = 16  # Absolute minimum
PURGE_EVERY_N_BATCHES = 64  # Less frequent purging
GPU_INDEX = 0
ENABLE_ULTRA = True
USE_TORCH_COMPILE = True  # Enable torch.compile for 30-50% speedup
COMPILE_MODE = "reduce-overhead"  # Options: "default", "reduce-overhead", "max-autotune"

def get_optimal_batch_size(vram_total_mib: int = 0) -> int:
    """
    Calcule automatiquement le batch size optimal basé sur la VRAM disponible

    Args:
        vram_total_mib: VRAM totale en MiB (0 = auto-detect)

    Returns:
        Batch size optimal pour cette GPU
    """
    if vram_total_mib <= 0:
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(GPU_INDEX)
                vram_total_mib = props.total_memory // (1024 * 1024)
            except Exception:
                vram_total_mib = 8000  # Default: assume 8GB
        else:
            return 32  # CPU mode: small batch

    # Batch size scaling based on VRAM
    # Formula: base + (vram_gb - 4) * scale_factor
    # Minimum 4GB required for reasonable performance
    vram_gb = vram_total_mib / 1024

    if vram_gb >= 24:      # RTX 4090, 3090, A6000 (24GB)
        return 512
    elif vram_gb >= 16:    # RTX 4080, A4000 (16GB)
        return 384
    elif vram_gb >= 12:    # RTX 3080 12GB, RTX 4070 Ti
        return 256
    elif vram_gb >= 10:    # RTX 3080 10GB
        return 192
    elif vram_gb >= 8:     # RTX 3070, 3060 Ti, GTX 1080 (8GB)
        return 128
    elif vram_gb >= 6:     # RTX 3060, GTX 1060 (6GB)
        return 64
    elif vram_gb >= 4:     # GTX 1650, older cards (4GB)
        return 32
    else:                  # < 4GB: very limited
        return 16

def get_gpu_tier() -> str:
    """
    Détermine le tier de la GPU pour ajuster les paramètres

    Returns:
        "high" (>=16GB), "medium" (8-16GB), "low" (4-8GB), "minimal" (<4GB), "cpu"
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        props = torch.cuda.get_device_properties(GPU_INDEX)
        vram_gb = props.total_memory / (1024**3)

        if vram_gb >= 16:
            return "high"
        elif vram_gb >= 8:
            return "medium"
        elif vram_gb >= 4:
            return "low"
        else:
            return "minimal"
    except Exception:
        return "medium"  # Default assumption

# Auto-detect optimal batch size at import time
DEFAULT_BATCH_SIZE = get_optimal_batch_size()

# Offline & Cache
OFFLINE_MODE: bool = False
MODELS_CACHE_DIR: Optional[str] = None

# Excel-safe
INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]")
EXCEL_CELL_MAX = 32767
FORMULA_PREFIXES = ('=', '+', '-', '@')

# Offload configuration
OFFLOAD_ENFORCE_TO_CPU = True
MIN_VRAM_FOR_ENFORCE_MIB = 8000
MIN_RAM_FOR_CPU_OFFLOAD_MIB = 12000

# Caches pour modèles
_fallback_cache = {}
_fast_en_cache = {}

# =========================
#    CONFIGURATION
# =========================

class TranslatorConfig:
    """Configuration pour le traducteur"""
    def __init__(
        self,
        model_name: str,
        target_lang: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        preset: str = "Quality+",
        offline_mode: bool = False,
        cache_dir: Optional[str] = None,
        quantization: str = "none"
    ):
        # Validation des paramètres obligatoires
        if not model_name:
            raise ValueError("model_name cannot be None or empty")
        if not target_lang:
            raise ValueError("target_lang cannot be None or empty")
        if quantization not in ["none", "int8", "int4"]:
            raise ValueError("quantization must be 'none', 'int8', or 'int4'")

        self.model_name = model_name
        self.target_lang = target_lang
        self.batch_size = max(MIN_BATCH_SIZE, batch_size)
        self.preset = preset
        self.offline_mode = offline_mode
        self.cache_dir = cache_dir
        self.quantization = quantization

        # Appliquer la configuration offline
        global OFFLINE_MODE, MODELS_CACHE_DIR
        OFFLINE_MODE = offline_mode
        MODELS_CACHE_DIR = cache_dir
        self._apply_offline_env()

    def _apply_offline_env(self):
        """Configure les variables d'environnement pour mode offline"""
        if self.offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.cache_dir, "transformers")
            os.environ["HF_DATASETS_CACHE"] = os.path.join(self.cache_dir, "datasets")

# =========================
#    UTILITAIRES
# =========================

def sanitize_cell(val: str) -> str:
    """Nettoie une cellule pour Excel"""
    if val is None:
        s = ""
    elif isinstance(val, str):
        s = val
    else:
        s = str(val)
    s = INVALID_XML_RE.sub(" ", s).replace("\r\n", "\n").replace("\r", "\n")
    if s and s[0] in FORMULA_PREFIXES:
        s = "'" + s
    if len(s) > EXCEL_CELL_MAX:
        s = s[:EXCEL_CELL_MAX - 1] + "…"
    return s

def sanitize_series_for_excel(series: pd.Series) -> pd.Series:
    """Nettoie une série pandas pour Excel"""
    return series.apply(sanitize_cell)

def log_verbose(msg: str):
    """Conditional logging based on VERBOSE_LOGGING flag"""
    if VERBOSE_LOGGING:
        print(msg)

def purge_vram(sync=False, force=False):
    """
    Nettoie la mémoire VRAM - Optimized version

    Args:
        sync: Whether to synchronize CUDA (slower but more thorough)
        force: Force aggressive cleanup (use sparingly)
    """
    if not force:
        # Light cleanup - just Python GC, don't touch CUDA unless needed
        gc.collect()
        return

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if sync:
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
        except Exception:
            pass

def print_vram_state(prefix="VRAM"):
    """Affiche l'état de la VRAM"""
    try:
        cuda_ok = torch.cuda.is_available()
        print(f"[{prefix}] cuda_available={cuda_ok}")
        if not cuda_ok:
            return
        torch.cuda.set_device(GPU_INDEX)
        name = torch.cuda.get_device_name(GPU_INDEX)
        free_b, total_b = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved() // (1024 * 1024)
        allocated = torch.cuda.memory_allocated() // (1024 * 1024)
        free = free_b // (1024 * 1024)
        total = total_b // (1024 * 1024)
        print(f"[{prefix}] device={name} | free={free} MiB / total={total} MiB | reserved={reserved} | allocated={allocated}")
    except Exception as e:
        print(f"[{prefix}] error: {e}")

def free_vram_mib() -> int:
    """Retourne la VRAM libre en MiB"""
    if not torch.cuda.is_available():
        return 0
    free_b, _ = torch.cuda.mem_get_info()
    return free_b // (1024 * 1024)

def free_ram_mib() -> int:
    """Retourne la RAM libre en MiB"""
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:
        return 0

def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système en temps réel (CPU, RAM, GPU, VRAM)"""
    metrics = {
        'cpu_percent': 0.0,
        'ram_used_gb': 0.0,
        'ram_total_gb': 0.0,
        'ram_percent': 0.0,
        'gpu_available': False,
        'gpu_name': 'N/A',
        'vram_used_gb': 0.0,
        'vram_total_gb': 0.0,
        'vram_percent': 0.0,
        'gpu_utilization': 0.0,
        'gpu_temperature': 0
    }

    try:
        import psutil

        # CPU
        metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)

        # RAM
        ram = psutil.virtual_memory()
        metrics['ram_used_gb'] = ram.used / (1024**3)
        metrics['ram_total_gb'] = ram.total / (1024**3)
        metrics['ram_percent'] = ram.percent

    except Exception as e:
        print(f"[METRICS] Error getting CPU/RAM: {e}")

    # GPU/VRAM
    if torch.cuda.is_available():
        try:
            metrics['gpu_available'] = True
            metrics['gpu_name'] = torch.cuda.get_device_name(GPU_INDEX)

            # VRAM
            free_b, total_b = torch.cuda.mem_get_info(GPU_INDEX)
            used_b = total_b - free_b
            metrics['vram_used_gb'] = used_b / (1024**3)
            metrics['vram_total_gb'] = total_b / (1024**3)
            metrics['vram_percent'] = (used_b / total_b) * 100 if total_b > 0 else 0

            # Utilisation GPU et température via pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['gpu_utilization'] = utilization.gpu
                try:
                    metrics['gpu_temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    pass
                pynvml.nvmlShutdown()
            except Exception as e:
                print(f"[METRICS] pynvml error: {e}")

        except Exception as e:
            print(f"[METRICS] Error getting GPU/VRAM: {e}")

    return metrics

# =========================
#    GPU DETECTION
# =========================

def get_gpu_info(index: Optional[int] = None) -> Dict[str, Any]:
    """Récupère les infos GPU"""
    info = dict(available=torch.cuda.is_available())
    if not info["available"]:
        return dict(available=False, index=None, name="CPU", total_mib=0, cc=(0, 0), bf16=False)

    if index is None:
        index = _best_gpu_index_by_memory_then_cc()

    props = torch.cuda.get_device_properties(index)
    total_mib = props.total_memory // (1024 * 1024)
    bf16 = False
    try:
        torch.cuda.set_device(index)
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        pass

    return dict(
        available=True, index=index, name=props.name, total_mib=total_mib,
        cc=(props.major, props.minor), bf16=bf16
    )

def _best_gpu_index_by_memory_then_cc() -> Optional[int]:
    """Sélectionne le meilleur GPU"""
    if not torch.cuda.is_available():
        return None
    best = None
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mib = props.total_memory // (1024 * 1024)
        cc = (props.major, props.minor)
        cand = (total_mib, cc, i)
        if best is None or cand > best:
            best = cand
    return best[2] if best else None

# =========================
#    DÉTECTION DE LANGUE
# =========================

# Pre-compiled regex patterns for faster language detection
_RE_JAPANESE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")
_RE_ARABIC = re.compile(r"[\u0600-\u06FF]")
_RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_RE_KOREAN = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]")
_RE_CJK = re.compile(r"[\u4E00-\u9FFF]")
_RE_NON_ASCII = re.compile(r"[^\x00-\x7F]")

ISO_HINT_EN = re.compile(
    r"\b(the|and|of|in|with|from|to|on|for|is|are|was|were|as|by|at|which|that|who|when|where|during|into|after|before|"
    r"over|under|between|about|around|through|without|because|if|then|but|so|there|here|also|however|although|this|"
    r"these|those|an|a|some|many|more|most|less|each|other)\b", re.IGNORECASE
)

# Larger cache for language detection (500K entries)
@lru_cache(maxsize=500_000)
def _detect_iso2_cached(head: str) -> str:
    """Détection de langue avec cache - optimized"""
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    return detect(head)

# Fast path cache for full language detection results
@lru_cache(maxsize=500_000)
def _detect_lang_cached(text_hash: str, text_sample: str) -> str:
    """Cached full language detection"""
    return _detect_lang_impl(text_sample)

def _detect_lang_impl(s: str) -> str:
    """Internal language detection implementation"""
    # Fast path: script-based detection (no langdetect needed)
    if _RE_JAPANESE.search(s):
        return "jpn_Jpan"
    if _RE_ARABIC.search(s):
        return "ara_Arab"
    if _RE_CYRILLIC.search(s):
        return "rus_Cyrl"
    if _RE_KOREAN.search(s):
        return "kor_Hang"

    # CJK without kana -> Chinese
    if _RE_CJK.search(s):
        head = s[:160]
        try:
            iso2 = _detect_iso2_cached(head)
            if iso2 in ("ja", "zh-cn", "zh"):
                return "zho_Hans"
        except Exception:
            pass
        return "zho_Hans"

    # Use langdetect for Latin/other scripts
    head = s[:160]
    try:
        iso2 = _detect_iso2_cached(head)
        code = ISO2_TO_NLLB.get(iso2)
        if code:
            return code
    except Exception:
        pass

    # Fallback: English hints
    if ISO_HINT_EN.search(s):
        return "eng_Latn"

    return "fra_Latn"

def detect_lang_nllb(text: str) -> str:
    """Détecte la langue d'un texte - Optimized with caching"""
    s = text.strip()
    if not s:
        return "eng_Latn"

    # Use hash + sample for cache key (handles long texts)
    text_hash = str(hash(s[:500]))
    return _detect_lang_cached(text_hash, s[:500])

def same_language(src_code: str, tgt_code: str) -> bool:
    """Vérifie si deux codes langue sont identiques"""
    return (src_code or "").split("_")[0] == (tgt_code or "").split("_")[0]

def looks_like_target(text: str, tgt_code: str) -> bool:
    """Vérifie si le texte est dans la langue cible - Optimized"""
    s = (text or "").strip() if isinstance(text, str) else ("" if text is None else str(text)).strip()
    if not s:
        return True
    if not tgt_code:
        return True

    try:
        iso = _detect_iso2_cached(s[:160])
        tgt_iso = tgt_code.split("_")[0][:2]
        return ISO2_TO_NLLB.get(iso, "").startswith(tgt_iso)
    except Exception:
        pass

    # Use pre-compiled regex patterns
    if tgt_code.endswith("Cyrl"):
        return bool(_RE_CYRILLIC.search(s))
    if tgt_code.endswith("Arab"):
        return bool(_RE_ARABIC.search(s))
    if tgt_code == "zho_Hans":
        return bool(_RE_CJK.search(s))
    if tgt_code == "jpn_Jpan":
        return bool(_RE_JAPANESE.search(s) or _RE_CJK.search(s))
    if tgt_code == "kor_Hang":
        return bool(_RE_KOREAN.search(s))
    if _RE_NON_ASCII.search(s):
        return False
    return True

# =========================
#    TOKENISATION
# =========================

def pick_sentence_column(df: pd.DataFrame) -> str:
    """Sélectionne la meilleure colonne de texte"""
    CAND_COLS = ["sentence", "Sentence", "text", "Text", "texte", "Texte", "content", "Content", "body", "Body"]
    for c in CAND_COLS:
        if c in df.columns:
            return c
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not text_cols:
        return df.columns[0]
    avg_len = {c: df[c].astype(str).str.len().mean() for c in text_cols}
    return max(avg_len, key=avg_len.get)

def chunk_by_tokens(text: str, tokenizer, max_tokens: int = MAX_TOKENS_PER_CHUNK):
    """Découpe un texte en chunks selon les tokens"""
    if not text or text.isspace():
        return [text]
    parts = re.split(r"([\.!?;:])", text)
    sentences = []
    for i in range(0, len(parts), 2):
        seg = (parts[i] or "").strip()
        if not seg:
            continue
        if i + 1 < len(parts):
            seg += parts[i + 1] or ""
        sentences.append(seg.strip())
    chunks, buf, ids = [], [], []
    for s in sentences:
        t_ids = tokenizer(s, add_special_tokens=False).input_ids
        if len(ids) + len(t_ids) <= max_tokens:
            buf.append(s)
            ids.extend(t_ids)
        else:
            if buf:
                chunks.append(" ".join(buf).strip())
            buf = [s]
            ids = t_ids[:]
            while len(ids) > max_tokens:
                cut = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
                chunks.append(cut.strip())
                ids = ids[max_tokens:]
    if buf:
        chunks.append(" ".join(buf).strip())
    return chunks

def dynamic_max_new_tokens(tokenizer, model_cfg, texts, factor=1.25, floor=50, max_ceiling=512) -> int:
    """
    Calcule dynamiquement le max_new_tokens - OPTIMIZED with batch tokenization

    Args:
        tokenizer: Tokenizer du modèle
        model_cfg: Configuration du modèle
        texts: Liste des textes à traduire
        factor: Facteur de multiplication (1.25-1.30)
        floor: Minimum de tokens
        max_ceiling: Plafond maximum absolu (512 par défaut pour éviter OOM)
    """
    if not texts:
        return floor

    # OPTIMIZATION: Batch tokenization instead of individual tokenization
    # This is MUCH faster than tokenizing each text separately
    try:
        # Tokenize all texts at once (no padding needed, just get lengths)
        encodings = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
        max_in = max(len(ids) for ids in encodings.input_ids) if encodings.input_ids else 0
    except Exception:
        # Fallback: sample first few texts only
        sample = texts[:min(5, len(texts))]
        max_in = 0
        for t in sample:
            try:
                n = len(tokenizer(t, add_special_tokens=False).input_ids)
                max_in = max(max_in, n)
            except Exception:
                pass

    result = max(floor, min(int(max_in * factor), max_ceiling))
    log_verbose(f"  [MAX_NEW_TOKENS] Input: {max_in} tokens → Output ceiling: {result} tokens")
    return result

# =========================
#    CHARGEMENT MODÈLES
# =========================

def _tp_kwargs():
    """Retourne les kwargs pour from_pretrained"""
    if MODELS_CACHE_DIR and os.path.isdir(MODELS_CACHE_DIR):
        os.environ.setdefault("HF_HOME", MODELS_CACHE_DIR)
        os.environ.setdefault("TRANSFORMERS_CACHE", MODELS_CACHE_DIR)

    if OFFLINE_MODE:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)

    return dict(
        local_files_only=OFFLINE_MODE,
        cache_dir=MODELS_CACHE_DIR if MODELS_CACHE_DIR else None,
    )

def _resolve_local_repo(repo_or_path: str) -> str:
    """Résout le chemin local d'un repo"""
    if not OFFLINE_MODE:
        return repo_or_path

    paths = []
    if os.path.isabs(repo_or_path) and os.path.isdir(repo_or_path):
        paths.append(repo_or_path)

    if MODELS_CACHE_DIR:
        repo_norm = repo_or_path.replace("\\", "/").strip("/")
        paths.append(os.path.join(MODELS_CACHE_DIR, repo_norm.replace("/", os.sep)))
        paths.append(os.path.join(MODELS_CACHE_DIR, os.path.basename(repo_norm)))

    for p in paths:
        if os.path.isdir(p):
            return p

    return repo_or_path

def load_model(model_name: str, device: torch.device, quantization: str = "none"):
    """
    Charge un modèle de traduction avec option de quantization
    OPTIMIZED for RTX 4090 with torch.compile support

    Args:
        model_name: Nom du modèle HuggingFace
        device: Device torch (cuda/cpu)
        quantization: Type de quantization ("none", "int8", "int4")
    """
    print(f"🔧 Chargement modèle : {model_name}")
    if quantization != "none":
        print(f"⚡ Quantization activée : {quantization}")

    tp = _tp_kwargs()
    resolved = _resolve_local_repo(model_name)

    # Load tokenizer with fast tokenizer if available
    tokenizer = AutoTokenizer.from_pretrained(resolved, use_fast=True, **tp)

    # Configuration de base
    load_kwargs = {**tp}
    using_quantization = False

    if quantization == "int8" and device.type == "cuda":
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            using_quantization = True
            print(f"✅ Configuration int8 appliquée (réduction VRAM ~50%)")

        except ImportError:
            print(f"⚠️ bitsandbytes non disponible, fallback to BF16")
            quantization = "none"
        except Exception as e:
            print(f"⚠️ Erreur int8: {e}, fallback to BF16")
            quantization = "none"

    elif quantization == "int4" and device.type == "cuda":
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            using_quantization = True
            print(f"✅ Configuration int4 appliquée (réduction VRAM ~75%)")

        except ImportError:
            print(f"⚠️ bitsandbytes non disponible, fallback to BF16")
            quantization = "none"
        except Exception as e:
            print(f"⚠️ Erreur int4: {e}, fallback to BF16")
            quantization = "none"

    # Configuration pour chargement standard - OPTIMIZED
    if quantization == "none":
        load_kwargs["torch_dtype"] = torch.bfloat16 if device.type == "cuda" else torch.float32
        load_kwargs["attn_implementation"] = "sdpa"  # FlashAttention-2 compatible
        load_kwargs["low_cpu_mem_usage"] = True  # Faster loading

    try:
        print(f"📥 Chargement du modèle...")
        model = AutoModelForSeq2SeqLM.from_pretrained(resolved, **load_kwargs)
        print(f"✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur: {e}, tentative sans optimisations...")
        load_kwargs = {
            "torch_dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
            **tp
        }
        model = AutoModelForSeq2SeqLM.from_pretrained(resolved, **load_kwargs)
        using_quantization = False
        quantization = "none"

    model.eval()

    # Move to device if not using quantization
    if not using_quantization and device.type == "cuda":
        model.to(device)

    # ============================================
    # TORCH.COMPILE - Major speedup (30-50%)
    # Only for high-tier GPUs (Ampere+) to avoid compatibility issues
    # ============================================
    gpu_tier = get_gpu_tier()
    should_compile = (
        USE_TORCH_COMPILE and
        device.type == "cuda" and
        not using_quantization and
        gpu_tier in ("high", "medium")  # Only compile on >=8GB GPUs
    )

    if should_compile:
        try:
            # Check compute capability (need SM 7.0+ for best results)
            props = torch.cuda.get_device_properties(device)
            compute_cap = props.major + props.minor / 10

            if compute_cap >= 7.0:  # Volta and newer
                print(f"🚀 Compilation du modèle (mode={COMPILE_MODE})...")
                model = torch.compile(model, mode=COMPILE_MODE, fullgraph=False)
                print(f"✅ Modèle compilé - Warmup au premier batch")
            else:
                print(f"ℹ️ torch.compile ignoré (GPU SM {compute_cap:.1f} < 7.0)")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")

    # Enable CUDA graph capture for repeated inference patterns
    if device.type == "cuda":
        try:
            # Pre-warm CUDA
            torch.cuda.synchronize()
        except Exception:
            pass

    if VERBOSE_LOGGING and device.type == "cuda":
        print_vram_state("POST-LOAD")

    return tokenizer, model

def get_fallback_tokenizer_model(src_code: str, tgt_code: str, device: torch.device):
    """Récupère un modèle spécialiste"""
    key = (src_code, tgt_code)
    name = PAIR_SPECIALISTS.get(key)
    if not name:
        return None, None

    if key in _fallback_cache:
        tok, mdl = _fallback_cache[key]
        if mdl is not None:
            mdl.to(device)
        return tok, mdl

    tp = _tp_kwargs()
    resolved = _resolve_local_repo(name)

    tok = AutoTokenizer.from_pretrained(resolved, **tp)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        resolved,
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else None),
        attn_implementation="sdpa",
        **tp
    )

    mdl.eval()
    if device.type == "cuda":
        mdl.to(device)
    _fallback_cache[key] = (tok, mdl)
    return tok, mdl

def get_fast_en2tgt_tokenizer_model(tgt_code: str, device: torch.device):
    """Récupère le modèle rapide EN→TGT"""
    base_repo = "facebook/nllb-200-distilled-600M"
    key = ("en2tgt_fast", tgt_code)

    if key in _fast_en_cache:
        tok, mdl = _fast_en_cache[key]
        if mdl is not None:
            mdl.to(device)
        return tok, mdl

    tp = _tp_kwargs()
    resolved = _resolve_local_repo(base_repo)

    tok = AutoTokenizer.from_pretrained(resolved, **tp)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        resolved,
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else None),
        attn_implementation="sdpa",
        **tp
    )

    mdl.eval()
    if device.type == "cuda":
        mdl.to(device)
    _fast_en_cache[key] = (tok, mdl)
    return tok, mdl

# =========================
#    TRADUCTION
# =========================

def gen_params_for_preset(preset: str):
    """
    Retourne les paramètres de génération selon le preset
    OPTIMIZED for RTX 4090 - reduced beams for speed without major quality loss
    """
    if preset == "Speed":
        return dict(num_beams=1, do_sample=False, repetition_penalty=1.0, length_penalty=1.0, early_stopping=False)
    if preset == "Balanced":
        # Reduced from 3 to 2 beams for better GPU utilization
        return dict(num_beams=2, do_sample=False, repetition_penalty=1.05, length_penalty=1.02, early_stopping=True)
    # Quality+ preset: reduced from 5 to 3 beams (still good quality, much faster)
    return dict(num_beams=3, do_sample=False, repetition_penalty=1.1, length_penalty=1.05, early_stopping=True)

def is_m2m(model_name: str) -> bool:
    """Vérifie si c'est un modèle M2M"""
    return "m2m100" in model_name.lower()

def translate_batch_generic(
    model_name, tokenizer, model, device, src_code, tgt_code,
    texts, model_cfg, preset="Quality+", extra_gen_kwargs=None
):
    """
    Traduction générique par lots avec backoff OOM
    OPTIMIZED for RTX 4090 with reduced overhead
    """
    # Protection contre les codes de langue None
    if not src_code:
        src_code = "eng_Latn"
    if not tgt_code:
        tgt_code = "fra_Latn"

    log_verbose(f"[TRANSLATE] {src_code} → {tgt_code}, {len(texts)} texts")

    # Configuration M2M vs NLLB - Optimized with minimal logging
    if is_m2m(model_name):
        src = NLLB_TO_M2M.get(src_code, "auto")
        tgt = NLLB_TO_M2M.get(tgt_code, "en")
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt
        forced_bos = tokenizer.get_lang_id(tgt) if hasattr(tokenizer, "get_lang_id") else None
    else:
        # NLLB configuration
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_code
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt_code

        # Get forced_bos_token_id efficiently
        forced_bos = None

        # Method 1: lang_code_to_id (preferred for NLLB)
        if hasattr(tokenizer, "lang_code_to_id") and tgt_code in tokenizer.lang_code_to_id:
            forced_bos = tokenizer.lang_code_to_id[tgt_code]

        # Method 2: convert_tokens_to_ids fallback
        if forced_bos is None:
            try:
                forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
                if forced_bos == tokenizer.unk_token_id:
                    forced_bos = None
            except Exception:
                pass

        # Method 3: short code fallback
        if forced_bos is None:
            try:
                short_code = tgt_code.split("_")[0]
                forced_bos = tokenizer.convert_tokens_to_ids(short_code)
                if forced_bos == tokenizer.unk_token_id:
                    forced_bos = None
            except Exception:
                pass

    def _encode(_texts, _max_len=MAX_TOKENS_PER_CHUNK):
        """Encode texts to tensors - Optimized for GPU"""
        enc = tokenizer(_texts, return_tensors="pt", padding=True, truncation=True, max_length=_max_len)
        if device.type == "cuda":
            # Use non_blocking for async transfer
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        return enc

    def _gen_attempt(enc, genp, forced_bos, max_new, use_cache=True):
        """Generate translations - Optimized inference"""
        with torch.inference_mode(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            kwargs = dict(**enc, max_new_tokens=max_new, use_cache=use_cache, **genp)
            if forced_bos is not None:
                kwargs["forced_bos_token_id"] = forced_bos
            return model.generate(**kwargs)

    genp = gen_params_for_preset(preset)
    if extra_gen_kwargs:
        genp.update(extra_gen_kwargs)
    factor = 1.25 if preset != "Quality+" else 1.30

    # Language-specific optimizations
    if (src_code or "").startswith("rus_"):
        genp["num_beams"] = 1
        factor = min(factor, 1.15)

    # CJK languages: use 2 beams max for speed
    if src_code in ["jpn_Jpan", "zho_Hans", "kor_Hang"]:
        genp["num_beams"] = min(genp.get("num_beams", 2), 2)

    dyn_new = dynamic_max_new_tokens(tokenizer, model_cfg, texts, factor=factor, floor=MIN_NEW_TOKENS)
    enc = _encode(texts)

    # Simplified backoff plan for faster recovery
    attempts = [
        dict(genp=dict(genp), max_new=dyn_new, use_cache=True),
        dict(genp={**genp, "num_beams": 1}, max_new=dyn_new, use_cache=True),
        dict(genp={**genp, "num_beams": 1}, max_new=int(dyn_new * 0.7), use_cache=False),
    ]

    for step, plan in enumerate(attempts, 1):
        try:
            out_ids = _gen_attempt(enc, plan["genp"], forced_bos, plan["max_new"], use_cache=plan["use_cache"])
            return tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            log_verbose(f"⚠️ OOM (attempt {step}/{len(attempts)})")
            purge_vram(force=True)

    # Micro-batch split on OOM
    if len(texts) > 1:
        mid = len(texts) // 2
        left = translate_batch_generic(model_name, tokenizer, model, device, src_code, tgt_code, texts[:mid], model_cfg, preset=preset, extra_gen_kwargs=extra_gen_kwargs)
        right = translate_batch_generic(model_name, tokenizer, model, device, src_code, tgt_code, texts[mid:], model_cfg, preset=preset, extra_gen_kwargs=extra_gen_kwargs)
        return left + right

    # Last resort - minimal settings
    enc = _encode(texts)
    out_ids = _gen_attempt(enc, {"num_beams": 1}, forced_bos, max_new=max(MIN_NEW_TOKENS, int(dyn_new * 0.5)), use_cache=False)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)

def suggest_next_batch_size(curr_bs: int, free_mib: int, max_bs_cap: int = 1024, avg_input_length: int = 0) -> int:
    """
    Suggère la taille de batch suivante - AUTO-ADAPTIVE à toute GPU

    Utilise des pourcentages de VRAM libre plutôt que des seuils absolus
    """
    # Get total VRAM for percentage calculations
    total_vram_mib = 8000  # Default assumption
    if torch.cuda.is_available():
        try:
            _, total_b = torch.cuda.mem_get_info(GPU_INDEX)
            total_vram_mib = total_b // (1024 * 1024)
        except Exception:
            pass

    # Calculate percentage of VRAM free
    free_percent = (free_mib / total_vram_mib) * 100 if total_vram_mib > 0 else 50

    # Adjust cap based on input length (relative to GPU capability)
    gpu_tier = get_gpu_tier()
    if avg_input_length > 0:
        if avg_input_length > 300:
            # Very long texts: strict cap
            tier_caps = {"high": 128, "medium": 64, "low": 32, "minimal": 16, "cpu": 16}
            max_bs_cap = min(max_bs_cap, tier_caps.get(gpu_tier, 64))
        elif avg_input_length > 200:
            tier_caps = {"high": 256, "medium": 128, "low": 64, "minimal": 32, "cpu": 32}
            max_bs_cap = min(max_bs_cap, tier_caps.get(gpu_tier, 128))
        elif avg_input_length > 100:
            tier_caps = {"high": 384, "medium": 192, "low": 96, "minimal": 48, "cpu": 48}
            max_bs_cap = min(max_bs_cap, tier_caps.get(gpu_tier, 192))

    # Adaptive batch sizing based on % of VRAM free
    if free_percent >= 80:  # > 80% VRAM libre - très agressif
        increment = {"high": 128, "medium": 64, "low": 32, "minimal": 16, "cpu": 8}
        new_bs = min(curr_bs + increment.get(gpu_tier, 32), max_bs_cap)
    elif free_percent >= 60:  # 60-80% libre
        increment = {"high": 64, "medium": 32, "low": 16, "minimal": 8, "cpu": 4}
        new_bs = min(curr_bs + increment.get(gpu_tier, 16), max_bs_cap)
    elif free_percent >= 40:  # 40-60% libre
        new_bs = min(curr_bs, max_bs_cap)  # Stable
    elif free_percent >= 25:  # 25-40% libre
        new_bs = max(MIN_BATCH_SIZE, int(curr_bs * 0.8))  # Reduce 20%
    elif free_percent >= 15:  # 15-25% libre
        new_bs = max(MIN_BATCH_SIZE, curr_bs // 2)  # Reduce 50%
    else:  # < 15% libre - critique
        new_bs = MIN_BATCH_SIZE

    log_verbose(f"  [BATCH] {curr_bs}→{new_bs} (VRAM:{free_percent:.0f}% free, tier:{gpu_tier})")
    return new_bs

# =========================
#    CLASSE PRINCIPALE
# =========================

class ExcelTranslator:
    """Traducteur Excel multilingue"""

    def __init__(self, config: TranslatorConfig, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.device = torch.device(f"cuda:{GPU_INDEX}") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = None
        self.model = None
        self.model_cfg = None

    def _update_progress(self, message: str, progress: float = 0):
        """Met à jour la progression"""
        if self.progress_callback:
            self.progress_callback(message, progress)
        print(message)

    def load_model(self):
        """Charge le modèle - Optimized for RTX 4090"""
        self._update_progress(f"🔧 Chargement...")
        purge_vram(force=True)
        self.tokenizer, self.model = load_model(
            self.config.model_name,
            self.device,
            quantization=self.config.quantization
        )
        self.model_cfg = self.model.config
        self._update_progress("✅ Modèle prêt")

    def translate_file(self, input_path: str, output_path: str):
        """Traduit un fichier Excel - OPTIMIZED for RTX 4090"""
        self._update_progress(f"📖 Lecture du fichier...")
        df = pd.read_excel(input_path)

        text_col = pick_sentence_column(df)
        rows = df[text_col].astype(str).tolist()

        # Segment preparation
        work_items: List[Tuple[int, int, str, str]] = []
        keep_original = set()

        for i, s in enumerate(rows):
            s = (s or "").strip()
            if not s:
                continue
            src_code = detect_lang_nllb(s)
            if same_language(src_code, self.config.target_lang):
                keep_original.add(i)
                continue
            for j, ch in enumerate(chunk_by_tokens(s, self.tokenizer, MAX_TOKENS_PER_CHUNK)):
                work_items.append((i, j, ch, src_code))

        if not work_items:
            df_out = df.copy()
            df_out[text_col] = [sanitize_cell(x) for x in rows]
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                df_out.to_excel(writer, index=False, sheet_name="Sheet1")
                writer.book.strings_to_urls = False
            return

        # Group by source language
        groups: Dict[str, List[int]] = {}
        for idx, (row, order, ch, src) in enumerate(work_items):
            groups.setdefault(src, []).append(idx)

        outputs = [None] * len(work_items)
        batch_size = self.config.batch_size
        total_segments = len(work_items)
        processed = 0

        for src_lang, idx_list in groups.items():
            self._update_progress(f"🌐 {src_lang}→{self.config.target_lang} ({len(idx_list)} seg)")
            k = 0
            group_texts = [work_items[idx][2] for idx in idx_list]
            specialist_cap = 1024 if ((src_lang, self.config.target_lang) in PAIR_SPECIALISTS) else 768

            # OPTIMIZATION: Pre-calculate avg length with batch tokenization
            try:
                sample = group_texts[:min(20, len(group_texts))]
                enc = self.tokenizer(sample, add_special_tokens=False, padding=False, truncation=False)
                group_avg_len = sum(len(ids) for ids in enc.input_ids) // len(sample) if sample else 0
            except Exception:
                group_avg_len = 100

            while k < len(group_texts):
                free_mib = free_vram_mib()
                batch_size = suggest_next_batch_size(batch_size, free_mib, max_bs_cap=specialist_cap, avg_input_length=group_avg_len)
                bs = min(batch_size, len(group_texts) - k)
                batch_texts = group_texts[k:k + bs]

                max_retries = 2
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        out_txts = translate_batch_generic(
                            self.config.model_name, self.tokenizer, self.model, self.device,
                            src_lang, self.config.target_lang, batch_texts, self.model_cfg,
                            preset=self.config.preset
                        )

                        for j, txt in enumerate(out_txts):
                            outputs[idx_list[k + j]] = txt

                        k += bs
                        processed += bs
                        progress = (processed / total_segments) * 100
                        self._update_progress(f"📊 {processed}/{total_segments} ({progress:.0f}%)", progress)

                        if processed % PURGE_EVERY_N_BATCHES == 0:
                            purge_vram(force=True)
                        break

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            retry_count += 1
                            purge_vram(force=True)
                            if retry_count >= max_retries:
                                old_bs = batch_size
                                batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                                bs = min(batch_size, len(group_texts) - k)
                                batch_texts = group_texts[k:k + bs]
                                retry_count = 0
                                if bs <= MIN_BATCH_SIZE:
                                    raise Exception(f"OOM avec batch minimum")
                            else:
                                time.sleep(0.5)
                        else:
                            raise

        # Reconstruction
        by_row = {}
        for (row, order, _ch, _src), txt in zip(work_items, outputs):
            by_row.setdefault(row, {})[order] = txt

        translated = []
        for i, s in enumerate(rows):
            if i in keep_original:
                translated.append(s)
            elif i not in by_row:
                translated.append(s)
            else:
                parts = [by_row[i][k] for k in sorted(by_row[i].keys())]
                translated.append(" ".join(parts).strip())

        # Export
        df_out = df.copy()
        df_out[text_col] = [sanitize_cell(x) for x in translated]
        for col in df_out.columns:
            if df_out[col].dtype == "object":
                df_out[col] = sanitize_series_for_excel(df_out[col])

        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Sheet1")
            writer.book.strings_to_urls = False

        self._update_progress(f"✅ Terminé!", 100)


# =========================
#    DOCX TRANSLATOR
# =========================

class DocxTranslator:
    """Traducteur de documents Word - OPTIMIZED for RTX 4090"""

    def __init__(self, config: TranslatorConfig, progress_callback=None):
        self.config = config
        self.progress_callback = progress_callback
        self.device = torch.device(f"cuda:{GPU_INDEX}") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = None
        self.model = None
        self.model_cfg = None

    def _update_progress(self, message: str, progress: float = 0):
        if self.progress_callback:
            self.progress_callback(message, progress)

    def load_model(self):
        """Charge le modèle - Optimized"""
        self._update_progress(f"🔧 Chargement...")
        purge_vram(force=True)
        self.tokenizer, self.model = load_model(
            self.config.model_name,
            self.device,
            quantization=self.config.quantization
        )
        self.model_cfg = self.model.config
        self._update_progress("✅ Modèle prêt")

    def translate_file(self, input_path: str, output_path: str):
        """Traduit un fichier Word - OPTIMIZED"""
        from docx_handler import DocxProcessor

        self._update_progress(f"📖 Lecture...")
        texts, metadata, handler = DocxProcessor.extract_texts_for_translation(input_path)

        if not texts:
            handler.doc.save(output_path)
            return

        work_items: List[Tuple[int, int, str, str]] = []
        keep_original = set()

        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                keep_original.add(i)
                continue
            src_code = detect_lang_nllb(text)
            if same_language(src_code, self.config.target_lang):
                keep_original.add(i)
                continue
            for j, chunk in enumerate(chunk_by_tokens(text, self.tokenizer, MAX_TOKENS_PER_CHUNK)):
                work_items.append((i, j, chunk, src_code))

        if not work_items:
            handler.doc.save(output_path)
            return

        groups: Dict[str, List[int]] = {}
        for idx, (text_idx, order, chunk, src) in enumerate(work_items):
            groups.setdefault(src, []).append(idx)

        outputs = [None] * len(work_items)
        batch_size = self.config.batch_size
        total_segments = len(work_items)
        processed = 0

        for src_lang, idx_list in groups.items():
            self._update_progress(f"🌐 {src_lang}→{self.config.target_lang} ({len(idx_list)} seg)")
            k = 0
            group_texts = [work_items[idx][2] for idx in idx_list]
            specialist_cap = 1024 if ((src_lang, self.config.target_lang) in PAIR_SPECIALISTS) else 768

            # OPTIMIZATION: Batch tokenization for avg length
            try:
                sample = group_texts[:min(20, len(group_texts))]
                enc = self.tokenizer(sample, add_special_tokens=False, padding=False, truncation=False)
                group_avg_len = sum(len(ids) for ids in enc.input_ids) // len(sample) if sample else 0
            except Exception:
                group_avg_len = 100

            while k < len(group_texts):
                free_mib = free_vram_mib()
                batch_size = suggest_next_batch_size(batch_size, free_mib, max_bs_cap=specialist_cap, avg_input_length=group_avg_len)
                bs = min(batch_size, len(group_texts) - k)
                batch_texts = group_texts[k:k + bs]

                max_retries = 2
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        out_txts = translate_batch_generic(
                            self.config.model_name, self.tokenizer, self.model, self.device,
                            src_lang, self.config.target_lang, batch_texts, self.model_cfg,
                            preset=self.config.preset
                        )

                        for j, txt in enumerate(out_txts):
                            outputs[idx_list[k + j]] = txt

                        k += bs
                        processed += bs
                        progress = (processed / total_segments) * 100
                        self._update_progress(f"📊 {processed}/{total_segments} ({progress:.0f}%)", progress)

                        if processed % PURGE_EVERY_N_BATCHES == 0:
                            purge_vram(force=True)
                        break

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            retry_count += 1
                            purge_vram(force=True)
                            if retry_count >= max_retries:
                                old_bs = batch_size
                                batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                                bs = min(batch_size, len(group_texts) - k)
                                batch_texts = group_texts[k:k + bs]
                                retry_count = 0
                                if bs <= MIN_BATCH_SIZE:
                                    raise Exception(f"OOM avec batch minimum")
                            else:
                                time.sleep(0.5)
                        else:
                            raise

        # Reconstruction
        by_text_idx = {}
        for (text_idx, order, _chunk, _src), txt in zip(work_items, outputs):
            by_text_idx.setdefault(text_idx, {})[order] = txt

        translated_texts = []
        for i, original_text in enumerate(texts):
            if i in keep_original:
                translated_texts.append(original_text)
            elif i not in by_text_idx:
                translated_texts.append(original_text)
            else:
                parts = [by_text_idx[i][k] for k in sorted(by_text_idx[i].keys())]
                translated_texts.append(" ".join(parts).strip())

        DocxProcessor.apply_translations(handler, translated_texts, metadata, output_path)
        self._update_progress(f"✅ Terminé!", 100)
