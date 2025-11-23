# -*- coding: utf-8 -*-
"""
Core translation logic - Multilingue Excel Translator
Auteur : Renaud LOISON (optimisé et restructuré)
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import re, gc, time, warnings, threading, subprocess, shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as hf_logging

# Configuration CUDA/Performance
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.fp32_precision = "high"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

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

# Paramètres par défaut
MAX_TOKENS_PER_CHUNK = 420
DYNAMIC_FACTOR_OUT = 1.25
MIN_NEW_TOKENS = 50
DEFAULT_BATCH_SIZE = 256
MIN_BATCH_SIZE = 16
PURGE_EVERY_N_BATCHES = 16
GPU_INDEX = 0
ENABLE_ULTRA = True

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
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.target_lang = target_lang
        self.batch_size = max(MIN_BATCH_SIZE, batch_size)
        self.preset = preset
        self.offline_mode = offline_mode
        self.cache_dir = cache_dir

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

def purge_vram(sync=True):
    """Nettoie la mémoire VRAM"""
    gc.collect()
    if torch.cuda.is_available():
        try:
            if sync:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if sync:
                torch.cuda.synchronize()
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

ISO_HINT_EN = re.compile(
    r"\b(the|and|of|in|with|from|to|on|for|is|are|was|were|as|by|at|which|that|who|when|where|during|into|after|before|"
    r"over|under|between|about|around|through|without|because|if|then|but|so|there|here|also|however|although|this|"
    r"these|those|an|a|some|many|more|most|less|each|other)\b", re.IGNORECASE
)

@lru_cache(maxsize=100_000)
def _detect_iso2_cached(head: str) -> str:
    """Détection de langue avec cache"""
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    return detect(head)

def detect_lang_nllb(text: str) -> str:
    """Détecte la langue d'un texte"""
    s = text.strip()
    if not s:
        return "eng_Latn"

    # Détection japonais : Hiragana ou Katakana
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", s):
        print(f"[LANG DETECT] Japanese characters (Hiragana/Katakana) detected -> jpn_Jpan | Sample: {s[:50]}...")
        return "jpn_Jpan"

    # Détection arabe
    if re.search(r"[\u0600-\u06FF]", s):
        print(f"[LANG DETECT] Arabic characters detected -> ara_Arab")
        return "ara_Arab"

    # Détection cyrillique (russe)
    if re.search(r"[\u0400-\u04FF]", s):
        print(f"[LANG DETECT] Cyrillic characters detected -> rus_Cyrl")
        return "rus_Cyrl"

    # Détection coréen (Hangul)
    if re.search(r"[\uAC00-\uD7AF\u1100-\u11FF]", s):
        print(f"[LANG DETECT] Korean characters (Hangul) detected -> kor_Hang | Sample: {s[:50]}...")
        return "kor_Hang"

    # CJK characters sans Hiragana/Katakana -> probablement chinois, mais vérifier avec langdetect
    if re.search(r"[\u4E00-\u9FFF]", s):
        head = s[:160]
        try:
            iso2 = _detect_iso2_cached(head)
            # Si langdetect dit japonais mais pas de Hiragana/Katakana, c'est probablement du chinois
            if iso2 == "ja":
                print(f"[LANG DETECT] CJK + langdetect=ja but no kana -> likely Chinese -> zho_Hans")
                return "zho_Hans"
            elif iso2 == "zh-cn" or iso2 == "zh":
                print(f"[LANG DETECT] Chinese characters + langdetect={iso2} -> zho_Hans")
                return "zho_Hans"
        except Exception as e:
            print(f"[LANG DETECT] CJK characters detected, langdetect failed -> defaulting to zho_Hans")
            pass
        return "zho_Hans"

    # Utiliser langdetect pour les autres langues
    head = s[:160]
    detected_lang = None
    try:
        iso2 = _detect_iso2_cached(head)
        code = ISO2_TO_NLLB.get(iso2)
        if code:
            detected_lang = code
            print(f"[LANG DETECT] langdetect says '{iso2}' -> {code} | Sample: {head[:50]}...")
            return code
    except Exception as e:
        print(f"[LANG DETECT] langdetect failed: {e}")
        pass

    # Fallback basé sur des indices
    if ISO_HINT_EN.search(s):
        print(f"[LANG DETECT] English hints detected -> eng_Latn | Sample: {s[:50]}...")
        return "eng_Latn"

    print(f"[LANG DETECT] Fallback to fra_Latn | Sample: {s[:50]}...")
    return "fra_Latn"

def same_language(src_code: str, tgt_code: str) -> bool:
    """Vérifie si deux codes langue sont identiques"""
    return (src_code or "").split("_")[0] == (tgt_code or "").split("_")[0]

def looks_like_target(text: str, tgt_code: str) -> bool:
    """Vérifie si le texte est dans la langue cible"""
    s = (text or "").strip() if isinstance(text, str) else ("" if text is None else str(text)).strip()
    if not s:
        return True

    # Protection contre tgt_code None
    if not tgt_code:
        return True

    try:
        iso = _detect_iso2_cached(s[:160])
        tgt_iso = tgt_code.split("_")[0][:2]
        return ISO2_TO_NLLB.get(iso, "").startswith(tgt_iso)
    except Exception:
        pass
    if tgt_code.endswith("Cyrl"):
        return bool(re.search(r"[\u0400-\u04FF]", s))
    if tgt_code.endswith("Arab"):
        return bool(re.search(r"[\u0600-\u06FF]", s))
    if tgt_code == "zho_Hans":
        return bool(re.search(r"[\u4E00-\u9FFF]", s))
    if tgt_code == "jpn_Jpan":
        return bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", s))
    if tgt_code == "kor_Hang":
        return bool(re.search(r"[\uAC00-\uD7AF\u1100-\u11FF]", s))
    if re.search(r"[^\x00-\x7F]", s):
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

def dynamic_max_new_tokens(tokenizer, model_cfg, texts, factor=1.25, floor=50) -> int:
    """Calcule dynamiquement le max_new_tokens"""
    max_in = 0
    for t in texts:
        n = len(tokenizer(t, add_special_tokens=False).input_ids)
        if n > max_in:
            max_in = n
    ceilings = []
    for attr in ("max_length", "max_position_embeddings", "max_target_positions"):
        v = getattr(model_cfg, attr, None)
        if isinstance(v, int) and v > 0:
            ceilings.append(v)
    ceiling = min([1024] + ceilings)
    new_tokens = int(max(floor, min(int(max_in * factor), ceiling)))
    return max(floor, min(new_tokens, ceiling - 1))

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

def load_model(model_name: str, device: torch.device):
    """Charge un modèle de traduction"""
    print(f"🔧 Chargement modèle : {model_name}")
    tp = _tp_kwargs()
    resolved = _resolve_local_repo(model_name)

    tokenizer = AutoTokenizer.from_pretrained(resolved, **tp)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        resolved,
        torch_dtype=(torch.bfloat16 if device.type == "cuda" else None),
        attn_implementation="sdpa",
        **tp
    )

    model.eval()
    if device.type == "cuda":
        model.to(device)
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
    """Retourne les paramètres de génération selon le preset"""
    if preset == "Speed":
        return dict(num_beams=1, do_sample=False, repetition_penalty=1.0, length_penalty=1.0, early_stopping=False)
    if preset == "Balanced":
        return dict(num_beams=3, do_sample=False, repetition_penalty=1.05, length_penalty=1.02, early_stopping=True)
    return dict(num_beams=5, do_sample=False, repetition_penalty=1.1, length_penalty=1.05, early_stopping=True)

def is_m2m(model_name: str) -> bool:
    """Vérifie si c'est un modèle M2M"""
    return "m2m100" in model_name.lower()

def translate_batch_generic(
    model_name, tokenizer, model, device, src_code, tgt_code,
    texts, model_cfg, preset="Quality+", extra_gen_kwargs=None
):
    """Traduction générique par lots avec backoff OOM"""

    # Protection contre les codes de langue None
    if not src_code:
        src_code = "eng_Latn"
        print(f"[WARNING] src_code was None, defaulting to eng_Latn")
    if not tgt_code:
        tgt_code = "fra_Latn"
        print(f"[WARNING] tgt_code was None, defaulting to fra_Latn")

    # ⚠️ IMPORTANT: Configuration du tokenizer AVANT toute utilisation
    print(f"\n[TRANSLATION DEBUG]")
    print(f"  Model: {model_name}")
    print(f"  Source: {src_code}, Target: {tgt_code}")
    print(f"  Sample text: {texts[0][:100] if texts else 'N/A'}...")
    print(f"  Tokenizer has src_lang: {hasattr(tokenizer, 'src_lang')}")
    print(f"  Tokenizer has tgt_lang: {hasattr(tokenizer, 'tgt_lang')}")
    print(f"  Tokenizer has lang_code_to_id: {hasattr(tokenizer, 'lang_code_to_id')}")

    # Configuration M2M vs NLLB
    if is_m2m(model_name):
        src = NLLB_TO_M2M.get(src_code, "auto")
        tgt = NLLB_TO_M2M.get(tgt_code, "en")
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src
            print(f"  Set tokenizer.src_lang = {src}")
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt
            print(f"  Set tokenizer.tgt_lang = {tgt}")
        forced_bos = tokenizer.get_lang_id(tgt) if hasattr(tokenizer, "get_lang_id") else None
        print(f"[M2M] src={src_code}→{src}, tgt={tgt_code}→{tgt}, forced_bos={forced_bos}")
    else:
        # Pour NLLB, il faut absolument configurer src_lang et tgt_lang
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src_code
            print(f"  Set tokenizer.src_lang = {src_code}")
        else:
            print(f"  WARNING: tokenizer does not have 'src_lang' attribute!")

        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = tgt_code
            print(f"  Set tokenizer.tgt_lang = {tgt_code}")
        else:
            print(f"  WARNING: tokenizer does not have 'tgt_lang' attribute!")

        # Récupérer le forced_bos_token_id pour la langue cible
        # NLLB utilise des tokens spéciaux pour chaque langue
        forced_bos = None

        # Méthode 1 : via lang_code_to_id (NLLB-200)
        if hasattr(tokenizer, "lang_code_to_id"):
            if tgt_code in tokenizer.lang_code_to_id:
                forced_bos = tokenizer.lang_code_to_id[tgt_code]
                print(f"  forced_bos from lang_code_to_id[{tgt_code}] = {forced_bos}")
            else:
                print(f"  WARNING: {tgt_code} not found in tokenizer.lang_code_to_id!")
                print(f"  Available codes sample: {list(tokenizer.lang_code_to_id.keys())[:10]}")

        # Méthode 2 : via convert_tokens_to_ids (fallback)
        if forced_bos is None:
            try:
                # Les tokens de langue NLLB sont de la forme "fra_Latn"
                forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
                if forced_bos != tokenizer.unk_token_id:
                    print(f"  forced_bos from convert_tokens_to_ids({tgt_code}) = {forced_bos}")
                else:
                    forced_bos = None
            except Exception as e:
                print(f"  convert_tokens_to_ids failed: {e}")

        # Méthode 3 : essayer avec le code court (ex: "fra" au lieu de "fra_Latn")
        if forced_bos is None:
            short_code = tgt_code.split("_")[0]
            try:
                forced_bos = tokenizer.convert_tokens_to_ids(short_code)
                if forced_bos != tokenizer.unk_token_id:
                    print(f"  forced_bos from short code '{short_code}' = {forced_bos}")
                else:
                    forced_bos = None
            except Exception:
                pass

        if forced_bos is None:
            print(f"  ERROR: Could not find forced_bos for {tgt_code}!")
            print(f"  Tokenizer type: {type(tokenizer)}")
            print(f"  This will result in incorrect translations!")
        else:
            print(f"  ✓ Successfully set forced_bos = {forced_bos} for {tgt_code}")

        print(f"[NLLB] src={src_code}, tgt={tgt_code}, forced_bos={forced_bos}")

    def _encode(_texts, _max_len=MAX_TOKENS_PER_CHUNK):
        enc = tokenizer(_texts, return_tensors="pt", padding=True, truncation=True, max_length=_max_len)
        if device.type == "cuda":
            for k in enc:
                enc[k] = enc[k].to(device, non_blocking=True)
        return enc

    def _gen_attempt(enc, genp, forced_bos, max_new, use_cache=True):
        with torch.inference_mode():
            kwargs = dict(**enc, max_new_tokens=max_new, use_cache=use_cache, **genp)
            if forced_bos is not None:
                kwargs["forced_bos_token_id"] = forced_bos
                print(f"  Using forced_bos_token_id={forced_bos} in generation")
            else:
                print(f"  WARNING: No forced_bos_token_id set! Model may generate any language!")
            out_ids = model.generate(**kwargs)
            print(f"  First generated token IDs: {out_ids[0][:5].tolist() if len(out_ids) > 0 else 'N/A'}")
            return out_ids

    genp = gen_params_for_preset(preset)
    if extra_gen_kwargs:
        genp.update(extra_gen_kwargs)
    factor = 1.25 if preset != "Quality+" else 1.30
    if (src_code or "").startswith("rus_"):
        genp.setdefault("num_beams", 1)
        factor = min(factor, 1.15)

    dyn_new = dynamic_max_new_tokens(tokenizer, model_cfg, texts, factor=factor, floor=MIN_NEW_TOKENS)
    enc = _encode(texts)

    # Plan de backoff
    attempts = [
        dict(genp=dict(genp), max_new=dyn_new, use_cache=True),
        dict(genp={**genp, "num_beams": 1}, max_new=dyn_new, use_cache=True),
        dict(genp={**genp, "num_beams": 1}, max_new=int(dyn_new * 0.8), use_cache=True),
        dict(genp={**genp, "num_beams": 1}, max_new=int(dyn_new * 0.8), use_cache=False),
        dict(genp={**genp, "num_beams": 1, "no_repeat_ngram_size": 0}, max_new=int(dyn_new * 0.6), use_cache=False),
    ]

    for step, plan in enumerate(attempts, 1):
        try:
            out_ids = _gen_attempt(enc, plan["genp"], forced_bos, plan["max_new"], use_cache=plan["use_cache"])
            result = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            print(f"  Generated sample: {result[0][:100] if result else 'N/A'}...")
            return result
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"⚠️ OOM in generate (attempt {step}/{len(attempts)}): backoff…")
            purge_vram(sync=True)

    # Micro-batch split
    if len(texts) > 1:
        mid = len(texts) // 2
        left = translate_batch_generic(model_name, tokenizer, model, device, src_code, tgt_code, texts[:mid], model_cfg, preset=preset, extra_gen_kwargs=extra_gen_kwargs)
        purge_vram(sync=True)
        right = translate_batch_generic(model_name, tokenizer, model, device, src_code, tgt_code, texts[mid:], model_cfg, preset=preset, extra_gen_kwargs=extra_gen_kwargs)
        purge_vram(sync=True)
        return left + right

    # Dernier recours
    enc = _encode(texts)
    out_ids = _gen_attempt(enc, {"num_beams": 1}, forced_bos, max_new=max(MIN_NEW_TOKENS, int(dyn_new * 0.5)), use_cache=False)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)

def suggest_next_batch_size(curr_bs: int, free_mib: int, max_bs_cap: int = 1024) -> int:
    """Suggère la taille de batch suivante"""
    if free_mib >= 18000:
        return min(curr_bs + 64, max_bs_cap)
    if free_mib >= 12000:
        return min(curr_bs + 32, max_bs_cap)
    if free_mib >= 8000:
        return curr_bs
    if free_mib >= 4000:
        return max(MIN_BATCH_SIZE, int(curr_bs * 0.75))
    return max(MIN_BATCH_SIZE, curr_bs // 2)

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
        """Charge le modèle de traduction"""
        self._update_progress(f"🔧 Chargement du modèle {self.config.model_name}...")
        # Purge VRAM avant chargement pour maximiser la mémoire disponible
        purge_vram(sync=True)
        print_vram_state("VRAM avant chargement modèle")
        self.tokenizer, self.model = load_model(self.config.model_name, self.device)
        self.model_cfg = self.model.config
        purge_vram(sync=True)
        print_vram_state("VRAM après chargement modèle")
        self._update_progress("✅ Modèle chargé")

    def translate_file(self, input_path: str, output_path: str):
        """Traduit un fichier Excel"""
        self._update_progress(f"📖 Lecture du fichier {input_path}...")
        df = pd.read_excel(input_path)

        text_col = pick_sentence_column(df)
        self._update_progress(f"🧭 Colonne détectée : '{text_col}'")
        rows = df[text_col].astype(str).tolist()

        # Préparation des segments
        self._update_progress("🔍 Analyse et découpage du texte...")
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
            self._update_progress("ℹ️ Aucune donnée à traduire")
            df_out = df.copy()
            df_out[text_col] = [sanitize_cell(x) for x in rows]
            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                df_out.to_excel(writer, index=False, sheet_name="Sheet1")
                writer.book.strings_to_urls = False
            return

        # Groupement par langue source
        groups: Dict[str, List[int]] = {}
        for idx, (row, order, ch, src) in enumerate(work_items):
            groups.setdefault(src, []).append(idx)

        # Traduction
        outputs = [None] * len(work_items)
        batch_size = self.config.batch_size
        total_segments = len(work_items)
        processed = 0

        for src_lang, idx_list in groups.items():
            self._update_progress(f"🌐 Traduction {src_lang} → {self.config.target_lang} ({len(idx_list)} segments)")
            print(f"[DEBUG] Source détectée: {src_lang}, Cible: {self.config.target_lang}")
            k = 0
            group_texts = [work_items[idx][2] for idx in idx_list]

            specialist_cap = 1024 if ((src_lang, self.config.target_lang) in PAIR_SPECIALISTS) else 512

            while k < len(group_texts):
                free_mib = free_vram_mib()
                batch_size = suggest_next_batch_size(batch_size, free_mib, max_bs_cap=specialist_cap)

                bs = min(batch_size, len(group_texts) - k)
                batch_texts = group_texts[k:k + bs]

                # Traduction du batch avec gestion OOM robuste
                max_retries = 3
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        # Purge VRAM avant chaque batch pour maximiser mémoire disponible
                        if retry_count > 0 or k == 0:
                            purge_vram(sync=True)

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
                        self._update_progress(f"📊 Progression: {processed}/{total_segments} segments", progress)

                        # Purge périodique
                        if processed % PURGE_EVERY_N_BATCHES == 0:
                            purge_vram(sync=True)

                        break  # Succès, sortir de la boucle retry

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            retry_count += 1
                            purge_vram(sync=True)

                            if retry_count >= max_retries:
                                # Dernier recours: réduire batch size et réessayer
                                batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                                bs = min(batch_size, len(group_texts) - k)
                                batch_texts = group_texts[k:k + bs]
                                self._update_progress(f"⚠️ OOM persistant - Réduction batch à {batch_size}")
                                retry_count = 0  # Reset pour nouveau batch size
                                if bs == 1 and retry_count >= max_retries:
                                    raise Exception(f"Impossible de traduire même avec batch_size=1. GPU trop faible ou texte trop long.")
                            else:
                                self._update_progress(f"⚠️ OOM - Tentative {retry_count}/{max_retries}")
                                time.sleep(1)  # Pause courte pour laisser GPU se vider
                        else:
                            raise

        # Reconstruction
        self._update_progress("🔨 Reconstruction des textes...")
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
        self._update_progress("💾 Écriture du fichier...")
        df_out = df.copy()
        df_out[text_col] = [sanitize_cell(x) for x in translated]
        for col in df_out.columns:
            if df_out[col].dtype == "object":
                df_out[col] = sanitize_series_for_excel(df_out[col])

        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Sheet1")
            writer.book.strings_to_urls = False

        self._update_progress(f"✅ Traduction terminée: {output_path}", 100)
        purge_vram()


# =========================
#    DOCX TRANSLATOR
# =========================

class DocxTranslator:
    """Traducteur de documents Word multilingue"""

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
        """Charge le modèle de traduction"""
        self._update_progress(f"🔧 Chargement du modèle {self.config.model_name}...")
        # Purge VRAM avant chargement pour maximiser la mémoire disponible
        purge_vram(sync=True)
        print_vram_state("VRAM avant chargement modèle")
        self.tokenizer, self.model = load_model(self.config.model_name, self.device)
        self.model_cfg = self.model.config
        purge_vram(sync=True)
        print_vram_state("VRAM après chargement modèle")
        self._update_progress("✅ Modèle chargé")

    def translate_file(self, input_path: str, output_path: str):
        """Traduit un fichier Word"""
        from docx_handler import DocxProcessor

        self._update_progress(f"📖 Lecture du document {input_path}...")

        # Extraction des textes avec métadonnées
        texts, metadata, handler = DocxProcessor.extract_texts_for_translation(input_path)

        if not texts:
            self._update_progress("ℹ️ Aucun texte à traduire dans le document")
            handler.doc.save(output_path)
            return

        self._update_progress(f"🔍 Analyse: {len(texts)} segments à traduire...")

        # Préparation des segments
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

            # Découper le texte en chunks si nécessaire
            for j, chunk in enumerate(chunk_by_tokens(text, self.tokenizer, MAX_TOKENS_PER_CHUNK)):
                work_items.append((i, j, chunk, src_code))

        if not work_items:
            self._update_progress("ℹ️ Aucune traduction nécessaire (texte déjà dans la langue cible)")
            handler.doc.save(output_path)
            return

        # Groupement par langue source
        groups: Dict[str, List[int]] = {}
        for idx, (text_idx, order, chunk, src) in enumerate(work_items):
            groups.setdefault(src, []).append(idx)

        # Traduction
        outputs = [None] * len(work_items)
        batch_size = self.config.batch_size
        total_segments = len(work_items)
        processed = 0

        for src_lang, idx_list in groups.items():
            self._update_progress(f"🌐 Traduction {src_lang} → {self.config.target_lang} ({len(idx_list)} segments)")
            print(f"[DEBUG] Source détectée: {src_lang}, Cible: {self.config.target_lang}")
            k = 0
            group_texts = [work_items[idx][2] for idx in idx_list]

            specialist_cap = 1024 if ((src_lang, self.config.target_lang) in PAIR_SPECIALISTS) else 512

            while k < len(group_texts):
                free_mib = free_vram_mib()
                batch_size = suggest_next_batch_size(batch_size, free_mib, max_bs_cap=specialist_cap)

                bs = min(batch_size, len(group_texts) - k)
                batch_texts = group_texts[k:k + bs]

                # Traduction du batch avec gestion OOM robuste
                max_retries = 3
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        # Purge VRAM avant chaque batch pour maximiser mémoire disponible
                        if retry_count > 0 or k == 0:
                            purge_vram(sync=True)

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
                        self._update_progress(f"📊 Progression: {processed}/{total_segments} segments", progress)

                        # Purge périodique
                        if processed % PURGE_EVERY_N_BATCHES == 0:
                            purge_vram(sync=True)

                        break  # Succès, sortir de la boucle retry

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            retry_count += 1
                            purge_vram(sync=True)

                            if retry_count >= max_retries:
                                # Dernier recours: réduire batch size et réessayer
                                batch_size = max(MIN_BATCH_SIZE, batch_size // 2)
                                bs = min(batch_size, len(group_texts) - k)
                                batch_texts = group_texts[k:k + bs]
                                self._update_progress(f"⚠️ OOM persistant - Réduction batch à {batch_size}")
                                retry_count = 0  # Reset pour nouveau batch size
                                if bs == 1 and retry_count >= max_retries:
                                    raise Exception(f"Impossible de traduire même avec batch_size=1. GPU trop faible ou texte trop long.")
                            else:
                                self._update_progress(f"⚠️ OOM - Tentative {retry_count}/{max_retries}")
                                time.sleep(1)  # Pause courte pour laisser GPU se vider
                        else:
                            raise

        # Reconstruction des textes traduits
        self._update_progress("🔨 Reconstruction des textes...")
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

        # Application des traductions
        self._update_progress("💾 Écriture du document traduit...")
        DocxProcessor.apply_translations(handler, translated_texts, metadata, output_path)

        self._update_progress(f"✅ Traduction terminée: {output_path}", 100)
        purge_vram()
