# -*- coding: utf-8 -*-
"""
Traducteur Multilingue Excel — GUI Tkinter — Version complète (FR) + Hors‑ligne & Dossier cache
Auteur : Renaud LOISON (adapté avec options offline/cache)

Ajouts clés de cette version
----------------------------
- Case **Mode hors‑ligne** : bloque tout accès réseau Hugging Face (HF Hub/Transformers)
- Sélecteur **Dossier de stockage des modèles** : fixe HF_HOME/TRANSFORMERS_CACHE/DATASETS_CACHE
- Tous les chargements `from_pretrained(...)` utilisent `local_files_only=OFFLINE_MODE` et `cache_dir=MODELS_CACHE_DIR`

Le reste reprend exactement votre logique : auto‑tune GPU, découpage token‑aware, backoff anti‑OOM,
retry/ultra, enforce de langue, export Excel “safe”.
"""

# ---- anti-fragmentation CUDA, à définir AVANT torch
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import re, gc, csv, time, warnings, threading, subprocess, shutil, argparse, tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
from functools import lru_cache
from tqdm import tqdm

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging as hf_logging

# =========================
#    OFFLINE & CACHE (NEW)
# =========================
OFFLINE_MODE: bool = False
MODELS_CACHE_DIR: Optional[str] = None

def apply_offline_env(offline: bool, cache_dir: Optional[str]):
    """Configure les variables d'environnement HF pour mode hors‑ligne et cache dédié."""
    if offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

# =========================
#          CUDA / PERF
# =========================
# matmul en FP32: haute performance (autorise TF32 côté matmul)
torch.set_float32_matmul_precision("high")

# Nouvelle API TF32 (PyTorch >= 2.9) — tolérante aux versions plus anciennes
try:
    torch.backends.cuda.matmul.fp32_precision = "high"   # "ieee" si stricte
    torch.backends.cudnn.conv.fp32_precision = "tf32"    # "ieee_float32" si stricte
except Exception:
    pass

# Active SDPA/FlashAttention si dispo
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

# Réduit les logs Transformers + correctif warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# =========================
#   OFFLINE / CACHE (NEW)
# =========================
# Dossier où TU STOCKES les modèles (adapter si besoin)
MODELS_CACHE_DIR = r"C:\IA Test\models"
# Activer/désactiver explicitement le mode hors-ligne
OFFLINE_MODE = True

def _tp_kwargs():
    """
    kwargs communs pour from_pretrained(...).
    - force le cache_dir vers MODELS_CACHE_DIR
    - coupe le réseau si OFFLINE_MODE
    - pose les variables d'env cohérentes
    """
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


# =========================
#    MODÈLES & LANGUES
# =========================
MODELS = {
    "Fast (NLLB-200 distilled 600M)" : "facebook/nllb-200-distilled-600M",
    "Quality (NLLB-200 1.3B)"        : "facebook/nllb-200-1.3B",
    "Very High (M2M100 1.2B)"        : "facebook/m2m100_1.2B",
}

PAIR_SPECIALISTS = {
    ("rus_Cyrl","eng_Latn"): "Helsinki-NLP/opus-mt-ru-en",
    ("ukr_Cyrl","eng_Latn"): "Helsinki-NLP/opus-mt-uk-en",
    ("bul_Cyrl","eng_Latn"): "Helsinki-NLP/opus-mt-bg-en",
    ("ara_Arab","eng_Latn"): "Helsinki-NLP/opus-mt-ar-en",
    ("zho_Hans","eng_Latn"): "Helsinki-NLP/opus-mt-zh-en",
    ("jpn_Jpan","eng_Latn"): "Helsinki-NLP/opus-mt-ja-en",
    ("kor_Hang","eng_Latn"): "Helsinki-NLP/opus-mt-ko-en",
    ("fra_Latn","eng_Latn"): "Helsinki-NLP/opus-mt-fr-en",
    ("eng_Latn","fra_Latn"): "Helsinki-NLP/opus-mt-en-fr",
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
    "fr":"fra_Latn","en":"eng_Latn","es":"spa_Latn","de":"deu_Latn","it":"ita_Latn","pt":"por_Latn",
    "nl":"nld_Latn","pl":"pol_Latn","sv":"swe_Latn","no":"nob_Latn","da":"dan_Latn","fi":"fin_Latn",
    "cs":"ces_Latn","sk":"slk_Latn","sl":"slv_Latn","ro":"ron_Latn","bg":"bul_Cyrl","ru":"rus_Cyrl",
    "uk":"ukr_Cyrl","el":"ell_Grek","tr":"tur_Latn","ar":"ara_Arab","he":"heb_Hebr","iw":"heb_Hebr",
    "hi":"hin_Deva","zh":"zho_Hans","ja":"jpn_Jpan","ko":"kor_Hang"
}

NLLB_TO_M2M = {
    "eng_Latn":"en","fra_Latn":"fr","spa_Latn":"es","deu_Latn":"de","ita_Latn":"it","por_Latn":"pt",
    "nld_Latn":"nl","pol_Latn":"pl","swe_Latn":"sv","nob_Latn":"no","dan_Latn":"da","fin_Latn":"fi",
    "ces_Latn":"cs","slk_Latn":"sk","slv_Latn":"sl","ron_Latn":"ro","bul_Cyrl":"bg","rus_Cyrl":"ru",
    "ukr_Cyrl":"uk","ell_Grek":"el","tur_Latn":"tr","ara_Arab":"ar","heb_Hebr":"he","hin_Deva":"hi",
    "zho_Hans":"zh","jpn_Jpan":"ja","kor_Hang":"ko"
}

# =========================
#     PARAMÈTRES PERF
# =========================
MAX_TOKENS_PER_CHUNK = 420
DYNAMIC_FACTOR_OUT = 1.25
MIN_NEW_TOKENS = 50

DEFAULT_BATCH_SIZE = 256
MIN_BATCH_SIZE = 16

PURGE_EVERY_N_BATCHES = 16

GPU_INDEX = 0
MONITOR_INTERVAL_S = 1.0
PRINT_EVERY = 5
GPU_LOG_CSV = None

ENABLE_ULTRA = True
ENABLE_AUTO_SWEEP = False

CAND_COLS = ["sentence","Sentence","text","Text","texte","Texte","content","Content","body","Body"]

# Offload CPU pour Enforce si VRAM trop basse + RAM dispo
OFFLOAD_ENFORCE_TO_CPU = True
MIN_VRAM_FOR_ENFORCE_MIB = 8000
MIN_RAM_FOR_CPU_OFFLOAD_MIB = 12000

# ==== GPU-AWARE DEFAULTS (auto-tuned at runtime) ====
DEFAULT_MODEL_KEY = list(MODELS.keys())[0]
DEFAULT_PRESET = "Quality+"

# =========================
#        EXCEL-SAFE
# =========================
INVALID_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\uD800-\uDFFF\uFFFE\uFFFF]")
EXCEL_CELL_MAX = 32767
FORMULA_PREFIXES = ('=', '+', '-', '@')

def sanitize_cell(val: str) -> str:
    if val is None: s = ""
    elif isinstance(val, str): s = val
    else: s = str(val)
    s = INVALID_XML_RE.sub(" ", s).replace("\r\n","\n").replace("\r","\n")
    if s and s[0] in FORMULA_PREFIXES: s = "'" + s
    if len(s) > EXCEL_CELL_MAX: s = s[:EXCEL_CELL_MAX-1] + "…"
    return s

def sanitize_series_for_excel(series: pd.Series) -> pd.Series:
    return series.apply(sanitize_cell)

# =========================
#       MONITOR GPU
# =========================
class GPUMonitor:
    def __init__(self, index=0, interval=1.0, print_every=5, log_csv=None):
        self.index=index; self.interval=interval; self.print_every=print_every; self.log_csv=log_csv
        self._stop=threading.Event(); self._thr=None
        self._nvsmi_path=shutil.which("nvidia-smi")
        self.samples=[]

    def start(self):
        if not self._thr:
            self._thr=threading.Thread(target=self._run,daemon=True); self._thr.start()

    def stop(self):
        if self._thr: self._stop.set(); self._thr.join(timeout=5); self._thr=None

    def _run(self):
        n_print=0
        while not self._stop.is_set():
            try:
                ts=datetime.now().strftime("%H:%M:%S")
                if self._nvsmi_path:
                    q="utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
                    cmd=[self._nvsmi_path,f"--query-gpu={q}","--format=csv,noheader,nounits","-i",str(self.index)]
                    out=subprocess.check_output(cmd,stderr=subprocess.DEVNULL,text=True).strip()
                    vals=[v.strip() for v in out.split(",")]
                    data=(float(vals[0]),float(vals[1]),float(vals[2]),float(vals[3]),float(vals[4]),float(vals[5]))
                    self.samples.append((ts,*data))
                    n_print+=1
                    if n_print>=self.print_every:
                        n_print=0
                        print(f"[GPU{self.index} {ts}] util:{data[0]:5.1f}% | mem:{data[2]:.0f}/{data[3]:.0f} MiB | temp:{data[4]:.0f}°C | power:{data[5]:.0f}W")
            except Exception:
                pass
            time.sleep(self.interval)

    def summary(self):
        if not self.samples: return "No GPU samples recorded."
        gpu_avg=sum(s[1] for s in self.samples)/len(self.samples)
        mem_avg=sum(s[2] for s in self.samples)/len(self.samples)
        return f"GPU avg util: {gpu_avg:.1f}% | Mem avg util: {mem_avg:.1f}% (samples: {len(self.samples)})"

def purge_vram(sync=True):
    gc.collect()
    if torch.cuda.is_available():
        try:
            if sync: torch.cuda.synchronize()
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            if sync: torch.cuda.synchronize()
        except Exception: pass

def print_vram_state(prefix="VRAM"):
    try:
        cuda_ok=torch.cuda.is_available()
        print(f"[{prefix}] cuda_available={cuda_ok}")
        if not cuda_ok: return
        torch.cuda.set_device(GPU_INDEX)
        name=torch.cuda.get_device_name(GPU_INDEX)
        cc=".".join(map(str,torch.cuda.get_device_capability(GPU_INDEX)))
        free_b,total_b=torch.cuda.mem_get_info()
        reserved=torch.cuda.memory_reserved()//(1024*1024)
        allocated=torch.cuda.memory_allocated()//(1024*1024)
        free=free_b//(1024*1024); total=total_b//(1024*1024)
        print(f"[{prefix}] device={name} | free={free} MiB / total={total} MiB | torch_reserved={reserved} | torch_allocated={allocated}")
    except Exception as e:
        print(f"[{prefix}] error: {e}")

def print_gpu_processes():
    try:
        nvsmi=shutil.which("nvidia-smi")
        if not nvsmi:
            print("[GPU PROCS] nvidia-smi introuvable."); return
        q="pid,process_name,used_memory"
        out=subprocess.check_output([nvsmi,f"--query-compute-apps={q}","--format=csv,noheader,nounits","-i",str(GPU_INDEX)],
                                    stderr=subprocess.DEVNULL,text=True).strip()
        print("[GPU PROCS]\n"+(out if out else "Aucun processus compute sur le GPU."))
    except Exception as e:
        print(f"[GPU PROCS] error: {e}")
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated()//(1024*1024)
        print(f"[GPU SELF] torch_allocated={mem} MiB")

# =========================
#   GPU AUTODETECT & AUTOTUNE
# =========================
from typing import Any

def _best_gpu_index_by_memory_then_cc() -> Optional[int]:
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

def get_gpu_info(index: Optional[int]=None) -> Dict[str, Any]:
    info = dict(available=torch.cuda.is_available())
    if not info["available"]:
        return dict(available=False, index=None, name="CPU", total_mib=0, cc=(0,0), bf16=False)
    if index is None:
        index = _best_gpu_index_by_memory_then_cc()
    props = torch.cuda.get_device_properties(index)
    total_mib = props.total_memory // (1024*1024)
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

def apply_gpu_autotune(gpu):
    """
    Tiers:
      - <= 8 GiB     → 'Speed', chunks petits, purges + fréquentes, offload agressif
      - 8–12 GiB     → 'Balanced'
      - 12–24 GiB    → 'Quality+' (défaut)
      - > 24 GiB     → 'Quality+' + gros batch/chunk, purges moins fréquentes
    """
    global GPU_INDEX, DEFAULT_BATCH_SIZE, PURGE_EVERY_N_BATCHES
    global MAX_TOKENS_PER_CHUNK, DYNAMIC_FACTOR_OUT, DEFAULT_MODEL_KEY
    global DEFAULT_PRESET, MIN_VRAM_FOR_ENFORCE_MIB, MIN_RAM_FOR_CPU_OFFLOAD_MIB
    global OFFLOAD_ENFORCE_TO_CPU

    if not gpu["available"]:
        DEFAULT_MODEL_KEY = "Fast (NLLB-200 distilled 600M)"
        DEFAULT_PRESET = "Speed"
        DEFAULT_BATCH_SIZE = max(MIN_BATCH_SIZE, 32)
        MAX_TOKENS_PER_CHUNK = 320
        DYNAMIC_FACTOR_OUT = 1.20
        PURGE_EVERY_N_BATCHES = 8
        OFFLOAD_ENFORCE_TO_CPU = False
        MIN_VRAM_FOR_ENFORCE_MIB = 0
        print("🧠 Auto-tune: CPU mode → paramètres conservateurs.")
        return

    GPU_INDEX = gpu["index"]
    try:
        torch.cuda.set_device(GPU_INDEX)
    except Exception:
        pass

    vram = gpu["total_mib"]
    bf16_ok = gpu["bf16"]

    if vram <= 8192:
        DEFAULT_MODEL_KEY = "Fast (NLLB-200 distilled 600M)"
        DEFAULT_PRESET = "Speed"
        DEFAULT_BATCH_SIZE = 96 if bf16_ok else 64
        MAX_TOKENS_PER_CHUNK = 320
        DYNAMIC_FACTOR_OUT = 1.18
        PURGE_EVERY_N_BATCHES = 8
        OFFLOAD_ENFORCE_TO_CPU = True
        MIN_VRAM_FOR_ENFORCE_MIB = 2048
        MIN_RAM_FOR_CPU_OFFLOAD_MIB = 8000
        tier = "≤8 GiB"

    elif vram <= 12288:
        DEFAULT_MODEL_KEY = "Fast (NLLB-200 distilled 600M)"
        DEFAULT_PRESET = "Balanced"
        DEFAULT_BATCH_SIZE = 160 if bf16_ok else 128
        MAX_TOKENS_PER_CHUNK = 360
        DYNAMIC_FACTOR_OUT = 1.22
        PURGE_EVERY_N_BATCHES = 12
        OFFLOAD_ENFORCE_TO_CPU = True
        MIN_VRAM_FOR_ENFORCE_MIB = 4096
        MIN_RAM_FOR_CPU_OFFLOAD_MIB = 10000
        tier = "8–12 GiB"

    elif vram <= 24576:
        DEFAULT_MODEL_KEY = "Quality (NLLB-200 1.3B)"
        DEFAULT_PRESET = "Quality+"
        DEFAULT_BATCH_SIZE = 256 if bf16_ok else 192
        MAX_TOKENS_PER_CHUNK = 420
        DYNAMIC_FACTOR_OUT = 1.25
        PURGE_EVERY_N_BATCHES = 16
        OFFLOAD_ENFORCE_TO_CPU = True
        MIN_VRAM_FOR_ENFORCE_MIB = 6144
        MIN_RAM_FOR_CPU_OFFLOAD_MIB = 12000
        tier = "12–24 GiB"

    else:
        DEFAULT_MODEL_KEY = "Very High (M2M100 1.2B)"
        DEFAULT_PRESET = "Quality+"
        DEFAULT_BATCH_SIZE = 448 if bf16_ok else 320
        MAX_TOKENS_PER_CHUNK = 460
        DYNAMIC_FACTOR_OUT = 1.28
        PURGE_EVERY_N_BATCHES = 24
        OFFLOAD_ENFORCE_TO_CPU = True
        MIN_VRAM_FOR_ENFORCE_MIB = 8192
        MIN_RAM_FOR_CPU_OFFLOAD_MIB = 14000
        tier = ">24 GiB"

    print(f"🧠 Auto-tune: GPU='{gpu['name']}', VRAM={vram} MiB, BF16={bf16_ok} → tier {tier}")
    print(f"   • Model default: {DEFAULT_MODEL_KEY} | Preset default: {DEFAULT_PRESET}")
    print(f"   • Batch={DEFAULT_BATCH_SIZE} | Max tokens/chunk={MAX_TOKENS_PER_CHUNK} | dyn_out={DYNAMIC_FACTOR_OUT}")
    print(f"   • Purge every {PURGE_EVERY_N_BATCHES} batches | Enforce offload={OFFLOAD_ENFORCE_TO_CPU} (VRAM<{MIN_VRAM_FOR_ENFORCE_MIB} MiB)")

# =========================
#           UI (updated)
# =========================

def ask_options():
    global OFFLINE_MODE, MODELS_CACHE_DIR
    print("🪟 Options…")
    root=tk.Tk(); root.title("Options de traduction"); root.update_idletasks(); root.attributes("-topmost",True); root.lift()

    # Labels à gauche
    tk.Label(root,text="Modèle").grid(row=0,column=0,sticky="w",padx=8,pady=6)
    tk.Label(root,text="Langue cible").grid(row=1,column=0,sticky="w",padx=8,pady=6)
    tk.Label(root,text="Batch size (départ)").grid(row=2,column=0,sticky="w",padx=8,pady=6)
    tk.Label(root,text="Preset").grid(row=3,column=0,sticky="w",padx=8,pady=6)

    # Widgets à droite
    model_var=tk.StringVar(value=DEFAULT_MODEL_KEY)
    tk.OptionMenu(root, model_var, *MODELS.keys()).grid(row=0,column=1,padx=8,pady=6,sticky="ew")

    tgt_var=tk.StringVar(value="English (en)")
    tk.OptionMenu(root, tgt_var, *LANG_CODES.keys()).grid(row=1,column=1,padx=8,pady=6,sticky="ew")

    batch_var=tk.IntVar(value=DEFAULT_BATCH_SIZE)
    tk.Entry(root,textvariable=batch_var).grid(row=2,column=1,padx=8,pady=6,sticky="ew")

    preset_var=tk.StringVar(value=DEFAULT_PRESET)
    tk.OptionMenu(root, preset_var, "Speed", "Balanced", "Quality+").grid(row=3,column=1,padx=8,pady=6,sticky="ew")

    # --- Nouveaux contrôles Offline + Cache dir
    offline_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Mode hors‑ligne (aucun téléchargement / aucun réseau)", variable=offline_var).grid(row=4,column=0,columnspan=2,sticky="w",padx=8,pady=(12,0))

    tk.Label(root,text="Dossier de stockage des modèles").grid(row=5,column=0,sticky="w",padx=8,pady=(8,4))
    cache_var = tk.StringVar(value="" if MODELS_CACHE_DIR is None else MODELS_CACHE_DIR)
    ent = tk.Entry(root,textvariable=cache_var,width=52)
    ent.grid(row=5,column=1,sticky="ew",padx=8,pady=(8,4))
    def _choose_dir():
        d = filedialog.askdirectory(title="Choisir un dossier pour les modèles")
        if d:
            cache_var.set(d)
    tk.Button(root,text="Parcourir…",command=_choose_dir).grid(row=5,column=2,sticky="w",padx=(0,8),pady=(8,4))

    def _ok():
        try: root.attributes("-topmost",False)
        except Exception: pass
        root.destroy()
    tk.Button(root,text="OK",command=_ok).grid(row=6,column=0,columnspan=3,pady=12)

    root.columnconfigure(1, weight=1)
    root.mainloop()

    # Sortie
    OFFLINE_MODE = bool(offline_var.get())
    MODELS_CACHE_DIR = cache_var.get().strip() or None
    apply_offline_env(OFFLINE_MODE, MODELS_CACHE_DIR)  # appliquer avant tout chargement

    model_name=MODELS[model_var.get()]
    tgt_code = LANG_CODES[tgt_var.get()]
    batch=max(MIN_BATCH_SIZE, int(batch_var.get()))
    preset=preset_var.get()
    print(f"⚙️  Options → model={model_name} | tgt={tgt_code} | batch_start={batch} | preset={preset} | offline={OFFLINE_MODE} | cache={MODELS_CACHE_DIR or 'default'}")
    return model_name, tgt_code, batch, preset


def choose_input_file(cli_path: Optional[str]=None) -> Optional[str]:
    if cli_path and os.path.exists(cli_path):
        print(f"✅ Fichier (CLI): {cli_path}"); return cli_path
    print("📂 Choisir le fichier Excel…")
    root=tk.Tk(); root.withdraw(); root.update_idletasks(); root.attributes("-topmost",True); root.lift()
    path=filedialog.askopenfilename(title="Choisissez le fichier Excel à traduire",
                                    filetypes=[("Fichiers Excel","*.xlsx *.xls")])
    try: root.attributes("-topmost",False)
    except Exception: pass
    root.destroy()
    if not path:
        print("❌ Aucun fichier sélectionné.")
        try: messagebox.showwarning("Annulé","Aucun fichier sélectionné.")
        except Exception: pass
        return None
    print(f"✅ Fichier sélectionné: {path}")
    return path

# =========================
#     AIDES TOKEN/TEXTE
# =========================

def same_language(src_code: str, tgt_code: str) -> bool:
    return (src_code or "").split("_")[0] == (tgt_code or "").split("_")[0]

def pick_sentence_column(df: pd.DataFrame) -> str:
    CAND_COLS = ["sentence","Sentence","text","Text","texte","Texte","content","Content","body","Body"]
    for c in CAND_COLS:
        if c in df.columns: return c
    text_cols=[c for c in df.columns if df[c].dtype=="object"]
    if not text_cols: return df.columns[0]
    avg_len={c: df[c].astype(str).str.len().mean() for c in text_cols}
    return max(avg_len, key=avg_len.get)

def chunk_by_tokens(text: str, tokenizer, max_tokens:int=MAX_TOKENS_PER_CHUNK):
    if not text or text.isspace(): return [text]
    parts=re.split(r"([\.!?;:])", text)
    sentences=[]
    for i in range(0,len(parts),2):
        seg=(parts[i] or "").strip()
        if not seg: continue
        if i+1 < len(parts): seg += parts[i+1] or ""
        sentences.append(seg.strip())
    chunks,buf,ids=[],[],[]
    for s in sentences:
        t_ids=tokenizer(s, add_special_tokens=False).input_ids
        if len(ids)+len(t_ids) <= max_tokens:
            buf.append(s); ids.extend(t_ids)
        else:
            if buf: chunks.append(" ".join(buf).strip())
            buf=[s]; ids=t_ids[:]
            while len(ids)>max_tokens:
                cut=tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
                chunks.append(cut.strip()); ids=ids[max_tokens:]
    if buf: chunks.append(" ".join(buf).strip())
    return chunks

def dynamic_max_new_tokens(tokenizer, model_cfg, texts, factor=1.25, floor=50)->int:
    max_in=0
    for t in texts:
        n=len(tokenizer(t, add_special_tokens=False).input_ids)
        if n>max_in: max_in=n
    ceilings=[]
    for attr in ("max_length","max_position_embeddings","max_target_positions"):
        v=getattr(model_cfg,attr,None)
        if isinstance(v,int) and v>0: ceilings.append(v)
    ceiling=min([1024]+ceilings)
    new_tokens=int(max(floor, min(int(max_in*factor), ceiling)))
    return max(floor, min(new_tokens, ceiling-1))

# =========================
#   DÉTECTION DE LANGUE
# =========================
ISO_HINT_EN = re.compile(
    r"\b(the|and|of|in|with|from|to|on|for|is|are|was|were|as|by|at|which|that|who|when|where|during|into|after|before|"
    r"over|under|between|about|around|through|without|because|if|then|but|so|there|here|also|however|although|this|"
    r"these|those|an|a|some|many|more|most|less|each|other)\b", re.IGNORECASE
)

@lru_cache(maxsize=100_000)
def _detect_iso2_cached(head: str) -> str:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    return detect(head)

def detect_lang_nllb(text: str) -> str:
    s=text.strip()
    if not s: return "eng_Latn"
    if re.search(r"[\u0600-\u06FF]", s): return "ara_Arab"
    if re.search(r"[\u0400-\u04FF]", s): return "rus_Cyrl"
    if re.search(r"[\u4E00-\u9FFF]", s): return "zho_Hans"
    head = s[:160]
    try:
        iso2=_detect_iso2_cached(head)
        code=ISO2_TO_NLLB.get(iso2)
        if code: return code
    except Exception:
        pass
    if ISO_HINT_EN.search(s): return "eng_Latn"
    return "fra_Latn"

def looks_like_target(text: str, tgt_code: str) -> bool:
    s=(text or "").strip() if isinstance(text,str) else ("" if text is None else str(text)).strip()
    if not s: return True
    try:
        iso=_detect_iso2_cached(s[:160])
        tgt_iso = tgt_code.split("_")[0][:2]
        return ISO2_TO_NLLB.get(iso, "").startswith(tgt_iso)
    except Exception:
        pass
    if tgt_code.endswith("Cyrl"): return bool(re.search(r"[\u0400-\u04FF]", s))
    if tgt_code.endswith("Arab"): return bool(re.search(r"[\u0600-\u06FF]", s))
    if tgt_code == "zho_Hans":    return bool(re.search(r"[\u4E00-\u9FFF]", s))
    if re.search(r"[^\x00-\x7F]", s): return False
    return True

# =========================
#  CACHE MODÈLES SPÉCIALISTES (UPDATED)
# =========================
_fallback_cache = {}
def get_fallback_tokenizer_model(src_code: str, tgt_code: str, device: torch.device):
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

    try:
        tok = AutoTokenizer.from_pretrained(resolved, **tp)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            resolved,
            torch_dtype=(torch.bfloat16 if device.type=="cuda" else None),
            attn_implementation="sdpa",
            **tp
        )
    except Exception as e:
        if OFFLINE_MODE:
            print(f"⚠️ Spécialiste introuvable localement pour {name}: {e}")
            _explain_missing_model(name)
        raise

    mdl.eval()
    if device.type=="cuda":
        mdl.to(device)
    _fallback_cache[key] = (tok, mdl)
    return tok, mdl

# =========================
#  CHEMIN RAPIDE RU→EN→TGT (UPDATED)
# =========================
_fast_en_cache = {}
def get_fast_en2tgt_tokenizer_model(tgt_code: str, device: torch.device):
    base_repo = "facebook/nllb-200-distilled-600M"
    key = ("en2tgt_fast", tgt_code)

    if key in _fast_en_cache:
        tok, mdl = _fast_en_cache[key]
        if mdl is not None:
            mdl.to(device)
        return tok, mdl

    tp = _tp_kwargs()
    resolved = _resolve_local_repo(base_repo)

    try:
        tok = AutoTokenizer.from_pretrained(resolved, **tp)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            resolved,
            torch_dtype=(torch.bfloat16 if device.type=="cuda" else None),
            attn_implementation="sdpa",
            **tp
        )
    except Exception as e:
        if OFFLINE_MODE:
            print(f"⚠️ Modèle rapide EN↔TGT introuvable localement ({base_repo}): {e}")
            _explain_missing_model(base_repo)
        raise

    mdl.eval()
    if device.type == "cuda":
        mdl.to(device)
    _fast_en_cache[key] = (tok, mdl)
    return tok, mdl


def fast_ru_to_target(batch_texts: List[str], tgt_code: str, device: torch.device):
    tok_ru, mdl_ru = get_fallback_tokenizer_model("rus_Cyrl", "eng_Latn", device)
    enc = tok_ru(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS_PER_CHUNK)
    if device.type == "cuda":
        for k in enc: enc[k] = enc[k].to(device, non_blocking=True)
    with torch.inference_mode():
        out_ids = mdl_ru.generate(**enc, max_new_tokens=192, num_beams=1, do_sample=False, use_cache=True)
    mid_en = tok_ru.batch_decode(out_ids, skip_special_tokens=True)

    tok_fast, mdl_fast = get_fast_en2tgt_tokenizer_model(tgt_code, device)
    if hasattr(tok_fast, "src_lang"): tok_fast.src_lang = "eng_Latn"
    if hasattr(tok_fast, "tgt_lang"): tok_fast.tgt_lang = tgt_code
    forced_bos = getattr(getattr(tok_fast, "lang_code_to_id", {}), "get", lambda _:_)(tgt_code, None)
    enc2 = tok_fast(mid_en, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS_PER_CHUNK)
    if device.type == "cuda":
        for k in enc2: enc2[k] = enc2[k].to(device, non_blocking=True)
    with torch.inference_mode():
        kwargs = dict(**enc2, max_new_tokens=192, num_beams=1, do_sample=False, use_cache=True)
        if forced_bos is not None: kwargs["forced_bos_token_id"] = forced_bos
        out_ids2 = mdl_fast.generate(**kwargs)
    return tok_fast.batch_decode(out_ids2, skip_special_tokens=True)

# =========================
#     PARAM. GÉNÉRATION (ROBUSTE ANTI-OOM)
# =========================

def gen_params_for_preset(preset: str):
    if preset == "Speed":
        return dict(num_beams=1, do_sample=False, repetition_penalty=1.0, length_penalty=1.0, early_stopping=False)
    if preset == "Balanced":
        return dict(num_beams=3, do_sample=False, repetition_penalty=1.05, length_penalty=1.02, early_stopping=True)
    return dict(num_beams=5, do_sample=False, repetition_penalty=1.1, length_penalty=1.05, early_stopping=True)

def is_m2m(model_name: str) -> bool:
    return "m2m100" in model_name.lower()

def translate_batch_generic(model_name, tokenizer, model, device, src_code, tgt_code, texts, model_cfg, preset="Quality+", extra_gen_kwargs=None):
    """
    Traduction générique par lots, avec backoff OOM intégré:
      1) essaie params demandés
      2) si OOM: beams -> 1, puis use_cache -> False
      3) si OOM: réduit max_new_tokens (80%, puis 60%)
      4) si OOM: scinde le lot en 2 (récursif) et purge la VRAM entre tentatives
    """
    def _encode(_texts, _max_len=MAX_TOKENS_PER_CHUNK):
        enc = tokenizer(_texts, return_tensors="pt", padding=True, truncation=True, max_length=_max_len)
        if device.type == "cuda":
            for k in enc: enc[k] = enc[k].to(device, non_blocking=True)
        return enc

    def _gen_attempt(enc, genp, forced_bos, max_new, use_cache=True):
        with torch.inference_mode():
            kwargs = dict(**enc, max_new_tokens=max_new, use_cache=use_cache, **genp)
            if forced_bos is not None:
                kwargs["forced_bos_token_id"] = forced_bos
            return model.generate(**kwargs)

    # ---- params de départ
    genp = gen_params_for_preset(preset)
    if extra_gen_kwargs: genp.update(extra_gen_kwargs)
    factor = 1.25 if preset != "Quality+" else 1.30
    if (src_code or "").startswith("rus_"):
        genp.setdefault("num_beams", 1)
        factor = min(factor, 1.15)

    # M2M vs NLLB: config langue & forced_bos
    if is_m2m(model_name):
        src = NLLB_TO_M2M.get(src_code, "auto"); tgt = NLLB_TO_M2M.get(tgt_code, "en")
        if hasattr(tokenizer, "src_lang"): tokenizer.src_lang = src
        if hasattr(tokenizer, "tgt_lang"): tokenizer.tgt_lang = tgt
        forced_bos = tokenizer.get_lang_id(tgt) if hasattr(tokenizer, "get_lang_id") else None
    else:
        if hasattr(tokenizer, "src_lang"): tokenizer.src_lang = src_code
        if hasattr(tokenizer, "tgt_lang"): tokenizer.tgt_lang = tgt_code
        forced_bos = tokenizer.lang_code_to_id[tgt_code] if hasattr(tokenizer,"lang_code_to_id") and tgt_code in tokenizer.lang_code_to_id else None

    dyn_new = dynamic_max_new_tokens(tokenizer, model_cfg, texts, factor=factor, floor=MIN_NEW_TOKENS)
    enc = _encode(texts)

    # ---- plan de backoff
    attempts = [
        dict(genp=dict(genp), max_new=dyn_new, use_cache=True),
        dict(genp={**genp, "num_beams": 1}, max_new=dyn_new, use_cache=True),
        dict(genp={**genp, "num_beams": 1}, max_new=int(dyn_new*0.8), use_cache=True),
        dict(genp={**genp, "num_beams": 1}, max_new=int(dyn_new*0.8), use_cache=False),
        dict(genp={**genp, "num_beams": 1, "no_repeat_ngram_size": 0}, max_new=int(dyn_new*0.6), use_cache=False),
    ]

    for step, plan in enumerate(attempts, 1):
        try:
            out_ids = _gen_attempt(enc, plan["genp"], forced_bos, plan["max_new"], use_cache=plan["use_cache"])
            return tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"⚠️ OOM in generate (attempt {step}/{len(attempts)}): backoff…")
            purge_vram(sync=True)

    # ---- encore OOM → micro-batch (split en 2), récursif
    if len(texts) > 1:
        mid = len(texts)//2
        left  = translate_batch_generic(model_name, tokenizer, model, device, src_code, tgt_code, texts[:mid],  model_cfg, preset=preset, extra_gen_kwargs=extra_gen_kwargs)
        purge_vram(sync=True); print_vram_state("VRAM (split-L)")
        right = translate_batch_generic(model_name, tokenizer, model, device, src_code, tgt_code, texts[mid:], model_cfg, preset=preset, extra_gen_kwargs=extra_gen_kwargs)
        purge_vram(sync=True); print_vram_state("VRAM (split-R)")
        return left + right

    # ---- dernier recours: retenter avec cache off + max_new réduit
    enc = _encode(texts)
    out_ids = _gen_attempt(enc, {"num_beams":1}, forced_bos, max_new=max(MIN_NEW_TOKENS, int(dyn_new*0.5)), use_cache=False)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)


def translate_batch_ultra(model_name, tokenizer, model, device, src_code, tgt_code, texts, model_cfg):
    ultra = dict(num_beams=3, do_sample=False, repetition_penalty=1.12,
                 length_penalty=1.06, early_stopping=True, no_repeat_ngram_size=3)
    return translate_batch_generic(model_name, tokenizer, model, device, src_code, tgt_code, texts, model_cfg, preset="Quality+", extra_gen_kwargs=ultra)


def cleanup_english_outputs(tokenizer, model, device, texts, model_cfg):
    tgt_code = "eng_Latn"
    if hasattr(tokenizer, "tgt_lang"): tokenizer.tgt_lang = tgt_code
    forced_bos = getattr(tokenizer, "lang_code_to_id", {}).get(tgt_code, None)
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS_PER_CHUNK)
    if device.type=="cuda":
        for k in enc: enc[k] = enc[k].to(device, non_blocking=True)
    with torch.inference_mode():
        out_ids = model.generate(**enc, num_beams=2, do_sample=False, use_cache=True,
                                 length_penalty=1.02, repetition_penalty=1.05, early_stopping=True,
                                 forced_bos_token_id=forced_bos)
    return tokenizer.batch_decode(out_ids, skip_special_tokens=True)

# =========================
#    RAM & OFFLOAD HELPERS
# =========================

def free_vram_mib() -> int:
    if not torch.cuda.is_available(): return 0
    free_b, _ = torch.cuda.mem_get_info()
    return free_b // (1024 * 1024)

def free_ram_mib() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available // (1024 * 1024))
    except Exception:
        return 0

def _move_cached_models_to(device: torch.device):
    try:
        for key, val in _fallback_cache.items():
            tok, mdl = val
            if mdl is not None:
                mdl.to(device)
        for key, val in _fast_en_cache.items():
            tok, mdl = val
            if mdl is not None:
                mdl.to(device)
    except Exception:
        pass

# =========================
#    ENFORCE LANGUE CIBLE
# =========================

def enforce_target_language(model_name, tokenizer, model, device, model_cfg, work_items, outputs, tgt_code):
    """
    Garde-fou de langue:
    - Offload CPU temporaire si VRAM faible et RAM suffisante.
    - Beams adaptatifs pour limiter la conso mémoire.
    - Purges VRAM fréquentes (avant/après blocs).
    - 2 passes maximum.
    """
    import re as _re

    local_device = device
    offloaded = False
    if (device.type == "cuda" and OFFLINE_MODE is not None and True and True and OFFLOAD_ENFORCE_TO_CPU):
        vram = free_vram_mib()
        ram  = free_ram_mib()
        if vram < MIN_VRAM_FOR_ENFORCE_MIB and ram >= MIN_RAM_FOR_CPU_OFFLOAD_MIB:
            print(f"🧳 Offload Enforce → CPU (VRAM {vram} MiB < {MIN_VRAM_FOR_ENFORCE_MIB} MiB, RAM dispo {ram} MiB)")
            try:
                model.to(torch.device("cpu"))
                _move_cached_models_to(torch.device("cpu"))
                local_device = torch.device("cpu")
                offloaded = True
                purge_vram(sync=True)
                print_vram_state("VRAM (after enforce offload)")
            except Exception as e:
                print(f"⚠️ Offload CPU impossible ({e}) → poursuite sur GPU")

    def _batch_translate_src_to(src_lang, tgt_lang, idxs, extra=None):
        texts = [work_items[i][2] for i in idxs]
        use_specialist = ((src_lang, tgt_lang) in PAIR_SPECIALISTS)
        vfree = free_vram_mib()
        beam_enf = 1 if vfree < 6000 else (2 if vfree < 12000 else 3)

        if use_specialist:
            tok_s, mdl_s = get_fallback_tokenizer_model(src_lang, tgt_lang, local_device)
            if tok_s is not None and mdl_s is not None:
                enc = tok_s(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS_PER_CHUNK)
                if local_device.type == "cuda":
                    for k in enc: enc[k] = enc[k].to(local_device, non_blocking=True)
                with torch.inference_mode():
                    out_ids = mdl_s.generate(**enc, num_beams=beam_enf, do_sample=False, use_cache=True,
                                             length_penalty=1.06, repetition_penalty=1.08, early_stopping=True,
                                             no_repeat_ngram_size=(3 if beam_enf >= 2 else 0), max_new_tokens=160)
                res = tok_s.batch_decode(out_ids, skip_special_tokens=True)
                purge_vram(sync=True)
                return res

        extra_defaults = dict(num_beams=beam_enf, no_repeat_ngram_size=(3 if beam_enf >= 2 else 0),
                              length_penalty=1.04, repetition_penalty=1.06)
        res = translate_batch_generic(model_name, tokenizer, model, local_device, src_lang, tgt_lang, texts, model_cfg,
                                      preset="Quality+", extra_gen_kwargs=(extra or extra_defaults))
        purge_vram(sync=True)
        return res

    for pass_id in (1, 2):
        purge_vram(sync=True)

        bad = [i for i in range(len(work_items)) if not looks_like_target(outputs[i], tgt_code)]
        if not bad: break
        print(f"🧭 Enforce target [{tgt_code}] — pass {pass_id}: {len(bad)} segment(s)")

        grp = {}
        for i in bad:
            _, _, _chunk, src_lang = work_items[i]
            grp.setdefault(src_lang, []).append(i)

        for src_lang, idxs in grp.items():
            purge_vram(sync=True)

            if tgt_code != "eng_Latn":
                mid = _batch_translate_src_to(src_lang, "eng_Latn", idxs)
                vfree = free_vram_mib()
                beam_enf = 1 if vfree < 6000 else (2 if vfree < 12000 else 3)
                extra = dict(num_beams=beam_enf, no_repeat_ngram_size=(3 if beam_enf >= 2 else 0),
                             length_penalty=1.04, repetition_penalty=1.06)
                final = translate_batch_generic(model_name, tokenizer, model, local_device, "eng_Latn", tgt_code, mid, model_cfg,
                                                preset="Quality+", extra_gen_kwargs=extra)
                for j, i in enumerate(idxs):
                    outputs[i] = final[j]
                purge_vram(sync=True)
            else:
                final = _batch_translate_src_to(src_lang, "eng_Latn", idxs)
                rx_nonlatin = _re.compile(r"[\u0400-\u04FF\u4E00-\u9FFF\u0600-\u06FF]")
                fix_map = [i for i, txt in zip(idxs, final) if rx_nonlatin.search((txt or ""))]
                for j, i in enumerate(idxs):
                    outputs[i] = final[j]
                if fix_map:
                    to_fix = [outputs[i] for i in fix_map]
                    fixed = cleanup_english_outputs(tokenizer, model, local_device, to_fix, model_cfg)
                    for j, i in enumerate(fix_map):
                        outputs[i] = fixed[j]
                    purge_vram(sync=True)

        purge_vram(sync=True)

    if offloaded:
        try:
            model.to(device)
            _move_cached_models_to(device)
            purge_vram(sync=True)
            print_vram_state("VRAM (after enforce reload)")
        except Exception as e:
            print(f"⚠️ Retour GPU après Enforce impossible ({e})")

    purge_vram(sync=True)


# =========================
#  RESOLUTION LOCALE (NEW)
# =========================
def _candidate_local_paths(repo_or_path: str, base_dir: Optional[str]) -> List[str]:
    paths = []
    # 1) Chemin absolu déjà un dossier → ok
    if os.path.isabs(repo_or_path) and os.path.isdir(repo_or_path):
        paths.append(repo_or_path)
    # 2) Essais sous le dossier base (org/repo et repo)
    if base_dir:
        repo_norm = repo_or_path.replace("\\", "/").strip("/")
        paths.append(os.path.join(base_dir, repo_norm.replace("/", os.sep)))
        paths.append(os.path.join(base_dir, os.path.basename(repo_norm)))
    return [p for p in paths if os.path.isdir(p)]

def _resolve_local_repo(repo_or_path: str) -> str:
    """Si OFFLINE_MODE: renvoie un chemin local existant pour ce repo si trouvé, sinon repo_or_path."""
    if not OFFLINE_MODE:
        return repo_or_path
    for p in _candidate_local_paths(repo_or_path, MODELS_CACHE_DIR):
        return p
    return repo_or_path

def _explain_missing_model(repo_or_path: str):
    short = repo_or_path.replace("\\","/").split("/")[-1]
    base = MODELS_CACHE_DIR or "<non défini>"
    msg = (f"Modèle introuvable en local pour '{repo_or_path}'.\n"
           f"Mode hors-ligne actif → aucun téléchargement possible.\n\n"
           f"Vérifie la présence d'un dossier modèle sous :\n"
           f"  {base}\\facebook\\{short}  (ou  {base}\\{short})\n"
           f"et qu'il contient: config.json, tokenizer.*, poids (.bin/.safetensors).")
    try:
        from tkinter import messagebox
        messagebox.showerror("Modèle manquant (hors-ligne)", msg)
    except Exception:
        print("[OFFLINE] " + msg)


# =========================
#       CHARGEMENT MODÈLE (UPDATED)
# =========================
def load_model(model_name: str, device: torch.device):
    print(f"🔧 Chargement modèle : {model_name}")
    tp = _tp_kwargs()

    # Résout vers un dossier local si OFFLINE
    resolved = _resolve_local_repo(model_name)
    if OFFLINE_MODE and resolved == model_name:
        # Rien trouvé en local → avertir tout de suite (et on essaie quand même, pour log détaillé)
        _explain_missing_model(model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved, **tp)
    except Exception as e:
        if OFFLINE_MODE:
            print(f"⚠️ Tokenizer introuvable localement pour {model_name}: {e}")
            _explain_missing_model(model_name)
        raise

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            resolved,
            torch_dtype=(torch.bfloat16 if device.type=="cuda" else None),
            attn_implementation="sdpa",
            **tp
        )
    except Exception as e:
        if OFFLINE_MODE:
            print(f"⚠️ Poids introuvables localement pour {model_name}: {e}")
            _explain_missing_model(model_name)
        raise

    model.eval()
    if device.type=="cuda":
        model.to(device)
    return tokenizer, model

# =========================
#   AUTO TUNING DE BATCH
# =========================

def suggest_next_batch_size(curr_bs: int, free_mib: int, max_bs_cap: int = 1024) -> int:
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
#           MAIN
# =========================

def main():
    purge_vram(); print_vram_state("VRAM (startup)"); print_gpu_processes()

    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument("--input",type=str,default=None)
    args,_=parser.parse_known_args()

    # Auto-détection GPU + auto-tune paramètres
    gpu = get_gpu_info(None)
    apply_gpu_autotune(gpu)
    if gpu["available"]:
        print(f"🎯 Using GPU index {GPU_INDEX}: {gpu['name']} ({gpu['total_mib']} MiB)")

    input_path=choose_input_file(args.input)
    if not input_path: return
    base,_=os.path.splitext(input_path); base=str(base)

    model_name, tgt_code, batch_size, preset = ask_options()  # applique offline/env ici
    print_vram_state("VRAM (before model load)")
    output_path=f"{base}_translated_{tgt_code.split('_')[0]}.xlsx"

    print(f"torch {torch.__version__} | cuda={getattr(torch.version,'cuda',None)} | cuda dispo={torch.cuda.is_available()}")
    device=torch.device(f"cuda:{GPU_INDEX}") if torch.cuda.is_available() else torch.device("cpu")
    print("⚡ GPU CUDA détecté → BF16/SDPA") if device.type=="cuda" else print("⚠️ CPU (plus lent)")

    df=pd.read_excel(input_path)
    text_col=pick_sentence_column(df); print(f"🧭 Colonne détectée : '{text_col}'")
    rows=df[text_col].astype(str).tolist()
    lang_col="language" if "language" in df.columns else None

    tokenizer, model = load_model(model_name, device)
    model_cfg=model.config
    print_vram_state("VRAM (after model load)")

    # Préparation des segments
    work_items: List[Tuple[int,int,str,str]] = []
    keep_original = set()

    for i, s in enumerate(rows):
        s = (s or "").strip()
        if not s: continue
        src_code = detect_lang_nllb(s)
        if same_language(src_code, tgt_code):
            keep_original.add(i)
            continue
        for j, ch in enumerate(chunk_by_tokens(s, tokenizer, MAX_TOKENS_PER_CHUNK)):
            work_items.append((i, j, ch, src_code))

    if not work_items:
        print("ℹ️ Aucune donnée à traduire (tout est déjà dans la langue cible).")
        df_out = df.copy()
        df_out[text_col] = [sanitize_cell(x) for x in rows]
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Sheet1")
            writer.book.strings_to_urls = False
        print(f"✅ Fichier (identique) : {output_path}")
        return

    # Groupement par langue source
    groups: Dict[str, List[int]] = {}
    for idx,(row,order,ch,src) in enumerate(work_items):
        groups.setdefault(src, []).append(idx)

    # Monitoring GPU
    batch_size = max(MIN_BATCH_SIZE, int(batch_size))
    mon = None
    if device.type=="cuda" and GPU_LOG_CSV:
        print("🛰️ Monitoring GPU…")
        mon = GPUMonitor(index=GPU_INDEX, interval=MONITOR_INTERVAL_S, print_every=PRINT_EVERY, log_csv=GPU_LOG_CSV)
        mon.start()

    print_vram_state("VRAM (before translate)")
    t0=time.time()
    outputs=[None]*len(work_items)
    batch_idx=0
    pbar=tqdm(total=len(work_items), unit="seg")

    # =========================
    #     BOUCLE PRINCIPALE
    # =========================
    for src_lang, idx_list in groups.items():
        print(f"🌐 Source={src_lang} → Target={tgt_code} | seg={len(idx_list)}")
        k=0
        group_texts=[work_items[idx][2] for idx in idx_list]

        specialist_cap = 1024 if ((src_lang, tgt_code) in PAIR_SPECIALISTS) else 512

        while k < len(group_texts):
            free_mib = free_vram_mib()
            batch_size = suggest_next_batch_size(batch_size, free_mib, max_bs_cap=specialist_cap)

            bs = min(batch_size, len(group_texts)-k)
            batch_texts=group_texts[k:k+bs]

            sl = src_lang
            if tgt_code=="eng_Latn":
                join=" ".join(batch_texts)
                if re.search(r"[\u0400-\u04FF]", join): sl="rus_Cyrl"
                elif re.search(r"[\u4E00-\u9FFF]", join): sl="zho_Hans"
                elif re.search(r"[\u0600-\u06FF]", join): sl="ara_Arab"

            out_txts=None

            # 1) Spécialiste direct
            if (sl, tgt_code) in PAIR_SPECIALISTS:
                try:
                    tok_s, mdl_s = get_fallback_tokenizer_model(sl, tgt_code, device)
                    if tok_s is not None:
                        enc = tok_s(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS_PER_CHUNK)
                        if device.type=="cuda":
                            for k2 in enc: enc[k2]=enc[k2].to(device, non_blocking=True)
                        with torch.inference_mode():
                            out_ids = mdl_s.generate(**enc, max_new_tokens=192, num_beams=1, do_sample=False, use_cache=True)
                        out_txts = tok_s.batch_decode(out_ids, skip_special_tokens=True)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        purge_vram(); batch_size=max(MIN_BATCH_SIZE, batch_size//2); print(f"⚠️ OOM specialist → reduce batch to {batch_size}")
                        continue
                    else:
                        raise

            # 2) Chemin rapide ru→en→tgt
            if out_txts is None and sl.startswith("rus_") and tgt_code != "eng_Latn":
                try:
                    out_txts = fast_ru_to_target(batch_texts, tgt_code, device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        purge_vram(); batch_size=max(MIN_BATCH_SIZE, batch_size//2); print(f"⚠️ OOM ru-fast → reduce batch to {batch_size}")
                        continue
                    else:
                        raise

            # 3) Pipeline générique robuste
            if out_txts is None:
                try:
                    extra = None
                    if sl.startswith("rus_"):
                        extra = dict(num_beams=1)
                    out_txts = translate_batch_generic(model_name, tokenizer, model, device, sl, tgt_code, batch_texts, model_cfg, preset=preset, extra_gen_kwargs=extra)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        purge_vram(); batch_size=max(MIN_BATCH_SIZE, batch_size//2); print(f"⚠️ OOM generic → reduce batch to {batch_size}")
                        continue
                    else:
                        raise

            for j,txt in enumerate(out_txts):
                outputs[idx_list[k+j]]=txt

            k+=bs; pbar.update(bs); batch_idx+=1

            if device.type=="cuda" and (batch_idx % PURGE_EVERY_N_BATCHES == 0):
                print("🧹 Purge VRAM…"); purge_vram(); print_vram_state("VRAM (periodic purge)")

    pbar.close()

    # =========================
    #     RETRY RAPIDE
    # =========================
    need_retry=[i for i in range(len(work_items)) if not looks_like_target(outputs[i], tgt_code)]
    if need_retry:
        print(f"♻️ Retry (greedy): {len(need_retry)} segments")
        retry_groups={}
        for idx in need_retry:
            _,_,_,src_lang = work_items[idx]
            retry_groups.setdefault(src_lang, []).append(idx)
        for src_lang, idx_list in retry_groups.items():
            texts=[work_items[i][2] for i in idx_list]
            outs=translate_batch_generic(model_name, tokenizer, model, device, src_lang, tgt_code, texts, model_cfg, preset="Speed")
            for j,i in enumerate(idx_list): outputs[i]=outs[j]

    # =========================
    #      ULTRA-RETRY
    # =========================
    still_bad = [i for i in range(len(work_items)) if not looks_like_target(outputs[i], tgt_code)]
    hard_idxs = [i for i in still_bad if (not work_items[i][3].startswith("rus_")) and len((work_items[i][2] or "").split()) >= 12][:200]
    if ENABLE_ULTRA and hard_idxs:
        print(f"💪 Ultra-retry (lighter): {len(hard_idxs)} segments (cap 200)")
        grp = {}
        for i in hard_idxs:
            _, _, _, src_lang = work_items[i]
            grp.setdefault(src_lang, []).append(i)
        for src_lang, idxs in grp.items():
            texts = [work_items[i][2] for i in idxs]
            outs  = translate_batch_ultra(model_name, tokenizer, model, device, src_lang, tgt_code, texts, model_cfg)
            for j,i in enumerate(idxs): outputs[i] = outs[j]

    # =========================
    #   ENFORCE LANGUE CIBLE
    # =========================
    purge_vram(sync=True)
    enforce_target_language(model_name, tokenizer, model, device, model_cfg, work_items, outputs, tgt_code)
    purge_vram(sync=True)

    if mon is not None:
        mon.stop()
        print("📊", mon.summary())
    purge_vram(); print_vram_state("VRAM (final)")

    # =========================
    #   RECONSTRUCTION & EXPORT
    # =========================
    by_row={}
    for (row,order,_ch,_src), txt in zip(work_items, outputs):
        by_row.setdefault(row,{})[order]=txt
    translated = []
    for i, s in enumerate(rows):
        if i in keep_original:
            translated.append(s)
        elif i not in by_row:
            translated.append(s)
        else:
            parts = [by_row[i][k] for k in sorted(by_row[i].keys())]
            translated.append(" ".join(parts).strip())

    df_out=df.copy()
    df_out[text_col]=[sanitize_cell(x) for x in translated]
    for col in df_out.columns:
        if df_out[col].dtype=="object": df_out[col]=sanitize_series_for_excel(df_out[col])
    if lang_col:
        tgt_prefix=tgt_code.split("_")[0]
        df_out[lang_col]=[sanitize_cell(tgt_prefix if isinstance(x,(str,type(None))) else x) for x in df_out.get(lang_col,[])]

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Sheet1")
        writer.book.strings_to_urls = False

    dt=time.time()-t0
    print(f"✅ Fichier traduit (Excel-safe) : {output_path}")
    print(f"⏱️ Durée: {dt:.1f}s")
    try:
        messagebox.showinfo("Terminé", f"Traduction terminée !\nCible : {tgt_code}\nFichier :\n{output_path}")
    except Exception:
        pass

# Point d’entrée
if __name__ == "__main__":
    main()
