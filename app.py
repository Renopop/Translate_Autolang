# -*- coding: utf-8 -*-
"""
Application Streamlit - Traducteur Multilingue Excel
Auteur : Renaud LOISON
Interface moderne avec Streamlit
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import torch

from translator_core import (
    TranslatorConfig,
    ExcelTranslator,
    MODELS,
    LANG_CODES,
    get_gpu_info,
    DEFAULT_BATCH_SIZE,
    purge_vram,
    print_vram_state
)

# Configuration de la page
st.set_page_config(
    page_title="Traducteur Excel Multilingue",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown('<h1 class="main-header">🌐 Traducteur Excel Multilingue</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialisation de l'état de session
if 'translation_done' not in st.session_state:
    st.session_state.translation_done = False
if 'output_file' not in st.session_state:
    st.session_state.output_file = None
if 'gpu_info' not in st.session_state:
    st.session_state.gpu_info = get_gpu_info()

# Barre latérale - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Informations GPU
    gpu = st.session_state.gpu_info
    if gpu['available']:
        st.success(f"🎯 GPU Détecté: {gpu['name']}")
        st.info(f"💾 VRAM: {gpu['total_mib']} MiB")
        st.info(f"🔧 BF16: {'✅' if gpu['bf16'] else '❌'}")
    else:
        st.warning("⚠️ Mode CPU (plus lent)")

    st.markdown("---")

    # Sélection du modèle
    st.subheader("🤖 Modèle de traduction")
    model_choice = st.selectbox(
        "Choisissez le modèle",
        options=list(MODELS.keys()),
        index=0,
        help="Fast: rapide mais qualité moyenne | Quality: bon équilibre | Very High: meilleure qualité mais plus lent"
    )

    # Langue cible
    st.subheader("🌍 Langue cible")
    target_lang = st.selectbox(
        "Traduire vers",
        options=list(LANG_CODES.keys()),
        index=1,  # English par défaut
        help="Langue dans laquelle traduire le texte"
    )

    # Preset de qualité
    st.subheader("⚡ Preset de performance")
    preset = st.radio(
        "Qualité vs Vitesse",
        options=["Speed", "Balanced", "Quality+"],
        index=2,
        help="Speed: plus rapide | Balanced: équilibré | Quality+: meilleure qualité"
    )

    # Batch size
    st.subheader("📦 Taille de batch")
    batch_size = st.number_input(
        "Batch size",
        min_value=16,
        max_value=1024,
        value=DEFAULT_BATCH_SIZE,
        step=16,
        help="Plus grand = plus rapide mais consomme plus de mémoire"
    )

    st.markdown("---")

    # Mode hors-ligne
    st.subheader("🔒 Mode hors-ligne")
    offline_mode = st.checkbox(
        "Activer le mode hors-ligne",
        value=False,
        help="Empêche tout téléchargement de modèles (nécessite des modèles pré-téléchargés)"
    )

    # Dossier de cache
    st.subheader("📁 Dossier de cache")
    cache_dir = st.text_input(
        "Chemin du dossier de cache",
        value=r"C:\IA Test\models" if os.name == 'nt' else "/tmp/models",
        help="Dossier où stocker/lire les modèles"
    )

    # Vérification du dossier
    if cache_dir and os.path.isdir(cache_dir):
        st.success(f"✅ Dossier valide")
    elif cache_dir:
        st.warning(f"⚠️ Dossier introuvable")

    st.markdown("---")

    # Bouton pour purger la VRAM
    if st.button("🧹 Purger la VRAM", help="Libère la mémoire GPU"):
        purge_vram(sync=True)
        st.success("VRAM purgée!")

# Zone principale
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📂 Fichier à traduire")

    # Upload de fichier
    uploaded_file = st.file_uploader(
        "Choisissez un fichier Excel (.xlsx, .xls)",
        type=['xlsx', 'xls'],
        help="Le fichier Excel contenant le texte à traduire"
    )

    if uploaded_file is not None:
        st.success(f"✅ Fichier sélectionné: {uploaded_file.name}")

        # Affichage des informations du fichier
        file_size = uploaded_file.size / 1024  # En KB
        st.info(f"📊 Taille: {file_size:.2f} KB")

with col2:
    st.header("📋 Récapitulatif")

    if uploaded_file is not None:
        st.markdown(f"""
        <div class="info-box">
        <strong>Configuration:</strong><br>
        🤖 Modèle: {model_choice}<br>
        🌍 Langue: {target_lang}<br>
        ⚡ Preset: {preset}<br>
        📦 Batch: {batch_size}<br>
        🔒 Offline: {'✅' if offline_mode else '❌'}<br>
        📁 Cache: {os.path.basename(cache_dir) if cache_dir else 'Défaut'}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
        ⚠️ Veuillez sélectionner un fichier Excel pour commencer
        </div>
        """, unsafe_allow_html=True)

# Zone de traduction
st.markdown("---")

if uploaded_file is not None:
    # Bouton de traduction
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

    with col_btn2:
        if st.button("🚀 Lancer la traduction", type="primary", use_container_width=True):
            st.session_state.translation_done = False
            st.session_state.output_file = None

            # Zone de progression
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Sauvegarde temporaire du fichier uploadé
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_input:
                    tmp_input.write(uploaded_file.getvalue())
                    input_path = tmp_input.name

                # Préparation du fichier de sortie
                output_filename = f"{Path(uploaded_file.name).stem}_translated_{LANG_CODES[target_lang].split('_')[0]}.xlsx"
                output_path = os.path.join(tempfile.gettempdir(), output_filename)

                # Configuration du traducteur
                config = TranslatorConfig(
                    model_name=MODELS[model_choice],
                    target_lang=LANG_CODES[target_lang],
                    batch_size=batch_size,
                    preset=preset,
                    offline_mode=offline_mode,
                    cache_dir=cache_dir if cache_dir and os.path.isdir(cache_dir) else None
                )

                # Callback de progression
                def update_progress(message: str, progress: float = 0):
                    status_text.text(message)
                    if progress > 0:
                        progress_bar.progress(min(int(progress), 100) / 100)

                # Création du traducteur
                translator = ExcelTranslator(config, progress_callback=update_progress)

                # Chargement du modèle
                status_text.info("🔧 Chargement du modèle...")
                translator.load_model()

                # Traduction
                status_text.info("🌐 Traduction en cours...")
                translator.translate_file(input_path, output_path)

                # Succès
                progress_bar.progress(100)
                st.session_state.translation_done = True
                st.session_state.output_file = output_path
                st.session_state.output_filename = output_filename

                # Nettoyage du fichier temporaire d'entrée
                try:
                    os.unlink(input_path)
                except:
                    pass

                status_text.empty()
                progress_bar.empty()

                st.balloons()
                st.success("✅ Traduction terminée avec succès!")

            except Exception as e:
                st.error(f"❌ Erreur lors de la traduction: {str(e)}")
                import traceback
                with st.expander("Détails de l'erreur"):
                    st.code(traceback.format_exc())

# Zone de téléchargement
if st.session_state.translation_done and st.session_state.output_file:
    st.markdown("---")
    st.header("📥 Téléchargement")

    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])

    with col_dl2:
        try:
            with open(st.session_state.output_file, 'rb') as f:
                file_data = f.read()

            st.download_button(
                label="⬇️ Télécharger le fichier traduit",
                data=file_data,
                file_name=st.session_state.output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
                use_container_width=True
            )

            st.success(f"📄 Fichier prêt: {st.session_state.output_filename}")

        except Exception as e:
            st.error(f"Erreur lors de la préparation du téléchargement: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🌐 Traducteur Excel Multilingue - Powered by Transformers & Streamlit</p>
    <p style="font-size: 0.8rem;">Utilise NLLB-200 et M2M100 pour des traductions de haute qualité</p>
</div>
""", unsafe_allow_html=True)

# Informations de debug (masquées par défaut)
with st.expander("🔍 Informations de debug"):
    st.write("**Configuration système:**")
    st.write(f"- PyTorch version: {torch.__version__}")
    st.write(f"- CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.write(f"- CUDA version: {torch.version.cuda}")
        st.write(f"- Nombre de GPUs: {torch.cuda.device_count()}")
        st.write(f"- GPU actuel: {torch.cuda.current_device()}")

    st.write("**État de la session:**")
    st.json({
        "translation_done": st.session_state.translation_done,
        "has_output_file": st.session_state.output_file is not None,
        "gpu_available": st.session_state.gpu_info['available']
    })
