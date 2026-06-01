"""
BLUESTAR ENGINE v9.1 — Streamlit Interface
Dépose ce fichier à la racine du repo, au même niveau que ENGINE.V9.py
"""
import importlib.util
import sys
import tempfile
import os
import io
from pathlib import Path

import streamlit as st

# ── WeasyPrint (PDF natif côté serveur) ──────────────────────────────────
try:
    from weasyprint import HTML as _WeasyHTML
    _HAS_WEASYPRINT = True
except Exception:
    _HAS_WEASYPRINT = False

def _html_to_pdf_bytes(html: str) -> bytes:
    """Génère un PDF calibré en mémoire via WeasyPrint. Retourne les bytes."""
    buf = io.BytesIO()
    _WeasyHTML(string=html).write_pdf(buf)
    return buf.getvalue()

# ── Import du moteur (filename ENGINE.V9.py contient un point — import classique impossible) ──
def _load_engine():
    here = Path(__file__).parent
    candidates = ["ENGINE.V9.py", "ENGINE_V9.py", "bluestar_engine_v9.py"]
    for name in candidates:
        path = here / name
        if path.exists():
            spec = importlib.util.spec_from_file_location("bluestar_engine", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bluestar_engine"] = mod
            spec.loader.exec_module(mod)
            return mod
    return None

_engine = _load_engine()
if _engine is None:
    st.error("Moteur introuvable. Vérifie que ENGINE.V9.py est bien dans le repo.")
    st.stop()

run_pipeline = _engine.run_pipeline

# ── Config page ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BLUESTAR v9.1",
    page_icon="🔵",
    layout="wide",
)

st.title("🔵 BLUESTAR ENGINE v9.1")
st.caption("FX Institutional Desk · Deterministic DAG Pipeline")

if not _HAS_WEASYPRINT:
    st.warning(
        "⚠️ WeasyPrint indisponible — seul le téléchargement HTML sera proposé. "
        "Ajoute `weasyprint` dans requirements.txt et les libs système dans packages.txt."
    )

# ── Upload des fichiers ───────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    merged_file = st.file_uploader(
        "📊 Merged JSON (`bluestar_merged_*.json`)",
        type=["json"],
        key="merged",
    )

with col2:
    calendar_file = st.file_uploader(
        "📅 Calendar JSON (`calendar.json`)",
        type=["json"],
        key="calendar",
    )

# ── Bouton run ────────────────────────────────────────────────────────────
if merged_file and calendar_file:
    if st.button("▶ Générer le rapport", type="primary", use_container_width=True):
        with st.spinner("Pipeline en cours…"):
            # Écriture dans des fichiers temporaires (run_pipeline attend des paths)
            with tempfile.TemporaryDirectory() as tmpdir:
                merged_path = os.path.join(tmpdir, "merged.json")
                calendar_path = os.path.join(tmpdir, "calendar.json")
                output_path = os.path.join(tmpdir, "report.html")

                with open(merged_path, "wb") as f:
                    f.write(merged_file.getvalue())
                with open(calendar_path, "wb") as f:
                    f.write(calendar_file.getvalue())

                try:
                    html = run_pipeline(
                        merged_path=merged_path,
                        calendar_json_path=calendar_path,
                        output_path=output_path,
                    )
                    st.success("✅ Rapport généré")

                    # Affichage inline du HTML
                    st.components.v1.html(html, height=1800, scrolling=True)

                    # ── Téléchargements ───────────────────────────────
                    dl_col1, dl_col2 = st.columns(2)

                    with dl_col1:
                        st.download_button(
                            label="⬇️ Télécharger le rapport HTML",
                            data=html,
                            file_name="bluestar_report.html",
                            mime="text/html",
                            use_container_width=True,
                        )

                    with dl_col2:
                        if _HAS_WEASYPRINT:
                            with st.spinner("Génération PDF…"):
                                try:
                                    pdf_bytes = _html_to_pdf_bytes(html)
                                    st.download_button(
                                        label="⬇️ Télécharger PDF (calibré)",
                                        data=pdf_bytes,
                                        file_name="bluestar_report.pdf",
                                        mime="application/pdf",
                                        use_container_width=True,
                                    )
                                except Exception as pdf_err:
                                    st.error(f"Erreur PDF : {pdf_err}")
                        else:
                            st.info("PDF indisponible — WeasyPrint non installé.")

                except Exception as e:
                    st.error(f"Erreur pipeline : {e}")
                    st.exception(e)
else:
    st.info("⬆️ Upload les deux fichiers JSON pour lancer le pipeline.")
    st.markdown("""
    **Fichiers requis :**
    - `bluestar_merged_YYYYMMDD_HHMMutc.json` — output du merge engine
    - `calendar.json` — calendrier économique parsé
    """)
