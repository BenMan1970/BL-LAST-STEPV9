"""
BLUESTAR ENGINE v9.1 — Streamlit Interface
Dépose ce fichier à la racine du repo, au même niveau que ENGINE_V9.py
"""
import json
import tempfile
import os

import streamlit as st

# ── Import du moteur ──────────────────────────────────────────────────────
try:
    from ENGINE_V9 import run_pipeline
except ImportError as e:
    st.error(f"Impossible d'importer ENGINE_V9 : {e}")
    st.stop()

# ── Config page ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BLUESTAR v9.1",
    page_icon="🔵",
    layout="wide",
)

st.title("🔵 BLUESTAR ENGINE v9.1")
st.caption("FX Institutional Desk · Deterministic DAG Pipeline")

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

                    # Bouton de téléchargement
                    st.download_button(
                        label="⬇️ Télécharger le rapport HTML",
                        data=html,
                        file_name="bluestar_report.html",
                        mime="text/html",
                        use_container_width=True,
                    )

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
