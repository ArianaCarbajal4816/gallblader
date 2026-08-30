import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import base64
from PIL import Image
from datetime import datetime
from pathlib import Path
import io
from measurements import clean_class_1_mask

from config import (
    DEVICE, INPUT_SIZE, TEMP_DIR, OUTPUT_DIR,
    UNET_MULTICLASS_PATH, UNET_E1_PATH, UNET_E2_PATH,
    XGB_FULL_PATH, XGB_VESICLE_PATH
)
from segmentation import segment_video, find_best_frame
from radiomics import extract_features
from classifier import predict_label
from measurements import annotate_best_frame, clean_class_1_mask
from report import generate_report
from model_downloader import ensure_multiclass, ensure_cascade


st.set_page_config(
    page_title="Software de evaluación automática de vesícula biliar",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main { padding-top: 1rem; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    h1 { color: #1f3864; font-weight: 600; }
    h2 { color: #2d5a96; font-weight: 600; }
    h3 { color: #3d6ab5; font-weight: 500; }

    /* Modificaciones de la Barra Lateral */
    [data-testid="stSidebar"] { background-color: #f5f7fa; }
    
    /* Hace que todo el contenido del sidebar empiece más arriba */
    [data-testid="stSidebarUserContent"] {
        padding-top: 1.5rem !important;
    }
    
    /* Reduce el espacio vertical alrededor de las líneas divisorias del sidebar */
    [data-testid="stSidebar"] hr {
        margin: 0.8rem 0 !important;
    }

    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; color: #1f3864; }
    [data-testid="stMetricLabel"] { color: #666; font-size: 0.85rem; }

    .info-card {
        background-color: #ecf0f6;
        border-left: 4px solid #2d5a96;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
        font-size: 0.92rem;
    }
    .success-card {
        background-color: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .alert-card {
        background-color: #fff3e0;
        border-left: 4px solid #e65100;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .diag-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 8px 0;
    }
    .diag-positive { background-color: #ffebee; color: #b71c1c; border: 1px solid #ef9a9a; }
    .diag-negative { background-color: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button[kind="primary"] {
        background-color: #2d5a96;
        border-color: #2d5a96;
        color: white;
    }
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        background-color: #2d5a96;
        border-color: #2d5a96;
        color: white;
    }
    .stDownloadButton > button:hover {
        background-color: #244c7d;
        border-color: #244c7d;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background-color: #fafbfd;
        border: 2px dashed #2d5a96;
        border-radius: 8px;
        padding: 12px;
    }

    hr { margin: 1.5rem 0; border-color: #e0e6ed; }
</style>
""", unsafe_allow_html=True)


for key, default in [
    ("processed", False),
    ("frames_data", None),
    ("best_frame", None),
    ("features_result", None),
    ("classification_result", None),
    ("annotated_frame", None),
    ("video_path", None),
    ("seg_video_path", None),
    ("video_info", {}),
    ("seg_mode", "multiclass"),
    ("use_classifier", False),
    ("clf_mode", "full"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def classifier_available(mode):
    if mode == "full":
        return XGB_FULL_PATH.exists()
    return XGB_VESICLE_PATH.exists()


def render_video(video_bytes):
    b64 = base64.b64encode(video_bytes).decode()
    st.markdown(
        f"""
        <video controls autoplay loop muted style="width:100%; border-radius:8px;">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
            Tu navegador no soporta video HTML5.
        </video>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    "<h1 style='text-align:center; margin-bottom:0;'>Software de evaluación automática de vesícula biliar</h1>",
    unsafe_allow_html=True
)
st.markdown("---")


with st.sidebar:
    st.markdown("### Configuración del análisis")

    st.markdown("#### 1. Modelo de segmentación")
    seg_choice = st.radio(
        "Selecciona la arquitectura",
        options=["Multiclase (3 clases)", "Cascada binaria (2 etapas)"],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.seg_mode = "multiclass" if "Multiclase" in seg_choice else "cascade"

    with st.container(border=True):
        if st.session_state.seg_mode == "multiclass":
            st.write("**Segmentación Multiclase**")
            st.caption("Modelo UNet que realiza segmentación simultánea de tres clases: fondo, vesícula biliar y cálculos.")
        else:
            st.write("**Segmentación binaria**")
            st.caption("Dos modelos UNet binarios secuenciales. Etapa 1: localización de vesícula biliar. Etapa 2: detección de cálculos dentro de la región identificada.")
    

    st.markdown("#### 2. Clasificación")
    st.session_state.use_classifier = st.toggle(
        "Activar diagnóstico asistido",
        value=st.session_state.use_classifier,
        help="Extrae características radiómicas y predice litiasis vesicular con XGBoost"
    )

    if st.session_state.use_classifier:
        clf_choice = st.radio(
            "Tipo de clasificación",
            options=["Basado en segmentación ",
                     "Basado en radiómica"],
            index=0
        )
        st.session_state.clf_mode = "full" if "segmentación" in clf_choice else "vesicle"

  

    st.markdown("---")
    st.markdown("#### Leyenda")
    st.markdown(
        """
        <div style='font-size: 0.85rem; line-height: 1.6;'>
            <p style='margin: 4px 0;'>
                <span style='display:inline-block; width:14px; height:14px; background:#000; border:1px solid #999; vertical-align:middle; margin-right:8px;'></span>Fondo
            </p>
            <p style='margin: 4px 0;'>
                <span style='display:inline-block; width:14px; height:14px; background:rgb(0,114,178); vertical-align:middle; margin-right:8px;'></span>Vesícula
            </p>
            <p style='margin: 4px 0;'>
                <span style='display:inline-block; width:14px; height:14px; background:rgb(213,94,0); vertical-align:middle; margin-right:8px;'></span>Cálculos
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )



tab1, tab2, tab3 = st.tabs(["Análisis", "Resultados", "Reporte"])


with tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(
            "<div class='info-card'><b>Flujo del análisis</b><br>"
            "1. Carga un video ecográfico (.mp4, .avi, .mov)<br>"
            "2. Configura el modelo en la barra lateral<br>"
            "3. Procesa y revisa los resultados<br>"
            "4. Exporta el reporte clínico en PDF</div>",
            unsafe_allow_html=True
        )
    with col_b:
        clf_label = "Sí" if st.session_state.use_classifier else "No"
        st.markdown(
            f"<div class='info-card'><b>Configuración activa</b><br>"
            f"Segmentación: {seg_choice}<br>"
            f"Clasificación: {clf_label}<br>",
            unsafe_allow_html=True
        )

    st.markdown("### Carga del video")
    video_file = st.file_uploader(
        "Selecciona un video ecográfico",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if video_file is not None:
        temp_video = TEMP_DIR / "input.mp4"
        with open(temp_video, "wb") as f:
            f.write(video_file.read())
        st.session_state.video_path = str(temp_video)

        cap = cv2.VideoCapture(str(temp_video))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

