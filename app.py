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

from config import (
    DEVICE, INPUT_SIZE, TEMP_DIR, OUTPUT_DIR,
    UNET_MULTICLASS_PATH, UNET_E1_PATH, UNET_E2_PATH,
    XGB_FULL_PATH, XGB_VESICLE_PATH
)
from segmentation import segment_video, find_best_frame
from radiomics import extract_features
from classifier import predict_label
from measurements import annotate_best_frame
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

    [data-testid="stSidebar"] { background-color: #f5f7fa; }
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
    

    st.markdown("#### 2. Clasificación (opcional)")
    st.session_state.use_classifier = st.toggle(
        "Activar diagnóstico asistido",
        value=st.session_state.use_classifier,
        help="Extrae características radiómicas y predice litiasis vesicular con XGBoost"
    )

    if st.session_state.use_classifier:
        clf_choice = st.radio(
            "Tipo de clasificación",
            options=["Basado en segmentación (vesícula + cálculos)",
                     "Basado en características (solo vesícula)"],
            index=0
        )
        st.session_state.clf_mode = "full" if "segmentación" in clf_choice else "vesicle"

    st.markdown("#### Visualización")
    opacity = st.slider("Opacidad de la máscara", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.markdown("#### Leyenda")
    st.markdown(
        "<div style='font-size:0.85rem;'>"
        "<span style='display:inline-block; width:14px; height:14px; background:#000; border:1px solid #999; vertical-align:middle;'></span> Fondo<br>"
        "<span style='display:inline-block; width:14px; height:14px; background:rgb(0,114,178); vertical-align:middle;'></span> Vesícula<br>"
        "<span style='display:inline-block; width:14px; height:14px; background:rgb(213,94,0); vertical-align:middle;'></span> Cálculos"
        "</div>",
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
            f"Clasificación: {clf_label}<br>"
            f"Opacidad de máscara: {int(opacity*100)}%</div>",
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
        cap.release()

        st.session_state.video_info = {
            "fps": fps, "frames": frame_count,
            "width": w, "height": h, "duration": duration
        }

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Duración", f"{duration:.1f} s")
        m2.metric("Frames", f"{frame_count}")
        m3.metric("FPS", f"{fps}")
        m4.metric("Resolución", f"{w}x{h}")

        st.markdown("---")
        run_clicked = st.button(
            "Procesar video",
            type="primary",
            use_container_width=True
        )

        if run_clicked:
            progress = st.progress(0.0)
            status = st.empty()

            status.info("Preparando modelo...")
            try:
                if st.session_state.seg_mode == "multiclass":
                    seg_ok = ensure_multiclass()
                else:
                    seg_ok = ensure_cascade()
            except Exception:
                seg_ok = False

            if not seg_ok:
                status.error("No se pudo preparar el modelo de segmentación. Inténtalo de nuevo.")
                st.stop()

            if st.session_state.use_classifier and not classifier_available(st.session_state.clf_mode):
                status.error("El modelo de clasificación no está disponible.")
                st.stop()

            status.info("Segmentando frames...")

            seg_video_path = OUTPUT_DIR / f"segmentado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

            frames_data, fps = segment_video(
                st.session_state.video_path,
                seg_video_path,
                model_type=st.session_state.seg_mode,
                opacity=opacity,
                progress_callback=lambda p: progress.progress(min(p, 1.0))
            )

            st.session_state.frames_data = frames_data
            st.session_state.seg_video_path = str(seg_video_path)

            status.info("Identificando frame de mayor visualización...")
            best = find_best_frame(frames_data)
            st.session_state.best_frame = best

            if best is None or best["vesicle_area_px"] == 0:
                st.warning("No se detectó vesícula en ningún frame del video.")
                progress.progress(1.0)
                status.empty()
                st.stop()

            status.info("Extrayendo características radiómicas...")
            feat = extract_features(best["frame_rgb"], best["mask"])
            st.session_state.features_result = feat

            status.info("Generando frame anotado con mediciones...")
            annotated = annotate_best_frame(
                best["frame_rgb"], best["mask"],
                feat["vesicle_lines"], feat["calculi_info"]
            )
            st.session_state.annotated_frame = annotated

            if st.session_state.use_classifier:
                status.info("Realizando diagnóstico asistido...")
                clf = predict_label(feat["features"], st.session_state.clf_mode)
                st.session_state.classification_result = clf
            else:
                st.session_state.classification_result = None

            st.session_state.processed = True
            progress.progress(1.0)
            status.success("Análisis completado. Revisa la pestaña 'Resultados'.")


with tab2:
    if not st.session_state.processed:
        st.info("Procesa un video en la pestaña 'Análisis' para ver los resultados.")
    else:
        st.markdown("### Video segmentado completo")
        if st.session_state.seg_video_path and Path(st.session_state.seg_video_path).exists():
            with open(st.session_state.seg_video_path, "rb") as f:
                video_bytes = f.read()

            render_video(video_bytes)

            d1, d2, d3 = st.columns([1, 2, 1])
            with d2:
                st.download_button(
                    "Descargar video segmentado",
                    data=video_bytes,
                    file_name=Path(st.session_state.seg_video_path).name,
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown("### Frame de mayor visualización")

        feat = st.session_state.features_result
        best = st.session_state.best_frame
        ann = st.session_state.annotated_frame

        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(ann, caption=f"Frame #{best['idx']} - mayor área de vesícula", use_container_width=True)
        with c2:
            st.markdown("#### Mediciones principales")
            f = feat["features"]
            st.metric("Área vesicular", f"{f.get('ves_area_mm2', 0):.1f} mm²")
            st.metric("Largo (eje mayor)", f"{f.get('ves_major_mm', 0):.1f} mm")
            st.metric("Ancho (eje menor)", f"{f.get('ves_minor_mm', 0):.1f} mm")
            st.metric("Cálculos detectados", int(f.get('num_calculi', 0)))
            if f.get('has_calculi'):
                st.metric("Diámetro máximo cálculo", f"{f.get('max_calc_diam_mm', 0):.1f} mm")

        if st.session_state.classification_result:
            st.markdown("---")
            st.markdown("### Diagnóstico asistido")
            clf = st.session_state.classification_result
            badge_class = "diag-positive" if clf["prediction"] == 1 else "diag-negative"
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span class='diag-badge {badge_class}'>{clf['label']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            if clf.get("prob_normal") is not None:
                cc1, cc2 = st.columns(2)
                cc1.metric("Probabilidad normal", f"{clf['prob_normal']*100:.1f} %")
                cc2.metric("Probabilidad litiasis", f"{clf['prob_litiasis']*100:.1f} %")

        st.markdown("---")
        st.markdown("### Detalle de características")

        f = feat["features"]
        morpho_data = {
            "Característica": ["Área", "Largo", "Ancho", "Razón de aspecto", "Elongación",
                              "Esfericidad", "Aplanamiento"],
            "Valor": [
                f"{f.get('ves_area_mm2', 0):.2f} mm²",
                f"{f.get('ves_major_mm', 0):.2f} mm",
                f"{f.get('ves_minor_mm', 0):.2f} mm",
                f"{f.get('ves_aspect_ratio', 0):.3f}" if not pd.isna(f.get('ves_aspect_ratio')) else "N/D",
                f"{f.get('ves_elongation', 0):.3f}",
                f"{f.get('ves_sphericity', 0):.3f}",
                f"{f.get('ves_flatness', 0):.3f}",
            ]
        }
        texture_data = {
            "Característica": ["Intensidad media", "Desviación estándar",
                              "Entropía first-order", "Contraste GLCM",
                              "Homogeneidad GLCM", "Entropía de zona"],
            "Valor": [
                f"{f.get('ves_mean', 0):.2f}",
                f"{f.get('ves_std', 0):.2f}",
                f"{f.get('ves_entropy', 0):.3f}",
                f"{f.get('ves_contrast', 0):.3f}",
                f"{f.get('ves_homogeneity', 0):.3f}",
                f"{f.get('ves_zone_entropy', 0):.3f}",
            ]
        }
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Morfometría")
            st.table(pd.DataFrame(morpho_data))
        with c2:
            st.markdown("#### Textura")
            st.table(pd.DataFrame(texture_data))

        if feat["calculi_info"]:
            st.markdown("#### Detalle de cálculos")
            calc_df = pd.DataFrame([
                {"ID": f"C{c['id']}", "Diámetro (mm)": round(c['diam_mm'], 2),
                 "Área (px)": c['area_px']}
                for c in feat["calculi_info"]
            ])
            st.dataframe(calc_df, use_container_width=True, hide_index=True)


with tab3:
    if not st.session_state.processed:
        st.info("Procesa un video en la pestaña 'Análisis' para generar el reporte.")
    else:
        st.markdown("### Reporte clínico en PDF")
        st.markdown(
            "<div class='info-card'>El reporte incluye el frame anotado con las mediciones, "
            "todas las características radiómicas extraídas, el detalle de cálculos detectados y "
            "el resultado del diagnóstico asistido.</div>",
            unsafe_allow_html=True
        )

        if st.button("Generar reporte PDF", type="primary", use_container_width=True):
            pdf_path = OUTPUT_DIR / f"reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

            seg_name = ("UNet Multiclase" if st.session_state.seg_mode == "multiclass"
                        else "UNet Cascada Binaria")

            generate_report(
                str(pdf_path),
                st.session_state.annotated_frame,
                st.session_state.features_result["features"],
                st.session_state.features_result["calculi_info"],
                st.session_state.classification_result,
                seg_name,
                st.session_state.video_info
            )

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.success(f"Reporte generado: {pdf_path.name}")
            st.download_button(
                "Descargar reporte PDF",
                data=pdf_bytes,
                file_name=pdf_path.name,
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )


st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.78rem;'>"
    "Software de evaluación automática de vesícula biliar."
    "</div>",
    unsafe_allow_html=True
)
