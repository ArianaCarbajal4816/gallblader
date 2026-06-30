import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
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


st.set_page_config(
    page_title="GallBladder AI",
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


def model_files_status():
    return {
        "Multiclase (unet_multiclase.pth)": UNET_MULTICLASS_PATH.exists(),
        "Cascada Etapa 1 (unet_e1.pth)": UNET_E1_PATH.exists(),
        "Cascada Etapa 2 (unet_e2.pth)": UNET_E2_PATH.exists(),
        "XGBoost completo (xgboost_radiomics.pkl)": XGB_FULL_PATH.exists(),
        "XGBoost solo vesicula (xgboost_radiomics_std.pkl)": XGB_VESICLE_PATH.exists(),
    }


st.markdown("<h1 style='text-align:center; margin-bottom:0;'>GallBladder AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#666; font-size:1rem; margin-top:0;'>"
    "Plataforma de Analisis Ecografico Asistido por IA"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")


with st.sidebar:
    st.markdown("### Configuracion del Analisis")

    device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"<div class='info-card'><b>Dispositivo:</b> {device_label}</div>",
                unsafe_allow_html=True)

    st.markdown("#### 1. Modelo de Segmentacion")
    seg_choice = st.radio(
        "Selecciona la arquitectura",
        options=["Multiclase (3 clases)", "Cascada binaria (2 etapas)"],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.seg_mode = "multiclass" if "Multiclase" in seg_choice else "cascade"

    if st.session_state.seg_mode == "multiclass":
        st.markdown(
            "<div class='info-card'>UNet multiclase que segmenta fondo, vesicula y calculos en un solo paso. "
            "Dice validacion: 0.928 ± 0.033.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='info-card'>Dos UNet binarias secuenciales: la primera detecta vesicula, "
            "la segunda detecta calculos dentro de ella. Dice E1: 0.914 ± 0.009.</div>",
            unsafe_allow_html=True
        )

    st.markdown("#### 2. Clasificacion (opcional)")
    st.session_state.use_classifier = st.toggle(
        "Activar diagnostico asistido",
        value=st.session_state.use_classifier,
        help="Extrae caracteristicas radiomicas y predice litiasis vesicular con XGBoost"
    )

    if st.session_state.use_classifier:
        clf_choice = st.radio(
            "Tipo de clasificacion",
            options=["Basado en segmentacion (vesicula + calculos)",
                     "Basado en caracteristicas (solo vesicula)"],
            index=0
        )
        st.session_state.clf_mode = "full" if "segmentacion" in clf_choice else "vesicle"

    st.markdown("#### Visualizacion")
    opacity = st.slider("Opacidad de la mascara", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.markdown("#### Leyenda")
    st.markdown(
        "<div style='font-size:0.85rem;'>"
        "<span style='display:inline-block; width:14px; height:14px; background:#000; border:1px solid #999; vertical-align:middle;'></span> Fondo<br>"
        "<span style='display:inline-block; width:14px; height:14px; background:rgb(0,114,178); vertical-align:middle;'></span> Vesicula<br>"
        "<span style='display:inline-block; width:14px; height:14px; background:rgb(213,94,0); vertical-align:middle;'></span> Calculos"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### Estado de los modelos")
    statuses = model_files_status()
    for name, ok in statuses.items():
        icon = "OK" if ok else "FALTA"
        color = "#1b5e20" if ok else "#b71c1c"
        st.markdown(
            f"<div style='font-size:0.78rem; color:{color};'>[{icon}] {name}</div>",
            unsafe_allow_html=True
        )


tab1, tab2, tab3 = st.tabs(["Analisis", "Resultados", "Reporte"])


with tab1:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(
            "<div class='info-card'><b>Flujo del analisis</b><br>"
            "1. Carga un video ecografico (.mp4, .avi, .mov)<br>"
            "2. Configura el modelo en la barra lateral<br>"
            "3. Procesa y revisa los resultados<br>"
            "4. Exporta el reporte clinico en PDF</div>",
            unsafe_allow_html=True
        )
    with col_b:
        clf_label = "Si" if st.session_state.use_classifier else "No"
        st.markdown(
            f"<div class='info-card'><b>Configuracion activa</b><br>"
            f"Segmentacion: {seg_choice}<br>"
            f"Clasificacion: {clf_label}<br>"
            f"Opacidad mascara: {int(opacity*100)}%<br>"
            f"Dispositivo: {device_label}</div>",
            unsafe_allow_html=True
        )

    st.markdown("### Carga del video")
    video_file = st.file_uploader(
        "Selecciona un video ecografico",
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
        m1.metric("Duracion", f"{duration:.1f} s")
        m2.metric("Frames", f"{frame_count}")
        m3.metric("FPS", f"{fps}")
        m4.metric("Resolucion", f"{w}x{h}")

        st.markdown("---")
        run_clicked = st.button(
            "Procesar video",
            type="primary",
            use_container_width=True
        )

        if run_clicked:
            statuses = model_files_status()
            seg_ok = (statuses["Multiclase (unet_multiclase.pth)"]
                      if st.session_state.seg_mode == "multiclass"
                      else (statuses["Cascada Etapa 1 (unet_e1.pth)"] and statuses["Cascada Etapa 2 (unet_e2.pth)"]))
            if not seg_ok:
                st.error("Faltan los archivos de modelo para la segmentacion seleccionada. "
                         "Colocarlos en la carpeta 'models/'.")
                st.stop()
            if st.session_state.use_classifier:
                clf_ok = (statuses["XGBoost completo (xgboost_radiomics.pkl)"]
                          if st.session_state.clf_mode == "full"
                          else statuses["XGBoost solo vesicula (xgboost_radiomics_std.pkl)"])
                if not clf_ok:
                    st.error("Falta el archivo del modelo XGBoost seleccionado.")
                    st.stop()

            progress = st.progress(0.0)
            status = st.empty()
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

            status.info("Identificando frame de mayor visualizacion...")
            best = find_best_frame(frames_data)
            st.session_state.best_frame = best

            if best is None or best["vesicle_area_px"] == 0:
                st.warning("No se detecto vesicula en ningun frame del video.")
                progress.progress(1.0)
                status.empty()
                st.stop()

            status.info("Extrayendo caracteristicas radiomicas...")
            feat = extract_features(best["frame_rgb"], best["mask"])
            st.session_state.features_result = feat

            status.info("Generando frame anotado con mediciones...")
            annotated = annotate_best_frame(
                best["frame_rgb"], best["mask"],
                feat["vesicle_lines"], feat["calculi_info"]
            )
            st.session_state.annotated_frame = annotated

            if st.session_state.use_classifier:
                status.info("Realizando diagnostico asistido...")
                clf = predict_label(feat["features"], st.session_state.clf_mode)
                st.session_state.classification_result = clf
            else:
                st.session_state.classification_result = None

            st.session_state.processed = True
            progress.progress(1.0)
            status.success("Analisis completado. Revisa la pestaña 'Resultados'.")


with tab2:
    if not st.session_state.processed:
        st.info("Procesa un video en la pestaña 'Analisis' para ver los resultados.")
    else:
        st.markdown("### Video segmentado completo")
        if st.session_state.seg_video_path and Path(st.session_state.seg_video_path).exists():
            with open(st.session_state.seg_video_path, "rb") as f:
                video_bytes = f.read()
            c1, c2 = st.columns([3, 1])
            with c1:
                st.video(video_bytes)
            with c2:
                st.download_button(
                    "Descargar video segmentado",
                    data=video_bytes,
                    file_name=Path(st.session_state.seg_video_path).name,
                    mime="video/mp4",
                    use_container_width=True
                )

        st.markdown("---")
        st.markdown("### Frame de mayor visualizacion")

        feat = st.session_state.features_result
        best = st.session_state.best_frame
        ann = st.session_state.annotated_frame

        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(ann, caption=f"Frame #{best['idx']} - mayor area de vesicula", use_container_width=True)
        with c2:
            st.markdown("#### Mediciones principales")
            f = feat["features"]
            st.metric("Area vesicular", f"{f.get('ves_area_mm2', 0):.1f} mm²")
            st.metric("Largo (eje mayor)", f"{f.get('ves_major_mm', 0):.1f} mm")
            st.metric("Ancho (eje menor)", f"{f.get('ves_minor_mm', 0):.1f} mm")
            st.metric("Calculos detectados", int(f.get('num_calculi', 0)))
            if f.get('has_calculi'):
                st.metric("Diametro maximo calculo", f"{f.get('max_calc_diam_mm', 0):.1f} mm")

        if st.session_state.classification_result:
            st.markdown("---")
            st.markdown("### Diagnostico asistido")
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
                cc1.metric("Probabilidad Normal", f"{clf['prob_normal']*100:.1f} %")
                cc2.metric("Probabilidad Litiasis", f"{clf['prob_litiasis']*100:.1f} %")

        st.markdown("---")
        st.markdown("### Detalle de caracteristicas")

        f = feat["features"]
        morpho_data = {
            "Caracteristica": ["Area", "Largo", "Ancho", "Aspect ratio", "Elongacion",
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
            "Caracteristica": ["Intensidad media", "Desviacion estandar",
                              "Entropia first-order", "Contraste GLCM",
                              "Homogeneidad GLCM", "Entropia de zona"],
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
            st.markdown("#### Morfometria")
            st.table(pd.DataFrame(morpho_data))
        with c2:
            st.markdown("#### Textura")
            st.table(pd.DataFrame(texture_data))

        if feat["calculi_info"]:
            st.markdown("#### Detalle de calculos")
            calc_df = pd.DataFrame([
                {"ID": f"C{c['id']}", "Diametro (mm)": round(c['diam_mm'], 2),
                 "Area (px)": c['area_px']}
                for c in feat["calculi_info"]
            ])
            st.dataframe(calc_df, use_container_width=True, hide_index=True)


with tab3:
    if not st.session_state.processed:
        st.info("Procesa un video en la pestaña 'Analisis' para generar el reporte.")
    else:
        st.markdown("### Reporte clinico en PDF")
        st.markdown(
            "<div class='info-card'>El reporte incluye el frame anotado con las mediciones, "
            "todas las caracteristicas radiomicas extraidas, el detalle de calculos detectados y "
            "el resultado del diagnostico asistido (si fue activado).</div>",
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
                use_container_width=True
            )


st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.78rem;'>"
    "GallBladder AI - Sistema de apoyo diagnostico. No reemplaza el criterio clinico profesional."
    "</div>",
    unsafe_allow_html=True
)
