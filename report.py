import io
from datetime import datetime, timezone, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from PIL import Image
import numpy as np


SOFTWARE_NAME = "Desarrollo de un software para la evaluación automática de la vesícula biliar"
PERU_TZ = timezone(timedelta(hours=-5))


def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("Title", parent=base["Title"], fontSize=20,
                                textColor=colors.HexColor("#1f3864"),
                                alignment=TA_CENTER, spaceAfter=4, leading=24),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontSize=11,
                                   textColor=colors.HexColor("#666666"),
                                   alignment=TA_CENTER, spaceAfter=18, leading=14),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=13,
                             textColor=colors.HexColor("#2d5a96"), spaceBefore=12, spaceAfter=8),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontSize=10,
                               textColor=colors.HexColor("#222222"), leading=14),
        "small": ParagraphStyle("Small", parent=base["Normal"], fontSize=8,
                                textColor=colors.HexColor("#777777"),
                                alignment=TA_CENTER),
        "diag_positive": ParagraphStyle("DiagPos", parent=base["Normal"], fontSize=14,
                                        textColor=colors.HexColor("#b30000"),
                                        alignment=TA_CENTER, spaceAfter=6),
        "diag_negative": ParagraphStyle("DiagNeg", parent=base["Normal"], fontSize=14,
                                        textColor=colors.HexColor("#006600"),
                                        alignment=TA_CENTER, spaceAfter=6),
    }
    return styles


def array_to_flowable(arr, max_width=15 * cm):
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    w, h = img.size
    ratio = max_width / w
    return RLImage(buf, width=max_width, height=h * ratio)


def metric_table(rows):
    t = Table(rows, colWidths=[7 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def fmt(v, decimals=2, unit=""):
    if v is None:
        return "N/D"
    try:
        if isinstance(v, float) and (v != v):
            return "N/D"
        return f"{v:.{decimals}f}{unit}"
    except Exception:
        return str(v)


def classifier_label(mode):
    if mode == "full":
        return "XGBoost basado en segmentación"
    return "XGBoost basado en radiómica"


def generate_report(output_path, frame_annotated, features, calculi_info,
                    classification, segmentation_model_name, video_info):
    styles = build_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm
    )

    story = []

    story.append(Paragraph("Reporte de análisis ecográfico", styles["title"]))
    story.append(Paragraph(SOFTWARE_NAME, styles["subtitle"]))

    timestamp = datetime.now(PERU_TZ).strftime("%Y-%m-%d %H:%M:%S")
    meta_rows = [
        ["Parámetro", "Valor"],
        ["Fecha y hora de análisis", f"{timestamp} (Perú)"],
        ["Modelo de segmentación", segmentation_model_name],
    ]
    if classification:
        meta_rows.append(["Modelo de clasificación", classifier_label(classification.get("mode"))])
    story.append(metric_table(meta_rows))
    story.append(Spacer(1, 0.4 * cm))

    if classification is not None:
        story.append(Paragraph("Diagnóstico asistido", styles["h2"]))
        diag_style = styles["diag_positive"] if classification["prediction"] == 1 else styles["diag_negative"]
        story.append(Paragraph(classification["label"], diag_style))

        prob_rows = [["Etiqueta", "Probabilidad"]]
        if classification.get("prob_normal") is not None:
            prob_rows.append(["Vesícula normal", fmt(classification["prob_normal"] * 100, 1, " %")])
            prob_rows.append(["Litiasis vesicular", fmt(classification["prob_litiasis"] * 100, 1, " %")])
            story.append(metric_table(prob_rows))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Frame de mayor visualización", styles["h2"]))
    story.append(Paragraph(
        "Frame seleccionado automáticamente como el de mayor área vesicular. "
        "Las líneas indican el largo (naranja) y ancho (azul) máximos. "
        "Los cálculos detectados aparecen marcados en amarillo.",
        styles["body"]
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(array_to_flowable(frame_annotated, max_width=14 * cm))
    story.append(Spacer(1, 0.4 * cm))

    story.append(PageBreak())

    story.append(Paragraph("Características morfométricas - vesícula", styles["h2"]))
    morpho_rows = [
        ["Característica", "Valor"],
        ["Área", fmt(features.get("ves_area_mm2"), 2, " mm²")],
        ["Largo (eje mayor)", fmt(features.get("ves_major_mm"), 2, " mm")],
        ["Ancho (eje menor)", fmt(features.get("ves_minor_mm"), 2, " mm")],
        ["Razón de aspecto", fmt(features.get("ves_aspect_ratio"), 3)],
        ["Elongación", fmt(features.get("ves_elongation"), 3)],
        ["Esfericidad", fmt(features.get("ves_sphericity"), 3)],
        ["Aplanamiento", fmt(features.get("ves_flatness"), 3)],
    ]
    story.append(metric_table(morpho_rows))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Características de textura - vesícula", styles["h2"]))
    texture_rows = [
        ["Característica", "Valor"],
        ["Intensidad media", fmt(features.get("ves_mean"), 2)],
        ["Desviación estándar", fmt(features.get("ves_std"), 2)],
        ["Entropía (first-order)", fmt(features.get("ves_entropy"), 3)],
        ["Contraste (GLCM)", fmt(features.get("ves_contrast"), 3)],
        ["Homogeneidad (GLCM)", fmt(features.get("ves_homogeneity"), 3)],
        ["Entropía de zona", fmt(features.get("ves_zone_entropy"), 3)],
    ]
    story.append(metric_table(texture_rows))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Análisis de cálculos", styles["h2"]))
    if features.get("has_calculi") == 1 and calculi_info:
        summary = [
            ["Resumen", "Valor"],
            ["Cálculos detectados", str(int(features.get("num_calculi", 0)))],
            ["Diámetro máximo", fmt(features.get("max_calc_diam_mm"), 2, " mm")],
            ["Entropía (cálculo mayor)", fmt(features.get("calc_entropy"), 3)],
            ["Contraste (cálculo mayor)", fmt(features.get("calc_contrast"), 3)],
        ]
        story.append(metric_table(summary))
        story.append(Spacer(1, 0.3 * cm))

        detail_rows = [["ID", "Diámetro (mm)", "Área (px)"]]
        for c in calculi_info:
            detail_rows.append([f"C{c['id']}", fmt(c["diam_mm"], 2), str(c["area_px"])])
        story.append(Paragraph("Detalle por cálculo:", styles["body"]))
        detail_table = Table(detail_rows, colWidths=[3 * cm, 5 * cm, 5 * cm])
        detail_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(detail_table)
    else:
        story.append(Paragraph("No se detectaron cálculos en el frame analizado.", styles["body"]))

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "Aviso: este reporte es generado por un sistema de inteligencia artificial "
        "como apoyo diagnóstico. No reemplaza el criterio clínico de un profesional médico.",
        styles["small"]
    ))

    doc.build(story)
