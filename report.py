import io
from datetime import datetime, timezone, timedelta
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.enums import TA_CENTER
from PIL import Image
import numpy as np

SOFTWARE_NAME = "Desarrollo de un software para la evaluación automática de la vesícula biliar"
PERU_TZ = timezone(timedelta(hours=-5))

def build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("Title", parent=base["Title"], fontSize=15,
                                textColor=colors.HexColor("#1f3864"),
                                alignment=TA_CENTER, spaceAfter=2, leading=17),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontSize=8.5,
                                   textColor=colors.HexColor("#666666"),
                                   alignment=TA_CENTER, spaceAfter=8, leading=11),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=10.5,
                             textColor=colors.HexColor("#2d5a96"), spaceBefore=4, spaceAfter=3, leading=13),
        "body": ParagraphStyle("Body", parent=base["Normal"], fontSize=8.5,
                               textColor=colors.HexColor("#222222"), leading=11),
        "small": ParagraphStyle("Small", parent=base["Normal"], fontSize=6.5,
                                textColor=colors.HexColor("#777777"),
                                alignment=TA_CENTER, leading=8.5),
        "diag_positive": ParagraphStyle("DiagPos", parent=base["Normal"], fontSize=11,
                                        textColor=colors.HexColor("#b30000"),
                                        alignment=TA_CENTER, spaceAfter=3, leading=13),
        "diag_negative": ParagraphStyle("DiagNeg", parent=base["Normal"], fontSize=11,
                                        textColor=colors.HexColor("#006600"),
                                        alignment=TA_CENTER, spaceAfter=3, leading=13),
    }
    return styles

def array_to_flowable(arr, max_width=11 * cm):
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    w, h = img.size
    ratio = max_width / w
    return RLImage(buf, width=max_width, height=h * ratio)

def dense_table(rows, col_widths, align_center=False):
    t = Table(rows, colWidths=col_widths)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    if align_center:
        style_cmds.append(("ALIGN", (0, 0), (-1, -1), "CENTER"))
    t.setStyle(TableStyle(style_cmds))
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
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm
    )

    story = []

    story.append(Paragraph("Reporte de Análisis Ecográfico", styles["title"]))
    story.append(Paragraph(SOFTWARE_NAME, styles["subtitle"]))

    timestamp = datetime.now(PERU_TZ).strftime("%Y-%m-%d %H:%M:%S")
    meta_rows = [
        ["Parámetro", "Valor"],
        ["Fecha y hora", f"{timestamp} (Perú)"],
        ["Mod. Segmentación", segmentation_model_name],
    ]
    if classification:
        meta_rows.append(["Mod. Clasificación", classifier_label(classification.get("mode"))])
    
    table_meta = dense_table(meta_rows, col_widths=[4.0 * cm, 4.0 * cm])

    if classification is not None:
        diag_style = styles["diag_positive"] if classification["prediction"] == 1 else styles["diag_negative"]
        diag_p = Paragraph(f"<b>Diagnóstico:</b> {classification['label']}", diag_style)
        
        prob_rows = [["Etiqueta", "Probabilidad"]]
        if classification.get("prob_normal") is not None:
            prob_rows.append(["Vesícula normal", fmt(classification["prob_normal"] * 100, 1, " %")])
            prob_rows.append(["Litiasis vesicular", fmt(classification["prob_litiasis"] * 100, 1, " %")])
        table_prob = dense_table(prob_rows, col_widths=[4.5 * cm, 4.0 * cm])
        
        header_table = Table([[table_meta, [diag_p, table_prob]]], colWidths=[8.5 * cm, 8.9 * cm])
        header_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(header_table)
    else:
        story.append(table_meta)

    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Características Morfométricas y de Textura", styles["h2"]))
    
    combined_metrics = [
        ["Morfometría", "Valor", "Textura", "Valor"],
        ["Área", fmt(features.get("ves_area_mm2"), 2, " mm²"), "Intensidad media", fmt(features.get("ves_mean"), 2)],
        ["Largo (eje mayor)", fmt(features.get("ves_major_mm"), 2, " mm"), "Desv. estándar", fmt(features.get("ves_std"), 2)],
        ["Ancho (eje menor)", fmt(features.get("ves_minor_mm"), 2, " mm"), "Entropía (1st-ord)", fmt(features.get("ves_entropy"), 3)],
        ["Razón de aspecto", fmt(features.get("ves_aspect_ratio"), 3), "Contraste (GLCM)", fmt(features.get("ves_contrast"), 3)],
        ["Elongación", fmt(features.get("ves_elongation"), 3), "Homogeneidad", fmt(features.get("ves_homogeneity"), 3)],
        ["Esfericidad", fmt(features.get("ves_sphericity"), 3), "Entropía zona", fmt(features.get("ves_zone_entropy"), 3)],
        ["Aplanamiento", fmt(features.get("ves_flatness"), 3), "-", "-"],
    ]
    table_metrics = dense_table(combined_metrics, col_widths=[4.5 * cm, 4.2 * cm, 4.5 * cm, 4.2 * cm])
    story.append(table_metrics)

    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Frame de Mayor Visualización", styles["h2"]))
    img_flowable = array_to_flowable(frame_annotated, max_width=11 * cm)
    
    img_wrapper = Table([[img_flowable]], colWidths=[17.4 * cm])
    img_wrapper.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(img_wrapper)

    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Análisis de Cálculos", styles["h2"]))
    if features.get("has_calculi") == 1 and calculi_info:
        summary_rows = [
            ["Resumen", "Valor"],
            ["Detectados", str(int(features.get("num_calculi", 0)))],
            ["Diámetro máx.", fmt(features.get("max_calc_diam_mm"), 2, " mm")],
            ["Entropía (mayor)", fmt(features.get("calc_entropy"), 3)],
            ["Contraste (mayor)", fmt(features.get("calc_contrast"), 3)],
        ]
        table_calc_summary = dense_table(summary_rows, col_widths=[4.2 * cm, 4.0 * cm])

        detail_rows = [["ID", "Diámetro (mm)", "Área (px)"]]
        for c in calculi_info[:3]:
            detail_rows.append([f"C{c['id']}", fmt(c["diam_mm"], 2), str(c["area_px"])])
        table_calc_detail = dense_table(detail_rows, col_widths=[2.5 * cm, 3.5 * cm, 3.2 * cm], align_center=True)

        calc_side_by_side = Table([[table_calc_summary, table_calc_detail]], colWidths=[8.5 * cm, 8.9 * cm])
        calc_side_by_side.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(calc_side_by_side)
    else:
        story.append(Paragraph("No se detectaron cálculos en el frame analizado.", styles["body"]))

    story.append(Spacer(1, 0.3 * cm))
    
    story.append(Paragraph(
        "Este reporte ha sido generado utilizando modelos de deep learning de la tesis: \"Desarrollo de un software para la evaluación automática de la vesícula biliar utilizando imágenes de ultrasonido obtenidas mediante el protocolo de barrido volumétrico\". Es una herramienta de apoyo diagnóstico basado en IA y no reemplaza el criterio clínico profesional.",
        styles["small"]
    ))

    doc.build(story)
