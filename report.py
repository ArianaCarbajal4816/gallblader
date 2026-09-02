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
    

    style_title = ParagraphStyle("CustomTitle", parent=base["Title"], fontSize=14,
                                 textColor=colors.HexColor("#1f3864"),
                                 alignment=TA_CENTER, spaceAfter=2, leading=16)
    
    style_subtitle = ParagraphStyle("CustomSubtitle", parent=base["Normal"], fontSize=8,
                                    textColor=colors.HexColor("#666666"),
                                    alignment=TA_CENTER, spaceAfter=6, leading=10)
    
    style_h2 = ParagraphStyle("CustomH2", parent=base["Heading2"], fontSize=10,
                              textColor=colors.HexColor("#2d5a96"), spaceBefore=3, spaceAfter=2, leading=12)
    
    style_body = ParagraphStyle("CustomBody", parent=base["Normal"], fontSize=8,
                                textColor=colors.HexColor("#222222"), leading=10)
    
    style_table_text = ParagraphStyle("CustomTableText", parent=base["Normal"], fontSize=7.5,
                                      textColor=colors.HexColor("#222222"), leading=9)
    
    style_small = ParagraphStyle("CustomSmall", parent=base["Normal"], fontSize=6.5,
                                 textColor=colors.HexColor("#777777"),
                                 alignment=TA_CENTER, leading=8)
    
    style_diag_header = ParagraphStyle("CustomDiagHeader", parent=base["Normal"], fontSize=10,
                                       textColor=colors.HexColor("#1f3864"),
                                       alignment=TA_CENTER, spaceAfter=4, leading=12)

    
    base.add(style_title)
    base.add(style_subtitle)
    base.add(style_h2)
    base.add(style_body)
    base.add(style_table_text)
    base.add(style_small)
    base.add(style_diag_header)

    styles = {
        "title": base["CustomTitle"],
        "subtitle": base["CustomSubtitle"],
        "h2": base["CustomH2"],
        "body": base["CustomBody"],
        "table_text": base["CustomTableText"],
        "small": base["CustomSmall"],
        "diag_header": base["CustomDiagHeader"],
    }
    return styles

def array_to_flowable(arr, max_width=9.5 * cm, max_height=5.2 * cm):
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    w, h = img.size
    ratio = min(max_width / w, max_height / h)
    return RLImage(buf, width=w * ratio, height=h * ratio)

def dense_table(rows, col_widths, align_center=False):
    t = Table(rows, colWidths=col_widths)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
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
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm
    )

    story = []

    story.append(Paragraph("Reporte de Análisis Ecográfico", styles["title"]))
    story.append(Paragraph(SOFTWARE_NAME, styles["subtitle"]))

    timestamp = datetime.now(PERU_TZ).strftime("%Y-%m-%d %H:%M:%S")
    seg_model_p = Paragraph(segmentation_model_name, styles["table_text"])
    
    meta_rows = [
        ["Parámetro", "Valor"],
        ["Fecha y hora", f"{timestamp} (Perú)"],
        ["Mod. Segmentación", seg_model_p],
    ]
    if classification:
        class_model_p = Paragraph(classifier_label(classification.get("mode")), styles["table_text"])
        meta_rows.append(["Mod. Clasificación", class_model_p])
    
    table_meta = dense_table(meta_rows, col_widths=[6.0 * cm, 12.0 * cm])
    story.append(table_meta)
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Resultados de Diagnóstico Asistido", styles["h2"]))
    
    if classification:
        raw_label = classification.get('label', 'N/D')
        clean_label = raw_label.replace("Vesicula", "Vesícula").replace("vesicula", "vesícula")
        
        diag_p = Paragraph(f"<b>Diagnóstico:</b> {clean_label}", styles["diag_header"])
        story.append(diag_p)
        
        prob_rows = [["Etiqueta", "Probabilidad"]]
        if classification.get("prob_normal") is not None:
            prob_rows.append(["Vesícula sana", fmt(classification["prob_normal"] * 100, 1, " %")])
            prob_rows.append(["Litiasis vesicular", fmt(classification["prob_litiasis"] * 100, 1, " %")])
            
        table_prob = dense_table(prob_rows, col_widths=[9.0 * cm, 9.0 * cm], align_center=True)
        story.append(table_prob)
    else:
        story.append(Paragraph("Sin clasificación disponible para este estudio.", styles["body"]))

    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Frame de Mayor Visualización", styles["h2"]))
    img_flowable = array_to_flowable(frame_annotated, max_width=9.5 * cm, max_height=5.2 * cm)
    
    img_wrapper = Table([[img_flowable]], colWidths=[18.0 * cm])
    img_wrapper.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(img_wrapper)

    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Características Morfométricas y de Textura", styles["h2"]))
    
    combined_metrics = [
        ["Morfometría", "Valor", "Textura", "Valor"],
        ["Área", fmt(features.get("ves_area_mm2"), 2, " mm²"), "Intensidad media", fmt(features.get("ves_mean"), 2)],
        ["Largo (eje mayor)", fmt(features.get("ves_major_mm"), 2, " mm"), "Desv. estándar", fmt(features.get("ves_std"), 2)],
        ["Ancho (eje menor)", fmt(features.get("ves_minor_mm"), 2, " mm"), "Entropía ", fmt(features.get("ves_entropy"), 3)],
        ["Razón de aspecto", fmt(features.get("ves_aspect_ratio"), 3), "Contraste (GLCM)", fmt(features.get("ves_contrast"), 3)],
        ["Elongación", fmt(features.get("ves_elongation"), 3), "Homogeneidad", fmt(features.get("ves_homogeneity"), 3)],
        ["Esfericidad", fmt(features.get("ves_sphericity"), 3), "Entropía zona", fmt(features.get("ves_zone_entropy"), 3)],
        ["Aplanamiento", fmt(features.get("ves_flatness"), 3), "-", "-"],
    ]
    table_metrics = dense_table(combined_metrics, col_widths=[4.5 * cm, 4.5 * cm, 4.5 * cm, 4.5 * cm])
    story.append(table_metrics)

    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("Análisis de Cálculos", styles["h2"]))
    if features.get("has_calculi") == 1 and calculi_info:
        summary_rows = [
            ["Resumen", "Valor"],
            ["Número de cálculos", str(int(features.get("num_calculi", 0)))],
            ["Diámetro de cálculo mayor", fmt(features.get("max_calc_diam_mm"), 2, " mm")],
            ["Entropía de cálculo mayor", fmt(features.get("calc_entropy"), 3)],
            ["Contraste de cálculo mayor", fmt(features.get("calc_contrast"), 3)],
        ]
        table_calc_summary = dense_table(summary_rows, col_widths=[4.4 * cm, 4.4 * cm])

        detail_rows = [["ID", "Diámetro (mm)", "Área (px)"]]
        for c in calculi_info[:3]:
            detail_rows.append([f"C{c['id']}", fmt(c["diam_mm"], 2), str(c["area_px"])])
        table_calc_detail = dense_table(detail_rows, col_widths=[2.8 * cm, 3.2 * cm, 3.2 * cm], align_center=True)

        calc_side_by_side = Table([[table_calc_summary, table_calc_detail]], colWidths=[8.8 * cm, 9.2 * cm])
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

    story.append(Spacer(1, 0.2 * cm))
    
    story.append(Paragraph(
        "Este reporte ha sido generado utilizando modelos de deep learning de la tesis: \"Desarrollo de un software para la evaluación automática de la vesícula biliar utilizando imágenes de ultrasonido obtenidas mediante el protocolo de barrido volumétrico\". Es una herramienta de apoyo diagnóstico basado en IA y no reemplaza el criterio clínico profesional.",
        styles["small"]
    ))

    doc.build(story)
