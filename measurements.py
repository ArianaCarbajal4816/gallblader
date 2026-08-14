import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import label  
import streamlit as st  

from config import COLOR_LARGO, COLOR_ANCHO


import cv2
import numpy as np
from scipy.ndimage import label


def clean_class_1_mask(mask):
    mask_clean = mask.copy()
    mask_c1 = (mask_clean == 1).astype(np.uint8)

    if np.sum(mask_c1) == 0:
        return mask_clean

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask_c1_separated = cv2.morphologyEx(mask_c1, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask_c1_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 1:
        menor_error_elipse = float("inf")
        index_vesicula_real = -1

        for i, cnt in enumerate(contours):
            if len(cnt) < 5 or cv2.contourArea(cnt) < 100:
                continue

            ellipse = cv2.fitEllipse(cnt)

            ellipse_mask = np.zeros_like(mask_c1)
            cv2.ellipse(ellipse_mask, ellipse, 1, thickness=-1)

            diferencia = cv2.bitwise_xor(
                mask_c1_separated & (ellipse_mask == 0),
                ellipse_mask & (mask_c1_separated == 0),
            )
            error_ajuste = np.sum(diferencia) / cv2.contourArea(cnt)

            if error_ajuste < menor_error_elipse:
                menor_error_elipse = error_ajuste
                index_vesicula_real = i

        if index_vesicula_real != -1:
            mask_filtrada = np.zeros_like(mask_c1)
            cv2.drawContours(
                mask_filtrada, contours, index_vesicula_real, 1, thickness=-1
            )

            mask_clean[mask_clean == 1] = 0
            mask_clean[mask_filtrada == 1] = 1
    else:
        labeled_array, num_features = label(mask_c1)
        if num_features > 1:
            counts = np.bincount(labeled_array.ravel())
            counts[0] = 0
            largest_idx = np.argmax(counts)
            mask_clean[(mask_clean == 1) & (labeled_array != largest_idx)] = 0

    return mask_clean



def annotate_best_frame(frame_rgb, mask, vesicle_lines, calculi_info):
    h, w = frame_rgb.shape[:2]
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame_rgb)
    mask_cleaned = clean_class_1_mask(mask)

    overlay = np.zeros_like(frame_rgb)
    overlay[mask_cleaned == 1] = [0, 114, 178]
    overlay[mask_cleaned == 2] = [213, 94, 0]
    alpha_layer = np.zeros((h, w), dtype=np.float32)
    alpha_layer[mask_cleaned >= 1] = 0.25
    ax.imshow(overlay, alpha=alpha_layer)

    if vesicle_lines is not None:
        L1, L2 = vesicle_lines["L1"], vesicle_lines["L2"]
        A1, A2 = vesicle_lines["A1"], vesicle_lines["A2"]
        ax.plot([L1[0], L2[0]], [L1[1], L2[1]], '-', color=COLOR_LARGO, lw=2.5,
                label=f"Largo: {vesicle_lines['largo_mm']:.1f} mm")
        ax.plot([A1[0], A2[0]], [A1[1], A2[1]], '-', color=COLOR_ANCHO, lw=2.5,
                label=f"Ancho: {vesicle_lines['ancho_mm']:.1f} mm")

    for c in calculi_info:
        cx, cy = c["centroid"]
        ax.plot(cx, cy, 'o', markersize=10, markerfacecolor='none',
                markeredgecolor='yellow', markeredgewidth=2)
        ax.annotate(f"C{c['id']}: {c['diam_mm']:.1f}mm",
                   (cx, cy), textcoords="offset points", xytext=(8, -8),
                   fontsize=9, color='yellow', weight='bold')

    if vesicle_lines is not None:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.7)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = np.array(Image.open(buf))
    return img


def save_annotated_frame(annotated_array, path):
    Image.fromarray(annotated_array).save(path)


def save_annotated_frame(annotated_array, path):
    Image.fromarray(annotated_array).save(path)
