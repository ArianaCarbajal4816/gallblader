import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from config import COLOR_LARGO, COLOR_ANCHO
import cv2
import streamlit as st


def clean_class_1_mask(mask):
    mask_clean = mask.copy()
    mask_c1 = (mask_clean == 1).astype(np.uint8)

    if np.sum(mask_c1) == 0:
        return mask_clean

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_c1, connectivity=8)

    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + np.argmax(areas)
        main_component = (labels == largest_idx).astype(np.uint8)
    else:
        main_component = mask_c1.copy()

    dist = cv2.distanceTransform(main_component, cv2.DIST_L2, 5)
    max_dist = dist.max()

    if max_dist <= 0:
        return mask_clean

    threshold = 0.35 * max_dist
    sure_fg = np.zeros_like(main_component)
    sure_fg[dist >= threshold] = 1

    num_markers, markers = cv2.connectedComponents(sure_fg)

    if num_markers <= 2:
        threshold = 0.20 * max_dist
        sure_fg = np.zeros_like(main_component)
        sure_fg[dist >= threshold] = 1
        num_markers, markers = cv2.connectedComponents(sure_fg)

    if num_markers <= 2:
        candidate_mask = main_component.copy()
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return mask_clean

        best_contour = max(contours, key=cv2.contourArea)
        result = np.zeros_like(mask_c1)

        if cv2.contourArea(best_contour) > 100:
            cv2.drawContours(result, [best_contour], -1, 1, thickness=-1)

        mask_clean[mask_clean == 1] = 0
        mask_clean[result == 1] = 1

        return mask_clean

    markers = markers.astype(np.int32) + 1
    unknown = cv2.subtract(main_component, sure_fg.astype(np.uint8))
    markers[unknown == 1] = 0

    watershed_img = cv2.cvtColor((main_component * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(watershed_img, markers)

    region_ids = np.unique(markers)
    candidates = []

    for region_id in region_ids:
        if region_id <= 1:
            continue

        region = ((markers == region_id) & (main_component == 1)).astype(np.uint8)
        area = np.sum(region)

        if area < 100:
            continue

        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(contour)

        if contour_area < 100:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = 4 * np.pi * contour_area / (perimeter ** 2)
        else:
            circularity = 0

        ellipse_score = 0

        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (major_axis, minor_axis), angle = ellipse

                if major_axis > 0 and minor_axis > 0:
                    aspect_ratio = min(major_axis, minor_axis) / max(major_axis, minor_axis)
                    ellipse_area = np.pi * (major_axis / 2) * (minor_axis / 2)

                    if ellipse_area > 0:
                        area_similarity = min(contour_area / ellipse_area, ellipse_area / contour_area)
                    else:
                        area_similarity = 0

                    ellipse_score = 0.6 * area_similarity + 0.4 * aspect_ratio

            except cv2.error:
                ellipse_score = 0

        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx_region = M["m10"] / M["m00"]
            cy_region = M["m01"] / M["m00"]
        else:
            cx_region = 0
            cy_region = 0

        candidates.append({
            "region_id": region_id,
            "region": region,
            "area": contour_area,
            "circularity": circularity,
            "ellipse_score": ellipse_score,
            "cx": cx_region,
            "cy": cy_region
        })

    if len(candidates) == 0:
        mask_clean[mask_clean == 1] = 0
        mask_clean[main_component == 1] = 1
        return mask_clean

    max_area = max(c["area"] for c in candidates)

    for c in candidates:
        area_score = c["area"] / max_area
        c["final_score"] = 0.55 * c["ellipse_score"] + 0.25 * area_score + 0.20 * min(c["circularity"], 1.0)

    best = max(candidates, key=lambda x: x["final_score"])
    result = best["region"].copy()

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_small)

    mask_clean[mask_clean == 1] = 0
    mask_clean[result == 1] = 1

    return mask_clean


def filter_class_2_by_vesicle(mask, max_distance_px=25):
    result = mask.copy()

    vesicle = (result == 1).astype(np.uint8)
    calculi = (result == 2).astype(np.uint8)

    if np.sum(vesicle) == 0 or np.sum(calculi) == 0:
        return result

    distance = cv2.distanceTransform(1 - vesicle, cv2.DIST_L2, 5)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(calculi, connectivity=8)

    for i in range(1, num_labels):
        component = (labels == i).astype(np.uint8)
        ys, xs = np.where(component > 0)

        if len(xs) == 0:
            continue

        cx = int(round(centroids[i][0]))
        cy = int(round(centroids[i][1]))

        if cx < 0 or cx >= mask.shape[1] or cy < 0 or cy >= mask.shape[0]:
            result[component == 1] = 0
            continue

        component_distance = distance[ys, xs].min()

        if component_distance > max_distance_px:
            result[component == 2] = 0

    return result


def filter_calculi_info(calculi_info, vesicle_mask, max_distance_px=25):
    if vesicle_mask is None or np.sum(vesicle_mask == 1) == 0:
        return []

    vesicle = (vesicle_mask == 1).astype(np.uint8)
    distance = cv2.distanceTransform(1 - vesicle, cv2.DIST_L2, 5)

    h, w = vesicle.shape
    filtered = []

    for c in calculi_info:
        cx, cy = c["centroid"]
        x = int(round(cx))
        y = int(round(cy))

        if x < 0 or x >= w or y < 0 or y >= h:
            continue

        if distance[y, x] <= max_distance_px:
            filtered.append(c)

    return filtered


def annotate_best_frame(frame_rgb, mask, vesicle_lines, calculi_info):
    h, w = frame_rgb.shape[:2]
    dpi = 100

    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(frame_rgb)

    mask_cleaned = clean_class_1_mask(mask)
    mask_cleaned = filter_class_2_by_vesicle(mask_cleaned, max_distance_px=25)

    calculi_info = filter_calculi_info(calculi_info, mask_cleaned, max_distance_px=25)

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

        ax.annotate(f"C{c['id']}: {c['diam_mm']:.1f}mm", (cx, cy),
                    textcoords="offset points", xytext=(8, -8),
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
