import numpy as np
import cv2
from skimage.measure import label, regionprops, perimeter
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.spatial import ConvexHull, distance_matrix
from config import S_X, S_Y, NLEV, MIN_CLASS


def detect_classes(mask):
    vals, counts = np.unique(mask, return_counts=True)
    return sorted([(int(v), int(c)) for v, c in zip(vals, counts) if v != 0 and c >= MIN_CLASS], key=lambda x: -x[1])


def quantize(gray, mask):
    v = gray[mask].astype(float)
    if v.size == 0:
        return np.zeros(gray.shape, np.uint8)
    vmin, vmax = v.min(), v.max()
    q = np.zeros(gray.shape, np.uint8)
    if vmax <= vmin:
        q[mask] = 1
    else:
        q[mask] = 1 + np.clip((gray[mask].astype(float) - vmin) / (vmax - vmin) * (NLEV - 1), 0, NLEV - 1).astype(int)
    return q


def first_order(gray, mask):
    v = gray[mask].astype(float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    q = quantize(gray, mask)[mask]
    hist = np.bincount(q, minlength=NLEV + 1)[1:].astype(float)
    p = hist[hist > 0] / hist.sum() if hist.sum() > 0 else np.array([])
    entropy = float(-np.sum(p * np.log2(p))) if p.size > 0 else np.nan
    return float(v.mean()), float(v.std()), entropy


def glcm_features(gray, mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.nan, np.nan
    q = quantize(gray, mask)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    try:
        g = graycomatrix(q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=NLEV + 1, symmetric=True, normed=False).astype(float)
        g[0, :, :, :] = 0
        g[:, 0, :, :] = 0
        if g.sum() == 0:
            return np.nan, np.nan
        return float(graycoprops(g, 'contrast').mean()), float(graycoprops(g, 'homogeneity').mean())
    except Exception:
        return np.nan, np.nan


def zone_entropy(gray, mask):
    q = quantize(gray, mask)
    szm = {}
    for gl in range(1, NLEV + 1):
        lab, n = ndimage.label(q == gl)
        if n == 0:
            continue
        for s in ndimage.sum(np.ones_like(lab, float), lab, range(1, n + 1)):
            k = (gl, int(s))
            szm[k] = szm.get(k, 0) + 1
    if not szm:
        return np.nan
    c = np.array(list(szm.values()), float)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def feret_measurements(coords):
    pts = coords[:, ::-1].astype(float)
    pts_mm = pts * np.array([S_X, S_Y])

    if len(pts_mm) < 3:
        p = pts[0]
        return 0.0, 0.0, 0.0, 0.0, p, p, p, p

    centro_mm = pts_mm.mean(axis=0)
    C = pts_mm - centro_mm
    cov = np.cov(C.T)
    evals, evecs = np.linalg.eigh(cov)
    eje_mayor = evecs[:, np.argmax(evals)]
    eje_mayor = eje_mayor / np.linalg.norm(eje_mayor)
    eje_menor = np.array([-eje_mayor[1], eje_mayor[0]])
    eje_menor = eje_menor / np.linalg.norm(eje_menor)

    proj_largo = C @ eje_mayor
    proj_ancho = C @ eje_menor

    i_min = np.argmin(proj_largo)
    i_max = np.argmax(proj_largo)

    L1_mm = pts_mm[i_min]
    L2_mm = pts_mm[i_max]
    largo_mm = float(proj_largo[i_max] - proj_largo[i_min])

    n_bins = 100
    min_long = proj_largo.min()
    max_long = proj_largo.max()

    if max_long <= min_long:
        ancho_mm = 0.0
        pos_largo_mm = 0.0
        A1_mm = centro_mm
        A2_mm = centro_mm
    else:
        bins = np.linspace(min_long, max_long, n_bins + 1)
        mejor_ancho = 0.0
        mejor_pos = 0.0
        mejor_min = 0.0
        mejor_max = 0.0

        for k in range(n_bins):
            low = bins[k]
            high = bins[k + 1]
            sel = (proj_largo >= low) & (proj_largo < high)

            if sel.sum() < 2:
                continue

            ancho_min = proj_ancho[sel].min()
            ancho_max = proj_ancho[sel].max()
            ancho_actual = ancho_max - ancho_min

            if ancho_actual > mejor_ancho:
                mejor_ancho = float(ancho_actual)
                mejor_pos = float(np.mean(proj_largo[sel]))
                mejor_min = float(ancho_min)
                mejor_max = float(ancho_max)

        ancho_mm = mejor_ancho

        punto_interseccion_mm = centro_mm + mejor_pos * eje_mayor

        A1_mm = centro_mm + mejor_pos * eje_mayor + mejor_min * eje_menor
        A2_mm = centro_mm + mejor_pos * eje_mayor + mejor_max * eje_menor

    L1_px = L1_mm / np.array([S_X, S_Y])
    L2_px = L2_mm / np.array([S_X, S_Y])
    A1_px = A1_mm / np.array([S_X, S_Y])
    A2_px = A2_mm / np.array([S_X, S_Y])

    if evals[1] > 0:
        elong = float(np.sqrt(evals[0] / evals[1]))
        flat = float(evals[0] / evals[1])
    else:
        elong = 0.0
        flat = 0.0

    vector_largo = L2_mm - L1_mm
    vector_ancho = A2_mm - A1_mm
    norma_largo = np.linalg.norm(vector_largo)
    norma_ancho = np.linalg.norm(vector_ancho)

    if norma_largo > 0 and norma_ancho > 0:
        cos_angle = np.dot(vector_largo, vector_ancho) / (norma_largo * norma_ancho)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
    else:
        angle_deg = np.nan

    return largo_mm, ancho_mm, elong, flat, L1_px, L2_px, A1_px, A2_px


def extract_features(frame_rgb, mask):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    reales = detect_classes(mask)

    result = {
        "features": {},
        "vesicle_lines": None,
        "calculi_info": [],
        "vesicle_mask": None,
        "calculi_mask": None
    }

    if reales:
        real_vals = [v for v, _ in reales]
        ves = ndimage.binary_fill_holes(np.isin(mask, real_vals))
        lab = label(ves.astype(int))

        if lab.max() > 0:
            rv = max(regionprops(lab), key=lambda r: r.area)
            ves_big = lab == rv.label

            largo, ancho, elong, flat, L1, L2, A1, A2 = feret_measurements(rv.coords)
            mean, std, ent = first_order(gray, ves_big)
            contr, homog = glcm_features(gray, ves_big)
            zone_ent = zone_entropy(gray, ves_big)

            result["features"].update({
                "has_vesicle": 1,
                "ves_area_mm2": float(rv.area * S_X * S_Y),
                "ves_major_mm": largo,
                "ves_minor_mm": ancho,
                "ves_aspect_ratio": largo / ancho if ancho else np.nan,
                "ves_elongation": elong,
                "ves_sphericity": float(4 * np.pi * rv.area / (perimeter(ves_big) ** 2 + 1e-9)),
                "ves_flatness": flat,
                "ves_mean": mean,
                "ves_entropy": ent,
                "ves_std": std,
                "ves_contrast": contr,
                "ves_homogeneity": homog,
                "ves_zone_entropy": zone_ent
            })

            result["vesicle_lines"] = {
                "L1": L1,
                "L2": L2,
                "A1": A1,
                "A2": A2,
                "largo_mm": largo,
                "ancho_mm": ancho
            }

            result["vesicle_mask"] = ves_big
        else:
            ves_big = np.zeros(mask.shape, bool)
    else:
        ves_big = np.zeros(mask.shape, bool)
        result["features"]["has_vesicle"] = 0

        for k in ['ves_area_mm2', 'ves_major_mm', 'ves_minor_mm', 'ves_aspect_ratio', 'ves_elongation', 'ves_sphericity', 'ves_flatness', 'ves_mean', 'ves_entropy', 'ves_std', 'ves_contrast', 'ves_homogeneity', 'ves_zone_entropy']:
            result["features"][k] = np.nan

    calc_vals = [v for v, _ in reales[1:]] if len(reales) >= 2 else []
    calc = np.isin(mask, calc_vals) if calc_vals else np.zeros(mask.shape, bool)

    lab_c, n_c = ndimage.label(calc)

    components = []

    if n_c > 0 and np.any(ves_big):
        vesicle_uint8 = ves_big.astype(np.uint8)
        distance_from_vesicle = cv2.distanceTransform(1 - vesicle_uint8, cv2.DIST_L2, 5)

        for k in range(1, n_c + 1):
            component = lab_c == k
            area = int(component.sum())

            if area < MIN_CLASS:
                continue

            ys, xs = np.nonzero(component)

            if len(xs) == 0:
                continue

            cx = float(xs.mean())
            cy = float(ys.mean())

            cx_int = int(round(cx))
            cy_int = int(round(cy))

            if 0 <= cx_int < mask.shape[1] and 0 <= cy_int < mask.shape[0]:
                distance = float(distance_from_vesicle[cy_int, cx_int])
            else:
                distance = np.inf

            if distance <= 25:
                components.append((component, cx, cy))

    if components:
        diams = []
        calculi_details = []
        valid_components = []

        for ci, (c, cx, cy) in enumerate(components):
            ys, xs = np.nonzero(c)
            pts = np.column_stack([xs, ys]).astype(float)
            pts_mm = pts * np.array([S_X, S_Y])

            if len(pts_mm) >= 3:
                try:
                    hp = pts_mm[ConvexHull(pts_mm).vertices]
                    d = float(distance_matrix(hp, hp).max())
                except Exception:
                    d = 0.0
            else:
                d = 0.0

            if d <= 0:
                continue

            diams.append(d)
            valid_components.append(c)

            calculi_details.append({
                "id": len(calculi_details) + 1,
                "diam_mm": d,
                "area_px": int(c.sum()),
                "centroid": (cx, cy)
            })

        if valid_components:
            big_idx = int(np.argmax(diams))
            big = valid_components[big_idx]

            _, _, c_ent = first_order(gray, big)
            c_contr, _ = glcm_features(gray, big)

            result["features"].update({
                "has_calculi": 1,
                "num_calculi": len(valid_components),
                "max_calc_diam_mm": float(max(diams)),
                "calc_entropy": c_ent,
                "calc_contrast": c_contr
            })

            result["calculi_info"] = calculi_details

            calc_filtered = np.zeros(mask.shape, bool)

            for c in valid_components:
                calc_filtered[c] = True

            result["calculi_mask"] = calc_filtered
        else:
            result["features"].update({
                "has_calculi": 0,
                "num_calculi": 0,
                "max_calc_diam_mm": np.nan,
                "calc_entropy": np.nan,
                "calc_contrast": np.nan
            })
    else:
        result["features"].update({
            "has_calculi": 0,
            "num_calculi": 0,
            "max_calc_diam_mm": np.nan,
            "calc_entropy": np.nan,
            "calc_contrast": np.nan
        })

    return result
