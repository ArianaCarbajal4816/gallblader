import numpy as np
import cv2
from skimage.measure import label, regionprops, perimeter
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.spatial import ConvexHull, distance_matrix

from config import S_X, S_Y, NLEV, MIN_CLASS


def detect_classes(mask):
    vals, counts = np.unique(mask, return_counts=True)
    return sorted(
        [(int(v), int(c)) for v, c in zip(vals, counts) if v != 0 and c >= MIN_CLASS],
        key=lambda x: -x[1]
    )


def quantize(gray, mask):
    v = gray[mask].astype(float)
    if v.size == 0:
        return np.zeros(gray.shape, np.uint8)
    vmin, vmax = v.min(), v.max()
    q = np.zeros(gray.shape, np.uint8)
    if vmax <= vmin:
        q[mask] = 1
    else:
        q[mask] = 1 + np.clip(
            (gray[mask].astype(float) - vmin) / (vmax - vmin) * (NLEV - 1), 0, NLEV - 1
        ).astype(int)
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
        g = graycomatrix(
            q, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=NLEV + 1, symmetric=True, normed=False
        ).astype(float)
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
        return 0.0, 0.0, 0.0, 0.0, pts[0], pts[0], pts[0], pts[0]

    try:
        hull = ConvexHull(pts_mm)
        hull_pts_mm = pts_mm[hull.vertices]
        hull_pts_px = pts[hull.vertices]
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, pts[0], pts[0], pts[0], pts[0]

    D = distance_matrix(hull_pts_mm, hull_pts_mm)
    i, j = np.unravel_index(D.argmax(), D.shape)
    largo_mm = float(D[i, j])
    L1_px, L2_px = hull_pts_px[i], hull_pts_px[j]

    C = pts_mm - pts_mm.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(C.T))
    eje_mayor = evecs[:, np.argmax(evals)]
    eje_menor = evecs[:, np.argmin(evals)]
    proj_largo = C @ eje_mayor
    proj_ancho = C @ eje_menor

    n_bins = 60
    bins = np.linspace(proj_largo.min(), proj_largo.max(), n_bins + 1)
    idx = np.digitize(proj_largo, bins)
    ancho_mm = 0.0
    b_best = None
    for b in range(1, n_bins + 1):
        sel = idx == b
        if sel.sum() > 1:
            w = float(np.ptp(proj_ancho[sel]))
            if w > ancho_mm:
                ancho_mm = w
                b_best = b

    if b_best is None:
        A1_px, A2_px = pts[0], pts[0]
    else:
        sel = np.where(idx == b_best)[0]
        lo = sel[proj_ancho[sel].argmin()]
        hi = sel[proj_ancho[sel].argmax()]
        A1_px, A2_px = pts[lo], pts[hi]

    elong = float(np.sqrt(evals[0] / evals[1])) if evals[1] > 0 else 0.0
    flat = float(evals[0] / evals[1]) if evals[1] > 0 else 0.0

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
            result["vesicle_lines"] = {"L1": L1, "L2": L2, "A1": A1, "A2": A2,
                                      "largo_mm": largo, "ancho_mm": ancho}
            result["vesicle_mask"] = ves_big
    else:
        result["features"]["has_vesicle"] = 0
        for k in ['ves_area_mm2', 'ves_major_mm', 'ves_minor_mm', 'ves_aspect_ratio',
                  'ves_elongation', 'ves_sphericity', 'ves_flatness', 'ves_mean',
                  'ves_entropy', 'ves_std', 'ves_contrast', 'ves_homogeneity', 'ves_zone_entropy']:
            result["features"][k] = np.nan

    calc_vals = [v for v, _ in reales[1:]] if len(reales) >= 2 else []
    calc = np.isin(mask, calc_vals) if calc_vals else np.zeros(mask.shape, bool)
    lab_c, n_c = ndimage.label(calc)
    comps = [lab_c == k for k in range(1, n_c + 1) if (lab_c == k).sum() >= MIN_CLASS]

    if comps:
        diams = []
        calculi_details = []
        for ci, c in enumerate(comps):
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
            diams.append(d)
            cx, cy = float(xs.mean()), float(ys.mean())
            calculi_details.append({"id": ci + 1, "diam_mm": d, "area_px": int(c.sum()),
                                   "centroid": (cx, cy)})

        big_idx = int(np.argmax(diams))
        big = comps[big_idx]
        _, _, c_ent = first_order(gray, big)
        c_contr, _ = glcm_features(gray, big)
        result["features"].update({
            "has_calculi": 1,
            "num_calculi": len(comps),
            "max_calc_diam_mm": float(max(diams)),
            "calc_entropy": c_ent,
            "calc_contrast": c_contr
        })
        result["calculi_info"] = calculi_details
        result["calculi_mask"] = calc
    else:
        result["features"].update({
            "has_calculi": 0,
            "num_calculi": 0,
            "max_calc_diam_mm": np.nan,
            "calc_entropy": np.nan,
            "calc_contrast": np.nan
        })

    return result
