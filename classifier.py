import joblib
import numpy as np
import pandas as pd

from config import XGB_FULL_PATH, XGB_VESICLE_PATH, FEATURE_COLUMNS_FULL, FEATURE_COLUMNS_VESICLE


def load_classifier(mode):
    if mode == "full":
        return joblib.load(XGB_FULL_PATH)
    return joblib.load(XGB_VESICLE_PATH)


def get_feature_columns(mode):
    if mode == "full":
        return FEATURE_COLUMNS_FULL
    return FEATURE_COLUMNS_VESICLE


def predict_label(features_dict, mode):
    cols = get_feature_columns(mode)
    row = {c: features_dict.get(c, 0) for c in cols}
    df = pd.DataFrame([row])[cols].fillna(0)

    model = load_classifier(mode)
    pred = int(model.predict(df)[0])
    try:
        proba = model.predict_proba(df)[0]
        confidence = float(np.max(proba))
        prob_normal = float(proba[0])
        prob_litiasis = float(proba[1])
    except Exception:
        confidence = None
        prob_normal = None
        prob_litiasis = None

    label_text = "Litiasis Vesicular" if pred == 1 else "Vesicula Normal"

    return {
        "prediction": pred,
        "label": label_text,
        "confidence": confidence,
        "prob_normal": prob_normal,
        "prob_litiasis": prob_litiasis,
        "mode": mode
    }
