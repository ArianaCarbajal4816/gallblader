import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
TEMP_DIR = PROJECT_ROOT / "temp"
OUTPUT_DIR = PROJECT_ROOT / "output"

for d in [MODELS_DIR, TEMP_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = (384, 384)

S_X = 0.7414
S_Y = 0.5028
NLEV = 32
MIN_CLASS = 50

UNET_MULTICLASS_PATH = MODELS_DIR / "unet_multiclase.pth"
UNET_E1_PATH = MODELS_DIR / "unet_e1.pth"
UNET_E2_PATH = MODELS_DIR / "unet_e2.pth"
XGB_FULL_PATH = MODELS_DIR / "xgboost_radiomics.pkl"
XGB_VESICLE_PATH = MODELS_DIR / "xgboost_radiomics_std.pkl"

COLOR_BG = (0, 0, 0)
COLOR_VESICLE = (0, 114, 178)
COLOR_STONES = (213, 94, 0)
COLOR_LARGO = "#D55E00"
COLOR_ANCHO = "#0072B2"

CLASS_COLORS = {
    0: COLOR_BG,
    1: COLOR_VESICLE,
    2: COLOR_STONES
}

FEATURE_COLUMNS_FULL = [
    'has_vesicle', 'ves_area_mm2', 'ves_major_mm', 'ves_minor_mm', 'ves_aspect_ratio',
    'ves_elongation', 'ves_sphericity', 'ves_flatness', 'ves_mean', 'ves_entropy', 'ves_std',
    'ves_contrast', 'ves_homogeneity', 'ves_zone_entropy', 'has_calculi', 'num_calculi',
    'max_calc_diam_mm', 'calc_entropy', 'calc_contrast'
]

FEATURE_COLUMNS_VESICLE = [
    'has_vesicle', 'ves_area_mm2', 'ves_major_mm', 'ves_minor_mm', 'ves_aspect_ratio',
    'ves_elongation', 'ves_sphericity', 'ves_flatness', 'ves_mean', 'ves_entropy', 'ves_std',
    'ves_contrast', 'ves_homogeneity', 'ves_zone_entropy'
]
