import os
import sys
import gdown
from contextlib import contextmanager

from config import MODELS_DIR, DRIVE_MODELS


@contextmanager
def suppress_output():
    devnull = open(os.devnull, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        devnull.close()


def ensure_file(filename):
    target = MODELS_DIR / filename
    if target.exists():
        return True
    if filename not in DRIVE_MODELS:
        return False
    url = f"https://drive.google.com/uc?id={DRIVE_MODELS[filename]}"
    with suppress_output():
        gdown.download(url, str(target), quiet=True)
    return target.exists()


def ensure_multiclass():
    return ensure_file("unet_multiclase.pth")


def ensure_cascade():
    return ensure_file("unet_e1.pth") and ensure_file("unet_e2.pth")
