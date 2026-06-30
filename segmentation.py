import torch
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from torchvision import transforms

from models_arch import UNet
from config import DEVICE, INPUT_SIZE, CLASS_COLORS, UNET_MULTICLASS_PATH, UNET_E1_PATH, UNET_E2_PATH


def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb).resize(INPUT_SIZE)
    enhancer = ImageEnhance.Contrast(frame_pil)
    frame_pil = enhancer.enhance(1.5)
    frame_np = np.array(frame_pil)
    tensor = transforms.ToTensor()(frame_np).unsqueeze(0).to(DEVICE)
    return tensor, frame_np


def load_multiclass_model():
    model = UNet(in_channels=3, out_channels=3, filters=32)
    model.load_state_dict(torch.load(UNET_MULTICLASS_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def load_cascade_models():
    m1 = UNet(in_channels=3, out_channels=2, filters=32)
    m2 = UNet(in_channels=3, out_channels=2, filters=32)
    m1.load_state_dict(torch.load(UNET_E1_PATH, map_location=DEVICE))
    m2.load_state_dict(torch.load(UNET_E2_PATH, map_location=DEVICE))
    m1.to(DEVICE).eval()
    m2.to(DEVICE).eval()
    return m1, m2


def predict_multiclass(tensor, model):
    with torch.no_grad():
        out = model(tensor)
        mask = torch.argmax(out.squeeze(), dim=0).byte().cpu().numpy()
    return mask


def predict_cascade(tensor, m1, m2):
    with torch.no_grad():
        out1 = m1(tensor)
        mask1 = torch.argmax(out1.squeeze(), dim=0).byte().cpu().numpy()
        out2 = m2(tensor)
        mask2 = torch.argmax(out2.squeeze(), dim=0).byte().cpu().numpy()
        combined = np.zeros_like(mask1)
        combined[mask1 == 1] = mask2[mask1 == 1] + 1
    return combined


def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def overlay_mask(frame_rgb, mask, alpha=0.5):
    rgb_mask = mask_to_rgb(mask)
    blended = cv2.addWeighted(frame_rgb.astype(np.uint8), 1 - alpha, rgb_mask, alpha, 0)
    return blended


def segment_video(video_path, output_path, model_type, opacity=0.5, progress_callback=None):
    if model_type == "multiclass":
        model = load_multiclass_model()
        m1, m2 = None, None
    else:
        m1, m2 = load_cascade_models()
        model = None

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (INPUT_SIZE[0] * 2, INPUT_SIZE[1]))

    frames_data = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tensor, rgb_frame = preprocess_frame(frame)

        if model_type == "multiclass":
            mask = predict_multiclass(tensor, model)
        else:
            mask = predict_cascade(tensor, m1, m2)

        overlay = overlay_mask(rgb_frame, mask, alpha=opacity)
        combined = np.concatenate((rgb_frame, overlay), axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)

        vesicle_area = int(np.sum(mask >= 1))
        frames_data.append({
            "idx": idx,
            "frame_rgb": rgb_frame.copy(),
            "mask": mask.copy(),
            "vesicle_area_px": vesicle_area
        })

        idx += 1
        if progress_callback:
            progress_callback(idx / frame_count if frame_count else 0)

    cap.release()
    out.release()
    return frames_data, fps


def find_best_frame(frames_data):
    if not frames_data:
        return None
    return max(frames_data, key=lambda f: f["vesicle_area_px"])
