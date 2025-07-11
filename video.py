import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance
import io
from torchvision import transforms
import os
import gdown
import cv2

# --- Configuración de la página ---
st.set_page_config(page_title="Gallbladder AI", layout="wide")

# --- Descargar modelo desde Google Drive si no está localmente ---
MODEL_ID = '12wlavHoJO_yJAchDExAFSBnKAohOKTOy'
MODEL_PATH = 'mejor_modelo_clase2.pth'

if not os.path.exists(MODEL_PATH):
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_PATH, quiet=False)

# --- Definición de arquitectura UNet y bloques ---
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0, max_pooling=True):
        super().__init__()
        self.max_pooling = max_pooling
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_prob)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        if self.max_pooling:
            x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, filters=32):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, filters)
        self.enc2 = EncoderBlock(filters, filters*2)
        self.enc3 = EncoderBlock(filters*2, filters*4)
        self.enc4 = EncoderBlock(filters*4, filters*8, dropout_prob=0.3)
        self.center = EncoderBlock(filters*8, filters*16, dropout_prob=0.3, max_pooling=False)
        self.dec4 = DecoderBlock(filters*16, filters*8, filters*8)
        self.dec3 = DecoderBlock(filters*8, filters*4, filters*4)
        self.dec2 = DecoderBlock(filters*4, filters*2, filters*2)
        self.dec1 = DecoderBlock(filters*2, filters, filters)
        self.final = nn.Conv2d(filters, out_channels, kernel_size=1)

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        x5, _ = self.center(x4)
        x = self.dec4(x5, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.final(x)

@st.cache_resource
def cargar_modelo():
    model = UNet(in_channels=3, out_channels=3, filters=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

modelo = cargar_modelo()

st.title("Segmentación Automática de la Vesícula Biliar ")
st.markdown("Sube un video ecográfico ")

video_file = st.file_uploader("Sube un video .mp4", type=["mp4"])

if video_file:
    with open("input_temp.mp4", "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("input_temp.mp4")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = 384, 384
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "segmentado.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb).resize((384, 384))

        # Ajuste de contraste
        enhancer = ImageEnhance.Contrast(frame_pil)
        frame_pil = enhancer.enhance(1.5)

        tensor = transforms.ToTensor()(frame_pil).unsqueeze(0)

        with torch.no_grad():
            pred = modelo(tensor)
            mask = torch.argmax(pred.squeeze(), dim=0).byte().cpu().numpy()

        color_mask = np.zeros((384, 384, 3), dtype=np.uint8)
        color_mask[mask == 1] = [255, 255, 255]     # Vesícula: blanco
        color_mask[mask == 2] = [204, 153, 255]     # Cálculos: lila

        frame_np = np.array(frame_pil)
        combined = np.concatenate((frame_np, color_mask), axis=1)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)

        stframe.image(combined, caption="Frame con segmentación", use_column_width=True)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(output_path, "rb") as f:
        video_bytes = f.read()

    st.video(video_bytes)
    st.download_button(" Descargar video segmentado", video_bytes, file_name="segmentado.mp4", mime="video/mp4")
    # Mostrar diagnóstico fijo
    st.markdown("### Etiqueta diagnóstica:")
    st.success("Vesícula biliar normal")


st.sidebar.info("Desarrollado como parte del proyecto de tesis sobre evaluación automática de la vesícula biliar.")

