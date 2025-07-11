import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import io
from torchvision import transforms
import os
import gdown

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

# --- Cargar modelo ---
@st.cache_resource
def cargar_modelo():
    model = UNet(in_channels=3, out_channels=3, filters=32)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

modelo = cargar_modelo()

# --- Título ---
st.title("Evaluación Automática de la Vesícula Biliar")
st.markdown("Sube una imagen ecográfica para segmentar fondo, vesícula y cálculos.")

# --- Preprocesamiento seguro ---
def preprocesar_imagen(pil_img):
    try:
        img = pil_img.convert("RGB")
        img = ImageOps.exif_transpose(img)  # Corregir orientación
        img = img.resize((384, 384))
        tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, 3, 384, 384]
        return tensor, img
    except Exception as e:
        st.error(f"Error al procesar imagen: {e}")
        return None, None

# --- Subir imagen ---
archivo = st.file_uploader("Sube una imagen .png", type=["png", "jpg", "jpeg"])

if archivo:
    imagen_pil = Image.open(archivo)
    imagen_tensor, imagen_mostrable = preprocesar_imagen(imagen_pil)

    if imagen_tensor is not None:
        with torch.no_grad():
            salida = modelo(imagen_tensor)
            pred = torch.argmax(salida.squeeze(), dim=0).cpu().numpy()

        # --- Visualizar resultados ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen Original")
            st.image(imagen_mostrable)

        with col2:
            st.subheader("Segmentación")
            cmap = plt.get_cmap("tab10")
            fig, ax = plt.subplots()
            ax.imshow(pred, cmap=cmap, vmin=0, vmax=2)
            ax.set_title("0: Fondo, 1: Vesícula, 2: Cálculos")
            ax.axis("off")
            st.pyplot(fig)

st.sidebar.info("Desarrollado como parte del proyecto de tesis sobre evaluación automática de la vesícula biliar.")

