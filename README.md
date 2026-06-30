# GallBladder AI

Sistema de analisis ecografico asistido por IA para la vesicula biliar.

## Funcionalidades

- Segmentacion automatica del video ecografico (UNet multiclase o cascada binaria)
- Extraccion de caracteristicas radiomicas (morfometria y textura) del frame de mayor visualizacion
- Mediciones automaticas: largo, ancho, area de la vesicula, numero y diametro de calculos
- Clasificacion de litiasis vesicular con XGBoost (2 modelos disponibles)
- Reporte clinico exportable en PDF

## Estructura

```
gallbladder_ai/
  app.py                  Aplicacion Streamlit (entrada principal)
  config.py               Rutas y parametros globales
  models_arch.py          Arquitectura UNet
  segmentation.py         Inferencia de segmentacion
  radiomics.py            Extraccion de caracteristicas
  classifier.py           Carga y prediccion XGBoost
  measurements.py         Anotacion visual del frame
  report.py               Generacion del PDF
  requirements.txt
  models/
    unet_multiclase.pth
    unet_e1.pth
    unet_e2.pth
    xgboost_radiomics.pkl
    xgboost_radiomics_std.pkl
  temp/                   Archivos temporales (se crea sola)
  output/                 Resultados (se crea sola)
```

## Instalacion

```bash
git clone <repo-url>
cd gallbladder_ai
pip install -r requirements.txt
```

Asegurarse de que los 5 archivos de pesos esten dentro de `models/`.

## Ejecucion

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`.

## Modelos

| Archivo | Tipo | Descripcion |
|---|---|---|
| unet_multiclase.pth | Segmentacion | UNet con 3 clases (fondo, vesicula, calculos) |
| unet_e1.pth | Segmentacion | Etapa 1 de cascada (fondo vs vesicula) |
| unet_e2.pth | Segmentacion | Etapa 2 de cascada (vesicula vs calculos) |
| xgboost_radiomics.pkl | Clasificacion | Usa caracteristicas de vesicula + calculos |
| xgboost_radiomics_std.pkl | Clasificacion | Usa solo caracteristicas de vesicula |

## Flujo de la aplicacion

1. Selecciona la arquitectura de segmentacion (multiclase o cascada)
2. Activa la clasificacion si quieres diagnostico asistido y elige el tipo
3. Carga un video y procesalo
4. Revisa el video segmentado, el frame anotado con mediciones y las caracteristicas
5. Exporta el reporte en PDF

## Modos disponibles

Las cuatro combinaciones se forman al elegir uno de cada eje:

- Segmentacion: multiclase o cascada
- Clasificacion: basada en segmentacion (con calculos) o basada en caracteristicas (solo vesicula)

La clasificacion es opcional. Si se desactiva, el sistema entrega solo segmentacion + mediciones.

## Aviso clinico

Este sistema es una herramienta de apoyo diagnostico. No reemplaza el criterio de un profesional medico.
