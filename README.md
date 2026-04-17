#  Mutual Fund Style Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Descripción   

Sistema de Machine Learning no supervisado que clasifica automáticamente fondos de inversión y ETFs en el **Morningstar Style Box**, identificando si son **Growth, Value, Small Cap o Large Cap** usando solo datos históricos de precios.

##  Características

-  **Clustering Automático**: K-Means para clasificar fondos sin etiquetas previas
-  **Style Box Visual**: Heatmap interactivo del Morningstar Style Box
-  **Análisis Riesgo-Retorno**: Visualización de volatilidad vs rendimiento
-  **Sharpe Ratio**: Medición de eficiencia de inversión
-  **Datos en Vivo**: Integración con Yahoo Finance

##  Estructura del Proyecto
mutual-fund-style-classifier/
├── app/
│ └── main.py # Aplicación Streamlit
├── src/
│ ├── data_acquisition.py # Descarga de datos
│ ├── feature_engineering.py # Ingeniería de características
│ └── clustering_model.py # Modelo K-Means
├── data/
│ └── raw/ # Datos CSV (ignorados por git)
├── models/ # Modelos entrenados
├── requirements.txt # Dependencias
└── README.md

text

## Instalación Rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/mutual-fund-style-classifier.git
cd mutual-fund-style-classifier

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Pre-cargar datos
python preload_data_simple.py

# 5. Ejecutar la app
streamlit run app/main.py