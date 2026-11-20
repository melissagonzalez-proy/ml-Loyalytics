# Predicción de churn en e-commerce

Proyecto de ciencia de datos que analiza clientes de una plataforma de comercio electrónico y predice si abandonarán (`Churn`). Incluye EDA, imputación y preprocesamiento, entrenamiento, evaluación, calibración de umbrales y análisis PCA/UMAP.

## Estructura

```text
notebooks/       Notebooks numerados del análisis
src/             Utilidades compartidas
data/            Datos locales y estados intermedios
models/          Espacio reservado para artefactos locales
reports/         Tablas y resultados
reports/figures/ Figuras generadas
```

## Ejecutar en local (VS Code)

1. Cree el entorno virtual e instale dependencias:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Coloque el dataset en `data/raw/E Commerce Dataset.xlsx`.
3. Abra y ejecute los notebooks en orden: `1_eda`, `2_preprocessing`, `3_trainig`, `4_evaluation` y `5_pca_umap`.

## Ejecutar en Google Colab

1. Clone el repositorio y cambie al directorio del proyecto.
2. Instale las dependencias:

   ```python
   %pip install -r requirements.txt
   ```

3. Coloque el dataset en `data/raw/E Commerce Dataset.xlsx` o permita que el primer notebook lo descargue desde Kaggle.
4. Ejecute los notebooks en orden: `1_eda`, `2_preprocessing`, `3_trainig`, `4_evaluation` y `5_pca_umap`.

## Dataset

Fuente: [E-Commerce Customer Churn Analysis and Prediction Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction).

Coloque el archivo como `data/raw/E Commerce Dataset.xlsx`. El primer notebook también puede descargarlo desde Kaggle cuando esté disponible.
