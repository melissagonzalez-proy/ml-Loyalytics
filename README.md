# Loyalytics

Proyecto de Machine Learning desarrollado para la asignatura
**Modelos y Simulación de Sistemas II** de la Universidad de Antioquia.

El proyecto aborda el análisis y predicción de abandono de clientes
(customer churn) en un contexto de comercio electrónico.

---

## Estructura del repositorio

```text
ml-Loyalytics/
│
├── data/
│   └── raw/
│       ├── .gitkeep
│       └── E Commerce Dataset.xlsx
│
├── notebooks/
│   └── 01_exploracion_datos.ipynb
│
├── reports/
│   └── reporte_eda_loyalytics.html
│
├── .gitignore
├── README.md
└── requirements.txt
```

### `data/raw/`

Contiene el dataset original utilizado en el proyecto.

### `notebooks/`

Contiene los notebooks utilizados para la exploración y análisis
de los datos.

### `reports/`

Contiene los reportes generados durante el desarrollo del proyecto.

---

## Dataset

El proyecto utiliza el dataset **Ecommerce Customer Churn Analysis and Prediction**.

- **Fuente:** [Kaggle](https://www.kaggle.com/)
- **Archivo utilizado:** `E Commerce Dataset.xlsx`

El archivo se encuentra ubicado en:

```
data/raw/E Commerce Dataset.xlsx
```

---

## Requisitos

Para ejecutar el proyecto localmente se necesita:

- Python 3.10 o superior.
- Jupyter Notebook o Visual Studio Code con soporte para notebooks.

---

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   ```

2. Entrar al proyecto:

   ```bash
   cd ml-Loyalytics
   ```

3. Crear un entorno virtual (Windows):

   ```bash
   python -m venv .venv
   ```

4. Activar el entorno:

   ```bash
   .venv\Scripts\activate
   ```

5. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución en Visual Studio Code

1. Abrir la carpeta `ml-Loyalytics` en Visual Studio Code.
2. Instalar la extensión Jupyter si todavía no está instalada.
3. Seleccionar el intérprete de Python correspondiente al entorno virtual.
4. Abrir el notebook:

   ```
   notebooks/01_exploracion_datos.ipynb
   ```

5. Ejecutar las celdas del notebook en orden.

El notebook detectará automáticamente que se está ejecutando fuera de
Google Colab y utilizará el dataset ubicado en:

```
data/raw/E Commerce Dataset.xlsx
```

El reporte será generado automáticamente en:

```
reports/reporte_eda_loyalytics.html
```

---

## Ejecución en Google Colab

El notebook también puede ejecutarse utilizando Google Colab.

1. Abrir `01_exploracion_datos.ipynb` en Google Colab.
2. Ejecutar las celdas en orden.
3. Cuando el notebook lo solicite, seleccionar el archivo `E Commerce Dataset.xlsx`.
4. Continuar con la ejecución del notebook.

En Google Colab el reporte será generado como:

```
reporte_eda_loyalytics.html
```

El archivo puede descargarse y posteriormente incorporarse a la carpeta `reports/`.

---

## Exploración de datos

El notebook realiza una exploración inicial que incluye:

- Número de muestras.
- Número de variables.
- Tipos de datos.
- Estadísticas descriptivas.
- Valores faltantes.
- Registros duplicados.
- Distribución de la variable objetivo `Churn`.
- Perfilamiento automatizado mediante YData Profiling.

---

## Reporte de EDA

El análisis exploratorio automatizado se genera utilizando **YData Profiling**.

El reporte resultante se encuentra en:

```
reports/reporte_eda_loyalytics.html
```

---

## Entregables

Los resultados principales del proyecto se encuentran en la carpeta `reports/`.

Entre ellos se encuentra el reporte de análisis exploratorio y el
informe correspondiente al **Entregable I**.
