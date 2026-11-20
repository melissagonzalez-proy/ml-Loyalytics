import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_recall_curve


try:
    import umap
except ImportError:
    print("La libreria 'umap-learn' no se encontro. Instalando ahora...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "umap-learn"])
    import umap
    print("Instalacion completada con exito.")


try:
    import google.colab  # noqa: F401

    EN_COLAB = True
except ImportError:
    EN_COLAB = False


def _buscar_raiz():
    inicio = Path.cwd().resolve()
    for ruta in (inicio, *inicio.parents):
        if (ruta / "notebooks").is_dir() and (ruta / "src").is_dir():
            return ruta
    for entrada in map(Path, sys.path):
        try:
            entrada = entrada.resolve()
        except (OSError, RuntimeError):
            continue
        if (entrada / "notebooks").is_dir() and (entrada / "src").is_dir():
            return entrada
        if entrada.name == "src" and entrada.parent.joinpath("notebooks").is_dir():
            return entrada.parent
    return inicio


PROYECTO = _buscar_raiz()
RAIZ = PROYECTO
DATA_DIR = PROYECTO / "data"
MODELS_DIR = PROYECTO / "models"
REPORTS_DIR = PROYECTO / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DIR_FIGS = FIGURES_DIR
DIR_TABLAS = REPORTS_DIR
DATASET_NOMBRE = os.getenv("DATASET_NOMBRE", "E Commerce Dataset.xlsx")

for carpeta in (
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
):
    carpeta.mkdir(parents=True, exist_ok=True)

SEED = int(os.getenv("SEED", "42"))
BETA_UMBRAL = 2
COSTO_FN = 5.0
COSTO_FP = 1.0
UMBRALES_COSTO = np.linspace(0.001, 0.999, 999)


def guardar_estado(nombre, **objetos):
    destino = PROCESSED_DATA_DIR / f"estado_{nombre}.joblib"
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(objetos, destino)
    return destino


def cargar_estado(nombre):
    origen = PROCESSED_DATA_DIR / f"estado_{nombre}.joblib"
    if not origen.exists():
        raise FileNotFoundError(f"Corre primero el notebook {nombre}")
    return joblib.load(origen)


def guardar(fig, nombre):
    fig.savefig(DIR_FIGS / f"{nombre}.png", dpi=200, bbox_inches="tight")


def exportar(tabla, nombre):
    tabla.to_csv(DIR_TABLAS / f"{nombre}.csv", encoding="utf-8-sig")


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputador determinista libre de fugas que calcula la mediana de cada variable numerica
    condicionada a un atributo de segmentacion (por defecto 'PreferedOrderCat') dentro de cada fold de entrenamiento,
    usando la mediana global del fold como respaldo.
    """

    def __init__(self, group_col="PreferedOrderCat", num_cols=None):
        self.group_col = group_col
        self.num_cols = num_cols

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group_col)[self.num_cols].median()
        self.global_medians_ = X[self.num_cols].median()
        return self

    def transform(self, X):
        X_out = X[self.num_cols].copy()
        for col in self.num_cols:
            mapped_med = X[self.group_col].map(self.group_medians_[col])
            X_out[col] = X_out[col].fillna(mapped_med).fillna(self.global_medians_[col])
        return X_out.values


class UMAPTransformer(BaseEstimator, TransformerMixin):
    """Transformador UMAP compatible con Pipeline y validacion cruzada."""

    def __init__(
        self,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_epochs=200,
        random_state=SEED,
        n_jobs=1,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _crear_modelo(self):
        return umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X, y=None):
        self.model_ = self._crear_modelo()
        self.model_.fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.model_ = self._crear_modelo()
        return self.model_.fit_transform(X)

    def transform(self, X):
        return self.model_.transform(X)


_ALL_NUM_FEATURES = []
_ALL_CAT_FEATURES = []
_COLS_NA = []


def configurar_columnas(all_num, all_cat, cols_na):
    global _ALL_NUM_FEATURES, _ALL_CAT_FEATURES, _COLS_NA
    _ALL_NUM_FEATURES = list(all_num)
    _ALL_CAT_FEATURES = list(all_cat)
    _COLS_NA = list(cols_na)


def construir_preprocesador(estrategia):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if estrategia == "Mediana Simple (Sin Indicadores)":
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        return ColumnTransformer([
            ("num", num_pipe, _ALL_NUM_FEATURES),
            ("cat", ohe, _ALL_CAT_FEATURES),
        ])

    if estrategia == "Mediana Simple + MissingIndicators":
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        return ColumnTransformer([
            ("num", num_pipe, _ALL_NUM_FEATURES),
            ("cat", ohe, _ALL_CAT_FEATURES),
        ])

    if estrategia == "Mediana por Segmento (PreferedOrderCat) + MissingIndicators":
        group_num_pipe = Pipeline([
            ("group_imp", GroupMedianImputer(
                group_col="PreferedOrderCat",
                num_cols=_ALL_NUM_FEATURES,
            )),
            ("scaler", StandardScaler()),
        ])
        return ColumnTransformer([
            ("num_group", group_num_pipe, _ALL_NUM_FEATURES + ["PreferedOrderCat"]),
            ("na_ind", MissingIndicator(features="missing-only"), _COLS_NA),
            ("cat", ohe, _ALL_CAT_FEATURES),
        ])

    if estrategia == "KNNImputer (k=15) + MissingIndicators":
        num_pipe = Pipeline([
            ("scaler_pre", StandardScaler()),
            ("imp", KNNImputer(n_neighbors=15, weights="distance", add_indicator=True)),
        ])
        return ColumnTransformer([
            ("num", num_pipe, _ALL_NUM_FEATURES),
            ("cat", ohe, _ALL_CAT_FEATURES),
        ])

    if estrategia == "IterativeImputer (MICE) + MissingIndicators":
        num_pipe = Pipeline([
            ("imp", IterativeImputer(max_iter=15, random_state=SEED, add_indicator=True)),
            ("scaler", StandardScaler()),
        ])
        return ColumnTransformer([
            ("num", num_pipe, _ALL_NUM_FEATURES),
            ("cat", ohe, _ALL_CAT_FEATURES),
        ])

    if estrategia == "Nativo con Nulos (Sin Imputar) - Sin Indicadores":
        return ColumnTransformer([
            ("num_raw", "passthrough", _ALL_NUM_FEATURES),
            ("cat", ohe, _ALL_CAT_FEATURES),
        ])

    if estrategia == "Nativo con Nulos (Sin Imputar) + MissingIndicators":
        return ColumnTransformer([
            ("num_raw", "passthrough", _ALL_NUM_FEATURES),
            ("na_ind", MissingIndicator(features="missing-only"), _COLS_NA),
            ("cat", ohe, _ALL_CAT_FEATURES),
        ])

    raise ValueError(f"Estrategia no reconocida: {estrategia}")


def resumen_grid(nombre, grid, top=5):
    r = pd.DataFrame(grid.cv_results_)
    i = grid.best_index_
    print(f"\n{nombre}")
    print(
        f"  Mejores hiperparametros: {grid.best_params_}"
        f"  PR-AUC validacion CV: {r.loc[i, 'mean_test_score']:.4f} "
        f"+/- {r.loc[i, 'std_test_score']:.4f} | "
        f"entrenamiento: {r.loc[i, 'mean_train_score']:.4f} | "
        f"brecha train-val: {r.loc[i, 'mean_train_score'] - r.loc[i, 'mean_test_score']:.4f}"
    )
    display(
        r.sort_values("rank_test_score")
        .loc[:, ["params", "mean_train_score", "mean_test_score", "std_test_score"]]
        .head(top)
        .round(4)
    )


def umbral_fbeta(y, p, beta=BETA_UMBRAL):
    """Umbral que maximiza F-beta sobre una curva precision-exhaustividad."""
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    fb = (1 + beta**2) * prec * rec / np.clip(beta**2 * prec + rec, 1e-12, None)
    i = int(np.nanargmax(fb))
    return float(thr[i]), float(fb[i])


def costo_esperado(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return (
        COSTO_FN * ((yhat == 0) & (y == 1)).sum()
        + COSTO_FP * ((yhat == 1) & (y == 0)).sum()
    )


def umbral_costo(y, p, umbrales=None):
    candidatos = UMBRALES_COSTO if umbrales is None else umbrales
    costos = [costo_esperado(y, (p >= u).astype(int)) for u in candidatos]
    return float(candidatos[int(np.argmin(costos))])
