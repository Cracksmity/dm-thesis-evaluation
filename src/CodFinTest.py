
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "DataSet_completo.xlsx"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
CORR_THRESHOLD = 0.8
ARTICLE_DT_ACCURACY = 0.834


def reducir_por_correlacion(X: pd.DataFrame, umbral: float = 0.8):
    """Elimina columnas redundantes con |r| > umbral respecto a otra feature.

    Args:
        X: Matriz de características numéricas.
        umbral: Umbral absoluto de correlación para descartar columnas.

    Returns:
        Tupla (X_reducido, lista_columnas_eliminadas).
    """
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > umbral)]
    return X.drop(columns=to_drop), to_drop


def crear_gridsearch_mlp(X, y):
    """Ajusta un MLP con GridSearchCV y validación cruzada estratificada 5-fold.

    Construye un pipeline con escalado y MLP, explora la rejilla de
    hiperparámetros y devuelve el mejor estimador según accuracy.

    Args:
        X: Características de entrenamiento.
        y: Etiquetas de clase.

    Returns:
        Mejor estimador (Pipeline) encontrado por GridSearchCV.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=100,
            random_state=RANDOM_STATE,
        )),
    ])
    param_grid = {
        "mlp__hidden_layer_sizes": [(16, 8), (20, 10), (32, 16)],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate_init": [0.001, 0.01],
    }
    cv_inner = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    #
    gs = GridSearchCV(
    pipe, param_grid, cv=cv_inner, scoring="accuracy", n_jobs=1
    )
    gs.fit(X, y)
    print("Mejores hiperparámetros:", gs.best_params_)
    print(f"Accuracy CV interna (5-fold): {gs.best_score_:.3f}")
    return gs.best_estimator_


def evaluar_modelo(nombre, X, y, estimator):
    """Evalúa el estimador con RepeatedStratifiedKFold (5 folds × 10 repeticiones).

    Args:
        nombre: Etiqueta del escenario para la salida por consola.
        X: Características.
        y: Etiquetas.
        estimator: Modelo o pipeline ya configurado.

    Returns:
        Diccionario de scores devuelto por ``cross_validate``.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_estimator=False,
    )
    print(f"\n=== {nombre} ===")
    print(f"Accuracy media: {scores['test_accuracy'].mean():.3f} "
          f"± {scores['test_accuracy'].std():.3f}")
    print(f"F1 macro media: {scores['test_f1_macro'].mean():.3f}")
    print(f"Referencia artículo (DT): {ARTICLE_DT_ACCURACY:.1%}")
    return scores


def reporte_oof(nombre, X, y, estimator):
    """Genera reporte de clasificación con predicciones out-of-fold (5-fold).

    Evita leakage al predecir cada muestra solo con modelos entrenados
    en los otros folds.

    Args:
        nombre: Etiqueta del escenario.
        X: Características.
        y: Etiquetas verdaderas.
        estimator: Modelo o pipeline a evaluar.
    """
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    y_pred = cross_val_predict(estimator, X, y, cv=cv)
    print(f"\n{nombre} — Reporte (predicción OOF 5-fold):")
    print(classification_report(y, y_pred))
    print("Matriz de confusión:")
    #print(confusion_matrix(y, y_pred))
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Matriz de Confusión - MLP")
    plt.show()
    


def plot_loss_curve(estimator, X, y, titulo, archivo):
    """Reentrena un clon del estimador y guarda la curva de pérdida del MLP.

    Args:
        estimator: Pipeline con paso ``mlp`` (MLPClassifier).
        X: Características de entrenamiento.
        y: Etiquetas.
        titulo: Título de la figura.
        archivo: Ruta del archivo PNG de salida.
    """
    est = clone(estimator)
    est.fit(X, y)
    plt.figure(figsize=(8, 5))
    plt.plot(est.named_steps["mlp"].loss_curve_)
    plt.xlabel("Iteración")
    plt.ylabel("Pérdida")
    plt.title(titulo)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(archivo)
    plt.close()
    print(f"Guardada: {archivo}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Fase 1: Preprocesamiento -----
    print("=" * 60)
    print("FASE 1 — Preprocesamiento")
    print("=" * 60)

    df = pd.read_excel(DATA_PATH)
    df = df.dropna(axis=1, how="all")

    if "Longitud" not in df.columns or "Clase" not in df.columns:
        raise ValueError("El dataset debe incluir las columnas 'Longitud' y 'Clase'.")

    X_full = df.drop(columns=["Longitud", "Clase"])
    y = df["Clase"]

    print(f"Muestras: {len(df)} | Features (sin Longitud): {X_full.shape[1]}")
    print("\nDistribución de clases:")
    print(y.value_counts().sort_index())

    fig_clases = FIGURES_DIR / "fig_clases.png"
    y.value_counts().sort_index().plot(kind="bar", title="Muestras por clase")
    plt.xlabel("Clase")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(fig_clases)
    plt.close()
    print(f"Guardada: {fig_clases}")

    fig_corr_ini = FIGURES_DIR / "fig_corr_inicial.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_full.corr(), cmap="coolwarm", center=0)
    plt.title("Matriz de correlación (sin Longitud)")
    plt.tight_layout()
    plt.savefig(fig_corr_ini)
    plt.close()
    print(f"Guardada: {fig_corr_ini}")

    # ----- Fase 3: Reducción por correlación -----
    print("\n" + "=" * 60)
    print("FASE 3 — Análisis de correlación y reducción")
    print("=" * 60)

    X_red, dropped = reducir_por_correlacion(X_full, CORR_THRESHOLD)
    print(f"Umbral |r| > {CORR_THRESHOLD}")
    print(f"Características eliminadas ({len(dropped)}): {dropped}")
    print(f"Características restantes: {X_red.shape[1]}")

    dataset_reducido = RESULTS_DIR / "DataSet_reducido.xlsx"
    pd.concat([X_red, y], axis=1).to_excel(dataset_reducido, index=False)
    print(f"Guardado: {dataset_reducido} (features + Clase)")

    fig_corr_red = FIGURES_DIR / "fig_corr_reducido.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_red.corr(), cmap="coolwarm", center=0)
    plt.title("Matriz de correlación — dataset reducido")
    plt.tight_layout()
    plt.savefig(fig_corr_red)
    plt.close()
    print(f"Guardada: {fig_corr_red}")

    # ----- Fase 2: MLP con todas las features -----
    print("\n" + "=" * 60)
    print("FASE 2 — MLP con todas las características (sin Longitud)")
    print("=" * 60)

    best_mlp = crear_gridsearch_mlp(X_full, y)
    scores_fase2 = evaluar_modelo(
        "Fase 2 — features completas",
        X_full,
        y,
        best_mlp,
    )
    reporte_oof("Fase 2", X_full, y, best_mlp)

    # ----- Fase 4: MLP con dataset reducido -----
    print("\n" + "=" * 60)
    print("FASE 4 — MLP con características reducidas (mismos hiperparámetros)")
    print("=" * 60)

    best_mlp_red = clone(best_mlp)
    scores_fase4 = evaluar_modelo(
        "Fase 4 — features reducidas",
        X_red,
        y,
        best_mlp_red,
    )
    reporte_oof("Fase 4", X_red, y, best_mlp_red)

    plot_loss_curve(
        best_mlp,
        X_full,
        y,
        "Curva de pérdida — Fase 2 (features completas)",
        FIGURES_DIR / "fig_loss_fase2.png",
    )
    plot_loss_curve(
        best_mlp_red,
        X_red,
        y,
        "Curva de pérdida — Fase 4 (features reducidas)",
        FIGURES_DIR / "fig_loss_fase4.png",
    )

    # ----- Comparación final -----
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 60)

    comparacion = pd.DataFrame({
        "escenario": [
            "Fase 2 (sin Longitud)",
            "Fase 4 (reducido)",
            "Artículo DT",
        ],
        "accuracy_media": [
            scores_fase2["test_accuracy"].mean(),
            scores_fase4["test_accuracy"].mean(),
            ARTICLE_DT_ACCURACY,
        ],
        "accuracy_std": [
            scores_fase2["test_accuracy"].std(),
            scores_fase4["test_accuracy"].std(),
            np.nan,
        ],
        "f1_macro_media": [
            scores_fase2["test_f1_macro"].mean(),
            scores_fase4["test_f1_macro"].mean(),
            np.nan,
        ],
        "n_features": [X_full.shape[1], X_red.shape[1], 19],
        "features_eliminadas": [
            0,
            len(dropped),
            "N/A (artículo)",
        ],
    })
    print(comparacion.to_string(index=False))
    csv_path = RESULTS_DIR / "comparacion_resultados.csv"
    comparacion.to_csv(csv_path, index=False)
    print(f"\nGuardado: {csv_path}")


if __name__ == "__main__":
    main()
