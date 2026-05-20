"""Pipeline de evaluación MLP para clasificación EOG (Blink-to-Speak).



Ejecuta las fases 1 a 4 del proyecto: preprocesamiento, hold-out estratificado,

búsqueda de hiperparámetros con validación cruzada, reducción por correlación

(solo en train) y comparación con el dataset reducido.

Requiere ``data/DataSet_completo.xlsx``.

"""



from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.base import clone

from sklearn.model_selection import (

    StratifiedKFold,

    train_test_split,

    cross_val_predict,

    GridSearchCV,

)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (

    confusion_matrix,

    ConfusionMatrixDisplay,

    classification_report,

    accuracy_score,

    f1_score,

)



ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "DataSet_completo.xlsx"

RESULTS_DIR = ROOT / "results"

FIGURES_DIR = RESULTS_DIR / "figures"



RANDOM_STATE = 42

N_SPLITS = 5

TEST_SIZE = 0.2

CORR_THRESHOLD = 0.8

ARTICLE_DT_ACCURACY = 0.834

MAX_ITER = 800
EARLY_STOPPING = True
N_ITER_NO_CHANGE = 30
VALIDATION_FRACTION = 0.1





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

            max_iter=MAX_ITER,

            early_stopping=EARLY_STOPPING,

            n_iter_no_change=N_ITER_NO_CHANGE,

            validation_fraction=VALIDATION_FRACTION,

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

    gs = GridSearchCV(

        pipe, param_grid, cv=cv_inner, scoring="accuracy", n_jobs=-1

    )

    gs.fit(X, y)

    print("Mejores hiperparámetros:", gs.best_params_)

    print(f"Accuracy CV interna (5-fold): {gs.best_score_:.3f}")

    return gs.best_estimator_





def validacion_cruzada_estratificada(nombre, X, y, estimator, csv_path):

    """Evalúa el estimador con StratifiedKFold y guarda métricas por fold.



    Args:

        nombre: Etiqueta del escenario para la salida por consola.

        X: Características de entrenamiento.

        y: Etiquetas.

        estimator: Modelo o pipeline ya configurado.

        csv_path: Ruta del CSV con métricas por fold.



    Returns:

        DataFrame con accuracy y f1_macro por fold.

    """

    cv = StratifiedKFold(

        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE

    )

    filas = []

    print(f"\n=== {nombre} — Validación cruzada estratificada ({N_SPLITS}-fold) ===")

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), start=1):

        est = clone(estimator)

        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        y_pred = est.predict(X.iloc[val_idx])

        acc = accuracy_score(y.iloc[val_idx], y_pred)

        f1 = f1_score(y.iloc[val_idx], y_pred, average="macro")

        filas.append({"fold": fold, "accuracy": acc, "f1_macro": f1})

        print(f"  Fold {fold}: accuracy={acc:.3f}, f1_macro={f1:.3f}")



    df_cv = pd.DataFrame(filas)

    df_cv.to_csv(csv_path, index=False)

    print(f"Accuracy media: {df_cv['accuracy'].mean():.3f} "

          f"± {df_cv['accuracy'].std():.3f}")

    print(f"F1 macro media: {df_cv['f1_macro'].mean():.3f}")

    print(f"Referencia artículo (DT): {ARTICLE_DT_ACCURACY:.1%}")

    print(f"Guardado: {csv_path}")

    return df_cv





def guardar_matriz_confusion(y_true, y_pred, titulo: str, archivo: Path):

    """Guarda la matriz de confusión como imagen PNG en ``archivo``.

    Args:

        y_true: Etiquetas verdaderas.

        y_pred: Etiquetas predichas.

        titulo: Título de la figura.

        archivo: Ruta del PNG de salida.

    """

    labels = sorted(pd.Series(y_true).unique())

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(9, 8))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

    ax.set_title(titulo)

    fig.tight_layout()

    fig.savefig(archivo, dpi=160)

    plt.close(fig)

    print(f"Guardada: {archivo}")


def evaluar_holdout(nombre, X_train, y_train, X_test, y_test, estimator, archivo_cm=None):

    """Entrena en train completo y evalúa una sola vez en el hold-out test.



    Args:

        nombre: Etiqueta del escenario.

        X_train: Características de entrenamiento.

        y_train: Etiquetas de entrenamiento.

        X_test: Características de prueba (no usadas antes).

        y_test: Etiquetas de prueba.

        estimator: Modelo o pipeline a evaluar.

        archivo_cm: Ruta opcional para guardar la matriz de confusión como PNG.



    Returns:

        Accuracy en el conjunto de prueba.

    """

    est = clone(estimator)

    est.fit(X_train, y_train)

    y_pred = est.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== {nombre} — Hold-out test ===")

    print(f"Accuracy test: {acc:.3f}")

    print(classification_report(y_test, y_pred))

    print("Matriz de confusión (test):")

    print(confusion_matrix(y_test, y_pred))

    if archivo_cm is not None:

        guardar_matriz_confusion(

            y_test,

            y_pred,

            f"{nombre} — Hold-out test",

            archivo_cm,

        )

    return acc





def reporte_oof(nombre, X, y, estimator, archivo_cm=None):

    """Genera reporte de clasificación con predicciones out-of-fold (5-fold).



    Evita leakage al predecir cada muestra solo con modelos entrenados

    en los otros folds.



    Args:

        nombre: Etiqueta del escenario.

        X: Características.

        y: Etiquetas verdaderas.

        estimator: Modelo o pipeline a evaluar.

        archivo_cm: Ruta opcional para guardar la matriz de confusión como PNG.

    """

    cv = StratifiedKFold(

        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE

    )

    y_pred = cross_val_predict(estimator, X, y, cv=cv)

    print(f"\n{nombre} — Reporte (predicción OOF 5-fold):")

    print(classification_report(y, y_pred))

    print("Matriz de confusión:")

    print(confusion_matrix(y, y_pred))

    if archivo_cm is not None:

        guardar_matriz_confusion(

            y,

            y_pred,

            f"{nombre} — Predicción OOF (5-fold)",

            archivo_cm,

        )





def plot_loss_curve(estimator, X, y, titulo, archivo):

    """Reentrena un clon del estimador y guarda la curva de pérdida del MLP.

    Enfoca el eje X solo en las iteraciones ejecutadas (zoom) y, si hay early
    stopping, separa pérdida y accuracy en dos subplots compartiendo escala X.

    Args:

        estimator: Pipeline con paso ``mlp`` (MLPClassifier).

        X: Características de entrenamiento.

        y: Etiquetas.

        titulo: Título de la figura.

        archivo: Ruta del archivo PNG de salida.

    """

    est = clone(estimator)

    est.fit(X, y)

    mlp = est.named_steps["mlp"]

    curve = np.asarray(mlp.loss_curve_, dtype=float)

    n_iter = mlp.n_iter_

    max_iter = mlp.max_iter

    x = np.arange(1, len(curve) + 1)

    val_scores = getattr(mlp, "validation_scores_", None)

    has_val = (

        mlp.early_stopping

        and val_scores is not None

        and len(val_scores) > 0

    )

    pad = max(4, int(0.1 * n_iter))

    x_end = n_iter + pad

    color_loss = "#1d4ed8"

    color_val = "#c2410c"

    color_stop = "#b91c1c"

    color_best = "#15803d"

    if has_val:

        val_scores = np.asarray(val_scores, dtype=float)

        val_x = np.arange(1, len(val_scores) + 1)

        fig, (ax_loss, ax_val) = plt.subplots(

            2,

            1,

            sharex=True,

            figsize=(10, 7),

            gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.22},

        )

    else:

        fig, ax_loss = plt.subplots(figsize=(10, 4.5))

        ax_val = None

    ax_loss.plot(

        x,

        curve,

        color=color_loss,

        linewidth=2.2,

        label="Pérdida (entrenamiento)",

    )

    ax_loss.scatter(

        [n_iter],

        [curve[-1]],

        color=color_loss,

        s=55,

        zorder=5,

        edgecolors="white",

        linewidths=1.2,

    )

    ax_loss.set_ylabel("Pérdida")

    loss_lo, loss_hi = curve.min(), curve.max()

    loss_margin = max((loss_hi - loss_lo) * 0.12, 0.05)

    ax_loss.set_ylim(loss_lo - loss_margin, loss_hi + loss_margin)

    if ax_val is not None:

        ax_val.plot(

            val_x,

            val_scores,

            color=color_val,

            linewidth=2.2,

            label="Accuracy (validación)",

        )

        ax_val.scatter(

            [n_iter],

            [val_scores[-1]],

            color=color_val,

            s=55,

            zorder=5,

            edgecolors="white",

            linewidths=1.2,

        )

        best_iter = int(np.argmax(val_scores)) + 1

        ax_val.scatter(

            [best_iter],

            [val_scores[best_iter - 1]],

            color=color_best,

            s=70,

            marker="*",

            zorder=6,

            label=f"Mejor validación (iter. {best_iter})",

        )

        ax_val.set_ylabel("Accuracy validación")

        val_lo, val_hi = val_scores.min(), val_scores.max()

        val_margin = max((val_hi - val_lo) * 0.15, 0.03)

        ax_val.set_ylim(val_lo - val_margin, val_hi + val_margin)

    axes = [ax_loss] if ax_val is None else [ax_loss, ax_val]

    stop_label = f"Parada anticipada (iter. {n_iter})"

    for ax in axes:

        ax.axvline(

            n_iter,

            color=color_stop,

            linestyle="--",

            linewidth=2,

            label=stop_label,

        )

        ax.set_xlim(0.5, x_end)

        ax.grid(True, linestyle=":", alpha=0.45)

        ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    axes[-1].set_xlabel("Iteración")

    subtitle = (

        f"Early stopping: {n_iter} de {max_iter} iteraciones máx."

        if mlp.early_stopping and n_iter < max_iter

        else f"Entrenamiento completo: {n_iter} iteraciones"

    )

    fig.suptitle(f"{titulo}\n{subtitle}", fontsize=11, fontweight="bold", y=0.98)

    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.96)

    fig.savefig(archivo, dpi=160)

    plt.close(fig)

    stop_msg = (

        f"parada anticipada en iter. {n_iter}/{max_iter}"

        if mlp.early_stopping and n_iter < max_iter

        else f"completó {n_iter}/{max_iter} iteraciones"

    )

    print(f"Guardada: {archivo} ({stop_msg})")





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



    # ----- Hold-out estratificado (80/20) -----

    print("\n" + "=" * 60)

    print("HOLD-OUT — Separación train / test")

    print("=" * 60)



    X_train, X_test, y_train, y_test = train_test_split(

        X_full,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y,

    )

    print(f"Hold-out: train={len(X_train)} | test={len(X_test)}")

    print("Distribución train:\n", y_train.value_counts().sort_index())

    print("Distribución test:\n", y_test.value_counts().sort_index())



    fig_corr_ini = FIGURES_DIR / "fig_corr_inicial.png"

    plt.figure(figsize=(12, 10))

    sns.heatmap(X_train.corr(), cmap="coolwarm", center=0)

    plt.title("Matriz de correlación — train (sin Longitud)")

    plt.tight_layout()

    plt.savefig(fig_corr_ini)

    plt.close()

    print(f"Guardada: {fig_corr_ini}")



    # ----- Fase 3: Reducción por correlación (solo train) -----

    print("\n" + "=" * 60)

    print("FASE 3 — Análisis de correlación y reducción (solo train)")

    print("=" * 60)



    X_train_red, dropped = reducir_por_correlacion(X_train, CORR_THRESHOLD)

    X_test_red = X_test.drop(columns=dropped, errors="ignore")

    print(f"Umbral |r| > {CORR_THRESHOLD}")

    print(f"Características eliminadas ({len(dropped)}): {dropped}")

    print(f"Características restantes (train): {X_train_red.shape[1]}")



    dataset_reducido = RESULTS_DIR / "DataSet_reducido.xlsx"

    pd.concat([X_train_red, y_train], axis=1).to_excel(dataset_reducido, index=False)

    print(f"Guardado: {dataset_reducido} (train reducido + Clase; test no incluido)")



    fig_corr_red = FIGURES_DIR / "fig_corr_reducido.png"

    plt.figure(figsize=(10, 8))

    sns.heatmap(X_train_red.corr(), cmap="coolwarm", center=0)

    plt.title("Matriz de correlación — train reducido")

    plt.tight_layout()

    plt.savefig(fig_corr_red)

    plt.close()

    print(f"Guardada: {fig_corr_red}")



    # ----- Fase 2: MLP con todas las features -----

    print("\n" + "=" * 60)

    print("FASE 2 — MLP con todas las características (sin Longitud)")

    print("=" * 60)



    best_mlp = crear_gridsearch_mlp(X_train, y_train)

    cv_fase2 = validacion_cruzada_estratificada(

        "Fase 2 — features completas",

        X_train,

        y_train,

        best_mlp,

        RESULTS_DIR / "cv_fase2_train.csv",

    )

    reporte_oof(

        "Fase 2",

        X_train,

        y_train,

        best_mlp,

        FIGURES_DIR / "fig_cm_fase2_oof.png",

    )

    acc_test_fase2 = evaluar_holdout(

        "Fase 2 — features completas",

        X_train,

        y_train,

        X_test,

        y_test,

        best_mlp,

        FIGURES_DIR / "fig_cm_fase2_test.png",

    )



    # ----- Fase 4: MLP con dataset reducido -----

    print("\n" + "=" * 60)

    print("FASE 4 — MLP con características reducidas (mismos hiperparámetros)")

    print("=" * 60)



    best_mlp_red = clone(best_mlp)

    cv_fase4 = validacion_cruzada_estratificada(

        "Fase 4 — features reducidas",

        X_train_red,

        y_train,

        best_mlp_red,

        RESULTS_DIR / "cv_fase4_train.csv",

    )

    reporte_oof(

        "Fase 4",

        X_train_red,

        y_train,

        best_mlp_red,

        FIGURES_DIR / "fig_cm_fase4_oof.png",

    )

    acc_test_fase4 = evaluar_holdout(

        "Fase 4 — features reducidas",

        X_train_red,

        y_train,

        X_test_red,

        y_test,

        best_mlp_red,

        FIGURES_DIR / "fig_cm_fase4_test.png",

    )



    plot_loss_curve(

        best_mlp,

        X_train,

        y_train,

        "Curva de pérdida — Fase 2 (features completas)",

        FIGURES_DIR / "fig_loss_fase2.png",

    )

    plot_loss_curve(

        best_mlp_red,

        X_train_red,

        y_train,

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

        "accuracy_cv_media": [

            cv_fase2["accuracy"].mean(),

            cv_fase4["accuracy"].mean(),

            ARTICLE_DT_ACCURACY,

        ],

        "accuracy_cv_std": [

            cv_fase2["accuracy"].std(),

            cv_fase4["accuracy"].std(),

            np.nan,

        ],

        "accuracy_test": [

            acc_test_fase2,

            acc_test_fase4,

            np.nan,

        ],

        "f1_macro_cv_media": [

            cv_fase2["f1_macro"].mean(),

            cv_fase4["f1_macro"].mean(),

            np.nan,

        ],

        "n_features": [X_train.shape[1], X_train_red.shape[1], 19],

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


