# Evaluación de modelos MLP para clasificación EOG

Proyecto de la materia **Minería de Datos** orientado a la evaluación y comparación de modelos de red neuronal (MLP) para la clasificación de señales EOG en el contexto del lenguaje *Blink-to-Speak*. El pipeline implementa preprocesamiento, hold-out estratificado, búsqueda de hiperparámetros con validación cruzada, reducción de características por correlación (solo en train) y comparación con la referencia del artículo base (árbol de decisión ~83.4 % de accuracy).

Repositorio: [github.com/Cracksmity/dm-thesis-evaluation](https://github.com/Cracksmity/dm-thesis-evaluation)

## Estructura del repositorio

```
dm-thesis-evaluation/
├── data/                    # Datos de entrada
│   └── DataSet_completo.xlsx
├── src/                     # Código principal del proyecto
│   └── pruebaK_fold.py      # Pipeline completo (script principal)
├── results/                 # Salidas del pipeline
│   ├── figures/             # Gráficas generadas
│   ├── DataSet_reducido.xlsx
│   ├── cv_fase2_train.csv
│   ├── cv_fase4_train.csv
│   └── comparacion_resultados.csv
├── examples/                # Scripts opcionales de práctica
│   └── RedNeuronal.py       # Demo MLP con Iris (no es el pipeline EOG)
├── docs/
│   └── REFERENCIAS.md       # Enlaces a PDFs e instrucciones externas
├── requirements.txt
└── README.md
```

| Carpeta / archivo | Contenido |
|-------------------|-----------|
| `data/` | Dataset original (200 muestras, 10 clases, 20 por clase) con características EOG y columna `Clase`. |
| `src/pruebaK_fold.py` | Script principal: fases 1–4, hold-out, CV, gráficas y tablas de resultados. |
| `results/` | Artefactos generados al ejecutar el pipeline (figuras, CSV, Excel reducido). |
| `examples/` | Ejemplos didácticos independientes del flujo de evaluación. |
| `docs/` | Referencias bibliográficas y documentos del curso (enlaces externos). |

## Requisitos previos

- **Python** 3.10 o superior
- **Git**
- Conexión a internet para clonar e instalar dependencias

## Configuración

Clona el repositorio y prepara el entorno virtual desde la raíz del proyecto:

```bash
git clone https://github.com/Cracksmity/dm-thesis-evaluation.git
cd dm-thesis-evaluation
python -m venv venv
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si PowerShell bloquea la activación del entorno, usa una de estas alternativas:

```powershell
# Opción 1: CMD desde la raíz del proyecto
venv\Scripts\activate.bat

# Opción 2: permitir scripts locales (una sola vez)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux / macOS:**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Uso

Con el entorno virtual activado, ejecuta el script principal desde la raíz del repositorio:

```bash
python src/pruebaK_fold.py
```

### Qué hace el script

1. **Fase 1 — Preprocesamiento:** carga `data/DataSet_completo.xlsx`, explora la distribución de clases y guarda `results/figures/fig_clases.png`.
2. **Hold-out estratificado:** separación **80 % train / 20 % test** (`TEST_SIZE=0.2`). El test (4 muestras por clase) **no** se usa para entrenar, elegir hiperparámetros ni calcular correlaciones.
3. **Fase 3 — Reducción:** elimina características con correlación \|r\| > 0.8 calculada **solo en train**; genera `results/DataSet_reducido.xlsx` y mapas de correlación.
4. **Fase 2 — MLP con todas las features (sin `Longitud`):** `GridSearchCV` con `StratifiedKFold` (5-fold interno), validación cruzada en train, reporte OOF, evaluación hold-out en test y gráficas de pérdida / matriz de confusión.
5. **Fase 4 — MLP con features reducidas:** mismos hiperparámetros que Fase 2; misma evaluación (CV train, OOF, test).
6. **Comparación:** tabla resumen en consola y `results/comparacion_resultados.csv` (accuracy CV, accuracy test, F1, número de features).

### Configuración del MLP

Parámetros definidos al inicio de `src/pruebaK_fold.py`:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `MAX_ITER` | 800 | Máximo de iteraciones de entrenamiento |
| `EARLY_STOPPING` | `True` | Parada anticipada si la validación interna deja de mejorar |
| `N_ITER_NO_CHANGE` | 30 | Iteraciones sin mejora antes de parar |
| `VALIDATION_FRACTION` | 0.1 | 10 % del conjunto usado en cada `fit()` para early stopping |
| `N_SPLITS` | 5 | Folds en validación cruzada estratificada |
| `CORR_THRESHOLD` | 0.8 | Umbral de correlación para eliminar features |

La rejilla de `GridSearchCV` explora tamaños de capas ocultas, `alpha` y tasa de aprendizaje inicial.

### Archivos generados

#### Tablas y datos (`results/`)

| Archivo | Descripción |
|---------|-------------|
| `DataSet_reducido.xlsx` | Train tras reducción por correlación (sin test). |
| `cv_fase2_train.csv` | Accuracy y F1 macro por fold (Fase 2, train). |
| `cv_fase4_train.csv` | Igual para Fase 4 (features reducidas). |
| `comparacion_resultados.csv` | Resumen Fase 2 vs Fase 4 vs referencia del artículo. |

#### Gráficas (`results/figures/`)

| Archivo | Qué muestra |
|---------|-------------|
| `fig_clases.png` | Barras: cuántas muestras hay por clase en el dataset completo. |
| `fig_corr_inicial.png` | Mapa de calor de correlación entre features (**solo train**, sin `Longitud`). |
| `fig_corr_reducido.png` | Correlación tras eliminar columnas redundantes (train reducido). |
| `fig_loss_fase2.png` | Curva de pérdida (train) y accuracy de validación interna; línea de early stopping (Fase 2). |
| `fig_loss_fase4.png` | Igual para Fase 4 (features reducidas). |
| `fig_cm_fase2_oof.png` | Matriz de confusión con predicciones OOF en train (Fase 2). |
| `fig_cm_fase2_test.png` | Matriz de confusión en el **hold-out test** (Fase 2). |
| `fig_cm_fase4_oof.png` | Matriz de confusión OOF en train (Fase 4). |
| `fig_cm_fase4_test.png` | Matriz de confusión en test (Fase 4). |

**Cómo leer las matrices de confusión (test):** hay 10 clases y **4 muestras por clase** en test (20 % de 20). La **diagonal** indica aciertos; cada fila suma 4. Un **4** en la diagonal de una clase significa que se clasificaron correctamente los 4 casos de esa clase en test.

**Cómo leer las curvas de pérdida:** el panel superior es la pérdida en entrenamiento; el inferior, el accuracy en la validación interna del early stopping. La línea roja marca la iteración donde paró el entrenamiento.

### Resultados esperados en consola

- Mejores hiperparámetros del `GridSearchCV`.
- Métricas por fold (`Fold 1` … `Fold 5`) y medias de accuracy / F1.
- Reportes de clasificación y matrices numéricas (OOF y test).
- Tabla comparativa final.

La ejecución completa puede tardar **varios minutos** por la rejilla de hiperparámetros y la validación cruzada (`n_jobs=-1` usa todos los núcleos disponibles).

Puedes revisar los resultados ya versionados en `results/` sin volver a ejecutar el script.

### Script opcional

```bash
python examples/RedNeuronal.py
```

Entrenamiento rápido de un MLP con el dataset **Iris** (solo práctica de redes neuronales con scikit-learn).

## Documentación de referencia

Los PDF del artículo, la presentación y las instrucciones del proyecto **no** están en el repositorio. Consulta [docs/REFERENCIAS.md](docs/REFERENCIAS.md) para los enlaces compartidos del equipo.

## Colaboración

### Ramas

Crea una rama por tarea a partir de `main`:

```bash
git checkout main
git pull origin main
git checkout -b feature/nombre-tarea
```

Convención sugerida:

| Prefijo | Uso |
|---------|-----|
| `feature/` | Nueva funcionalidad o experimento |
| `fix/` | Corrección de errores |
| `docs/` | Solo documentación |
| `chore/` | Mantenimiento, dependencias, estructura |

### Commits

Usa mensajes claros en estilo [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: integrar optimización PSO para hiperparámetros
fix: corregir lectura de columnas en el dataset
docs: actualizar instrucciones de instalación en README
chore: actualizar dependencias en requirements.txt
```

### Flujo recomendado

1. Rama desde `main` → cambios → commit(s).
2. Push de la rama y apertura de **Pull Request** en GitHub.
3. Revisión por un compañero → merge a `main`.

## Licencia y autoría

Proyecto académico del equipo de Minería de Datos. Ajusta esta sección si el curso exige una licencia o créditos específicos.
