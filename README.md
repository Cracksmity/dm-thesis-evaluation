# Evaluación de modelos MLP para clasificación EOG

Proyecto de la materia **Minería de Datos** orientado a la evaluación y comparación de modelos de red neuronal (MLP) para la clasificación de señales EOG en el contexto del lenguaje *Blink-to-Speak*. El pipeline implementa preprocesamiento, búsqueda de hiperparámetros con validación cruzada, reducción de características por correlación y comparación con la referencia del artículo base (árbol de decisión ~83.4 % de accuracy).

Repositorio: [github.com/Cracksmity/dm-thesis-evaluation](https://github.com/Cracksmity/dm-thesis-evaluation)

## Estructura del repositorio

```
dm-thesis-evaluation/
├── data/                    # Datos de entrada
│   └── DataSet_completo.xlsx
├── src/                     # Código principal del proyecto
│   └── pruebaK_fold.py
├── results/                 # Salidas del pipeline (referencia versionada)
│   ├── figures/             # Gráficas (distribución, correlación, pérdida)
│   ├── DataSet_reducido.xlsx
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
| `data/` | Dataset original con características EOG y columna `Clase`. |
| `src/` | Script principal con las fases 1–4 y validación cruzada. |
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

1. **Fase 1 — Preprocesamiento:** carga `data/DataSet_completo.xlsx`, explora clases y guarda gráficas en `results/figures/`.
2. **Fase 3 — Reducción:** elimina características con correlación \|r\| > 0.8 y genera `results/DataSet_reducido.xlsx`.
3. **Fase 2 — MLP completo:** `GridSearchCV` (5-fold interno) + evaluación con `RepeatedStratifiedKFold` (5 × 10).
4. **Fase 4 — MLP reducido:** mismos hiperparámetros sobre el dataset reducido.
5. **Comparación:** tabla resumen en consola y `results/comparacion_resultados.csv`.

### Resultados esperados

- Salida detallada en consola: hiperparámetros óptimos, accuracy/F1 con media y desviación, reportes OOF y matrices de confusión.
- Archivos actualizados en `results/` y `results/figures/`.
- La ejecución completa puede tardar **varios minutos** por la rejilla de hiperparámetros y la validación cruzada repetida (`n_jobs=-1` usa todos los núcleos disponibles).

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
