# EO_yield_predictor
Repositorio para predicción de rendimiento con datos de observación terrestre

# Modelos lineales multibanda basados en GDD para predicción de rendimiento

Este módulo permite ajustar **modelos lineales, Ridge o Lasso** sobre datos
multibanda en una dimensión temporal **GDD (Growing Degree Days)** para predecir
rendimiento agrícola. Es totalmente **genérico**: puede trabajar con **1..N bandas**
y múltiples puntos en el tiempo.

---

## 📂 Estructura de datos

- **Cubo de entrada:**  
  `cube` con forma `(T, B, Y, X)`  
  - `T`: índices temporales o GDD.
  - `B`: bandas espectrales.
  - `Y, X`: dimensiones espaciales (pixeles).

- **Rendimiento:**  
  `target` con forma `(Y, X)`, alineado espacialmente con el cubo.

---

## 🚀 Funcionalidades principales

### 1. `fit_at_gdd(cube, target, gdd_index, bands_idx, spec)`
Ajusta un modelo en **un GDD específico** usando un subconjunto de bandas.

- Manejo automático de `NaN`s.
- Estandarización opcional de variables.
- Validación cruzada K-Fold.
- Búsqueda automática de `alpha` para Ridge/Lasso.

**Retorna:**  
`FitResult` con coeficientes, métricas de validación y estadísticas de estandarización.

---

### 2. `predict_map(cube, fit)`
Genera un **mapa de rendimiento predicho** para el mismo GDD donde se ajustó el modelo.

---

### 3. `fit_many(cube, target, gdd_indices, band_subsets)`
Permite barrer:
- Múltiples puntos GDD.
- Subconjuntos arbitrarios de bandas (e.g., todas las combinaciones de 2 bandas).

---

### 4. `best_by_metric(results, key="cv_score_mean")`
Selecciona el mejor modelo según una métrica (`cv_score_mean`, `cv_score_std`, etc.).

---

## ⚙️ Parámetros del modelo (`ModelSpec`)

| Parámetro        | Descripción                                  | Por defecto        |
|------------------|----------------------------------------------|--------------------|
| `model_type`      | `"linear"`, `"ridge"`, `"lasso"`             | `"linear"`         |
| `alpha`           | Regularización para Ridge/Lasso              | `None` (grid search)|
| `alpha_grid`      | Valores a explorar si `alpha=None`           | `[1e-3,...,100]`    |
| `standardize`     | Estandarizar variables (media=0, var=1)       | `True`             |
| `kfold_splits`    | Número de folds en K-Fold CV                 | `5`                |
| `scoring`         | Métrica sklearn (`"r2"`, `"neg_rmse"`, etc.) | `"r2"`             |

---

## 🧪 Ejemplo rápido

```python
from itertools import combinations
import numpy as np
from module_name import fit_many, best_by_metric, predict_map, ModelSpec

# Datos sintéticos
T, B, Y, X = 5, 4, 50, 60
cube = np.random.randn(T, B, Y, X).astype(np.float32)
target = np.random.randn(Y, X).astype(np.float32)

# Configuración del modelo
spec = ModelSpec(model_type="linear", standardize=True, kfold_splits=5)

# Ajustar todas las combinaciones de 2 bandas en GDD=3
results = fit_many(
    cube, target,
    gdd_indices=[3],
    band_subsets=combinations(range(B), 2),
    spec=spec
)

# Seleccionar mejor modelo
best = best_by_metric(results, key="cv_score_mean")
y_hat = predict_map(cube, best)
