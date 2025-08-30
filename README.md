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
```

##  Ejemplo rápido mejorado: importación de datos satelitales + ERA5

En este ejemplo mostramos cómo:
1. **Leer datos satelitales** desde MODIS, Landsat 8-9, Sentinel-2, PlanetScope y fuentes climáticas como ERA5.
2. Construir un cubo `(T, B, Y, X)` con bandas y fechas/GDD.
3. Ajustar modelos y generar mapas de rendimiento predicho.

```python
import numpy as np
import rasterio
from itertools import combinations
from module_name import fit_many, best_by_metric, predict_map, ModelSpec

# --- Función auxiliar para apilar escenas satelitales en cubo -------- #
def load_scenes(file_list, bands_idx):
    """
    Lee múltiples escenas satelitales y retorna un cubo (T, B, Y, X).

    file_list : lista de escenas; cada escena es lista de rutas a bandas
    bands_idx : índices de bandas a usar
    
    Retorna: np.ndarray con forma (T, B, Y, X)
    """
    scenes = []
    for scene_paths in file_list:
        bands = []
        for b in bands_idx:
            with rasterio.open(scene_paths[b]) as src:
                bands.append(src.read(1).astype(np.float32))
        scenes.append(np.stack(bands, axis=0))  # (B, Y, X)
    return np.stack(scenes, axis=0)  # (T, B, Y, X)

#  ── 1) MODIS — producto MOD09GA (HDF-EOS2, bandas 1-7) ─────────────── #
# Se obtiene en formato HDF-EOS2 (MODIS Surface Reflectance L2G) :contentReference[oaicite:1]{index=1}
modis_files = [
    ["MOD09GA_date1_B1.hdf", "MOD09GA_date1_B2.hdf"],
    # ...
]
cube_modis = load_scenes(modis_files, bands_idx=[0, 1])

#  ── 2) Landsat 8-9 (GeoTIFF) ──────────────────────────────────────── #
landsat_files = [
    ["LC08_date1_B4.tif", "LC08_date1_B5.tif"],
    # ...
]
cube_landsat = load_scenes(landsat_files, bands_idx=[0, 1])

#  ── 3) Sentinel-2 SAFE o .jp2 (JPEG2000), nivel L2A ──────────────── #
# Puedes abrir directamente subdatasets JP2 con rasterio :contentReference[oaicite:2]{index=2}
s2_files = [
    ["S2_date1_B4.jp2", "S2_date1_B8.jp2"],
    # ...
]
cube_s2 = load_scenes(s2_files, bands_idx=[0, 1])

#  ── 4) PlanetScope (GeoTIFF o archivos específicos, según proveedor) ─ #
ps_files = [
    ["PS_date1_B3.tif", "PS_date1_B4.tif"],
    # ...
]
cube_ps = load_scenes(ps_files, bands_idx=[0, 1])

#  ── 5) ERA5 (reanalisis climático, formato NetCDF) ─────────────────── #
import xarray as xr
# Ejemplo: lectura de una variable de ERA5 desde archivos NetCDF
ds = xr.open_mfdataset(["era5_var1.nc", "era5_var2.nc"])  # lee múltiples archivos :contentReference[oaicite:3]{index=3}
# Selección de variable y regiones (opcional)
da = ds['temperature']  # por ejemplo
# Para usar como banda o índice, podrías convertirlo a numpy y reescalarlo:
cube_era5 = da.values  # forma (T, Y, X) o más dimensiones según variables

#  ── 6) Concatenar todos los cubos (asumir misma resolución/área) ──── #
# Si las resoluciones difieren, reproyectar y remuestrear antes de concatenar
cube_all = np.concatenate([cube_modis, cube_landsat, cube_s2, cube_ps], axis=0)

#  ── 7) Cargar mapa de rendimiento (observado) ──────────────────────── #
with rasterio.open("yield_map.tif") as src:
    target = src.read(1).astype(np.float32)

#  ── 8) Ajustar modelo lineal (todas las combinaciones de 2 bandas, GDD índice 0) ─ #
spec = ModelSpec(model_type="linear", standardize=True, kfold_splits=5, scoring="r2")
results = fit_many(
    cube=cube_all,
    target=target,
    gdd_indices=[0],
    band_subsets=combinations(range(cube_all.shape[1]), 2),
    spec=spec
)

#  ── 9) Seleccionar y predecir ──────────────────────────────────────── #
best = best_by_metric(results, key="cv_score_mean", maximize=True)
y_hat = predict_map(cube_all, best)
print("Mejor subset:", best.bands_idx, " | CV R²:", best.cv_score_mean)
print("Mapa predicho con shape:", y_hat.shape)
