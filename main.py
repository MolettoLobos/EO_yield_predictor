"""
GDD-based multiband linear models for yield prediction (generic, 1..∞ bands)
-----------------------------------------------------------------------------
Author: Italo Moletto-Lobos

This module lets you:
  1) Feed an EO data cube shaped as (T, B, Y, X) where T indexes GDD (or time),
     B are spectral bands (>=1), and (Y, X) are spatial pixels.
  2) Provide a yield target raster shaped as (Y, X) aligned to the cube.
  3) Fit linear (or Ridge/Lasso) models at a *specific* GDD slice using any
     subset of bands (1..N) and evaluate via KFold CV.
  4) Predict a full-resolution yield map for that GDD slice.

Design choices (kept explicit and documented):
- The model is *global-in-space at a fixed GDD*: we pool all valid pixels
  (Y,X) from the chosen GDD slice to learn a single set of coefficients.
- Features are the raw band reflectances (or indices you computed beforehand),
  one column per band in the selected subset.
- Targets are per-pixel yields (same grid, same CRS/extent/resolution).
- NaNs are handled by masking rows with any NaN in features or target.
- Optionally standardize X (mean=0, std=1); y is left unscaled by default.

If your yield target is *region-aggregated* (e.g., one value per polygon),
this pixelwise learning setup is not directly applicable. See the TODO note at
`fit_at_gdd()` for a precise extension using zonal statistics to aggregate X.

Dependencies:
    numpy, scikit-learn (no hard dependency on xarray, but supported I/O)

Example usage is at the bottom of this file under `if __name__ == "__main__":`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any, Union

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ArrayLike4D = Union[np.ndarray]
ArrayLike2D = Union[np.ndarray]

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


@dataclass
class ModelSpec:
    """Holds all knobs for model fitting and evaluation.

    Attributes
    ----------
    model_type : str
        One of {"linear", "ridge", "lasso"}.
    alpha : Optional[float]
        Regularization strength for ridge/lasso. If None and model is ridge/lasso,
        an internal grid search via cross-validation will be performed over `alpha_grid`.
    alpha_grid : Sequence[float]
        Grid of alpha values to try when alpha is None (ridge/lasso only).
    standardize : bool
        Whether to standardize X columns (mean=0, std=1) prior to fitting.
    kfold_splits : int
        Number of folds for KFold CV.
    random_state : int
        Random seed for KFold shuffling (for reproducibility).
    shuffle : bool
        Whether to shuffle in KFold.
    scoring : str
        sklearn scoring string, e.g., "r2" (default) or "neg_root_mean_squared_error".
    """
    model_type: str = "linear"
    alpha: Optional[float] = None
    alpha_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1, 10, 100)
    standardize: bool = True
    kfold_splits: int = 5
    random_state: int = 42
    shuffle: bool = True
    scoring: str = "r2"


@dataclass
class FitResult:
    """Container for fitted model artifacts and diagnostics."""
    bands_idx: Tuple[int, ...]
    gdd_index: int
    coef_: np.ndarray
    intercept_: float
    cv_scores: np.ndarray
    cv_score_mean: float
    cv_score_std: float
    model_type: str
    alpha: Optional[float]
    feature_means: Optional[np.ndarray]
    feature_stds: Optional[np.ndarray]


# -------------------------- core utilities -------------------------- #

def _ensure_4d(arr: ArrayLike4D) -> np.ndarray:
    """Ensure a numpy array with ndim==4 (T, B, Y, X).

    Accepted inputs:
        - np.ndarray with shape (T, B, Y, X)
    """
    a = np.asarray(arr)
    if a.ndim != 4:
        raise ValueError(f"Expected (T, B, Y, X), got shape {a.shape}")
    return a


def _ensure_2d(arr: ArrayLike2D) -> np.ndarray:
    """Ensure a numpy array with ndim==2 (Y, X)."""
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"Expected (Y, X) for target, got shape {a.shape}")
    return a


def _slice_time(cube: np.ndarray, gdd_index: int) -> np.ndarray:
    """Return the slice at a given GDD/time index: shape (B, Y, X)."""
    if not (0 <= gdd_index < cube.shape[0]):
        raise IndexError(f"gdd_index {gdd_index} is out of range [0, {cube.shape[0]-1}]")
    return cube[gdd_index]  # (B, Y, X)


def _stack_features(bands_slice: np.ndarray, bands_idx: Sequence[int]) -> np.ndarray:
    """Stack selected bands into (N_pixels, N_features).

    Parameters
    ----------
    bands_slice : np.ndarray
        Array with shape (B, Y, X) at a single GDD/time.
    bands_idx : Sequence[int]
        Indices of bands to use as features.

    Returns
    -------
    X : np.ndarray
        Array with shape (Y*X, len(bands_idx)).
    mask_valid : np.ndarray (bool)
        Flattened boolean mask of valid rows (no NaNs across selected bands).
    """
    B, Y, X = bands_slice.shape
    chosen = np.stack([bands_slice[i] for i in bands_idx], axis=0)  # (k, Y, X)
    Xmat = chosen.reshape(len(bands_idx), -1).T  # (Y*X, k)
    mask_valid = ~np.isnan(Xmat).any(axis=1)
    return Xmat, mask_valid


def _prep_target(target: np.ndarray) -> np.ndarray:
    """Flatten target (Y, X) -> (Y*X,), keep NaNs for masking alignment."""
    return target.reshape(-1)


def _standardize_train(X: np.ndarray, standardize: bool) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if not standardize:
        return X, None, None
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    return Xs, scaler.mean_.copy(), scaler.scale_.copy()


def _standardize_apply(X: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    if mean is None or std is None:
        return X
    # Avoid division by zero
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe


def _mk_model(spec: ModelSpec, alpha_override: Optional[float] = None):
    mtype = spec.model_type.lower()
    alpha = spec.alpha if alpha_override is None else alpha_override
    if mtype == "linear":
        return LinearRegression(n_jobs=None, fit_intercept=True)
    elif mtype == "ridge":
        if alpha is None:
            # Will be set in outer grid-search loop
            return Ridge(fit_intercept=True)
        return Ridge(alpha=float(alpha), fit_intercept=True)
    elif mtype == "lasso":
        if alpha is None:
            return Lasso(fit_intercept=True, max_iter=10000)
        return Lasso(alpha=float(alpha), fit_intercept=True, max_iter=10000)
    else:
        raise ValueError("model_type must be one of {'linear','ridge','lasso'}")


# -------------------------- public API -------------------------- #

def fit_at_gdd(
    cube: ArrayLike4D,
    target: ArrayLike2D,
    gdd_index: int,
    bands_idx: Sequence[int],
    spec: Optional[ModelSpec] = None,
) -> FitResult:
    """Fit a global-in-space linear model at a specific GDD index.

    Parameters
    ----------
    cube : (T, B, Y, X) array
        EO cube with time/GDD dimension first.
    target : (Y, X) array
        Yield values aligned to cube's spatial grid.
    gdd_index : int
        Index along T to select the time/GDD slice.
    bands_idx : Sequence[int]
        Band indices (0-based) to use as features.
    spec : Optional[ModelSpec]
        Model/validation configuration.

    Returns
    -------
    FitResult
        Coefficients, CV metrics, and standardization stats for later prediction.

    Notes
    -----
    - If your ground truth is region-level (one value per polygon), extend this by
      aggregating pixels inside each region to region-level features, e.g.,
      mean/median over (Y,X) within each polygon, then fit on regions instead of pixels.
      # TODO(placeholder): If needed, add a `regions: GeoDataFrame` and an
      # aggregation function to compute region-level X and y.
    """
    spec = spec or ModelSpec()

    cube = _ensure_4d(cube)
    target = _ensure_2d(target)

    bands_slice = _slice_time(cube, gdd_index)  # (B, Y, X)
    Xmat, maskX = _stack_features(bands_slice, bands_idx)
    yvec = _prep_target(target)

    mask = maskX & ~np.isnan(yvec)
    X = Xmat[mask]
    y = yvec[mask]

    if X.size == 0:
        raise ValueError("No valid samples after masking NaNs in X or y.")

    # Standardize if requested
    Xs, mu, sigma = _standardize_train(X, spec.standardize)

    # Choose model (with optional alpha grid-search for ridge/lasso)
    if spec.model_type in {"ridge", "lasso"} and spec.alpha is None:
        best_alpha = None
        best_score = -np.inf
        for a in spec.alpha_grid:
            model = _mk_model(spec, alpha_override=a)
            kf = KFold(n_splits=spec.kfold_splits, shuffle=spec.shuffle, random_state=spec.random_state)
            scores = cross_val_score(model, Xs, y, cv=kf, scoring=spec.scoring, n_jobs=None)
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score, best_alpha = mean_score, a
        # Refit final model on all data with best alpha
        final_model = _mk_model(spec, alpha_override=best_alpha)
        final_model.fit(Xs, y)
        # CV again for reporting with chosen alpha
        kf = KFold(n_splits=spec.kfold_splits, shuffle=spec.shuffle, random_state=spec.random_state)
        cv_scores = cross_val_score(final_model, Xs, y, cv=kf, scoring=spec.scoring, n_jobs=None)
        alpha_used = float(best_alpha) if best_alpha is not None else None
    else:
        model = _mk_model(spec)
        kf = KFold(n_splits=spec.kfold_splits, shuffle=spec.shuffle, random_state=spec.random_state)
        cv_scores = cross_val_score(model, Xs, y, cv=kf, scoring=spec.scoring, n_jobs=None)
        model.fit(Xs, y)
        final_model = model
        alpha_used = float(spec.alpha) if spec.model_type in {"ridge","lasso"} else None

    coef_ = getattr(final_model, "coef_", None)
    intercept_ = float(getattr(final_model, "intercept_", 0.0))

    return FitResult(
        bands_idx=tuple(bands_idx),
        gdd_index=int(gdd_index),
        coef_=np.array(coef_, dtype=float),
        intercept_=intercept_,
        cv_scores=np.array(cv_scores, dtype=float),
        cv_score_mean=float(np.mean(cv_scores)),
        cv_score_std=float(np.std(cv_scores)),
        model_type=spec.model_type,
        alpha=alpha_used,
        feature_means=mu,
        feature_stds=sigma,
    )


def predict_map(
    cube: ArrayLike4D,
    fit: FitResult,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Predict a full-resolution yield map at the fitted GDD index.

    Parameters
    ----------
    cube : (T, B, Y, X)
        EO cube used for training (or with same layout/values at `fit.gdd_index`).
    fit : FitResult
        Output from `fit_at_gdd()`.
    fill_value : float
        Value to assign where features contain NaNs.

    Returns
    -------
    y_hat : (Y, X) np.ndarray
        Predicted yield map at `fit.gdd_index`.
    """
    cube = _ensure_4d(cube)
    bands_slice = _slice_time(cube, fit.gdd_index)  # (B, Y, X)

    Xmat, maskX = _stack_features(bands_slice, fit.bands_idx)  # (Y*X, k)

    Xs = _standardize_apply(Xmat, fit.feature_means, fit.feature_stds)
    y_pred = np.full((Xmat.shape[0],), fill_value, dtype=float)

    # Linear prediction: y = beta0 + X * beta
    beta = fit.coef_.reshape(-1)
    y_lin = fit.intercept_ + Xs @ beta

    # Assign only where inputs were valid
    y_pred[maskX] = y_lin[maskX]

    # Reshape back to (Y, X)
    B, Y, X = bands_slice.shape
    return y_pred.reshape(Y, X)


def fit_many(
    cube: ArrayLike4D,
    target: ArrayLike2D,
    gdd_indices: Sequence[int],
    band_subsets: Optional[Iterable[Sequence[int]]] = None,
    spec: Optional[ModelSpec] = None,
) -> List[FitResult]:
    """Fit many models across multiple GDD indices and band subsets.

    Parameters
    ----------
    cube : (T, B, Y, X)
    target : (Y, X)
    gdd_indices : list[int]
        Which time/GDD indices to evaluate.
    band_subsets : iterable of sequences of ints, optional
        If None, defaults to using *all* bands as a single subset.
        To explore e.g. all 2-band pairs, pass:
            (combinations(range(B), 2))
    spec : ModelSpec, optional

    Returns
    -------
    results : list[FitResult]
    """
    cube = _ensure_4d(cube)
    B = cube.shape[1]
    if band_subsets is None:
        band_subsets = [tuple(range(B))]

    results: List[FitResult] = []
    for t in gdd_indices:
        for subset in band_subsets:
            res = fit_at_gdd(cube, target, gdd_index=int(t), bands_idx=tuple(subset), spec=spec)
            results.append(res)
    return results


def best_by_metric(results: Sequence[FitResult], key: str = "cv_score_mean", maximize: bool = True) -> FitResult:
    """Pick the best FitResult by a metric.

    Parameters
    ----------
    results : list[FitResult]
    key : str
        One of {"cv_score_mean", "cv_score_std"}.
    maximize : bool
        If True (default), pick max; else min.
    """
    if not results:
        raise ValueError("Empty results list.")
    vals = np.array([getattr(r, key) for r in results], dtype=float)
    idx = int(np.nanargmax(vals) if maximize else np.nanargmin(vals))
    return results[idx]


# -------------------------- example usage -------------------------- #
if __name__ == "__main__":
    # Minimal runnable demo with synthetic data (small shapes for speed).
    rng = np.random.default_rng(0)
    T, B, Y, X = 5, 4, 50, 60  # 5 GDD nodes, 4 bands, 50x60 pixels

    # Build a synthetic cube where band 0 and 2 at GDD index 3 carry most signal
    cube = rng.normal(size=(T, B, Y, X)).astype(np.float32)
    true_beta = np.array([2.0, -1.0])  # for bands (0,2) at t=3
    intercept = 5.0

    signal = intercept + true_beta[0]*cube[3, 0] + true_beta[1]*cube[3, 2]
    noise = 0.5 * rng.normal(size=(Y, X))
    target = (signal + noise).astype(np.float32)

    # Introduce a few NaNs
    cube[:, :, 0:2, 0:2] = np.nan
    target[0:2, 0:2] = np.nan

    # Fit all 2-band subsets at GDD index 3 using linear regression
    spec = ModelSpec(model_type="linear", standardize=True, kfold_splits=5, scoring="r2")
    results = fit_many(
        cube=cube,
        target=target,
        gdd_indices=[3],
        band_subsets=combinations(range(B), 2),
        spec=spec,
    )

    # Pick the best by mean CV R^2
    best = best_by_metric(results, key="cv_score_mean", maximize=True)
    print("Best subset:", best.bands_idx, "GDD index:", best.gdd_index)
    print("CV R^2 mean=", best.cv_score_mean, "+/-", best.cv_score_std)

    # Predict a full map at that GDD
    y_hat = predict_map(cube, best)
    print("Pred map shape:", y_hat.shape, " | nan%:", np.isnan(y_hat).mean()*100)
