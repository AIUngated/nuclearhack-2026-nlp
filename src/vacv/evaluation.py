from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .chemistry import max_train_similarity
from .models import build_model, prediction_values


def regression_metrics(y, predictions) -> dict:
    mse = float(mean_squared_error(y, predictions))
    return {"mse": mse, "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, predictions)),
            "r2": float(r2_score(y, predictions)) if len(y) > 1 else None}


def make_splits(metadata: pd.DataFrame, mode: str, n_splits: int, seed: int):
    column = {"compound": "compound_group", "scaffold": "scaffold"}.get(mode)
    if column is None:
        raise ValueError("Split must be compound or scaffold.")
    groups = metadata[column].to_numpy()
    if len(set(groups)) < n_splits:
        raise ValueError(f"Only {len(set(groups))} groups for {n_splits} folds; no silent random fallback.")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(metadata, groups=groups))
    for train, valid in splits:
        if set(metadata.iloc[train]["compound_group"]) & set(metadata.iloc[valid]["compound_group"]):
            raise RuntimeError("A compound group crosses training and validation.")
        if mode == "scaffold" and set(groups[train]) & set(groups[valid]):
            raise RuntimeError("A scaffold crosses training and validation.")
    return splits


def cross_validate(x, y, metadata, config, progress=print):
    summary, fold_metrics, prediction_frames, split_frames = [], [], [], []
    for mode in config["validation_splits"]:
        splits = make_splits(metadata, mode, config["n_splits"], config["seed"])
        fold_ids = np.full(len(y), -1, dtype=int)
        similarity = np.zeros(len(y))
        for fold, (train, valid) in enumerate(splits):
            fold_ids[valid] = fold
            similarity[valid] = max_train_similarity(x[valid], x[train], config["n_bits"])
        split_frame = metadata.copy()
        split_frame.insert(0, "row_id", np.arange(len(y)))
        split_frame["split"], split_frame["seed"], split_frame["fold"] = mode, config["seed"], fold_ids
        split_frames.append(split_frame)
        for name in config["models"]:
            oof = np.full(len(y), np.nan)
            for fold, (train, valid) in enumerate(splits):
                # Imputation, scaling, estimator and target transforms fit only on this fold's training rows.
                estimator = build_model(name, config)
                estimator.fit(x[train], y[train])
                oof[valid] = prediction_values(estimator, x[valid], config["clip_min"])
                fold_metrics.append(dict(model=name, split=mode, seed=config["seed"], fold=fold,
                                         train_rows=len(train), valid_rows=len(valid),
                                         **regression_metrics(y[valid], oof[valid])))
            if not np.isfinite(oof).all():
                raise RuntimeError("Incomplete out-of-fold predictions.")
            metrics = regression_metrics(y, oof)
            summary.append(dict(model=name, split=mode, seed=config["seed"], **metrics))
            progress(f"{mode:8s} seed={config['seed']} {name:18s} MSE={metrics['mse']:.3f} RMSE={metrics['rmse']:.3f}", flush=True)
            prediction_frames.append(pd.DataFrame({
                "row_id": np.arange(len(y)), "model": name, "split": mode,
                "seed": config["seed"], "fold": fold_ids, "target": y, "prediction": oof,
                "max_train_tanimoto": similarity,
            }))
    return (pd.DataFrame(summary), pd.DataFrame(fold_metrics),
            pd.concat(prediction_frames, ignore_index=True), pd.concat(split_frames, ignore_index=True))
