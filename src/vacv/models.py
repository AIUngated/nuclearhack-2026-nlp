from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .chemistry import tanimoto_matrix


class TanimotoRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, alpha: float = 1.0, n_bits: int = 2048):
        self.alpha = alpha
        self.n_bits = n_bits

    def fit(self, x, y):
        self.reference_ = np.asarray(x).copy()
        self.model_ = KernelRidge(alpha=self.alpha, kernel="precomputed")
        self.model_.fit(tanimoto_matrix(x, x, self.n_bits), y)
        return self

    def predict(self, x):
        return self.model_.predict(tanimoto_matrix(x, self.reference_, self.n_bits))


class SimilarityKNN(RegressorMixin, BaseEstimator):
    def __init__(self, n_neighbors: int = 5, n_bits: int = 2048, power: float = 3.0):
        self.n_neighbors = n_neighbors
        self.n_bits = n_bits
        self.power = power

    def fit(self, x, y):
        self.reference_ = np.asarray(x).copy()
        self.targets_ = np.asarray(y, dtype=float).copy()
        return self

    def predict(self, x):
        similarities = tanimoto_matrix(x, self.reference_, self.n_bits)
        k = min(self.n_neighbors, len(self.targets_))
        indices = np.argsort(similarities, axis=1)[:, -k:]
        weights = np.take_along_axis(similarities, indices, axis=1) ** self.power
        sums = weights.sum(axis=1)
        numerator = (self.targets_[indices] * weights).sum(axis=1)
        return np.divide(numerator, sums, out=np.full(len(x), self.targets_.mean()), where=sums > 0)


def build_model(name: str, config: dict):
    if name.endswith("_log"):
        return TransformedTargetRegressor(regressor=build_model(name[:-4], config),
                                          func=np.log1p, inverse_func=np.expm1)
    if name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if name.startswith("ridge_"):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=float(name.split("_")[1])))
    if name.startswith("tanimoto_"):
        return TanimotoRegressor(alpha=float(name.split("_")[1]), n_bits=config["n_bits"])
    if name.startswith("knn_"):
        return SimilarityKNN(n_neighbors=int(name.split("_")[1]), n_bits=config["n_bits"])
    tree_args = dict(n_estimators=config["n_estimators"], min_samples_leaf=2,
                     random_state=config["seed"], n_jobs=config.get("n_jobs", 2))
    if name == "random_forest":
        model = RandomForestRegressor(max_features=0.7, **tree_args)
    elif name == "extra_trees":
        model = ExtraTreesRegressor(max_features=1.0, **tree_args)
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(SimpleImputer(strategy="median"), model)


def prediction_values(model, x, clip_min: float = 0.0) -> np.ndarray:
    predictions = np.maximum(np.asarray(model.predict(x), dtype=float).reshape(-1), clip_min)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions.")
    return predictions
