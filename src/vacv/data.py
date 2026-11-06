from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input {path.suffix}; provide CSV or Excel (Excel requires its engine).")


def validate_frame(df: pd.DataFrame, config: dict, training: bool = False) -> None:
    if df.empty:
        raise ValueError("Input is empty.")
    smiles = config["smiles_column"]
    if smiles not in df:
        raise ValueError(f"Required column is missing: {smiles}")
    if df[smiles].isna().any() or not df[smiles].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Every row must contain a non-empty SMILES string.")
    if training:
        target = config["target_column"]
        if target not in df:
            raise ValueError(f"Required target is missing: {target}")
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Training SI must be finite, numeric and non-negative. Censored values require review.")


def submission_from_template(template: pd.DataFrame, test: pd.DataFrame,
                             predictions: np.ndarray, config: dict) -> pd.DataFrame:
    id_column, target = config["id_column"], config["target_column"]
    if list(template.columns) != [id_column, target]:
        raise ValueError(f"Template columns must be exactly {[id_column, target]} in that order.")
    if id_column not in test or len(test) != len(template):
        raise ValueError("Test identifiers are absent or test/template lengths differ.")
    if template[id_column].isna().any() or test[id_column].isna().any():
        raise ValueError("Submission identifiers must not be missing.")
    if template[id_column].duplicated().any() or test[id_column].duplicated().any():
        raise ValueError("Submission identifiers must be unique.")
    if set(template[id_column]) != set(test[id_column]):
        raise ValueError("Test identifiers and template identifiers differ.")
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if len(predictions) != len(test) or not np.isfinite(predictions).all():
        raise ValueError("Predictions must have one finite value for every test row.")
    if (predictions < 0).any():
        raise ValueError("SI predictions must be non-negative.")
    mapping = pd.Series(predictions, index=test[id_column].to_numpy())
    result = template.copy()
    # Never use sample prediction values. Only its schema and identifier order matter.
    result[target] = template[id_column].map(mapping).to_numpy()
    return result
