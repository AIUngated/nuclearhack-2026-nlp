"""Compare archived experiments without hiding cross-platform numerical drift."""
from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd

KEY_COLUMNS = ["model", "split", "seed"]
METRIC_COLUMNS = ["mse", "rmse", "mae", "r2"]
RTOL = 1e-12
ATOL = 1e-9


def compare_submissions(reference: pd.DataFrame, reproduced: pd.DataFrame) -> dict:
    """Validate both CSV tables before comparing ordered IDs and SI values."""
    for label, frame in (("reference", reference), ("reproduced", reproduced)):
        if frame.columns.tolist() != ["index", "SI VACV"]:
            raise ValueError(f"Invalid submission columns in {label}; expected index,SI VACV.")
        if frame.empty:
            raise ValueError(f"Empty submission in {label}.")
        identifiers = frame["index"]
        if identifiers.isna().any() or identifiers.duplicated().any():
            raise ValueError(f"Submission index must be present and unique in {label}.")
        if identifiers.map(lambda value: isinstance(value, str) and not value.strip()).any():
            raise ValueError(f"Submission index must be non-empty in {label}.")
        values = pd.to_numeric(frame["SI VACV"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Submission SI must be finite, numeric and non-negative in {label}.")
    np.testing.assert_array_equal(reference["index"], reproduced["index"])
    old = reference["SI VACV"].to_numpy(dtype=float)
    new = reproduced["SI VACV"].to_numpy(dtype=float)
    np.testing.assert_allclose(old, new, rtol=RTOL, atol=ATOL, equal_nan=False)
    return {"rows": len(reference), "max_absolute_difference": float(np.max(np.abs(old - new))),
            "relative_tolerance": RTOL, "absolute_tolerance": ATOL}


def verify_model_submission(model_path, frame, template, reference, *, expected_name,
                            expected_config, expected_train_sha256) -> dict:
    """Check the actual deployed model against an independent control CSV."""
    from .data import submission_from_template
    from .pipeline import load_bundle, predict_frame
    from .utils import sha256_file

    bundle = load_bundle(model_path)
    if (bundle["model_name"] != expected_name or bundle["config"] != expected_config
            or bundle["training_sha256"] != expected_train_sha256):
        raise ValueError("Saved model identity/configuration/training source differs from the control.")
    predictions, _ = predict_frame(bundle, frame[[expected_config["smiles_column"]]])
    actual = submission_from_template(template, frame, predictions, expected_config)
    result = compare_submissions(reference, actual)
    result["model_sha256"] = sha256_file(model_path)
    return result


def resolve_manifest_path(root: str | Path, relative_path: str) -> Path:
    """Resolve Windows/POSIX manifest paths within the resolved reference root."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("A manifest path must be a non-empty relative string.")
    portable = PurePosixPath(relative_path.replace("\\", "/"))
    if (not portable.parts or portable.is_absolute()
            or PureWindowsPath(relative_path).drive or ".." in portable.parts):
        raise ValueError("Manifest paths must stay inside the reference directory.")
    reference = Path(root).resolve()
    resolved = reference.joinpath(*portable.parts).resolve()
    if resolved == reference or not resolved.is_relative_to(reference):
        raise ValueError("Manifest paths must stay inside the reference directory.")
    return resolved


def compare_metric_tables(reference: pd.DataFrame, reproduced: pd.DataFrame,
                          selected_model: str) -> tuple[pd.DataFrame, dict]:
    """Require the same experiments; report every metric and baseline match.

    Callers must fail if selected_model_matches is false. Differences in other
    candidates remain explicit evidence, rather than being rounded away.
    """
    required = KEY_COLUMNS + METRIC_COLUMNS
    for name, frame in (("reference", reference), ("reproduced", reproduced)):
        if not set(required).issubset(frame.columns):
            raise ValueError(f"Missing experiment columns in {name}.")
        if frame.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"Duplicate experiment keys in {name}.")
        if not np.isfinite(frame[METRIC_COLUMNS].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite experiment metrics in {name}.")
    left = reference.set_index(KEY_COLUMNS).sort_index()
    right = reproduced.set_index(KEY_COLUMNS).sort_index()
    if not left.index.equals(right.index):
        raise ValueError("The reference and reproduced experiment keys differ.")
    if selected_model not in left.index.get_level_values("model"):
        raise ValueError("The selected model is missing from experiment metrics.")
    rows = []
    for key in left.index:
        for metric in METRIC_COLUMNS:
            old, new = float(left.loc[key, metric]), float(right.loc[key, metric])
            rows.append(dict(zip(KEY_COLUMNS, key), metric=metric,
                             reference=old, reproduced=new, delta=new - old,
                             absolute_difference=abs(new - old),
                             within_tolerance=bool(np.isclose(old, new, rtol=RTOL, atol=ATOL))))
    comparison = pd.DataFrame(rows)
    selected = comparison[comparison.model.eq(selected_model)]
    summary = {
        "selected_model": selected_model,
        "selected_model_matches": bool(selected.within_tolerance.all()),
        "all_candidates_match": bool(comparison.within_tolerance.all()),
        "different_candidates": sorted(comparison.loc[~comparison.within_tolerance, "model"].unique()),
        "relative_tolerance": RTOL,
        "absolute_tolerance": ATOL,
    }
    return comparison, summary
