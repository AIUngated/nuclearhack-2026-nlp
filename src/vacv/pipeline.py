from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .chemistry import make_features, max_train_similarity
from .data import read_table, submission_from_template, validate_frame
from .evaluation import cross_validate
from .models import build_model, prediction_values
from .utils import environment, sha256_file, utc_now, write_json


def train_pipeline(train_path, config, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = read_table(train_path)
    validate_frame(train, config, training=True)
    x, metadata = make_features(train[config["smiles_column"]].tolist(), config)
    y = train[config["target_column"]].to_numpy(dtype=float)
    scores, folds, oof, splits = cross_validate(x, y, metadata, config)
    scores.to_csv(output / "metrics.csv", index=False)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    splits.to_csv(output / "splits.csv", index=False)
    eligible = scores[scores["split"].eq(config["primary_split"])].sort_values("mse", kind="stable")
    selected = eligible.iloc[0]
    name = selected["model"]
    model = build_model(name, config)
    model.fit(x, y)
    bundle = {"format_version": 1, "model": model, "config": config, "model_name": name,
              "train_features": x, "train_metadata": metadata, "train_mean": float(y.mean()),
              "environment": environment(), "training_sha256": sha256_file(train_path)}
    joblib.dump(bundle, output / "model.joblib", compress=3)
    run = {"created_at": utc_now(), "config": config, "train_rows": len(train),
           "train_sha256": sha256_file(train_path), "environment": environment(),
           "selected_model": name, "selection_split": config["primary_split"],
           "selected_oof_metrics": {key: float(selected[key]) for key in ("mse", "rmse", "mae", "r2")},
           "feature_sources": [config["smiles_column"]],
           "excluded_columns": [c for c in train.columns if c != config["smiles_column"]],
           "limitations": ["Selection and reported scores reuse the same CV; no nested unbiased final estimate.",
                           "MSE is measured on the official raw SI labels, including 10000 values.",
                           "No independent cytotoxicity or clinical safety model is trained."]}
    write_json(output / "run.json", run)
    print(f"Selected {name}; saved {output / 'model.joblib'}", flush=True)
    return run


def load_bundle(path):
    # joblib uses pickle: load only model files produced by this project or another trusted source.
    bundle = joblib.load(path)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model format.")
    return bundle


def predict_frame(bundle, frame):
    config = bundle["config"]
    validate_frame(frame, config)
    x, metadata = make_features(frame[config["smiles_column"]].tolist(), config)
    predictions = prediction_values(bundle["model"], x, config["clip_min"])
    diagnostics = metadata.copy()
    diagnostics.insert(0, "row_id", np.arange(len(frame)))
    diagnostics["prediction_si"] = predictions
    diagnostics["max_train_tanimoto"] = max_train_similarity(x, bundle["train_features"], config["n_bits"])
    reference = bundle["train_metadata"]
    diagnostics["seen_canonical_structure"] = metadata["canonical_smiles"].isin(reference["canonical_smiles"])
    diagnostics["seen_compound_group"] = metadata["compound_group"].isin(reference["compound_group"])
    # A nearest-neighbour similarity is an applicability diagnostic, not a probability or a calibrated interval.
    return predictions, diagnostics


def predict_pipeline(model_path, input_path, template_path, output_path):
    bundle, frame = load_bundle(model_path), read_table(input_path)
    predictions, diagnostics = predict_frame(bundle, frame)
    result = submission_from_template(read_table(template_path), frame, predictions, bundle["config"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    diagnostics.to_csv(output.with_name(output.stem + "_diagnostics.csv"), index=False)
    write_json(output.with_name(output.stem + "_manifest.json"), {
        "created_at": utc_now(), "model_sha256": sha256_file(model_path),
        "input_sha256": sha256_file(input_path), "template_sha256": sha256_file(template_path),
        "submission_sha256": sha256_file(output), "rows": len(result), "columns": result.columns.tolist(),
        "model_name": bundle["model_name"], "target_scale": "original SI", "feature_sources": ["SMILES"]})
    print(f"Saved {len(result)} predictions to {output}", flush=True)
    return result
