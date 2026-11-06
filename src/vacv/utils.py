from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "rdkit"):
        versions[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("feature_columns") != [config.get("smiles_column")]:
        raise ValueError("Only the explicitly named SMILES column may be used as input features.")
    if config.get("metric") != "mse":
        raise ValueError("The supplied competition specifies MSE on the original SI scale.")
    if config.get("primary_split") not in config.get("validation_splits", []):
        raise ValueError("primary_split must appear in validation_splits.")
    return config
