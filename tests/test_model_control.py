import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from vacv.chemistry import make_features
from vacv.models import build_model
from vacv.pipeline import predict_frame
from vacv.verification import verify_model_submission


@pytest.fixture
def saved_control(tmp_path):
    config = {"smiles_column": "SMILES", "target_column": "SI VACV", "id_column": "index",
              "radius": 2, "n_bits": 64, "use_chirality": True,
              "n_estimators": 8, "n_jobs": 1, "seed": 42, "clip_min": 0.0}
    frame = pd.DataFrame({"index": [2, 0, 1], "SMILES": ["CCO", "c1ccccc1", "c1ccncc1"]})
    x, metadata = make_features(frame.SMILES.tolist(), config)
    model = build_model("random_forest", config).fit(x, np.array([1., 10., 100.]))
    bundle = {"format_version": 1, "model_name": "random_forest", "model": model,
              "config": config, "training_sha256": "software-fixture",
              "train_features": x, "train_metadata": metadata}
    predictions, _ = predict_frame(bundle, frame)
    reference = pd.DataFrame({"index": frame["index"], "SI VACV": predictions})
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    return path, bundle, frame, reference


def check(saved_control):
    path, bundle, frame, reference = saved_control
    return verify_model_submission(path, frame, reference, reference,
                                   expected_name="random_forest", expected_config=bundle["config"],
                                   expected_train_sha256="software-fixture")


def test_saved_deployed_model_matches_independent_control(saved_control):
    assert check(saved_control)["max_absolute_difference"] == 0


def test_valid_bundle_with_wrong_estimator_is_rejected(saved_control):
    path, bundle, frame, reference = saved_control
    changed = copy.copy(bundle)
    changed["model"] = DummyRegressor(strategy="constant", constant=123).fit(
        bundle["train_features"], reference["SI VACV"])
    joblib.dump(changed, path)
    with pytest.raises(AssertionError):
        check(saved_control)


def test_saved_model_training_identity_is_checked(saved_control):
    path, bundle, frame, reference = saved_control
    changed = copy.copy(bundle)
    changed["training_sha256"] = "different-source"
    joblib.dump(changed, path)
    with pytest.raises(ValueError, match="training source"):
        check(saved_control)
