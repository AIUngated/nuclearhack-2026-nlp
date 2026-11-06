from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="Install requirements-app.txt to test the optional demo")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "artifacts/baseline/model.joblib").exists(), reason="Train the real baseline before testing the demo")
def test_demo_uses_real_model_and_clears_stale_results():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    app.text_input[0].set_value("c1ccccc1").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.dataframe[0].value["SMILES"].tolist() == ["c1ccccc1"]
    assert app.dataframe[0].value["prediction_si"].iloc[0] >= 0
    app.text_input[0].set_value("c1ccncc1").run()
    assert len(app.dataframe) == 0
    app.button[0].click().run()
    assert not app.exception
    assert app.dataframe[0].value["SMILES"].tolist() == ["c1ccncc1"]


@pytest.mark.skipif(not (ROOT / "artifacts/baseline/model.joblib").exists(), reason="Train the real baseline before testing the demo")
def test_demo_rejects_invalid_smiles_and_recovers():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.text_input[0].set_value("not_a_smiles").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.error
    assert len(app.dataframe) == 0
    app.text_input[0].set_value("CCO").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.error) == 0
    assert app.dataframe[0].value["SMILES"].tolist() == ["CCO"]


@pytest.mark.skipif(not all((ROOT / relative).exists() for relative in (
    "artifacts/baseline/model.joblib", "artifacts/submission.csv",
    "data/raw/test.csv", "data/raw/sample_submit.csv")),
    reason="The real baseline and competition CSV files are needed")
def test_demo_predicts_all_test_rows_and_keeps_export_order():
    import pandas as pd
    import numpy as np
    from vacv.pipeline import load_bundle, predict_frame

    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
    app.button[0].click().run()
    assert not app.exception
    result = app.session_state["result"]
    test = pd.read_csv(ROOT / "data/raw/test.csv")
    predictions, _ = predict_frame(load_bundle(ROOT / "artifacts/baseline/model.joblib"), test[["SMILES"]])
    assert result["index"].tolist() == test["index"].tolist()
    np.testing.assert_allclose(result.prediction_si, predictions, rtol=0, atol=1e-9)
    assert len(result) == 597
    reference = pd.read_csv(ROOT / "artifacts/submission.csv")
    actual = result.set_index("index").loc[reference["index"], "prediction_si"]
    np.testing.assert_allclose(actual, reference["SI VACV"], rtol=1e-12, atol=1e-9)
    assert len(app.get("download_button")) == 2
    app.slider[0].set_value(1.0).run()
    assert len(app.session_state["result"]) == 597


@pytest.mark.skipif(not (ROOT / "artifacts/baseline/model.joblib").exists(), reason="Train the real baseline before testing the demo")
def test_demo_works_without_private_test_csv(monkeypatch):
    original_exists = Path.exists
    def without_test_csv(path):
        return False if path == ROOT / "data/raw/test.csv" else original_exists(path)
    monkeypatch.setattr(Path, "exists", without_test_csv)
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.dataframe[0].value["SMILES"].tolist() == ["c1ccccc1"]
