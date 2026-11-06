import copy
import json

import numpy as np
import pandas as pd
import pytest

from vacv.chemistry import make_features, molecular_metadata, molecular_record, tanimoto_matrix
from vacv.collect import ChEMBLClient, normalize_activity, to_micromolar
from vacv.data import submission_from_template, validate_frame
from vacv.evaluation import make_splits
from vacv.pipeline import load_bundle, predict_frame, predict_pipeline, train_pipeline
from vacv.utils import read_config


@pytest.fixture
def config():
    return {"smiles_column": "SMILES", "target_column": "SI VACV", "id_column": "index",
            "feature_columns": ["SMILES"], "metric": "mse", "radius": 2, "n_bits": 128,
            "use_chirality": True, "n_splits": 3, "seed": 42, "primary_split": "compound",
            "validation_splits": ["compound", "scaffold"], "models": ["dummy_mean", "random_forest"],
            "n_estimators": 8, "n_jobs": 1, "clip_min": 0.0}


def test_chemical_grouping_preserves_duplicate_identity():
    assert molecular_record("c1ccccc1")["canonical_smiles"] == molecular_record("C1=CC=CC=C1")["canonical_smiles"]
    assert molecular_record("CC[NH3+].[Cl-]")["compound_group"] == molecular_record("CCN")["compound_group"]
    assert molecular_record("C[C@H](O)F")["compound_group"] == molecular_record("C[C@@H](O)F")["compound_group"]
    assert molecular_record("CCCO")["scaffold"] == molecular_record("CCN")["scaffold"] == "__ACYCLIC__"


@pytest.mark.parametrize("smiles", ["not_a_smiles", "", "[Na+].[Cl-].not_valid"])
def test_invalid_smiles_fails_with_row_number(smiles, config):
    with pytest.raises(ValueError, match="row 1"):
        make_features(["CCO", smiles], config)


@pytest.mark.parametrize("mode", ["compound", "scaffold"])
def test_fold_groups_never_cross(mode):
    smiles = ["c1ccccc1", "C1=CC=CC=C1", "Oc1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccoc1", "c1ccsc1"]
    metadata = molecular_metadata(smiles)
    splits = make_splits(metadata, mode, 3, 42)
    held_out = []
    for train, valid in splits:
        assert not set(metadata.iloc[train].compound_group) & set(metadata.iloc[valid].compound_group)
        if mode == "scaffold":
            assert not set(metadata.iloc[train].scaffold) & set(metadata.iloc[valid].scaffold)
        held_out.extend(valid.tolist())
    assert sorted(held_out) == list(range(len(smiles)))


def test_insufficient_scaffolds_has_no_random_fallback():
    metadata = molecular_metadata(["Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1"])
    with pytest.raises(ValueError, match="no silent random fallback"):
        make_splits(metadata, "scaffold", 3, 42)


def test_tanimoto_is_exact_binary_similarity(config):
    x, _ = make_features(["c1ccccc1", "c1ccncc1"], config)
    similarity = tanimoto_matrix(x, x, config["n_bits"])
    np.testing.assert_allclose(np.diag(similarity), 1)
    np.testing.assert_allclose(similarity, similarity.T)
    assert 0 <= similarity.min() <= similarity.max() <= 1


def test_config_rejects_target_leakage(tmp_path, config):
    config["feature_columns"].append("CC50_uM")
    path = tmp_path / "leaky.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Only"):
        read_config(path)


def test_submission_aligns_ids_ignores_template_predictions(config):
    template = pd.DataFrame({"index": [2, 0, 1], "SI VACV": [999, 999, 999]})
    test = pd.DataFrame({"index": [0, 1, 2]})
    output = submission_from_template(template, test, np.array([5, 6, 7]), config)
    assert output["SI VACV"].tolist() == [7, 5, 6]
    assert output.columns.tolist() == ["index", "SI VACV"]
    with pytest.raises(ValueError, match="finite"):
        submission_from_template(template, test, np.array([5, np.nan, 7]), config)
    with pytest.raises(ValueError, match="unique"):
        submission_from_template(template, pd.DataFrame({"index": [0, 0, 2]}), np.ones(3), config)


def test_non_numeric_or_censored_training_target_is_rejected(config):
    with pytest.raises(ValueError, match="Censored"):
        validate_frame(pd.DataFrame({"SMILES": ["CCO"], "SI VACV": [">100"]}), config, training=True)


def test_train_save_reload_inference_and_forbidden_column_invariance(tmp_path, config):
    # Artificial labels here are a software fixture, never a biological experiment.
    train = pd.DataFrame({"SMILES": ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccoc1", "c1ccsc1", "C1CCCC1"],
                          "SI VACV": [1., 2., 3., 4., 5., 6.], "CC50_uM": [999.] * 6,
                          "IC50_uM_VACV": [111.] * 6})
    path = tmp_path / "train.csv"
    train.to_csv(path, index=False)
    run = train_pipeline(path, config, tmp_path / "model")
    assert run["feature_sources"] == ["SMILES"]
    bundle = load_bundle(tmp_path / "model/model.joblib")
    test = pd.DataFrame({"index": [10, 20], "SMILES": ["Oc1ccccc1", "Cc1ccncc1"],
                         "CC50_uM": [1., 2.], "IC50_uM_VACV": [1., 1.]})
    before, _ = predict_frame(bundle, test)
    altered = test.copy()
    altered["CC50_uM"], altered["IC50_uM_VACV"] = 1e12, np.nan
    altered["Title"], altered["Culture"], altered["strain"] = "anything", "anything", "anything"
    after, _ = predict_frame(bundle, altered)
    np.testing.assert_array_equal(before, after)
    input_path, template_path = tmp_path / "test.csv", tmp_path / "sample.csv"
    test.to_csv(input_path, index=False)
    pd.DataFrame({"index": [20, 10], "SI VACV": [-123., -456.]}).to_csv(template_path, index=False)
    result = predict_pipeline(tmp_path / "model/model.joblib", input_path, template_path, tmp_path / "submission.csv")
    np.testing.assert_allclose(result["SI VACV"], before[::-1])
    assert (tmp_path / "submission_manifest.json").is_file()


def activity(**kwargs):
    record = {"canonical_smiles": "CCO", "standard_type": "SI", "standard_relation": "=",
              "standard_value": "25", "standard_units": None, "target_organism": "Orthopoxvirus vaccinia",
              "activity_id": 1}
    record.update(kwargs)
    return record


def test_molar_units_and_no_implicit_mass_conversion():
    assert to_micromolar("1000", "nM") == 1.
    assert to_micromolar("1", "mM") == 1000.
    assert to_micromolar("1", "ug.mL-1") is None
    assert to_micromolar("nan", "uM") is None


def test_collection_does_not_confuse_ec50_and_ic50():
    record = normalize_activity(activity(standard_type="EC50", standard_units="nM"), {}, {})
    assert record["IC50"] is None and record["SI"] is None
    assert not record["eligible_for_training"]
    row = normalize_activity(activity(), {"description": "Selectivity index CC50/EC50"}, {"doi": "10.example/test"})
    assert row["SI"] == 25 and row["si_definition"] == "CC50/EC50"
    assert not row["eligible_for_training"]


@pytest.mark.parametrize("relation", [">", ">=", "<", "~", ""])
def test_censored_external_si_is_not_an_exact_training_label(relation):
    row = normalize_activity(activity(standard_relation=relation), {"description": "SI = CC50/IC50"}, {})
    assert row["relation"] == relation
    assert not row["eligible_for_training"]


def test_only_explicit_unambiguous_ratio_is_eligible():
    assert normalize_activity(activity(), {"description": "SI = CC50/IC50"}, {})["eligible_for_training"]
    assert not normalize_activity(activity(), {"description": "selectivity index"}, {})["eligible_for_training"]
    assert not normalize_activity(activity(), {"description": "CC50/IC50 or CC50/EC50"}, {})["eligible_for_training"]
    assert not normalize_activity(activity(potential_duplicate=1), {"description": "CC50/IC50"}, {})["eligible_for_training"]


def test_collector_retains_all_pages_and_rejects_pagination_cycles(tmp_path, monkeypatch):
    client = ChEMBLClient(tmp_path, offline=True)
    pages = iter([{"activities": [{"id": 1}], "page_meta": {"next": "/chembl/api/data/activity.json?offset=1000"}},
                  {"activities": [{"id": 2}], "page_meta": {"next": None}}])
    monkeypatch.setattr(client, "get", lambda url: next(pages))
    assert client.pages("activity", {}, "activities") == [{"id": 1}, {"id": 2}]
    monkeypatch.setattr(client, "get", lambda url: {"activities": [], "page_meta": {"next": "/chembl/api/data/activity.json?limit=1000"}})
    with pytest.raises(RuntimeError, match="cycle"):
        client.pages("activity", {}, "activities")
