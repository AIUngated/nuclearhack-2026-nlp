import pandas as pd
import pytest

from vacv.verification import compare_metric_tables, compare_submissions, resolve_manifest_path


@pytest.mark.parametrize("problem", ["nan", "infinity", "negative", "duplicate", "missing_index", "empty", "columns"])
def test_reproduction_never_accepts_matching_invalid_submissions(problem):
    frame = pd.DataFrame({"index": [0, 1], "SI VACV": [1., 2.]})
    if problem in ("nan", "infinity", "negative"):
        frame.loc[0, "SI VACV"] = {"nan": float("nan"), "infinity": float("inf"), "negative": -1.}[problem]
    elif problem == "duplicate":
        frame["index"] = 0
    elif problem == "missing_index":
        frame["index"] = [0., float("nan")]
    elif problem == "empty":
        frame = frame.iloc[:0]
    else:
        frame = frame.rename(columns={"SI VACV": "prediction"})
    with pytest.raises(ValueError):
        compare_submissions(frame, frame.copy())


def test_reproduction_preserves_strict_prediction_and_index_comparison():
    frame = pd.DataFrame({"index": [2, 1], "SI VACV": [10., 20.]})
    assert compare_submissions(frame, frame.copy())["max_absolute_difference"] == 0
    with pytest.raises(AssertionError):
        compare_submissions(frame, frame.iloc[::-1])
    changed = frame.copy()
    changed.loc[0, "SI VACV"] += 0.001
    with pytest.raises(AssertionError):
        compare_submissions(frame, changed)


@pytest.mark.parametrize("relative", ["baseline/fold_metrics.csv", "baseline\\fold_metrics.csv"])
def test_manifest_paths_are_portable(tmp_path, relative):
    expected = tmp_path / "baseline" / "fold_metrics.csv"
    expected.parent.mkdir()
    expected.write_text("model,mse\nrandom_forest,100\n", encoding="utf-8")
    assert resolve_manifest_path(tmp_path, relative).read_bytes() == expected.read_bytes()


@pytest.mark.parametrize("relative", [
    "../outside.csv", "..\\outside.csv", "/outside.csv", "C:\\outside.csv",
    "C:outside.csv", "\\\\server\\share\\outside.csv", "nested\\..\\outside.csv",
    "", " ", ".", "./", None,
])
def test_manifest_paths_cannot_leave_reference_directory(tmp_path, relative):
    with pytest.raises(ValueError):
        resolve_manifest_path(tmp_path, relative)


def test_manifest_paths_cannot_escape_through_directory_link(tmp_path):
    reference = tmp_path / "reference"
    outside = tmp_path / "outside"
    reference.mkdir()
    outside.mkdir()
    (outside / "external.csv").write_text("external data", encoding="utf-8")
    link = reference / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"Directory links are unavailable: {error}")
    with pytest.raises(ValueError):
        resolve_manifest_path(reference, "linked/external.csv")


def metrics():
    return pd.DataFrame([
        {"model": model, "split": split, "seed": 42,
         "mse": 100., "rmse": 10., "mae": 8., "r2": 0.2}
        for model in ("random_forest", "knn_5") for split in ("compound", "scaffold")])


def test_comparison_distinguishes_selected_baseline_from_other_candidates():
    original = metrics()
    current = original.copy()
    current.loc[current.model.eq("knn_5"), "mse"] += 1
    comparison, result = compare_metric_tables(original, current, "random_forest")
    assert result["selected_model_matches"]
    assert not result["all_candidates_match"]
    assert result["different_candidates"] == ["knn_5"]
    assert comparison.loc[~comparison.within_tolerance, "delta"].tolist() == [1., 1.]
    current.loc[current.model.eq("random_forest"), "mse"] += 1
    assert not compare_metric_tables(original, current, "random_forest")[1]["selected_model_matches"]


def test_comparison_checks_all_metrics_and_accepts_row_reordering():
    original = metrics()
    current = original.iloc[::-1].copy()
    assert compare_metric_tables(original, current, "random_forest")[1]["all_candidates_match"]
    current.loc[current.model.eq("random_forest"), "mae"] += 1
    assert not compare_metric_tables(original, current, "random_forest")[1]["selected_model_matches"]


@pytest.mark.parametrize("problem", ["missing_row", "duplicate_row", "nonfinite"])
def test_comparison_rejects_incomplete_or_invalid_results(problem):
    original = metrics()
    if problem == "missing_row":
        current = original.iloc[:-1]
    elif problem == "duplicate_row":
        current = pd.concat([original, original.iloc[:1]])
    else:
        current = original.copy()
        current.loc[0, "mse"] = float("nan")
    with pytest.raises(ValueError):
        compare_metric_tables(original, current, "random_forest")
