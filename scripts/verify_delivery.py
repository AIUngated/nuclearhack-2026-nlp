#!/usr/bin/env python3
"""Verify the delivered baseline strictly; record all other candidate differences."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from vacv.data import submission_from_template
from vacv.pipeline import load_bundle, predict_frame
from vacv.utils import environment, sha256_file, utc_now, write_json
from vacv.verification import ATOL, RTOL, compare_metric_tables, compare_submissions, resolve_manifest_path, verify_model_submission


def report_reference(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-all-candidates", action="store_true",
                        help="Also fail on numerical drift in any exploratory candidate.")
    parser.add_argument("--reproduced-dir", default="artifacts/reproduced",
                        help="Existing reproduction output to verify; relative to the project root.")
    parser.add_argument("--report-dir", default="reports",
                        help="Write verification evidence here without replacing previous reports.")
    args = parser.parse_args(argv)
    reproduced_dir = (ROOT / args.reproduced_dir).resolve()
    report_dir = (ROOT / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    test_run = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "pytest", "-q"], cwd=ROOT,
        text=True, encoding="utf-8", capture_output=True,
        env={**os.environ, "PYTHONUTF8": "1"})
    (report_dir / "test_results.txt").write_text(test_run.stdout + test_run.stderr, encoding="utf-8")
    if test_run.returncode:
        raise RuntimeError(test_run.stdout + test_run.stderr)

    control = ROOT / "artifacts/control_original"
    baseline = control / "baseline" if control.exists() else ROOT / "artifacts/baseline"
    source_verification = control / "verification.json" if control.exists() else None
    original_run = json.loads((baseline / "run.json").read_text(encoding="utf-8"))
    repeated_model = reproduced_dir / "model"
    repeated_run = json.loads((repeated_model / "run.json").read_text(encoding="utf-8"))
    selected = original_run["selected_model"]
    assert repeated_run["selected_model"] == selected, "Baseline model selection changed."
    assert original_run["config"] == repeated_run["config"], "Baseline configuration changed."

    comparison, metric_check = compare_metric_tables(
        pd.read_csv(baseline / "metrics.csv"),
        pd.read_csv(repeated_model / "metrics.csv"), selected)
    comparison.to_csv(report_dir / "reproduction_metric_comparison.csv", index=False)
    assert metric_check["selected_model_matches"], "Selected baseline metrics changed."
    for filename in ("splits.csv", "oof_predictions.csv", "fold_metrics.csv"):
        old, new = pd.read_csv(baseline / filename), pd.read_csv(repeated_model / filename)
        if "model" in old:
            old, new = old[old.model.eq(selected)], new[new.model.eq(selected)]
        pd.testing.assert_frame_equal(old.reset_index(drop=True), new.reset_index(drop=True),
                                      rtol=RTOL, atol=ATOL)

    raw = ROOT / "data/raw"
    test = pd.read_csv(raw / "test.csv")
    template = pd.read_csv(raw / "sample_submit.csv")
    submitted = pd.read_csv(ROOT / "artifacts/submission.csv")
    reproduced = pd.read_csv(reproduced_dir / "submission.csv")
    compare_submissions(submitted, reproduced)
    assert submitted.columns.tolist() == ["index", "SI VACV"]
    assert submitted["index"].tolist() == template["index"].tolist()
    assert len(submitted) == len(test) == 597
    assert submitted["index"].is_unique
    assert np.isfinite(submitted["SI VACV"]).all() and submitted["SI VACV"].ge(0).all()
    np.testing.assert_array_equal(submitted["index"], reproduced["index"])
    np.testing.assert_allclose(submitted["SI VACV"], reproduced["SI VACV"], rtol=RTOL, atol=ATOL)

    demo_check = verify_model_submission(
        ROOT / "artifacts/baseline/model.joblib", test, template, submitted,
        expected_name=selected, expected_config=original_run["config"],
        expected_train_sha256=sha256_file(raw / "train.csv"))
    original_bundle = load_bundle(baseline / "model.joblib")
    reloaded_bundle = load_bundle(repeated_model / "model.joblib")
    assert original_bundle["config"]["feature_columns"] == ["SMILES"]
    assert reloaded_bundle["training_sha256"] == sha256_file(raw / "train.csv")
    from_all, _ = predict_frame(original_bundle, test)
    from_smiles_only, _ = predict_frame(original_bundle, test[["SMILES"]])
    from_reloaded, _ = predict_frame(reloaded_bundle, test[["SMILES"]])
    np.testing.assert_allclose(from_all, from_smiles_only, rtol=0, atol=ATOL)
    for predictions in (from_all, from_reloaded):
        aligned = submission_from_template(template, test, predictions, reloaded_bundle["config"])
        np.testing.assert_allclose(aligned["SI VACV"], submitted["SI VACV"], rtol=RTOL, atol=ATOL)
    shuffled = test.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_predictions, _ = predict_frame(reloaded_bundle, shuffled[["SMILES"]])
    poisoned_template = template.copy()
    poisoned_template["SI VACV"] = -123456.0
    aligned = submission_from_template(poisoned_template, shuffled, shuffled_predictions, reloaded_bundle["config"])
    np.testing.assert_array_equal(aligned["index"], template["index"])
    np.testing.assert_allclose(aligned["SI VACV"], submitted["SI VACV"], rtol=RTOL, atol=ATOL)

    source_files = {p.name: sha256_file(p) for p in sorted(raw.glob("*.csv"))}
    control_files_checked = 0
    if control.exists():
        manifest = json.loads((control / "control_manifest.json").read_text(encoding="utf-8-sig"))
        for entry in manifest["files"]:
            assert sha256_file(resolve_manifest_path(control, entry["path"])) == entry["sha256"], entry["path"]
            control_files_checked += 1
        archived = json.loads(source_verification.read_text(encoding="utf-8"))
        assert source_files == archived["source_files"], "Competition source files changed."
        assert sha256_file(ROOT / "artifacts/submission.csv") == archived["submission"]["sha256"]

    # Verify existing snapshots only; this command performs no collection.
    collection = json.loads((ROOT / "data/external/collection_manifest.json").read_text(encoding="utf-8"))
    for request in collection["requests"]:
        assert sha256_file(ROOT / "data/external/cache" / request["cache_file"]) == request["sha256"]
    for name, expected in collection["files"].items():
        assert sha256_file(ROOT / "data/external" / name) == expected
    curated = pd.read_csv(ROOT / "data/external/external_curated.csv")
    np.testing.assert_allclose(curated.SI, curated.CC50 / curated.IC50)
    assert not curated.included_in_baseline_training.any()
    assert not curated.eligible_as_new_training_data.any()
    assert (curated.overlap_train_parent | curated.overlap_test_parent).all()
    result = {
        "recorded_at": utc_now(), "scope": "Existing baseline engineering verification; no new data or models.",
        "environment": environment(),
        "pytest": {"command": "python -X utf8 -m pytest -q", "exit_code": test_run.returncode,
                   "output": test_run.stdout.strip(), "log": report_reference(report_dir / "test_results.txt")},
        "submission": {"rows": len(submitted), "columns": submitted.columns.tolist(),
                       "finite_nonnegative": True, "template_order_verified": True,
                       "shuffled_input_order_verified": True, "template_values_ignored": True,
                       "sha256": sha256_file(ROOT / "artifacts/submission.csv")},
        "full_retraining": {"executed": True, "selected_model": selected,
                            "artifact_directory": report_reference(reproduced_dir),
                            "selected_model_oof_and_folds_match": True, "split_assignments_match": True,
                            "metric_table_matches": metric_check["all_candidates_match"],
                            "metric_comparison": metric_check,
                            "comparison_csv": report_reference(report_dir / "reproduction_metric_comparison.csv"),
                            "prediction_max_absolute_difference": float(np.max(np.abs(submitted['SI VACV'] - reproduced['SI VACV']))),
                            "relative_tolerance": RTOL, "absolute_tolerance": ATOL},
        "model_reload": {"original_model_matches_submission": True,
                         "retrained_saved_model_matches_submission": True,
                         "actual_demo_model_matches_submission": True,
                         "actual_demo_model_check": demo_check},
        "real_test_forbidden_column_removal": {"rows_checked": len(test), "input_retained": ["SMILES"],
                                               "max_absolute_difference": float(np.max(np.abs(from_all - from_smiles_only)))},
        "control_copy": {"available": control.exists(), "files_checked": control_files_checked,
                         "checksums_valid": True if control.exists() else None},
        "external_data": {"records": collection["records"], "doi_rows": collection["doi_rows"],
                          "cache_responses_checked": len(collection["requests"]), "cache_checksums_valid": True,
                          "offline_collection_executed_this_run": False,
                          "existing_curated_pairs": len(curated), "new_nonoverlapping_pairs": 0,
                          "external_labels_used_in_baseline": 0, "further_research": "paused_by_user"},
        "source_files": source_files,
        "kaggle_submission": {"uploaded": False, "score": None}, "github": {"published": False},
    }
    write_json(report_dir / "verification.json", result)
    print(test_run.stdout.strip())
    print("Verified: baseline OOF/folds, model reload, 597-row CSV, shuffled index alignment, SMILES-only input and original checksums.")
    if not metric_check["all_candidates_match"]:
        print("Exploratory candidates differ across environments: " + ", ".join(metric_check["different_candidates"]))
        print(f"All differences are recorded in {report_reference(report_dir / 'reproduction_metric_comparison.csv')}; metric_table_matches=false.")
    if args.strict_all_candidates and not metric_check["all_candidates_match"]:
        raise SystemExit("Strict all-candidate verification failed; selected baseline verification passed.")


if __name__ == "__main__":
    main()
