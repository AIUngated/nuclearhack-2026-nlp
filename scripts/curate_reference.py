#!/usr/bin/env python3
"""Rebuild five literature-checked, paired point estimates without contaminating model training.

Reference source: PMC6335999, Table 1 and footnotes a-c; DOI 10.1039/C8MD00442K.
This is a fixed source mapping. Independent manual confirmation is still pending.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from vacv.chemistry import molecular_record
from vacv.utils import sha256_file, utc_now, write_json

# compound label, molecule ChEMBL ID, CC50 mean (uM), IC50 mean (uM), rounded published SI
REVIEWED = [
    ("8", "CHEMBL4591812", 854.3, 69.8, 12),
    ("9", "CHEMBL4516405", 1430.1, 5.1, 280),
    ("10", "CHEMBL4513542", 287.8, 11.1, 26),
    ("11", "CHEMBL4559597", 285.0, 38.7, 7),
    ("cidofovir", "CHEMBL5314375", 475.3, 40.0, 12),
]
SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6335999/"


def main():
    queue_path = ROOT / "data/external/pair_review_queue.csv"
    queue = pd.read_csv(queue_path)
    train, test = pd.read_csv(ROOT / "data/raw/train.csv"), pd.read_csv(ROOT / "data/raw/test.csv")
    train_groups = {molecular_record(s)["compound_group"] for s in train.SMILES}
    test_groups = {molecular_record(s)["compound_group"] for s in test.SMILES}
    rows = []
    for label, molecule_id, cc50, ic50, reported_si in REVIEWED:
        matched = queue[(queue.molecule_id == molecule_id)
                        & queue.doi_ic50.str.lower().eq("10.1039/c8md00442k")
                        & queue.same_known_cell_type & queue.both_exact
                        & np.isclose(queue.CC50, cc50) & np.isclose(queue.IC50, ic50)]
        if len(matched) != 1:
            raise ValueError(f"Reviewed source row is no longer unique/consistent: {molecule_id}")
        row = matched.iloc[0]
        if row.assay_id_ic50 != "CHEMBL4369494" or row.assay_id_cc50 != "CHEMBL4369495":
            raise ValueError("Reviewed assay identifiers changed; re-review required.")
        group = molecular_record(row.smiles_ic50)["compound_group"]
        seen_train, seen_test = group in train_groups, group in test_groups
        rows.append({
            "smiles": row.smiles_ic50, "IC50": ic50, "CC50": cc50, "SI": cc50 / ic50,
            "doi": "10.1039/C8MD00442K", "concentration_unit": "uM", "relation": "=",
            "SI_reported_rounded": reported_si, "si_definition": "CC50/IC50",
            "cell_type": "Vero", "strain": "Vaccinia virus Copenhagen", "exposure": "4 days",
            "molecule_id": molecule_id, "publication_compound": label,
            "ic50_assay_id": row.assay_id_ic50, "cc50_assay_id": row.assay_id_cc50,
            "source_url": SOURCE, "source_locator": "Table 1; footnotes a-c; ChEMBL assay metadata",
            "reviewer": "pending independent manual confirmation",
            "review_date": "2026-11-06", "review_status": "paired_point_estimates_literature_checked",
            "measurement_uncertainty": "Published means; experimental intervals are in the source table; not clinical evidence",
            "compound_group": group, "overlap_train_parent": seen_train, "overlap_test_parent": seen_test,
            "eligible_as_new_training_data": not seen_train and not seen_test,
            "included_in_baseline_training": False,
        })
    result = pd.DataFrame(rows)
    path = ROOT / "data/external/external_curated.csv"
    result.to_csv(path, index=False)
    write_json(path.with_name("curation_manifest.json"), {
        "created_at": utc_now(), "source_url": SOURCE, "doi": "10.1039/C8MD00442K",
        "locator": "Table 1 and definition footnotes", "rows": len(result),
        "queue_sha256": sha256_file(queue_path), "curated_sha256": sha256_file(path),
        "not_used_for_model_training": True, "novel_nonoverlapping_parent_groups": int(result.eligible_as_new_training_data.sum()),
        "excluded_record": "CHEMBL4591029 / compound 15: publication describes a diastereomeric mixture; mapping to one molecular record requires further review.",
        "point_estimate_policy": "SI is computed from matched means; SI_reported_rounded preserves the publication rounding.",
        "scope": "Five paired point estimates verified against one primary article. No wet-lab protocol is supplied."})
    print(f"Saved {len(result)} literature-checked pairs, {int(result.eligible_as_new_training_data.sum())} novel non-overlapping parent groups. Baseline unchanged.")


if __name__ == "__main__":
    main()
