#!/usr/bin/env python3
"""Create a review queue; never treat assay co-occurrence as verified selectivity."""
from pathlib import Path
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/external/external_candidates.csv")
parser.add_argument("--output", default="data/external/pair_review_queue.csv")
args = parser.parse_args()
frame = pd.read_csv(args.input)
columns = ["molecule_id", "document_id", "doi", "smiles", "assay_id", "cell_type",
           "assay_organism", "strain", "assay_description", "source_url", "relation"]
ic = frame[frame.measurement_type.eq("IC50") & frame.IC50.notna()][columns + ["IC50"]]
cc = frame[frame.measurement_type.eq("CC50") & frame.CC50.notna()][columns + ["CC50"]]
pairs = ic.merge(cc, on=["molecule_id", "document_id"], suffixes=("_ic50", "_cc50"))
if len(pairs):
    pairs["same_known_cell_type"] = pairs.cell_type_ic50.notna() & pairs.cell_type_cc50.notna() & pairs.cell_type_ic50.eq(pairs.cell_type_cc50)
    pairs["both_exact"] = pairs.relation_ic50.eq("=") & pairs.relation_cc50.eq("=")
    # The ratio deliberately remains uncomputed until the scientific context has been reviewed.
    pairs["SI"] = float("nan")
    pairs["review_status"] = "pending: verify cell line, exposure, assay endpoint, strain and publication table"
    pairs["reviewer"], pairs["evidence"], pairs["include_in_training"] = "", "", False
    pairs = pairs.sort_values(["same_known_cell_type", "both_exact"], ascending=False)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
pairs.to_csv(args.output, index=False)
print(f"Saved {len(pairs)} candidate pairs for review to {args.output}. None are approved or used for training.")
