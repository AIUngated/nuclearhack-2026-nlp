#!/usr/bin/env python3
"""Compare regenerated prediction values and identifiers with the delivered CSV."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from vacv.verification import compare_submissions

parser = argparse.ArgumentParser()
parser.add_argument("reference")
parser.add_argument("reproduced")
args = parser.parse_args()
reference, reproduced = pd.read_csv(args.reference), pd.read_csv(args.reproduced)
result = compare_submissions(reference, reproduced)
print(f"Reproduction verified: {result['rows']} rows; max absolute difference={result['max_absolute_difference']:.3g}; values agree within 1e-9 absolute / 1e-12 relative.")
