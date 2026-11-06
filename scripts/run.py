#!/usr/bin/env python3
"""Run from the project root, including before an editable package install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vacv.cli import main

if __name__ == "__main__":
    main()
