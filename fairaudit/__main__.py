"""Audit predictions in a CSV with the columns label, prediction, group, and optionally score.

Usage: python -m fairaudit results.csv "Model name" > card.md
"""
import csv
import sys

import numpy as np

from . import model_card


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["label"]) for r in rows])
    p = np.array([int(r["prediction"]) for r in rows])
    g = np.array([r["group"] for r in rows])
    s = np.array([float(r["score"]) for r in rows]) if "score" in rows[0] else None
    print(model_card(argv[2] if len(argv) > 2 else "Model", "Describe the intended use here.", y, p, g, s))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
