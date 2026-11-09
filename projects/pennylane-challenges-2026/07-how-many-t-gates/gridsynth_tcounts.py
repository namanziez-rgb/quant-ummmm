"""Part 2: what one rotation costs as an exact Clifford+T sequence.

Runs gridsynth (Ross and Selinger) on random RZ angles at several precisions, and writes the
numbers to gridsynth_tcounts.json for the figures.

Run: python gridsynth_tcounts.py   (about 10 s)
"""
import json
import os
import random

import mpmath
from pygridsynth import gridsynth_gates

HERE = os.path.dirname(__file__)

if __name__ == "__main__":
    random.seed(2026)
    rows = []
    for exp in range(4, 13):
        counts = []
        for _ in range(8):
            theta = mpmath.mpf(random.uniform(0, 2 * 3.141592653589793))
            gates = gridsynth_gates(theta=theta, epsilon=mpmath.mpf(10) ** (-exp))
            counts.append(str(gates).count("T"))
        mean = sum(counts) / len(counts)
        log2_inv = exp * 3.321928094887362
        rows.append({"eps_exponent": exp, "t_counts": counts, "mean": mean, "log2_inv_eps": log2_inv})
        print(f"eps = 1e-{exp:<2}  T-counts {counts}  mean {mean:6.1f}   3 log2(1/eps) = {3 * log2_inv:6.1f}"
              f"   RUS fit 1.149 log2(1/eps) + 9.2 = {1.149 * log2_inv + 9.2:5.1f}")
    with open(os.path.join(HERE, "gridsynth_tcounts.json"), "w") as f:
        json.dump(rows, f, indent=1)
