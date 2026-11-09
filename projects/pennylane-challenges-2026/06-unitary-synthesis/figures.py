"""Figure for the synthesis challenge. Run: python figures.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle as ps
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qp

from solution import submission_circ
from synthesis_tools import baseline_cost, cost_and_counts, haar_unitary

HERE = os.path.dirname(__file__)
ns = [2, 3, 4, 5]


def default_qubit_unitary(u):
    qp.QubitUnitary(u, wires=range(int(round(np.log2(len(u))))))


rows = {"PennyLane's default QubitUnitary": [], "my construction (4ⁿ − 1 rotations)": []}
for n in ns:
    u = haar_unitary(n, seed=7 + n)
    for name, circ in zip(rows, [default_qubit_unitary, submission_circ]):
        rows[name].append(100 * (1 - cost_and_counts(circ, u)[0] / baseline_cost(n)))
print({k: [round(v, 2) for v in vs] for k, vs in rows.items()})

fig, ax = plt.subplots(figsize=(7.4, 4.4))
for grade, level in [("Bronze", 5), ("Silver", 20), ("Gold", 60)]:
    ax.axhline(level, color=ps.INK_2, lw=1, ls="--")
    ax.annotate(f"{grade} needs {level}% at n = 5", (3.05 if grade == "Bronze" else 2.0, level), xytext=(0, 4),
                textcoords="offset points", fontsize=9, color=ps.INK_2)
ax.axhline(0, color=ps.INK_2, lw=0.8)
for (name, ys), color in zip(rows.items(), [ps.ORANGE, ps.BLUE]):
    ax.plot(ns, ys, "-o", color=color, label=name, markeredgecolor=ps.SURFACE, markeredgewidth=1.5)
    ps.label_end(ax, ns[-1], ys[-1], f"{ys[-1]:.2f}%", color, dx=7, dy=9 if ys[-1] > 10 else 0)
ax.set_xticks(ns)
ax.set_xlabel("number of qubits n")
ax.set_ylabel("saving against the Shannon baseline (%)")
ax.set_title("4ⁿ − 1 rotations is the floor, so 21.06% is the most any rotation circuit can save")
ax.set_ylim(-5, 66)
ax.set_xlim(1.9, 5.45)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
fig.savefig(os.path.join(HERE, "fig_savings.png"))
print("wrote fig_savings.png")
