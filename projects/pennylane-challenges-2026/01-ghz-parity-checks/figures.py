"""Figures for the GHZ challenge. Run: python figures.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle as ps
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qp

from solution import ghz, mixture_score, plus_product, score_of, wrong_phase

HERE = os.path.dirname(__file__)

# 1. The circuit and the checks, for n = 4
n = 4
dev = qp.device("default.qubit", wires=n)


@qp.qnode(dev)
def circuit():
    ghz(n)
    return qp.expval(qp.prod(*[qp.X(w) for w in range(n)]))


fig, ax = qp.draw_mpl(circuit, style="pennylane")()
fig.suptitle("GHZ$_4$: one Hadamard, then a CNOT fan-out", fontsize=12)
fig.savefig(os.path.join(HERE, "fig_circuit.png"))
plt.close(fig)

# 2. Scores of GHZ and the three decoys
ns = np.arange(2, 11)
rows = {
    "GHZ": [1.0 for _ in ns],
    "classical mixture": [mixture_score(k) for k in ns],
    "wrong phase": [score_of(wrong_phase, k) for k in ns],
    r"$|+\rangle^{\otimes n}$": [score_of(plus_product, k) for k in ns],
}
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for (name, ys), color in zip(rows.items(), ps.SERIES):
    ax.plot(ns, ys, "-o", color=color, label=name, markeredgecolor=ps.SURFACE, markeredgewidth=1.5)
    ps.label_end(ax, ns[-1], ys[-1], name, color)
ax.set_xlabel("number of qubits n")
ax.set_ylabel("mean of the n parity checks")
ax.set_title("Only GHZ passes all n checks")
ax.set_ylim(-0.05, 1.08)
ax.set_xlim(1.6, 13.2)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=4, fontsize=9.5)
fig.savefig(os.path.join(HERE, "fig_scores.png"))
print("wrote fig_circuit.png, fig_scores.png")
