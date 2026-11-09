"""Figure for the QPE resource-budget challenge. Run: python figures.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle as ps
import matplotlib.pyplot as plt
import numpy as np

from solution import GATE_SET, ROTATION_PRECISION, qre

HERE = os.path.dirname(__file__)
config = qre.ResourceConfig()
config.set_single_qubit_rot_precision(ROTATION_PRECISION)


def t_count(base, m):
    res = qre.estimate(qre.QPE(base(), num_estimation_wires=m), gate_set=GATE_SET, config=config)
    return int(res.gate_counts.get("T", 0))


ms = np.arange(1, 15)
series = {"generic 2-qubit unitary": [t_count(lambda: qre.QubitUnitary(num_wires=2), m) for m in ms],
          "single RZ rotation": [t_count(lambda: qre.RZ(), m) for m in ms]}

fig, ax = plt.subplots(figsize=(7.2, 4.2))
for (name, ys), color in zip(series.items(), ps.SERIES):
    ax.plot(ms, ys, "-o", color=color, label=name, markeredgecolor=ps.SURFACE, markeredgewidth=1.5)
    ps.label_end(ax, ms[-1], ys[-1], name, color, dx=6)
for budget, text, x, dy, va in [(1e6, "budget of test 1: 1 s at 1 µs per T", 1, 4, "bottom"),
                                (6e4, "budget of test 2: 60 s at 1 ms per T", 7.6, -4, "top")]:
    ax.axhline(budget, color=ps.INK_2, lw=1, ls="--")
    ax.annotate(text, (x, budget), xytext=(0, dy), textcoords="offset points", va=va,
                fontsize=9, color=ps.INK_2)
generic = series["generic 2-qubit unitary"]
for m_best in (9, 5):
    ax.annotate(f"answer: m = {m_best}\n{generic[m_best - 1]:,} T", (m_best, generic[m_best - 1]),
                xytext=(10, -8), textcoords="offset points", ha="left", va="top",
                fontsize=9, color=ps.INK)
ax.set_xticks(ms)
ax.set_yscale("log")
ax.set_xlabel("estimation wires m  (phase resolution 2⁻ᵐ)")
ax.set_ylabel("T-count")
ax.set_title("Each extra bit of phase doubles the work, unless the powers of U collapse")
ax.set_xlim(0.5, 18.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.savefig(os.path.join(HERE, "fig_tcount.png"))
for m, a, b in zip(ms, *series.values()):
    print(f"m={m:2d}: generic {a:>12,}   RZ {b:>8,}   ratio to previous m: {a / max(1, series['generic 2-qubit unitary'][m - 2]) if m > 1 else 0:.3f}")
print("wrote fig_tcount.png")
