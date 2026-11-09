"""Figures for the T-count study. Run: python figures.py  (after the two data scripts)"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle as ps
import matplotlib.pyplot as plt
import numpy as np

from bounds import five_qubit_estimates

HERE = os.path.dirname(__file__)
grid = json.load(open(os.path.join(HERE, "gridsynth_tcounts.json")))
counting = {r["n"]: r for r in json.load(open(os.path.join(HERE, "clifford_t_counting.json")))}

# 1. One rotation, three prices
x = np.array([r["log2_inv_eps"] for r in grid])
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axvspan(10 * math.log2(10), 11 * math.log2(10), color=ps.GRID, alpha=0.7, lw=0)
ax.annotate("per-rotation precision\nthis challenge needs", (11 * math.log2(10), 8), xytext=(-6, 0),
            textcoords="offset points", ha="right", fontsize=9, color=ps.INK_2)
ax.plot(x, [r["mean"] for r in grid], "o", color=ps.BLUE, label="gridsynth, measured (no ancilla)",
        markeredgecolor=ps.SURFACE, markeredgewidth=1.5, zorder=3)
ax.plot(x, 3 * x, color=ps.BLUE, lw=1.2, ls="--", label="3·log₂(1/ε), ancilla-free lower bound")
ax.plot(x, 1.149 * x + 9.2, color=ps.ORANGE, label="repeat-until-success fit, 1.149·log₂(1/ε) + 9.2")
ax.axhline(44, color=ps.INK_2, lw=1, ls=":")
ax.annotate("the challenge charges 44", (x[0], 44), xytext=(0, 4), textcoords="offset points",
            fontsize=9, color=ps.INK_2)
ax.set_xlabel("log₂(1/ε)")
ax.set_ylabel("T gates per rotation")
ax.set_title("An exact Clifford+T rotation costs about 3·log₂(1/ε) T gates")
ax.set_ylim(0, 135)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
fig.savefig(os.path.join(HERE, "fig_rotation_cost.png"))
plt.close(fig)

# 2. Bits per T gate, measured
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for (n, r), color in zip(counting.items(), ps.SERIES):
    c = r["cosets_by_T_count"]
    bits = [math.log2(c[0])] + [math.log2(c[i] / c[i - 1]) for i in range(1, len(c))]
    N = np.arange(1, len(c) + 1)
    ax.plot(N, bits, "-o", color=color, label=f"{n} qubit{'s' if n > 1 else ''}, measured",
            markeredgecolor=ps.SURFACE, markeredgewidth=1.5)
    ceiling = math.log2(r["upper_bound_per_T"])
    ax.axhline(ceiling, color=color, lw=1, ls="--")
    ax.annotate(f"ceiling log₂(4ⁿ − 1) = {ceiling:.2f}", (len(c) + 0.2 if n == 2 else 7, ceiling),
                xytext=(0, 4), textcoords="offset points", fontsize=9, color=ps.INK_2)
    ps.label_end(ax, N[-1], bits[-1], f"{bits[-1]:.2f} bits", color, dy=-9)
ax.set_xlabel("T-count N")
ax.set_ylabel("bits of new choice per T gate")
ax.set_title("One T gate steers 1 bit on one qubit, about 3.3 bits on two")
ax.set_xticks(range(1, 13))
ax.set_ylim(0, 4.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
fig.savefig(os.path.join(HERE, "fig_bits_per_t.png"))
plt.close(fig)

# 3. Where Gold would have to live, at five qubits (eps = 1e-10 per gate)
eps_exp = 10
g10 = next(r["mean"] for r in grid if r["eps_exponent"] == eps_exp)
rows = [
    ("exact rotations via gridsynth, one by one", 1023 * g10, ps.INK_2),
    ("Shannon baseline", 57024, ps.INK_2),
    ("my circuit: 1023 rotations at 44 each", 45012, ps.BLUE),
    ("every single-qubit gate synthesized whole\n(perfect packing, counting estimate)", 1023 * eps_exp * math.log2(10), ps.INK_2),
    ("whole-unitary Clifford+T, counting estimate", five_qubit_estimates(eps_exp, counting)[1], ps.AQUA),
]
fig, ax = plt.subplots(figsize=(8.0, 4.2))
labels = [r[0] for r in rows][::-1]
vals = [r[1] for r in rows][::-1]
cols = [r[2] for r in rows][::-1]
ax.barh(labels, vals, color=cols, height=0.55)
for i, v in enumerate(vals):
    ax.annotate(f"{v:,.0f}", (v, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
ax.axvline(0.4 * 57024, color=ps.INK, lw=1.4, ls="--")
ax.text(0.4 * 57024, 1.01, "  Gold: 22,810", transform=ax.get_xaxis_transform(), fontsize=9.5,
        color=ps.INK, va="bottom")
ax.set_xscale("log")
ax.set_xlim(1500, 3e5)
ax.set_xlabel("cost at five qubits, T-gate equivalents (log scale)")
ax.set_title("Only whole-unitary synthesis fits under Gold, and nobody knows how to do it", pad=18)
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(HERE, "fig_where_gold_lives.png"))
print("wrote fig_rotation_cost.png, fig_bits_per_t.png, fig_where_gold_lives.png")
