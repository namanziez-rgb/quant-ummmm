"""Fig. 3: accuracy of the SR-trained RBM across the field h at N = 10 (run step4.py first)."""
import numpy as np
import matplotlib.pyplot as plt

rows = np.load("step4_scan.npy")       # h, E_ED/N, E_RBM/N, relative error, fidelity
INK, MUTED, GRID, BLUE = "#15171c", "#4a5160", "#e6e6e1", "#2a78d6"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8,
})
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
ax.semilogy(rows[:, 0], rows[:, 3], color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.2)
ax.axvline(1.0, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(1.03, rows[:, 3].min() * 1.1, "critical point", color=MUTED, fontsize=10)
ax.set_xlabel("h / J")
ax.set_ylabel("relative energy error of the trained RBM")
ax.set_title("RBM with stochastic reconfiguration, N = 10, against exact diagonalization",
             loc="left", color=INK, fontsize=12)
fig.savefig("fig3_scan.png", dpi=200, facecolor="white")
print("saved fig3_scan.png")
