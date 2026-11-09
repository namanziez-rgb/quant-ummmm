"""Plot the per-round logical error rate against physical error rate (run threshold.py first)."""
import numpy as np
import matplotlib.pyplot as plt

rows = np.load("threshold_data.npy")          # d, p, p_L/round, error, n_errors, shots
INK, MUTED, GRID = "#15171c", "#4a5160", "#e6e6e1"
COLORS = {3: "#2a78d6", 5: "#eb6834", 7: "#1baf7a"}
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False,
})

fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
for d, color in COLORS.items():
    r = rows[rows[:, 0] == d]
    ax.errorbar(100 * r[:, 1], r[:, 2], yerr=r[:, 3], color=color, lw=2, marker="o", ms=6,
                mec="white", mew=1.2, capsize=0, label=f"d = {d}")
    ax.text(100 * r[0, 1] - 0.06, r[0, 2], f"d = {d}", color=INK, ha="right", va="center")
ax.axvline(1.15, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(1.19, 1.5e-5 * 3, "threshold ≈ 1.1–1.2%", color=MUTED, fontsize=10)
ax.set_yscale("log")
ax.set_xlim(0, 2.1)
ax.set_xlabel("physical error rate p (%)")
ax.set_ylabel("logical error rate per round")
ax.set_title("Rotated surface code, circuit-level noise, MWPM decoding", loc="left", color=INK, fontsize=12)
ax.legend(loc="lower right")
fig.savefig("fig_threshold.png", dpi=200, facecolor="white")
print("saved fig_threshold.png")
