"""Fig. 2: training the RBM at N = 10, h = 1 with plain gradient descent vs stochastic reconfiguration.

Run `python rbm.py gd` and `python rbm.py sr` first; they save the training histories.
"""
import numpy as np
import matplotlib.pyplot as plt

from ed import ground_state

N, H_FIELD = 10, 1.0
E_ED = ground_state(N, H_FIELD)[0] / N
gd = np.load("history_gd_h1.npy")   # columns: E/N, var(E_loc)
sr = np.load("history_sr_h1.npy")

def running_mean(x, w=10):
    return np.convolve(x, np.ones(w) / w, mode="valid")

INK, MUTED, GRID = "#15171c", "#4a5160", "#e6e6e1"
SR, GD = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False,
})
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

# (a) energy per site, smoothed over 10 iterations
it = np.arange(len(gd))
w = 10
ax_a.axhline(E_ED, color=INK, lw=1.5, ls=(0, (4, 3)), label="exact (ED)")
ax_a.plot(it[w - 1:], running_mean(gd[:, 0], w), color=GD, lw=2, label="plain gradient descent")
ax_a.plot(it[w - 1:], running_mean(sr[:, 0], w), color=SR, lw=2, label="stochastic reconfiguration")
ax_a.set_ylim(E_ED - 0.01, -1.20)
ax_a.set_xlabel("training iteration")
ax_a.set_ylabel(r"energy per site, $E/NJ$ (10-iteration mean)")
ax_a.set_title("(a) Energy", loc="left", color=INK, fontsize=12)
ax_a.legend(loc="upper right")

# (b) variance of the local energy: zero only for an exact eigenstate
ax_b.semilogy(it, gd[:, 1], color=GD, lw=1.5, label="plain gradient descent")
ax_b.semilogy(it, sr[:, 1], color=SR, lw=1.5, label="stochastic reconfiguration")
ax_b.text(230, 0.5, "gradient descent", color=INK, ha="center", va="bottom")
ax_b.text(230, 0.0022, "SR", color=INK, ha="center", va="top")
ax_b.set_ylim(1e-3, 30)
ax_b.set_xlabel("training iteration")
ax_b.set_ylabel(r"variance of the local energy, var$(E_\mathrm{loc})\,/\,J^2$")
ax_b.set_title("(b) Distance from an eigenstate", loc="left", color=INK, fontsize=12)
ax_b.legend(loc="upper right")

fig.savefig("fig2_training.png", dpi=200, facecolor="white")
print("saved fig2_training.png")
