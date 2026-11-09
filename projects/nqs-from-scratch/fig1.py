"""Fig. 1: ground-state energy per site of the TFIM chain vs h/J.

(a) exact diagonalization at N = 10 against the exact infinite-chain result.
(b) finite-size error E0/N - e_inf for periodic and open boundaries.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from ed import ground_state

N = 10
J = 1.0
H_VALUES = np.linspace(0.05, 2.5, 50)

# Exact infinite chain (Jordan-Wigner / Pfeuty 1970):
# e_inf(h) = -(1/2pi) * integral_{-pi}^{pi} sqrt(J^2 + h^2 - 2 J h cos k) dk
def e_infinite(h, J=1.0):
    integrand = lambda k: np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k))
    value, _ = quad(integrand, -np.pi, np.pi, points=[0.0])
    return -value / (2 * np.pi)

e_inf = np.array([e_infinite(h, J) for h in H_VALUES])
e_pbc = np.array([ground_state(N, h, J, periodic=True)[0] / N for h in H_VALUES])
e_obc = np.array([ground_state(N, h, J, periodic=False)[0] / N for h in H_VALUES])

np.savez("fig1_data.npz", h=H_VALUES, e_inf=e_inf, e_pbc=e_pbc, e_obc=e_obc)

# Style: one ink for text and the reference curve, two validated series colors.
INK, MUTED, GRID = "#15171c", "#4a5160", "#e6e6e1"
PBC, OBC = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False,
})

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

# (a) energy per site
ax_a.plot(H_VALUES, e_inf, color=INK, lw=2, label="exact, infinite chain")
ax_a.plot(H_VALUES[::3], e_pbc[::3], "o", ms=6.5, color=PBC, mec="white", mew=1.2,
          label=f"ED, N = {N}, periodic")
ax_a.set_xlabel("h / J")
ax_a.set_ylabel(r"energy per site, $E_0 / NJ$")
ax_a.set_title("(a) Energy per site", loc="left", color=INK, fontsize=12)
ax_a.legend(loc="lower left")

# (b) finite-size error
ax_b.axhline(0, color=MUTED, lw=1)
ax_b.plot(H_VALUES, e_obc - e_inf, color=OBC, lw=2, label=f"open, N = {N}")
ax_b.plot(H_VALUES, e_pbc - e_inf, color=PBC, lw=2, label=f"periodic, N = {N}")
ax_b.text(H_VALUES[-1], (e_obc - e_inf)[-1] + 0.004, "open", color=INK, ha="right", va="bottom")
ax_b.text(H_VALUES[-1], (e_pbc - e_inf)[-1] - 0.004, "periodic", color=INK, ha="right", va="top")
ax_b.set_xlabel("h / J")
ax_b.set_ylabel(r"finite-size error, $(E_0/N - e_\infty)\,/\,J$")
ax_b.set_title("(b) Error relative to the infinite chain", loc="left", color=INK, fontsize=12)
ax_b.legend(loc="upper right")

for ax in (ax_a, ax_b):
    ax.axvline(1.0, color=MUTED, lw=1, ls=(0, (3, 3)))
ax_a.text(1.03, ax_a.get_ylim()[1], "critical point", color=MUTED, va="top", fontsize=10)

fig.savefig("fig1_energy.png", dpi=200, facecolor="white")
print("saved fig1_energy.png")
i = np.argmin(abs(H_VALUES - 1.0))
print(f"h={H_VALUES[i]:.2f}  e_inf={e_inf[i]:.5f}  pbc err={e_pbc[i]-e_inf[i]:+.4f}  obc err={e_obc[i]-e_inf[i]:+.4f}")
print("max |pbc err| at h =", H_VALUES[np.argmax(abs(e_pbc - e_inf))].round(2), " value", abs(e_pbc - e_inf).max().round(4))
