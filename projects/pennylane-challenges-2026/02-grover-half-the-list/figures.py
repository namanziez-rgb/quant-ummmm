"""Figures for the Grover challenge. Run: python figures.py"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle as ps
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qp

from solution import diffuser, generator, pair_oracle, probabilities, single_bit_oracle

HERE = os.path.dirname(__file__)
PAIRS = list(itertools.combinations([(0, 0), (0, 1), (1, 0), (1, 1)], 2))

# 1. The rotation picture: stuck at M/N = 1/2, exact at M/N = 1/4
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
for ax, alpha, steps, title in [
        (axes[0], 45, 3, "2 qubits, M/N = 1/2: every step lands 45° from |good⟩"),
        (axes[1], 30, 2, "add one qubit, M/N = 2/8: one step lands on |good⟩")]:
    t = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=ps.GRID, lw=1.2)
    ax.plot(np.cos(t), -np.sin(t), color=ps.GRID, lw=1.2)
    ax.axhline(0, color=ps.INK_2, lw=0.8)
    ax.axvline(0, color=ps.INK_2, lw=0.8)
    ax.annotate("|bad⟩", (1.04, -0.03), color=ps.INK_2, ha="left", va="top")
    ax.annotate("|good⟩", (0.03, 1.06), color=ps.INK_2, ha="left")
    for k in range(steps):
        ang = np.radians((2 * k + 1) * alpha)
        ax.annotate("", xy=(np.cos(ang), np.sin(ang)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ps.BLUE, lw=2, mutation_scale=14))
        p = np.sin(ang) ** 2
        name = "start" if k == 0 else f"after {k} step" + ("s" if k > 1 else "")
        if abs(np.degrees(ang) - 90) < 1:      # pointing straight at |good>: label to the right
            spot, ha = (0.12, 0.88), "left"
        else:
            spot, ha = (1.22 * np.cos(ang), 1.18 * np.sin(ang)), "center"
        ax.annotate(f"{name}\nP = {p:.2f}", spot, ha=ha, va="center", fontsize=9, color=ps.INK)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.3, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
fig.suptitle("Each Grover step rotates the state by 2α, where sin α = √(M/N)", fontsize=12, y=1.0)
fig.savefig(os.path.join(HERE, "fig_geometry.png"))
plt.close(fig)


# 2. The design choices that matter, scored on all six solution pairs
@qp.qnode(qp.device("default.qubit", wires=3))
def variant(kind, oracle):
    if kind == "plain":
        generator()
        oracle()
        diffuser(generator)
        return qp.probs(wires=(0, 1))
    full = qp.prod(qp.Hadamard(0), qp.Hadamard(1), qp.Hadamard(2))
    generator()
    if kind == "minus":
        qp.PauliX(2)
    qp.Hadamard(2)
    qp.ctrl(oracle, control=2)()
    if kind == "small_diffuser":
        diffuser(generator)
    else:
        qp.Reflection(full)
    return qp.probs(wires=(0, 1))


def success(probs, pair):
    return float(sum(probs[2 * a + b] for a, b in pair))


rows = []
for kind, label in [("plain", "plain Grover,\n2 qubits"),
                    ("small_diffuser", "aux |+⟩, but the\n2-qubit diffuser"),
                    ("minus", "aux |−⟩\n(tutorial habit)")]:
    vals = [success(variant(kind, pair_oracle(p)), p) for p in PAIRS]
    rows.append((label, min(vals), max(vals)))
answer = [success(probabilities(generator, pair_oracle(p), diffuser), p) for p in PAIRS]
rows.append(("aux |+⟩ and the\n3-qubit diffuser", min(answer), max(answer)))

fig, ax = plt.subplots(figsize=(7.2, 3.6))
labels = [r[0] for r in rows]
vals = [r[1] for r in rows]
colors = [ps.INK_2, ps.INK_2, ps.INK_2, ps.BLUE]
bars = ax.bar(labels, [max(v, 0.012) for v in vals], color=colors, width=0.56)
for bar, (label, lo, hi) in zip(bars, rows):
    ax.annotate(f"{lo:.2f}", (bar.get_x() + bar.get_width() / 2, max(lo, 0.012)),
                xytext=(0, 4), textcoords="offset points", ha="center", color=ps.INK)
    assert abs(hi - lo) < 1e-9          # identical for every one of the six pairs
ax.set_ylim(0, 1.15)
ax.set_ylabel("P(measure a solution)")
ax.set_title("Same value for all six ways of marking 2 of 4 states")
ax.grid(axis="x", visible=False)
fig.savefig(os.path.join(HERE, "fig_variants.png"))
plt.close(fig)

# 3. The submitted circuit, drawn for one oracle
fig, ax = qp.draw_mpl(probabilities, style="pennylane", decimals=None)(
    generator, single_bit_oracle(0), diffuser)
fig.suptitle("Uniform state on 3 wires, oracle controlled by wire 2, reflection about the same state",
             fontsize=11)
fig.savefig(os.path.join(HERE, "fig_circuit.png"))
print("wrote fig_geometry.png, fig_variants.png, fig_circuit.png")
print("variant values:", [(r[0].replace(chr(10), " "), round(r[1], 6)) for r in rows])
