"""Figures for the QAOA challenge. Run: python figures.py  (about 2 minutes)"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle as ps
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qp
from scipy.optimize import minimize

from solution import approximation_ratio

HERE = os.path.dirname(__file__)
GRAPHS = {"test 1: triangle with a tail": ([[0, 1], [1, 2], [0, 2], [2, 3]], 0.58160924),
          "test 2: 4-cycle": ([[0, 1], [1, 2], [2, 3], [3, 0]], 0.94746734)}


def classical_costs(edges, n):
    """H_cost is diagonal: its eigenvalue for every bitstring (bit 1 = in the cover)."""
    costs = np.zeros(2 ** n)
    for idx, bits in enumerate(itertools.product([0, 1], repeat=n)):
        z = [1 - 2 * b for b in bits]
        costs[idx] = 0.75 * sum(z[i] * z[j] + z[i] + z[j] for i, j in edges) - sum(z)
    return costs


def fast_qaoa(x, costs, n, depth=4):
    """Plain-NumPy QAOA with the same conventions as qp.qaoa: exp(-i g H_C), exp(-i a sum X)."""
    psi = np.ones(2 ** n, complex) / np.sqrt(2 ** n)
    for k in range(depth):
        psi = psi * np.exp(-1j * x[k] * costs)
        a = x[depth + k]
        rx = np.array([[np.cos(a), -1j * np.sin(a)], [-1j * np.sin(a), np.cos(a)]])
        psi = psi.reshape([2] * n)
        for q in range(n):
            psi = np.moveaxis(np.tensordot(rx, psi, axes=([1], [q])), 0, q)
        psi = psi.reshape(-1)
    return float(np.real(np.vdot(psi, costs * psi)))


# 1. The two graphs, with a minimum vertex cover
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, (name, (edges, _)) in zip(axes, GRAPHS.items()):
    costs = classical_costs(edges, 4)
    best = np.flatnonzero(np.isclose(costs, costs.min()))
    cover = [q for q, b in enumerate(format(best[0], "04b")) if b == "1"]
    g = nx.Graph([tuple(e) for e in edges])
    pos = nx.spring_layout(g, seed=3) if "tail" in name else nx.circular_layout(g)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=ps.INK_2, width=2)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=620, edgecolors=ps.INK,
                           node_color=[ps.BLUE if v in cover else ps.SURFACE for v in g.nodes])
    nx.draw_networkx_labels(g, pos, ax=ax, font_color=ps.INK, font_size=10,
                            labels={v: str(v) for v in g.nodes})
    ax.set_title(f"{name}\nmin cover {cover} (one of {len(best)}), E_min = {costs.min():g}", fontsize=10)
    ax.margins(0.15)
    ax.axis("off")
fig.savefig(os.path.join(HERE, "fig_graphs.png"))
plt.close(fig)

# 2. The optimization path, from the submitted code itself, run past 800 steps
STEPS = 1200
fig, ax = plt.subplots(figsize=(7.4, 4.0))
for (name, (edges, expected)), color in zip(GRAPHS.items(), ps.SERIES):
    hist = []
    approximation_ratio(edges, steps=STEPS, history=hist)
    emin = classical_costs(edges, 4).min()
    ratio = np.array(hist) / emin
    ax.axhspan(expected * 0.95, expected * 1.05, color=color, alpha=0.12, lw=0)
    ax.plot(np.arange(STEPS), ratio, color=color, label=name)
    ax.plot([800], [ratio[800]], "o", color=color, markeredgecolor=ps.SURFACE, markeredgewidth=1.5)
    ps.label_end(ax, STEPS, ratio[-1], f"{ratio[-1]:.3f}", color)
    print(f"{name}: ratio at 800 = {ratio[800]:.6f}, at 1000 = {ratio[1000]:.4f}, at {STEPS} = {ratio[-1]:.4f}")
ax.axvline(800, color=ps.INK_2, lw=1, ls="--")
ax.annotate("800 steps: matches the\nexpected outputs", (800, 0.08), xytext=(-8, 0),
            textcoords="offset points", ha="right", fontsize=9, color=ps.INK_2)
ax.set_xlabel("gradient-descent step (stepsize 0.01, all angles start at 0.5)")
ax.set_ylabel("approximation ratio")
ax.set_title("Shaded: the grader's ±5% band. The optimizer is still moving at 800")
ax.set_xlim(0, STEPS + 110)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.savefig(os.path.join(HERE, "fig_optimization.png"))
plt.close(fig)

# 3. The landscape: 200 random restarts with BFGS, against the grader's reference value
rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
for ax, (name, (edges, expected)), color in zip(axes, GRAPHS.items(), ps.SERIES):
    costs = classical_costs(edges, 4)
    # cross-check the fast simulator against PennyLane once
    x = rng.uniform(0, 1, 8)
    cost_h = 0.75 * qp.sum(*[qp.Z(i) @ qp.Z(j) + qp.Z(i) + qp.Z(j) for i, j in edges]) \
        - qp.sum(*[qp.Z(v) for v in range(4)])
    mixer_h = qp.sum(*[qp.X(v) for v in range(4)])

    @qp.qnode(qp.device("default.qubit", wires=4))
    def ref(x):
        for w in range(4):
            qp.Hadamard(w)
        for k in range(4):
            qp.qaoa.cost_layer(x[k], cost_h)
            qp.qaoa.mixer_layer(x[4 + k], mixer_h)
        return qp.expval(cost_h)

    assert np.isclose(fast_qaoa(x, costs, 4), float(ref(x)))
    finals = [minimize(lambda v: fast_qaoa(v, costs, 4), rng.uniform(0, np.pi, 8), method="BFGS").fun
              / costs.min() for _ in range(200)]
    ax.hist(finals, bins=np.linspace(0, 1, 41), color=color, edgecolor=ps.SURFACE, linewidth=1)
    ax.axvline(expected, color=ps.INK, lw=1.2, ls="--")
    ax.annotate(f"grader's value\n{expected:.3f}", (expected, ax.get_ylim()[1] * 0.92), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=9)
    ax.set_title(f"{name}\nbest of 200 restarts: {max(finals):.3f}", fontsize=10)
    ax.set_xlabel("ratio after BFGS from a random start")
    print(f"{name}: best restart {max(finals):.4f}, median {np.median(finals):.4f}")
axes[0].set_ylabel("restarts")
fig.savefig(os.path.join(HERE, "fig_landscape.png"))
print("wrote fig_graphs.png, fig_optimization.png, fig_landscape.png")
