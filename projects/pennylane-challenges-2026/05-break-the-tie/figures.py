"""Figure for the Break the Tie challenge. Run: python figures.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle as ps
import matplotlib.pyplot as plt
import numpy as np

from solution import PUBLIC, estimate_mean

HERE = os.path.dirname(__file__)
titles = ["test 1: decaying distribution, squared quantity", "test 2: uniform distribution"]

fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
for ax, (probs, vals, n, expected), title in zip(axes, PUBLIC, titles):
    d = {}
    est = estimate_mean(probs, vals, n, details=d)
    readout = np.array(d["readout"])
    bins = np.arange(len(readout))
    k, mirror = d["k"], 2 ** n - d["k"]
    # the bin the direct estimate points to: k = 2^n * arccos(1 - 2a) / pi
    k_direct = 2 ** n * np.arccos(1 - 2 * d["a_direct"]) / np.pi
    keep = min((k, mirror), key=lambda b: abs(b - k_direct))
    drop = mirror if keep == k else k
    ax.bar(bins, readout, width=1.0, color=ps.INK_2)
    ax.bar([keep], [readout[keep]], width=2.2, color=ps.BLUE)
    for b, text, color in [(keep, "kept", ps.BLUE), (drop, "same height, rejected", ps.INK_2)]:
        a = (1 - np.cos(np.pi * b / 2 ** n)) / 2
        ax.annotate(f"k = {b}: a = {a:.4f}\n{text}", (b, readout[b]), xytext=(8, -4),
                    textcoords="offset points", va="top", fontsize=9, color=ps.INK)
    ax.axvline(k_direct, color=ps.BLUE, lw=1, ls=":")
    ax.annotate(f"direct circuit:\na = {d['a_direct']:.4f}", (k_direct, 0.36), xytext=(-6, 0),
                textcoords="offset points", ha="right", fontsize=9, color=ps.INK)
    ax.set_title(f"{title}   →   estimate {est:.6f}", fontsize=10.5)
    ax.set_ylabel("probability")
    ax.set_ylim(0, 0.56)
    ax.grid(axis="x", visible=False)
axes[-1].set_xlabel(f"estimation-register bin k  (n = 8 wires, θ = k / 256)")
axes[-1].set_xlim(-2, 258)
fig.suptitle("Two peaks of exactly equal height. The cheap direct circuit picks one.", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_readout.png"))
print("wrote fig_readout.png")
