"""Part 1: the rotation floor. Every one of the 4^n - 1 rotation angles moves U in an
independent direction, so none can be removed.

Takes the circuit from 06-unitary-synthesis, treats its rotation angles as free variables, and
computes the rank of the Jacobian of U(angles) by central finite differences.

Run: python rotation_floor.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "06-unitary-synthesis"))
from solution import build_ops                          # noqa: E402
from synthesis_tools import haar_unitary                 # noqa: E402

I2 = np.eye(2)


def single(gate, wire, n):
    mats = [I2] * n
    mats[wire] = gate
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def cnot(c, t, n):
    dim = 2 ** n
    out = np.zeros((dim, dim))
    for x in range(dim):
        y = x ^ (1 << (n - 1 - t)) if (x >> (n - 1 - c)) & 1 else x
        out[y, x] = 1
    return out


def rz(a):
    return np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def ry(a):
    return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]])


def unitary_from(ops, angles, phase, n):
    u = np.eye(2 ** n, dtype=complex)
    k = 0
    for name, _, wires in ops:
        if name == "CNOT":
            u = cnot(*wires, n) @ u
        else:
            u = single((rz if name == "RZ" else ry)(angles[k]), wires[0], n) @ u
            k += 1
    return np.exp(1j * phase) * u


def jacobian_rank(ops, angles, phase, n, frozen=None, h=1e-6):
    """Rank of d vec(U) / d(angles, phase), real and imaginary parts stacked."""
    free = [i for i in range(len(angles)) if i != frozen]
    cols = []
    for i in free + ["phase"]:
        a_plus, a_minus, p_plus, p_minus = angles.copy(), angles.copy(), phase, phase
        if i == "phase":
            p_plus, p_minus = phase + h, phase - h
        else:
            a_plus[i] += h
            a_minus[i] -= h
        d = (unitary_from(ops, a_plus, p_plus, n) - unitary_from(ops, a_minus, p_minus, n)) / (2 * h)
        cols.append(np.concatenate([d.real.ravel(), d.imag.ravel()]))
    s = np.linalg.svd(np.array(cols).T, compute_uv=False)
    return int((s > 1e-6 * s[0]).sum()), s[-1] / s[0]


if __name__ == "__main__":
    for n in (2, 3):
        u = haar_unitary(n, seed=11 + n)
        ops, minus_phase = build_ops(u)
        angles = np.array([a for name, a, _ in ops if name != "CNOT"])
        phase = minus_phase                       # GlobalPhase(-psi) multiplies by e^{+i psi}
        assert np.abs(unitary_from(ops, angles, phase, n) - u).max() < 1e-12
        rank, cond = jacobian_rank(ops, angles, phase, n)
        rank_frozen, _ = jacobian_rank(ops, angles, phase, n, frozen=len(angles) // 2)
        print(f"n={n}: {len(angles)} angles + 1 phase.  rank {rank} of a possible {4 ** n}"
              f" (smallest/largest singular value {cond:.2e}).  Freeze one angle: rank {rank_frozen}")
        assert len(angles) == 4 ** n - 1 and rank == 4 ** n and rank_frozen == 4 ** n - 1
    print("Every angle is independent, and removing any one leaves a lower-dimensional family.")
