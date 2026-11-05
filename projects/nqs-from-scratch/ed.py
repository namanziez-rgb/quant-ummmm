"""Exact diagonalization of the transverse-field Ising chain.

H = -J sum_i sz_i sz_{i+1} - h sum_i sx_i
Boundaries are periodic by default; pass periodic=False for an open chain.
"""
import numpy as np

SX = np.array([[0.0, 1.0], [1.0, 0.0]])
SZ = np.array([[1.0, 0.0], [0.0, -1.0]])
I2 = np.eye(2)


def embed(ops, n):
    """Tensor product over n sites. ops maps site -> 2x2 operator; identity everywhere else."""
    out = np.array([[1.0]])
    for j in range(n):
        out = np.kron(out, ops.get(j, I2))
    return out


def tfim_hamiltonian(n, h, J=1.0, periodic=True):
    dim = 2 ** n
    H = np.zeros((dim, dim))
    n_bonds = n if periodic else n - 1
    for i in range(n_bonds):
        H -= J * embed({i: SZ, (i + 1) % n: SZ}, n)  # bond i, i+1 (wraps if periodic)
    for i in range(n):
        H -= h * embed({i: SX}, n)                    # field on site i
    return H


def ground_state(n, h, J=1.0, periodic=True):
    energies, states = np.linalg.eigh(tfim_hamiltonian(n, h, J, periodic))
    return energies[0], states[:, 0]


if __name__ == "__main__":
    # Check 1: periodic ground-state energy per site against reference values
    reference = {
        (8, 0.5): -1.063635, (8, 1.0): -1.281458, (8, 2.0): -2.127271,
        (10, 0.5): -1.063560, (10, 1.0): -1.278491, (10, 2.0): -2.127121,
    }
    for (n, h), ref in reference.items():
        e0, _ = ground_state(n, h)
        print(f"N={n:2d} h={h:.1f}  E0/N={e0 / n:+.6f}  ref={ref:+.6f}  diff={abs(e0 / n - ref):.1e}")

    # Check 2: where the probability weight sits, N = 10
    n = 10
    for h in (0.1, 1.0, 5.0):
        _, psi = ground_state(n, h)
        p = psi ** 2
        print(f"h={h:.1f}  all up: {p[0]:.3f}  all down: {p[-1]:.3f}  "
              f"smallest: {p.min():.1e}  uniform: {1 / 2 ** n:.1e}")
