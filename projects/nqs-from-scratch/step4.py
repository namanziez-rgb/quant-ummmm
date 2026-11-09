"""Step 4: benchmark the SR-trained RBM across the field h, and far beyond exact diagonalization.

Part A (N = 10): train at several h; compare the trained RBM's exact energy and fidelity with ED.
Part B (h = 1, the critical point): N = 20 and 40 against the exact finite-N Jordan-Wigner energy,
where 2^N amplitudes could never be stored (2^40 is about 10^12).
Uses the vectorized sampler in fast_vmc.py.  Usage: python step4.py
"""
import numpy as np

from ed import ground_state, tfim_hamiltonian
from fast_vmc import estimate_energy, train_fast
from rbm import all_configs


def e_jordan_wigner(n, h, J=1.0):
    """Exact ground-state energy per site of the periodic chain (even-parity sector, antiperiodic momenta)."""
    k = (2 * np.arange(1, n + 1) - 1) * np.pi / n
    return -np.sum(np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k))) / n


def scan(n=10, h_values=(0.2, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0)):
    configs = all_configs(n)
    rows = []
    for h in h_values:
        rbm, _, _ = train_fast(n, h, n_iter=300, diag_shift=1e-3, log_every=10**9)
        psi = np.exp([rbm.log_psi(s) for s in configs])
        psi /= np.linalg.norm(psi)
        e0, psi_ed = ground_state(n, h)
        e_rbm = psi @ tfim_hamiltonian(n, h) @ psi
        rel, fid = abs((e_rbm - e0) / e0), abs(psi @ psi_ed) ** 2
        rows.append((h, e0 / n, e_rbm / n, rel, fid))
        print(f"N={n} h={h:.2f}  E_ED/N={e0 / n:+.6f}  E_RBM/N={e_rbm / n:+.6f}  "
              f"rel.err={rel:.1e}  fidelity={fid:.5f}", flush=True)
    np.save("step4_scan.npy", np.array(rows))


def large(h=1.0, sizes=(20, 40)):
    rows = []
    for n in sizes:
        rbm, sampler, _ = train_fast(n, h, n_iter=600, log_every=10**9)
        e, err, var = estimate_energy(rbm, sampler, h, n_sweeps=400)
        exact = e_jordan_wigner(n, h)
        rel = abs((e - exact) / exact)
        rows.append((n, exact, e, err, rel, var))
        print(f"N={n} h={h:.1f}  exact E/N={exact:+.6f}  VMC E/N={e:+.6f} ± {err:.6f}  "
              f"rel.err={rel:.1e}  var(E_loc)/N={var / n:.2e}", flush=True)
    np.save("step4_large.npy", np.array(rows))


if __name__ == "__main__":
    scan()
    large()
