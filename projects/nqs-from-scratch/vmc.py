"""Variational Monte Carlo for the TFIM chain (periodic boundaries).

A wavefunction is any function log_psi(s) -> log ψ(s), where s is an array of
spins s_i in {+1, -1}. We only ever need ratios ψ(s')/ψ(s), so ψ never has
to be normalized. Working with log ψ keeps the ratios numerically safe.
"""
import numpy as np


def local_energy(s, log_psi, h, J=1.0):
    """E_loc(s) = -J sum_i s_i s_{i+1} - h sum_i ψ(s with spin i flipped) / ψ(s)."""
    n = len(s)
    diagonal = -J * np.sum(s * np.roll(s, -1))
    log_psi_s = log_psi(s)
    off_diagonal = 0.0
    for i in range(n):
        s[i] *= -1                                   # flip spin i
        off_diagonal += np.exp(log_psi(s) - log_psi_s)
        s[i] *= -1                                   # flip it back
    return diagonal - h * off_diagonal


def metropolis(log_psi, n, n_samples, n_burn=200, rng=None):
    """Sample s from p(s) ∝ |ψ(s)|² with single-spin-flip Metropolis.

    One sample is recorded per sweep (n proposed flips), which reduces the
    correlation between consecutive samples.
    """
    rng = np.random.default_rng() if rng is None else rng
    s = rng.choice([-1, 1], size=n)
    log_psi_s = log_psi(s)
    samples, accepted, proposed = [], 0, 0
    for sweep in range(n_burn + n_samples):
        for _ in range(n):
            i = rng.integers(n)
            s[i] *= -1
            log_psi_new = log_psi(s)
            # accept with probability min(1, |ψ(s')/ψ(s)|²)
            if rng.random() < np.exp(2 * (log_psi_new - log_psi_s)):
                log_psi_s = log_psi_new
                accepted += 1
            else:
                s[i] *= -1                           # reject: undo the flip
            proposed += 1
        if sweep >= n_burn:
            samples.append(s.copy())
    return np.array(samples), accepted / proposed


def vmc_energy(log_psi, n, h, J=1.0, n_samples=4000, rng=None, n_bins=20):
    """Estimate E/N with an error bar from binned sample means."""
    samples, acceptance = metropolis(log_psi, n, n_samples, rng=rng)
    e_loc = np.array([local_energy(s, log_psi, h, J) for s in samples])
    bin_means = e_loc.reshape(n_bins, -1).mean(axis=1)
    return {
        "E/N": e_loc.mean() / n,
        "error": bin_means.std(ddof=1) / np.sqrt(n_bins) / n,
        "var(E_loc)": e_loc.var(),
        "acceptance": acceptance,
    }


# ---- Helpers for testing with ED states as lookup tables ----

def config_to_index(s):
    """Spin config -> basis index, matching ed.py: site 0 is the leftmost bit, up = 0."""
    bits = (s < 0).astype(int)
    return int(bits @ (1 << np.arange(len(s))[::-1]))


def lookup_log_psi(psi):
    """Turn a vector of 2^N amplitudes into a log_psi function."""
    log_amps = np.log(np.abs(psi))  # TFIM ground state: all amplitudes share one sign
    return lambda s: log_amps[config_to_index(s)]


if __name__ == "__main__":
    from ed import ground_state, tfim_hamiltonian

    n, h = 10, 1.0
    rng = np.random.default_rng(seed=7)
    H = tfim_hamiltonian(n, h)

    def exact_energy(psi):
        return psi @ H @ psi / (psi @ psi) / n

    tests = {
        "exact ground state, h = 1.0": ground_state(n, 1.0)[1],
        "uniform state": np.ones(2 ** n),
        "ground state of h = 0.5": ground_state(n, 0.5)[1],
    }
    for name, psi in tests.items():
        result = vmc_energy(lookup_log_psi(psi), n, h, rng=rng)
        print(f"{name}")
        print(f"   VMC  E/N = {result['E/N']:+.5f} ± {result['error']:.5f}"
              f"   exact = {exact_energy(psi):+.5f}")
        print(f"   var(E_loc) = {result['var(E_loc)']:.2e}   acceptance = {result['acceptance']:.2f}")
