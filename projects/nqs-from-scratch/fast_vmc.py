"""Vectorized VMC for the RBM, for chains too long for the one-sample-at-a-time code in vmc.py.

Same physics as vmc.py and rbm.py, but many Markov chains run in parallel with NumPy, and the RBM's
hidden-unit inputs θ are updated incrementally: flipping spin i only changes θ by -2 W[:, i] s_i.
Checked against the slow code at N = 10 before use (see __main__).
"""
import numpy as np

from rbm import RBM, sr_step


def log_cosh2(x):
    return np.logaddexp(x, -x)  # log(2 cosh x), overflow-safe


class Sampler:
    def __init__(self, rbm, n, n_chains, rng):
        self.rbm, self.n, self.rng = rbm, n, rng
        self.s = rng.choice([-1.0, 1.0], size=(n_chains, n))
        self.refresh()

    def refresh(self):
        """Recompute θ after the parameters change."""
        self.theta = self.s @ self.rbm.W.T + self.rbm.b

    def sweep(self):
        """n single-spin-flip Metropolis proposals per chain."""
        C = len(self.s)
        rows = np.arange(C)
        for _ in range(self.n):
            i = self.rng.integers(self.n, size=C)
            si = self.s[rows, i]
            dtheta = -2 * si[:, None] * self.rbm.W[:, i].T            # (C, M)
            dlog = -2 * self.rbm.a[i] * si + np.sum(log_cosh2(self.theta + dtheta) - log_cosh2(self.theta), axis=1)
            accept = self.rng.random(C) < np.exp(2 * dlog)
            self.s[rows[accept], i[accept]] *= -1
            self.theta[accept] += dtheta[accept]

    def sample(self, n_sweeps):
        out = []
        for _ in range(n_sweeps):
            self.sweep()
            out.append(self.s.copy())
        return np.concatenate(out)


def local_energies(rbm, samples, h, J=1.0):
    """E_loc for a batch of configurations, all single-spin flips at once."""
    theta = samples @ rbm.W.T + rbm.b                                  # (S, M)
    diag = -J * np.sum(samples * np.roll(samples, -1, axis=1), axis=1)
    dtheta = -2 * samples[:, :, None] * rbm.W.T[None, :, :]           # (S, N, M)
    dlog = (-2 * samples * rbm.a[None, :]
            + np.sum(log_cosh2(theta[:, None, :] + dtheta) - log_cosh2(theta)[:, None, :], axis=2))
    return diag - h * np.sum(np.exp(dlog), axis=1)


def train_fast(n, h, J=1.0, alpha=1, n_iter=400, n_chains=100, sweeps=10, lr=0.02, diag_shift=1e-2,
               seed=0, log_every=100):
    rng = np.random.default_rng(seed)
    rbm = RBM(n, alpha, rng=rng)
    sampler = Sampler(rbm, n, n_chains, rng)
    for _ in range(50):
        sampler.sweep()                                                # burn-in
    history = []
    for it in range(n_iter):
        samples = sampler.sample(sweeps)
        e_loc = local_energies(rbm, samples, h, J)
        rbm.update(-lr * sr_step(e_loc, rbm.log_derivatives(samples), diag_shift))
        sampler.refresh()
        history.append((e_loc.mean() / n, e_loc.var()))
        if it % log_every == 0 or it == n_iter - 1:
            print(f"iter {it:4d}   E/N = {e_loc.mean() / n:+.5f}   var(E_loc) = {e_loc.var():.3e}", flush=True)
    return rbm, sampler, np.array(history)


def estimate_energy(rbm, sampler, h, J=1.0, n_sweeps=200, n_bins=20):
    samples = sampler.sample(n_sweeps)
    e_loc = local_energies(rbm, samples, h, J)
    n = samples.shape[1]
    bins = e_loc.reshape(n_bins, -1).mean(axis=1)
    return e_loc.mean() / n, bins.std(ddof=1) / np.sqrt(n_bins) / n, e_loc.var()


if __name__ == "__main__":
    # Consistency check against the slow, one-sample-at-a-time code
    from vmc import local_energy
    rng = np.random.default_rng(1)
    rbm = RBM(10, 1, init_scale=0.3, rng=rng)
    configs = rng.choice([-1.0, 1.0], size=(50, 10))
    slow = np.array([local_energy(c.copy(), rbm.log_psi, 1.0) for c in configs])
    fast = local_energies(rbm, configs, 1.0)
    print("max |E_loc fast - slow| =", np.abs(fast - slow).max())
