"""Restricted Boltzmann machine wavefunction, trained with variational Monte Carlo.

log ψ(s) = Σ_i a_i s_i + Σ_j log(2 cosh θ_j),   θ_j = b_j + Σ_i W_ji s_i

Real parameters are enough for the TFIM, whose ground-state amplitudes are all positive.
"""
import numpy as np

from vmc import local_energy, metropolis


class RBM:
    def __init__(self, n_visible, alpha=1, init_scale=0.01, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        n_hidden = alpha * n_visible
        self.a = np.zeros(n_visible)
        self.b = np.zeros(n_hidden)
        self.W = init_scale * rng.standard_normal((n_hidden, n_visible))

    @property
    def n_params(self):
        return self.a.size + self.b.size + self.W.size

    def log_psi(self, s):
        theta = self.b + self.W @ s
        return self.a @ s + np.sum(np.logaddexp(theta, -theta))  # log(2 cosh θ), overflow-safe

    def log_derivatives(self, samples):
        """O_k(s) = ∂ log ψ(s) / ∂ param_k, one row per sample, ordered as (a, b, W)."""
        t = np.tanh(samples @ self.W.T + self.b)                       # (S, M)
        o_W = (t[:, :, None] * samples[:, None, :]).reshape(len(samples), -1)
        return np.hstack([samples, t, o_W])

    def update(self, delta):
        n, m = self.a.size, self.b.size
        self.a += delta[:n]
        self.b += delta[n:n + m]
        self.W += delta[n + m:].reshape(self.W.shape)


def energy_gradient(e_loc, O):
    """∂E/∂θ_k = 2 Cov(E_loc, O_k) over the samples (real wavefunction)."""
    return 2 * ((e_loc - e_loc.mean())[:, None] * (O - O.mean(axis=0))).mean(axis=0)


def sr_step(e_loc, O, diag_shift):
    """Stochastic reconfiguration: solve (S + shift·1) δ = F, with
    S_kl = Cov(O_k, O_l)  (how much ψ changes when you move θ_k and θ_l), and
    F_k  = Cov(E_loc, O_k) (half the energy gradient)."""
    dO = O - O.mean(axis=0)
    S = dO.T @ dO / len(O)
    F = ((e_loc - e_loc.mean())[:, None] * dO).mean(axis=0)
    return np.linalg.solve(S + diag_shift * np.eye(len(F)), F)


def train(n, h, J=1.0, alpha=1, n_iter=300, method="sr", lr=None, diag_shift=1e-3,
          n_samples=1000, seed=0, log_every=25):
    """method="gd": plain gradient descent (default lr 0.05).
    method="sr": stochastic reconfiguration (default lr 0.02)."""
    if lr is None:
        lr = 0.05 if method == "gd" else 0.02
    rng = np.random.default_rng(seed)
    rbm = RBM(n, alpha, rng=rng)
    history = []
    for it in range(n_iter):
        samples, acceptance = metropolis(rbm.log_psi, n, n_samples, n_burn=50, rng=rng)
        e_loc = np.array([local_energy(s, rbm.log_psi, h, J) for s in samples])
        O = rbm.log_derivatives(samples)
        if method == "gd":
            rbm.update(-lr * energy_gradient(e_loc, O))
        else:
            rbm.update(-lr * sr_step(e_loc, O, diag_shift))
        history.append((e_loc.mean() / n, e_loc.var()))
        if it % log_every == 0 or it == n_iter - 1:
            print(f"iter {it:4d}   E/N = {e_loc.mean() / n:+.5f}   var(E_loc) = {e_loc.var():.3e}"
                  f"   acceptance = {acceptance:.2f}")
    return rbm, np.array(history)


def all_configs(n):
    """All 2^n spin configurations, in the same order as the ED basis (site 0 = leftmost bit, up = 0)."""
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> np.arange(n)[::-1]) & 1
    return 1 - 2 * bits


if __name__ == "__main__":
    from ed import ground_state, tfim_hamiltonian

    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else "sr"
    n, h = 10, 1.0
    rbm, history = train(n, h, method=method)
    np.save(f"history_{method}_h1.npy", history)

    # At N = 10 we can still check the trained RBM exactly, with no sampling noise.
    configs = all_configs(n)
    psi_rbm = np.exp([rbm.log_psi(s) for s in configs])
    psi_rbm /= np.linalg.norm(psi_rbm)
    e0, psi_ed = ground_state(n, h)
    e_rbm = psi_rbm @ tfim_hamiltonian(n, h) @ psi_rbm
    print(f"\nparameters: {rbm.n_params}  (vs {2 ** n} amplitudes)")
    print(f"RBM exact energy  E/N = {e_rbm / n:+.6f}")
    print(f"ED ground state   E/N = {e0 / n:+.6f}")
    print(f"relative error        = {abs((e_rbm - e0) / e0):.1e}")
    print(f"fidelity |<RBM|ED>|²  = {abs(psi_rbm @ psi_ed) ** 2:.5f}")
