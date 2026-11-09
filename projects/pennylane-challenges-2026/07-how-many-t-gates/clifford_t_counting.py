"""Part 4: how much information does one T gate carry?

Counts the distinct Clifford+T operators at each exact T-count, for 1 and 2 qubits. Operators that
differ by a Clifford on the right are grouped into one coset gC, so the count is of cosets: the number
of genuinely different things you can reach with N T gates. Two cosets gC and g'C are equal exactly when
g P g† and g' P g'† give the same set of Paulis up to sign, which makes a cheap canonical key.

The 1-qubit count is checked against the known result (Matsumoto and Amano): 3 * 2^(N-1) cosets.
Writes clifford_t_counting.json for the figures.

Run: python clifford_t_counting.py   (about 15 s, mostly the last 2-qubit level)
"""
import json
import os
import time
from functools import reduce

import numpy as np

HERE = os.path.dirname(__file__)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])


def on(gate, wire, n):
    return reduce(np.kron, [gate if w == wire else I2 for w in range(n)])


def phase_key(m):
    """Canonical bytes of a matrix modulo global phase (+0.0 removes negative zeros)."""
    f = m.flat[np.argmax(np.abs(m).ravel() > 1e-9)]
    return (np.round(m * (abs(f) / f), 7) + 0.0).tobytes()


def clifford_group(n):
    gens = [on(H, w, n) for w in range(n)] + [on(S, w, n) for w in range(n)]
    if n == 2:
        gens.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex))
    group = {phase_key(np.eye(2 ** n, dtype=complex)): np.eye(2 ** n, dtype=complex)}
    frontier = list(group.values())
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                m = h @ g
                k = phase_key(m)
                if k not in group:
                    group[k] = m
                    new.append(m)
        frontier = new
    return list(group.values()), group


def count(n, max_t):
    cliffords, by_key = clifford_group(n)
    paulis = np.array([reduce(np.kron, p) for p in
                       __import__("itertools").product([I2, X, Y, Z], repeat=n)][1:])
    t0 = on(T, 0, n)
    # successors of a coset g C are g D T C, and D only matters modulo K = {D : T† D T is Clifford}
    K = [D for D in cliffords if phase_key(t0.conj().T @ D @ t0) in by_key]
    reps, seen = [], set()
    for D in cliffords:
        if phase_key(D) in seen:
            continue
        reps.append(D)
        seen.update(phase_key(D @ k) for k in K)
    step = np.array([D @ t0 for D in reps])

    def keys(G):
        q = np.einsum("bij,pjk,blk->bpil", G, paulis, G.conj()).reshape(len(G), len(paulis), -1)
        f = np.take_along_axis(q, np.argmax(np.abs(q) > 1e-9, axis=2)[..., None], axis=2)[..., 0]
        sign = np.where(np.abs(f.real) > 1e-9, np.sign(f.real), np.sign(f.imag))
        r = np.round(q * sign[..., None], 7) + 0.0
        return [tuple(sorted(r[b, p].tobytes() for p in range(len(paulis)))) for b in range(len(G))]

    level = np.eye(2 ** n, dtype=complex)[None]
    known = set(keys(level))
    sizes = []
    for N in range(1, max_t + 1):
        t = time.time()
        cand = np.einsum("gij,rjk->grik", level, step).reshape(-1, 2 ** n, 2 ** n)
        new = [i for i, k in enumerate(keys(cand)) if not (k in known or known.add(k))]
        level = cand[new]
        sizes.append(len(new))
        ratio = sizes[-1] / (sizes[-2] if len(sizes) > 1 else 1)
        print(f"  n={n}, T-count {N}: {len(new):>8,} new cosets  (x{ratio:5.2f}, {np.log2(ratio):.2f} bits)"
              f"  [{time.time() - t:.1f} s]", flush=True)
    return {"n": n, "clifford_group_size": len(cliffords), "K_size": len(K),
            "upper_bound_per_T": len(reps), "cosets_by_T_count": sizes}


if __name__ == "__main__":
    results = []
    for n, max_t in [(1, 12), (2, 5)]:
        r = count(n, max_t)
        print(f"n={n}: |Clifford| = {r['clifford_group_size']} (mod phase), new cosets per T at most "
              f"{r['upper_bound_per_T']} = 4^n - 1")
        results.append(r)
    one = results[0]["cosets_by_T_count"]
    assert one == [3 * 2 ** (N - 1) for N in range(1, len(one) + 1)], one      # Matsumoto-Amano
    assert results[1]["clifford_group_size"] == 11520
    print("1-qubit counts match 3 * 2^(N-1); the 2-qubit Clifford group has the known 11,520 elements.")
    with open(os.path.join(HERE, "clifford_t_counting.json"), "w") as f:
        json.dump(results, f, indent=1)
