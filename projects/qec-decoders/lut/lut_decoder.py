"""Lookup-table decoder for the distance-3 rotated surface code, checked against PyMatching.

Layout: 9 data qubits on a 3x3 grid, qubit index = 3*row + col.
Code-capacity noise: independent errors on data qubits, perfect syndrome measurement.
X errors are caught by the Z checks and Z errors by the X checks, so each type gets its own
16-entry table (4 checks -> 4-bit syndrome).
"""
import itertools
import numpy as np
import pymatching

N_QUBITS = 9
# Z-type checks (detect X errors) and X-type checks (detect Z errors)
Z_CHECKS = [[1, 2, 4, 5], [3, 4, 6, 7], [0, 3], [5, 8]]
X_CHECKS = [[0, 1, 3, 4], [4, 5, 7, 8], [1, 2], [6, 7]]
# A residual X error is a logical flip if it anticommutes with logical Z (top row),
# and a residual Z error if it anticommutes with logical X (left column).
LOGICAL_Z = [0, 1, 2]
LOGICAL_X = [0, 3, 6]


def check_matrix(checks):
    H = np.zeros((len(checks), N_QUBITS), dtype=np.uint8)
    for row, support in enumerate(checks):
        H[row, support] = 1
    return H


def support_vector(support):
    v = np.zeros(N_QUBITS, dtype=np.uint8)
    v[support] = 1
    return v


def syndrome_index(syndrome_bits):
    """Bit k of the index is check k."""
    return int(sum(int(b) << k for k, b in enumerate(syndrome_bits)))


ALL_ERRORS = np.array(list(itertools.product([0, 1], repeat=N_QUBITS)), dtype=np.uint8)


def build_lut(H):
    """For every syndrome, store a minimum-weight error that produces it."""
    lut = {}
    for e in sorted(ALL_ERRORS, key=lambda e: e.sum()):
        s = syndrome_index(H @ e % 2)
        lut.setdefault(s, e)
    assert len(lut) == 2 ** H.shape[0], "every syndrome must be reachable"
    return lut


def failures(H, logical, decode):
    """Decode every one of the 512 error patterns; True where the residual is a logical error."""
    L = support_vector(logical)
    out = np.zeros(len(ALL_ERRORS), dtype=bool)
    for i, e in enumerate(ALL_ERRORS):
        residual = (e + decode(H @ e % 2)) % 2
        assert not (H @ residual % 2).any(), "correction must clear the syndrome"
        out[i] = residual @ L % 2 == 1
    return out


def exact_failure_probability(fail, p):
    """P_fail(p) = sum over failing patterns of p^w (1-p)^(9-w). Exact, no sampling."""
    w = ALL_ERRORS.sum(axis=1)[fail]
    return np.sum(p ** w * (1 - p) ** (N_QUBITS - w))


if __name__ == "__main__":
    results = {}
    for name, checks, logical in [("X errors", Z_CHECKS, LOGICAL_Z), ("Z errors", X_CHECKS, LOGICAL_X)]:
        H = check_matrix(checks)
        lut = build_lut(H)
        matching = pymatching.Matching(H)
        fail_lut = failures(H, logical, lambda s: lut[syndrome_index(s)])
        fail_mwpm = failures(H, logical, lambda s: matching.decode(s).astype(np.uint8))
        weights = ALL_ERRORS.sum(axis=1)
        print(f"{name}: LUT and PyMatching agree on all 512 patterns: {np.array_equal(fail_lut, fail_mwpm)}")
        print(f"   all weight-1 errors corrected: {not fail_lut[weights == 1].any()}")
        print(f"   failing weight-2 patterns: {fail_lut[weights == 2].sum()} of {np.sum(weights == 2)}")
        results[name] = (lut, fail_lut)

    np.save("lut_failures.npy", np.array([results["X errors"][1], results["Z errors"][1]]))
    for p in (1e-3, 1e-2, 0.05):
        pf = exact_failure_probability(results["X errors"][1], p)
        print(f"p = {p:g}: exact logical failure probability (X errors) = {pf:.3e}")
