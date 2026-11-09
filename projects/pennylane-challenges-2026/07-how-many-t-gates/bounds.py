"""Part 5: what the counting allows. Rough lower bounds on T-count, and where Gold would have to live.

Counting argument: to get within eps of every unitary in a D-dimensional space you need about
(1/eps)^D distinct circuits, i.e. D log2(1/eps) bits. If N T gates can reach at most 2^(bN) different
cosets, then N >= (D log2(1/eps) - log2|Clifford|) / b. Constant factors (volumes of balls and of the
group) are ignored, so these are estimates of the scale, not sharp bounds.

Run: python bounds.py   (needs clifford_t_counting.json and gridsynth_tcounts.json)
"""
import json
import math
import os

HERE = os.path.dirname(__file__)
GOLD_LIMIT = 0.4 * 57024          # 60% saving at five qubits
BASELINE_5 = 57024
OURS_5 = 1023 * 44


def clifford_size_log2(n):
    """|Clifford_n| modulo phase = 2^(n^2 + 2n) prod_j (4^j - 1)."""
    return n * n + 2 * n + sum(math.log2(4 ** j - 1) for j in range(1, n + 1))


def two_qubit_bound(eps, counts):
    """Extrapolate the measured coset counts with the last measured growth ratio. The ratios are still
    falling, so this overestimates how many circuits exist, and the T bound is on the safe side."""
    c = counts["cosets_by_T_count"]
    ratio = c[-1] / c[-2]
    need = 15 * math.log2(1 / eps)
    for N in range(1, 10_000):
        level = c[N - 1] if N <= len(c) else c[-1] * ratio ** (N - len(c))
        total_log2 = math.log2(counts["clifford_group_size"]) + math.log2(level / (1 - 1 / ratio))
        if total_log2 >= need:
            return N, ratio


def measured_fraction(counting):
    """Bits per T actually measured, as a fraction of the log2(4^n - 1) ceiling, averaged over n = 1, 2."""
    fr = []
    for n in (1, 2):
        c = counting[n]["cosets_by_T_count"]
        fr.append(math.log2(c[-1] / c[-2]) / math.log2(4 ** n - 1))
    return sum(fr) / len(fr)


def five_qubit_estimates(eps_exponent, counting):
    """(T bound with the ceiling per T, T estimate with the measured efficiency)."""
    bits = 1023 * eps_exponent * math.log2(10) - clifford_size_log2(5)
    ceiling = math.log2(4 ** 5 - 1)
    return bits / ceiling, bits / (ceiling * measured_fraction(counting))


if __name__ == "__main__":
    counting = {r["n"]: r for r in json.load(open(os.path.join(HERE, "clifford_t_counting.json")))}
    grid = {r["eps_exponent"]: r["mean"] for r in json.load(open(os.path.join(HERE, "gridsynth_tcounts.json")))}

    print("Two qubits (15 parameters):")
    for exp in (8, 10):
        N, ratio = two_qubit_bound(10.0 ** -exp, counting[2])
        print(f"  eps = 1e-{exp}: counting needs at least ~{N} T gates  (growth x{ratio:.2f} per T)."
              f"  Rotations at 44 each: 660.  Gridsynth per rotation: {15 * grid[exp]:.0f}")

    print("\nFive qubits (1023 parameters), against the Gold limit of "
          f"{GOLD_LIMIT:,.0f} T-equivalents:")
    frac = measured_fraction(counting)                       # at most 4^n - 1 new cosets per T gate
    for exp in (8, 10):
        lo, mid = five_qubit_estimates(exp, counting)
        print(f"  eps = 1e-{exp}: counting needs at least ~{lo:,.0f} T gates with the"
              f" {math.log2(1023):.1f}-bit ceiling per T, ~{mid:,.0f} with"
              f" the measured efficiency ({frac:.2f} of the ceiling at n = 1, 2)")
    per_param_gold = GOLD_LIMIT / 1023
    print(f"\n  Gold allows {per_param_gold:.1f} T-equivalents per parameter.")
    for exp in (9, 10, 11):
        direct = 1023 * exp * math.log2(10)                  # 1 bit per T on one qubit: log2(1/eps) per parameter
        print(f"  eps = 1e-{exp}: rotation by rotation with gridsynth {1023 * grid[exp]:>9,.0f}"
              f"   single-qubit gates synthesized whole, perfectly packed {direct:>8,.0f}"
              f" ({100 * (1 - direct / BASELINE_5):.1f}% saving)")
    print(f"\n  for reference: baseline {BASELINE_5:,}, my circuit {OURS_5:,} (21.06%), Gold limit {GOLD_LIMIT:,.0f}")
