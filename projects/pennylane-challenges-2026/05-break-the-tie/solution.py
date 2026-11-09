"""Quantum Monte Carlo with amplitude estimation, and breaking its two-fold ambiguity.

Run: python solution.py
"""
import pennylane as qp
import pennylane.numpy as np


def run_circuit(num_wires, circuit):
    """Runs a no-argument circuit function on an exact simulator and returns plain floats."""
    device = qp.device("default.qubit", wires=num_wires)
    return [float(v) for v in np.asarray(qp.QNode(circuit, device)()).ravel()]


def estimate_mean(probabilities, values, n, details=None):
    """Amplitude-estimation estimate of E[f(X)], with the tie broken by a direct circuit.

    `details`, if given a dict, collects the intermediate numbers (for the figures).
    """
    probabilities = np.array(probabilities, dtype=float)
    values = np.array(values, dtype=float)
    m = int(round(np.log2(len(probabilities))))     # number of data wires

    # rescaling the quantity to the non-negative bound [0, 1]
    f_given = np.max(values)
    if f_given == 0.0:
        return 0.0
    f_rescaled = values / f_given

    # amplitude estimation, on m data wires + 1 objective wire (x = m + 1) and n estimation wires
    x = m + 1

    def circuit():
        def func(i):
            return f_rescaled[i]

        qp.QuantumMonteCarlo(probabilities, func, target_wires=range(x),
                             estimation_wires=range(x, x + n))
        return qp.probs(wires=range(x, x + n))

    regi_readout = run_circuit(x + n, circuit)

    # possible candidates: the two equal peaks at k and 2^n - k give a and 1 - a
    k = np.argmax(regi_readout)
    theta = k / (2 ** n)

    possible_candidates = [
        (1 - np.cos(np.pi * theta)) / 2,
        (1 + np.cos(np.pi * theta)) / 2
    ]

    # tie-break circuit: a normal sampling circuit, shallow, only used to choose between the two
    # possible candidates (which are at least 0.13 apart in the test cases)
    def breakthe_tie():
        qp.StatePrep(np.sqrt(probabilities), wires=range(m))
        for i in range(len(probabilities)):
            control_values = [(i >> (m - 1 - j)) & 1 for j in range(m)]

            # RY rotation angle loads amplitude sqrt(f_rescaled[i]) on |1>
            val_clipped = np.clip(f_rescaled[i], 0.0, 1.0)
            angle = 2.0 * np.arcsin(np.sqrt(val_clipped))
            qp.ctrl(
                qp.RY(angle, wires=m),
                control=range(m),
                control_values=control_values
            )
        return qp.probs(wires=m)

    tie_readout = run_circuit(x, breakthe_tie)
    a_win = tie_readout[1]

    # winner takes it all
    winner = min(possible_candidates, key=lambda c: abs(c - a_win))

    if details is not None:
        details.update(readout=regi_readout, k=int(k), candidates=[float(c) for c in possible_candidates],
                       a_direct=float(a_win), f_max=float(f_given))
    return float(f_given * winner)


PUBLIC = [
    ([0.4008104395612045, 0.24310382032676792, 0.14744992052145606, 0.0894328975684541,
      0.054243794362206846, 0.03290052437982574, 0.019955176756987287, 0.012103426523097706],
     [0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25], 8, 1.160949660702857),
    ([0.125] * 8, [0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0], 8, 0.8745681972617296),
]

if __name__ == "__main__":
    for probs, vals, n, expected in PUBLIC:
        d = {}
        got = estimate_mean(probs, vals, n, details=d)
        exact = float(np.dot(probs, vals))
        print(f"estimate {got!r}  expected {expected!r}  |diff| {abs(got - expected):.1e}"
              f"   (peaks at k = {d['k']} and {2 ** n - d['k']}, a_direct = {d['a_direct']:.6f},"
              f" exact mean {exact:.6f})")
        assert abs(got - expected) <= 1e-9
    print("All checks passed.")
