"""Certifying a GHZ state with n parity checks.

Run: python solution.py
"""
import numpy as np
import pennylane as qp


def parity_checks(n):
    """The global X parity plus the n-1 neighbouring ZZ parities."""
    # one global X parity: it sees the coherence between |0...0> and |1...1>
    checks = [qp.prod(*[qp.X(w) for w in range(n)])]
    # n-1 neighbouring ZZ parities: they pin the populations onto those two strings
    checks += [qp.Z(i) @ qp.Z(i + 1) for i in range(n - 1)]
    return checks


def prepare_and_build_checks(n):
    """Prepares GHZ_n on wires 0..n-1 and returns the n checks to measure."""
    # Hadamard on wire 0, then a CNOT fan-out copies that bit onto every other wire
    qp.Hadamard(wires=0)
    for target in range(1, n):
        qp.CNOT(wires=[0, target])
    return parity_checks(n)


def wrapper(n):
    """Mean of the n parity values, measured on the prepared state."""
    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def parities():
        return [qp.expval(obs) for obs in prepare_and_build_checks(n)]

    return float(np.mean(parities())), parities


# ---------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------
def pauli_word(obs):
    """An observable reduced to a Pauli word like '0X1X2X', or None if it is not one."""
    sentence = qp.pauli.pauli_sentence(obs)
    (word, coeff), = sentence.items()
    assert np.isclose(complex(coeff), 1.0)
    return "".join(f"{w}{word[w]}" for w in sorted(word))


def score_of(prepare, n):
    """Mean parity score of any state, given a function that prepares it."""
    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def circuit():
        prepare(n)
        return [qp.expval(obs) for obs in parity_checks(n)]

    return float(np.mean(circuit()))


def ghz(n):
    qp.Hadamard(wires=0)
    for target in range(1, n):
        qp.CNOT(wires=[0, target])


def plus_product(n):            # |+>^n: perfect X parity, no entanglement
    for w in range(n):
        qp.Hadamard(wires=w)


def wrong_phase(n):             # (|0...0> - |1...1>)/sqrt(2)
    ghz(n)
    qp.Z(wires=0)


def all_ones(n):
    for w in range(n):
        qp.X(wires=w)


def mixture_score(n):
    """The coin-flip mixture of |0...0> and |1...1>. Expectation values are linear,
    so its score is the average of the two pure-state scores."""
    return 0.5 * (score_of(lambda n: None, n) + score_of(all_ones, n))


if __name__ == "__main__":
    for n in (3, 5):
        value, qnode = wrapper(n)
        tape = qp.workflow.construct_tape(qnode)()
        words = sorted(pauli_word(m.obs) for m in tape.measurements)
        expected = sorted(["".join(f"{w}X" for w in range(n))]
                          + [f"{i}Z{i + 1}Z" for i in range(n - 1)])
        gates = [op.name for op in tape.operations]
        print(f"n={n}: score {value:.12f}, checks {words}")
        assert np.isclose(value, 1.0, atol=1e-5) and words == expected
        assert gates == ["Hadamard"] + ["CNOT"] * (n - 1), gates   # no stray gates on the tape

    print("\nDecoys (score, then the closed form):")
    for n in range(3, 9):
        p, mix, wp = score_of(plus_product, n), mixture_score(n), score_of(wrong_phase, n)
        print(f"n={n}: |+>^n {p:.4f} ({1 / n:.4f})   mixture {mix:.4f} ({(n - 1) / n:.4f})"
              f"   wrong phase {wp:.4f} ({(n - 2) / n:.4f})")
        assert np.isclose(p, 1 / n) and np.isclose(mix, (n - 1) / n) and np.isclose(wp, (n - 2) / n)
    print("\nAll checks passed.")
