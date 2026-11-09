"""Grover search when half the list is marked, with one auxiliary qubit and one shot.

Run: python solution.py
"""
import itertools

import numpy as np
import pennylane as qp

dev = qp.device("default.qubit", wires=3)


def grover_with_aux(generator, oracle, diffuser):
    """The circuit body I submitted. Data on wires 0 and 1, auxiliary qubit on wire 2."""

    # the given generator plus a Hadamard on the auxiliary wire: the 3-qubit uniform state
    @qp.prod
    def groversolfunc():
        generator()
        qp.Hadamard(wires=2)

    # prepare the uniform state, then mark solutions only where the auxiliary qubit is |1>
    def preppytime():
        groversolfunc()
        magicaloracle = qp.ctrl(oracle, control=2)
        magicaloracle()

    preppytime()

    # reflect about the same 3-qubit state we started in
    diffuser(groversolfunc)


@qp.set_shots(1)
@qp.qnode(dev)
def circuit(generator, oracle, diffuser):
    grover_with_aux(generator, oracle, diffuser)
    return qp.counts(wires=(0, 1))


@qp.qnode(dev)
def probabilities(generator, oracle, diffuser):
    grover_with_aux(generator, oracle, diffuser)
    return qp.probs(wires=(0, 1))


# ---------------------------------------------------------------------------------------
# The pieces the challenge hands you, rebuilt from its description
# ---------------------------------------------------------------------------------------
@qp.prod
def generator():
    qp.Hadamard(0)
    qp.Hadamard(1)


def diffuser(gen):
    qp.Reflection(gen())


def single_bit_oracle(case):
    """The four oracles the grader draws from: plus or minus Z on wire 0 or wire 1."""
    def oracle():
        wire = 0 if case in (0, 1) else 1
        if case in (1, 3):
            qp.PauliX(wires=wire)
        qp.PauliZ(wires=wire)
        if case in (1, 3):
            qp.PauliX(wires=wire)
    return oracle


SOLUTIONS_OF_CASE = {0: {"10", "11"}, 1: {"00", "01"}, 2: {"01", "11"}, 3: {"00", "10"}}


def pair_oracle(pair):
    """Any of the six ways to mark 2 of the 4 basis states."""
    def oracle():
        for state in pair:
            qp.FlipSign(list(state), wires=[0, 1])
    return oracle


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    # the grader's test: many random oracles, one shot each, every shot must be a solution
    for _ in range(1000):
        case = int(rng.integers(0, 4))          # all four cases (the grader only ever drew three)
        result = circuit(generator, single_bit_oracle(case), diffuser)
        assert set(result) <= SOLUTIONS_OF_CASE[case], (case, result)
    print("1000 random single-shot runs: every outcome was a solution")

    # stronger: the exact success probability for all six possible solution pairs
    for pair in itertools.combinations([(0, 0), (0, 1), (1, 0), (1, 1)], 2):
        probs = probabilities(generator, pair_oracle(pair), diffuser)
        success = sum(probs[2 * a + b] for a, b in pair)
        print(f"solutions {pair}: P(success) = {float(success):.12f}")
        assert np.isclose(success, 1.0)
    print("All checks passed.")
