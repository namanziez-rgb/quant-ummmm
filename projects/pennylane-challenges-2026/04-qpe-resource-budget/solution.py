"""The largest quantum phase estimation that fits a wall-clock budget.

Run: python solution.py
"""
import pennylane as qp

qre = qp.estimator

GATE_SET = {"Hadamard", "S", "CNOT", "T"}
ROTATION_PRECISION = 1e-6


def affordable_precision(t_gate_seconds, budget_seconds, max_estimation_wires=40):
    """Returns (m, 2**-m, T-count) for the largest affordable number of estimation wires m."""
    config = qre.ResourceConfig()
    config.set_single_qubit_rot_precision(ROTATION_PRECISION)

    best = None

    # estimating QPE on a generic 2-qubit base unitary for increasing estimation wires
    # binary search would also work, but at these sizes a scan suffices
    for m in range(1, max_estimation_wires + 1):
        resources = qre.estimate(
            qre.QPE(qre.QubitUnitary(num_wires=2), num_estimation_wires=m),
            gate_set=GATE_SET,
            config=config,
        )
        t_count = int(resources.gate_counts.get("T", 0))

        # for m=9 we have 684108 T and for m=10 we get 1366956 T (almost double), as expected

        # the runtime then should be t count * t_gate_seconds
        runtime = t_count * t_gate_seconds

        if runtime > budget_seconds:
            break

        best = (m, 2.0 ** -m, t_count)

    # why stopping is safe: the gate count scales as order 2**m, since for m wires we apply the
    # controlled U (2**m)-1 times, so once one size overruns nothing after it can fit
    return best


if __name__ == "__main__":
    public = [((1e-06, 1.0), (9, 0.001953125, 684108)),
              ((0.001, 60.0), (5, 0.03125, 42252))]
    for (t_gate, budget), expected in public:
        got = affordable_precision(t_gate, budget)
        print(f"t_T = {t_gate:g} s, budget {budget:g} s -> {got}   expected {expected}")
        assert got == expected
    print("All checks passed.")
