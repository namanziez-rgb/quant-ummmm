"""The cost model and linear algebra for the synthesis challenge, rebuilt from its description.

None of this is the challenge's own grader code. It follows the stated rules: decompose into
RX, RY, RZ, T, S, H, X, Y, Z, CNOT and GlobalPhase, charge 44 per rotation and 1 per T gate,
and double the cost for every auxiliary wire.
"""
import numpy as np
import pennylane as qp
from scipy.linalg import cossin

GATE_SET = {"RX", "RY", "RZ", "T", "S", "H", "X", "Y", "Z", "CNOT", "GlobalPhase"}
COSTS = {"RX": 44, "RY": 44, "RZ": 44, "T": 1}


def cs_decomposition(u_mat):
    """u = [[l0, 0], [0, l1]] @ [[C, -S], [S, C]] @ [[r0, 0], [0, r1]], with C = cos(theta)."""
    half = len(u_mat) // 2
    (l0, l1), theta, (r0, r1) = cossin(u_mat, p=half, q=half, separate=True)
    return l0, l1, theta, r0, r1


def haar_unitary(num_qubits, seed):
    """A Haar-random unitary, by QR of a complex Gaussian matrix with the phases fixed."""
    rng = np.random.default_rng(seed)
    dim = 2 ** num_qubits
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


def baseline_cost(n):
    """Shende, Bullock and Markov's Shannon decomposition: 44 (21/16 4^n - 3 2^(n-1))."""
    return 44 * ((21 * 4 ** n) // 16 - 3 * 2 ** (n - 1))


def compile_ops(circuit, *args):
    """The gates `circuit(*args)` queues, decomposed into the gate set."""
    tape = qp.tape.make_qscript(circuit)(*args)
    (decomposed,), _ = qp.decompose(tape, gate_set=GATE_SET)
    return decomposed


def cost_and_counts(circuit, u_mat):
    tape = compile_ops(circuit, u_mat)
    n = int(round(np.log2(len(u_mat))))
    names = [op.name for op in tape.operations]
    aux = len(set(tape.wires.tolist()) - set(range(n)))
    cost = sum(COSTS.get(name, 0) for name in names) * 2 ** aux
    return cost, {name: names.count(name) for name in set(names)}, aux


def max_error(circuit, u_mat):
    """Largest entrywise distance between the circuit's matrix and u_mat, global phase included."""
    n = int(round(np.log2(len(u_mat))))
    tape = compile_ops(circuit, u_mat)
    mat = qp.matrix(tape, wire_order=range(n))
    return float(np.abs(mat - u_mat).max())
