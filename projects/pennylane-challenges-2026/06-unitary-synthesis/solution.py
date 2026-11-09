"""Unitary synthesis with exactly 4^n - 1 rotations: Silver (and therefore Bronze).

Run: python solution.py
"""
import numpy as np
import pennylane as qp

from synthesis_tools import cs_decomposition


def decomp(v):
    """v = e^{i phi} RZ(a) RY(b) RZ(c) for a 2x2 unitary v. Returns a, b, c, phi."""
    phi = 0.5 * np.angle(np.linalg.det(v))
    w = v * np.exp(-1j * phi)
    alpha = w[0, 0]
    beta = w[1, 0]
    b = 2.0 * np.arctan2(np.abs(beta), np.abs(alpha))
    s = -2.0 * np.angle(alpha)
    d = 2.0 * np.angle(beta)
    return (s + d) / 2.0, b, (s - d) / 2.0, phi


# use CXs to build multiplexed RZ RY gates in recursion
def multirot(kind, angles, target, controls, templist):
    if not controls:
        templist.append((kind, float(angles[0]), [target]))
        return

    even = angles[0::2]
    odd = angles[1::2]

    multirot(kind, (even + odd) / 2.0, target, controls[:-1], templist)
    templist.append(("CNOT", None, [controls[-1], target]))
    multirot(kind, (even - odd) / 2.0, target, controls[:-1], templist)
    templist.append(("CNOT", None, [controls[-1], target]))

# for the last control qubit == 0, we have (e+o)/2 + (e-o)/2 = e, the even angles
# for last control == 1, we have (e+o)/2 - (e-o)/2 = o, the odd angles, because the CXs
# flip the sign of the second half (X R(t) X = R(-t) for both RY and RZ)


# approach is to keep splitting blocks of blocks until we get a pure sequence of just
# uniformly controlled gates
def unicon_gates(blockuni, k, n):
    if k == n - 1:
        return [(n - 1, blockuni)]  # already a uniformly controlled U gate on the last wire

    right_block = []
    left_block = []
    RotY_multi = []
    for block in blockuni:
        l0, l1, theta, r0, r1 = cs_decomposition(block)
        right_block.extend([r0, r1])
        left_block.extend([l0, l1])
        for th in theta:
            cos_theta, sin_theta = np.cos(th), np.sin(th)
            RotY_multi.append(np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=complex))
        # all the cos, sin angles (theta) become the 2x2 block RY(2 theta) for one control state

    # U = L CS R (applied from right to left ---> right_block -> multiplexed RY -> left_block)
    return unicon_gates(right_block, k + 1, n) + [(k, RotY_multi)] + unicon_gates(left_block, k + 1, n)


def build_ops(uni_mat):
    """The gate list for uni_mat, as (name, angle, wires) tuples, plus the global phase.

    Passing through the uniformly controlled U gates, preserving the diagonal as a phase:
    the minimum for any exact rotation circuit is 4**n - 1 rotations, and this reaches it.
    """
    n = int(round(np.log2(len(uni_mat))))
    templist = []
    psi = np.zeros(2 ** n, dtype=float)

    for t, mats in unicon_gates([np.array(uni_mat)], 0, n):
        controls = list(range(t)) + list(range(t + 1, n))
        lowbits = n - 1 - t
        rotz = []
        roty = []

        for c, g in enumerate(mats):
            x0 = ((c >> lowbits) << (lowbits + 1)) | (c & ((1 << lowbits) - 1))
            x1 = x0 | (1 << lowbits)

            # merge and update carry: the carried diagonal acts first, so it multiplies on the right
            g = g @ np.diag([np.exp(1j * psi[x0]), np.exp(1j * psi[x1])])

            a, b, cz, phi = decomp(g)
            rotz.append(cz)
            roty.append(b)

            psi[x0] = phi - a / 2.0
            psi[x1] = phi + a / 2.0

        multirot("RZ", np.array(rotz), t, controls, templist)
        multirot("RY", np.array(roty), t, controls, templist)

    # diagonal calc: the leftover diagonal costs 2**n - 1 rotations, for 4**n - 1 in total
    for m in range(n, 0, -1):
        lamb = psi[1::2] - psi[0::2]
        multirot("RZ", lamb, m - 1, list(range(m - 1)), templist)
        psi = (psi[0::2] + psi[1::2]) / 2.0

    return templist, float(psi[0])


def submission_circ(u_mat):
    templist, phase = build_ops(u_mat)
    for namefunc, angle, wires in templist:
        if namefunc == "CNOT":
            qp.CNOT(wires=wires)
        elif namefunc == "RZ":
            qp.RZ(angle, wires=wires)
        elif namefunc == "RY":
            qp.RY(angle, wires=wires)
    qp.GlobalPhase(-phase, wires=0)


if __name__ == "__main__":
    from synthesis_tools import baseline_cost, cost_and_counts, haar_unitary, max_error

    savings = []
    for n in (2, 3, 4, 5):
        costs, errors = [], []
        for sample in range(3):
            u = haar_unitary(n, seed=2026 + 1000 * n + sample)
            cost, counts, aux = cost_and_counts(submission_circ, u)
            costs.append(cost)
            errors.append(max_error(submission_circ, u))
            assert aux == 0
        base = baseline_cost(n)
        saving = 100 * (1 - np.mean(costs) / base)
        savings.append(saving)
        rot = counts.get("RZ", 0) + counts.get("RY", 0)
        print(f"n={n}: cost {np.mean(costs):>7.0f} vs baseline {base:>6}  saving {saving:5.2f}%"
              f"   rotations {rot} = 4^n - 1 = {4 ** n - 1}   CNOTs {counts.get('CNOT', 0)}"
              f"   max error {max(errors):.1e}")
        assert rot == 4 ** n - 1 and max(errors) < 1e-8 and np.mean(costs) <= base
    assert all(b >= a - 1e-9 for a, b in zip(savings, savings[1:]))     # saving never shrinks
    grade = "Gold" if savings[-1] >= 60 else "Silver" if savings[-1] >= 20 else "Bronze"
    print(f"grade at 5 qubits: {grade} ({savings[-1]:.2f}%)")
    print("All checks passed.")
