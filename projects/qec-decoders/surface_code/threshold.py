"""Circuit-level threshold of the rotated surface code under MWPM decoding.

For each distance d, Stim builds a memory-Z experiment with d rounds of syndrome
extraction and uniform depolarizing noise of strength p on every operation.
PyMatching decodes the detector error model, and we count logical failures.
"""
import numpy as np
import pymatching
import stim

DISTANCES = (3, 5, 7)
P_VALUES = np.array([0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.010, 0.012, 0.014, 0.016, 0.020])
MAX_SHOTS = 200_000
TARGET_ERRORS = 400          # stop early once this many failures are seen


def memory_circuit(d, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def logical_error_rate(d, p, rng_seed=0):
    circuit = memory_circuit(d, p)
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True))
    sampler = circuit.compile_detector_sampler(seed=rng_seed)
    shots = errors = 0
    while shots < MAX_SHOTS and errors < TARGET_ERRORS:
        batch = 20_000
        detections, observables = sampler.sample(batch, separate_observables=True)
        predictions = matching.decode_batch(detections)
        errors += int(np.sum(np.any(predictions != observables, axis=1)))
        shots += batch
    p_block = errors / shots
    err_block = np.sqrt(p_block * (1 - p_block) / shots)
    return per_round(p_block, d), per_round_error(p_block, err_block, d), errors, shots


def per_round(p_block, rounds):
    """Per-round logical error rate. If each round flips the logical with probability q,
    the block fails when an odd number of rounds flip it: P = (1 - (1 - 2q)^r) / 2."""
    return (1 - (1 - 2 * p_block) ** (1 / rounds)) / 2


def per_round_error(p_block, err_block, rounds):
    """Propagate the binomial error bar through per_round()."""
    return err_block * (1 - 2 * p_block) ** (1 / rounds - 1) / rounds


if __name__ == "__main__":
    rows = []
    for d in DISTANCES:
        for p in P_VALUES:
            pl, err, n_err, shots = logical_error_rate(d, p)
            rows.append((d, p, pl, err, n_err, shots))
            print(f"d={d}  p={p:.3f}  p_L/round={pl:.2e} ± {err:.1e}  ({n_err} errors / {shots} shots)")
    np.save("threshold_data.npy", np.array(rows))
