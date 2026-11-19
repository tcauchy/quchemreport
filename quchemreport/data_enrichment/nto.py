"""NTOs from the printed excitation amplitudes (X-only convention).

Indices in cclib transitions are zero-based. Missing printed amplitudes are
unknown, not evidence of zero; weights are relative to the available X data.
"""
import numpy as np

# With cclib's conventions (Gaussian restricted amplitudes already scaled by
# sqrt(2)), sum(X^2) is 1 for TDA/CIS and slightly above 1 for full TD-DFT
# (X^2 - Y^2 = 1) when every amplitude is printed. A lower value means that
# the log omits part of the excitation (printing threshold of the program).
X_NORM_COMPLETE_THRESHOLD = 0.99


def build_nto_pairs(transitions, mo_coefficients, homos, max_pairs=2, min_weight=0.01):
    """Return globally weighted, spin-resolved NTO pairs in AO coefficients."""
    matrices = {}
    coefficient_arrays = [np.asarray(coefficients, dtype=float) for coefficients in mo_coefficients]
    for (occupied, spin_i), (virtual, spin_a), amplitude in transitions:
        if spin_i != spin_a or spin_i not in (0, 1):
            raise ValueError("Spin-changing or unknown-spin transitions are not supported")
        if spin_i >= len(mo_coefficients) or spin_i >= len(homos):
            raise ValueError("Transition spin is absent from the reference orbitals")
        coefficients = coefficient_arrays[spin_i]
        if occupied < 0 or virtual < 0 or occupied >= len(coefficients) or virtual >= len(coefficients):
            raise ValueError("Transition refers to an orbital absent from the reference coefficients")
        if occupied > homos[spin_i] or virtual <= homos[spin_i]:
            raise ValueError("Transition indices do not match the reference occupied/virtual orbitals")
        if not np.isfinite(amplitude):
            raise ValueError("Non-finite transition amplitude")
        matrices.setdefault(spin_i, {})[(occupied, virtual)] = (
            matrices.setdefault(spin_i, {}).get((occupied, virtual), 0.0) + float(amplitude)
        )

    decompositions = []
    total = 0.0
    for spin, entries in matrices.items():
        occ = sorted({i for i, _ in entries})
        vir = sorted({a for _, a in entries})
        occ_index = {value: idx for idx, value in enumerate(occ)}
        vir_index = {value: idx for idx, value in enumerate(vir)}
        matrix = np.zeros((len(occ), len(vir)))
        for (i, a), value in entries.items():
            matrix[occ_index[i], vir_index[a]] = value
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        total += float(np.dot(singular, singular))
        decompositions.append((spin, occ, vir, u, singular, vh))
    if total <= 1e-20:
        raise ValueError("No nonzero excitation amplitudes were found")

    pairs = []
    for spin, occ, vir, u, singular, vh in decompositions:
        coefficients = coefficient_arrays[spin]
        for k, value in enumerate(singular):
            weight = float(value * value / total)
            if weight < min_weight or value <= 1e-12:
                continue
            pairs.append({
                "spin": "restricted" if len(homos) == 1 else ("alpha" if spin == 0 else "beta"),
                "weight": weight,
                "hole": u[:, k] @ coefficients[occ],
                "electron": vh[k, :] @ coefficients[vir],
            })
    pairs.sort(key=lambda pair: pair["weight"], reverse=True)
    return pairs[:max_pairs], total
