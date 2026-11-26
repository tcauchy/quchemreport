"""Open-shell molecules and report layout details found on O2 and HCFClBr (26/11/2026).

Logs: tests/data/gaussian/o2/TD.log (triplet O2, UKS TD-DFT, PBE0/6-31G(d)) and
tests/data/orca6/o2/OPT_ES1.out (triplet O2, ORCA 6.1, optimized excited state).
"""
import numpy as np
import pytest

from quchemreport.data_enrichment import TD2UVvis
from quchemreport.data_enrichment.compute import degenerate_partners, show_degenerate_pairs
from quchemreport.report_generator.generate_report import fukui_table_atoms, fmt_spatial_symmetry
from quchemreport.utility_services.naming import state_labels
from quchemreport.visualization_engine.visu_plots import spread_labels
from tests.helpers import read_steps


@pytest.mark.parametrize("path, solver", [("gaussian/o2/TD.log", "gaussian"), ("orca6/o2/OPT_ES1.out", "orca")])
def test_states_of_a_triplet_reference(path, solver):
    """cclib gives '3.002-?Sym' (Gaussian) or '???' (ORCA): the states are named after the
    multiplicity of the reference (T1, T2...) and their <S^2> is read in the log."""
    es = read_steps(path, solver)[-1].results.excited_states
    assert state_labels(es.et_sym)[:3] == ["T1", "T2", "T3"]
    assert len(es.et_s2) == len(es.et_energies)
    assert all(1.99 < s2 < 2.1 for s2 in es.et_s2[:8])
    if solver == "orca":
        # states 9-10 of the ORCA optimized state: 3.47, strongly contaminated (shown in bold)
        assert es.et_s2[8] > 3.4


def test_spatial_symmetry_of_linear_molecules():
    assert fmt_spatial_symmetry("sigma.u") == r"$\Sigma_{u}$"
    assert fmt_spatial_symmetry("B2") == "B2"


def test_fukui_table_lists_every_atom_of_small_or_symmetric_molecules():
    # O2: equal atoms, formerly an empty table (outliers only)
    assert fukui_table_atoms(2, [0.5, 0.5], [0.5, 0.5], [0.0, 0.0]) == ([0, 1], True)
    big = [0.0] * 30
    big[7] = 0.5
    atoms, all_atoms = fukui_table_atoms(30, big, [0.0] * 30, big)
    assert atoms == [7] and not all_atoms


def test_spectrum_grid_covers_every_band():
    """O2: the absorption curve stopped at 100000 cm-1, its bands at 125000 cm-1 were left out."""
    TD2UVvis.set_user_range(None)
    start, end, n = TD2UVvis.spectrum_grid([49534, 124733], 3000)
    assert start <= 49534 - 9000 and end >= 124733 + 9000
    assert np.isclose((end - start) / (n - 1), TD2UVvis.GRID_STEP)
    TD2UVvis.set_user_range((10000, 30000))
    start, end, _ = TD2UVvis.spectrum_grid([49534], 3000)
    assert start <= 10000 and end >= 49534 + 9000
    TD2UVvis.set_user_range(None)


def test_spread_labels():
    # S1/S2 of HCFClBr at 6.28/6.38 eV: moved apart around their middle; S0 does not move
    out = spread_labels([6.28, 6.38, 0.0], 0.25)
    assert out[2] == 0.0
    assert np.isclose(out[1] - out[0], 0.25) and np.isclose(out[0] + out[1], 6.28 + 6.38)


def test_degenerate_pi_pair_seen_alike():
    """px and py seen along y: py points at the camera; both become combinations at 45 degrees."""
    g = np.linspace(-4, 4, 33)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    envelope = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 2)
    voxels = np.array([X * envelope, Y * envelope])
    energies = [-8.68, -8.68, 6.23]
    assert degenerate_partners(energies, 0) == [1]
    rotated = show_degenerate_pairs(voxels, [0, 1], energies, X, Y, Z, (0, 1, 0), (0, 0, 0), None)
    assert rotated == [(0, 1)]
    depth2 = Y ** 2
    q = [np.sum(v ** 2 * depth2) for v in voxels]
    assert np.isclose(q[0], q[1], rtol=1e-6)
    assert np.isclose(abs(np.sum(voxels[0] * X * envelope)) / np.sum((X * envelope) ** 2), np.sqrt(0.5), rtol=1e-6)


def test_degenerate_partner_computed_when_not_requested():
    g = np.linspace(-4, 4, 21)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    envelope = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / 2)
    voxels = np.array([X * envelope])
    asked = []
    show_degenerate_pairs(voxels, [3], [0, 1, 2, -5.0, -5.0], X, Y, Z, (0, 1, 0), (0, 0, 0),
                          lambda k: asked.append(k) or Y * envelope)
    assert asked == [4]
    assert np.isclose(np.sum(voxels[0] ** 2 * Y ** 2), np.sum(voxels[0] ** 2 * X ** 2), rtol=1e-6)


def test_mep_ring_of_a_linear_molecule():
    """O2: the positive belt is one ring, formerly two arbitrary points on the hidden side."""
    from quchemreport.data_enrichment.mep_surface import merge_rings
    coords = [[0, 0, 1.136], [0, 0, -1.136]]
    belt = [{"kind": "max", "value": 0.0091, "position": [-1.85, -2.6, 0.0], "atom": 0, "distance": 3.4,
             "direction": [0, 0, 1]},
            {"kind": "max", "value": 0.0091, "position": [-2.6, -1.85, 0.0], "atom": 0, "distance": 3.4,
             "direction": [0, 0, 1]},
            {"kind": "min", "value": -0.0046, "position": [0, 0, 3.0], "atom": 0, "distance": 1.9,
             "direction": [0, 0, 1]}]
    merged = merge_rings(belt, coords, up=(0, 1, 0))
    rings = [e for e in merged if e.get("ring")]
    assert len(merged) == 2 and len(rings) == 1
    assert np.allclose(rings[0]["position"], [0, np.hypot(1.85, 2.6), 0])
    # not linear: unchanged
    assert merge_rings(belt, coords + [[1, 0, 0]], up=(0, 1, 0)) == belt


def test_prominence_filters_ripples_not_weak_sites():
    """Prominence (V_ext - median of V 1.5-2.5 bohr around): a weak but isolated maximum
    (sigma-hole near 0) stands out; a bump of a few 1e-5 a.u. on a plateau does not."""
    from quchemreport.data_enrichment.mep_surface import prominences, PROMINENCE_MIN
    g = np.linspace(-5, 5, 101)
    X, Y = np.meshgrid(g, g, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    r2 = X ** 2 + Y ** 2
    weak_site = 0.002 * np.exp(-r2) - 0.006 * (1 - np.exp(-r2))     # +1.3 kcal/mol, deep surroundings
    ripple = -0.02 + 3e-5 * np.exp(-r2)                              # flat -12.6 kcal/mol + ripple
    centre = int(np.argmin(r2.ravel()))
    p_site = prominences(points, weak_site.ravel(), [("max", centre)])[0]
    p_ripple = prominences(points, ripple.ravel(), [("max", centre)])[0]
    assert p_site > PROMINENCE_MIN > p_ripple > 0
    assert prominences(points, -weak_site.ravel(), [("min", centre)])[0] == p_site
