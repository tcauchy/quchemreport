"""Transition densities and transition dipoles of the excited states (27/11/2026).

calc_orb.TD computes the transition density rho_T = f sum_ia (X_ia + Y_ia) phi_i phi_a through
the natural transition orbitals of X + Y (calc_orb.transition_orbital_pairs), and its dipole;
compute.transition_dipole turns it into the charge density with the phase of the dipole of the
log and measures how far the printed amplitudes are from that dipole. The overlap density
sum C^2 phi_i phi_a used before lost the signs of the amplitudes (3 to 53 degrees away from
the dipole of Gaussian on HCFClBr, validation/calculs_20261127/tdip).

Data: gaussian/h2co (default printing: X above 0.1), gaussian/h2co/TD_full (IOp(9/40=4): all
the X and Y amplitudes), orca6/h2co. The de-excitations (Y) and the ORCA transition dipoles
are read by cclib with the patches of the cclib branches deexcitation-amplitudes and
transition-dipoles; without them the corresponding tests are skipped.

Run (see tests/README.md): tests/run_tests.sh -k transition_density
"""
import numpy as np
import pytest

from tests.helpers import read_steps


def model(path, solver="gaussian"):
    return read_steps(path, solver)[-1]


def td_output(m, states, step=0.15):
    from quchemreport.data_enrichment import calc_orb
    es = m.results.excited_states
    transitions = [es.et_transitions[k] for k in states]
    deexcitations = [es.et_deexcitations[k] for k in states] if es.et_deexcitations else None
    out, X, Y, Z = calc_orb.TD(m, transitions, grid_step=step, nproc=4, deexcitations=deexcitations)
    return out, transitions, X


def test_split_deexcitations():
    """cclib stores Y with the orbitals reversed (Q-Chem convention); they are kept apart."""
    from types import SimpleNamespace
    from quchemreport.parser_engine.scanlog import split_deexcitations
    es = SimpleNamespace(et_transitions=[[[(7, 0), (8, 0), 0.7], [(8, 0), (7, 0), -0.05], [(6, 1), (9, 1), 0.1]]],
                         et_deexcitations=None)
    split_deexcitations(es)
    assert es.et_transitions == [[((7, 0), (8, 0), 0.7), ((6, 1), (9, 1), 0.1)]]
    assert es.et_deexcitations == [[((7, 0), (8, 0), -0.05)]]


def test_edd_amplitudes_keep_the_default_printing():
    from quchemreport.data_enrichment.calc_orb import edd_amplitudes
    t = [((7, 0), (8, 0), 0.69), ((6, 0), (8, 0), 0.05), ((7, 0), (9, 0), -0.12)]
    assert edd_amplitudes(t) == [t[0], t[2]]
    assert edd_amplitudes([((7, 0), (8, 0), 0.05)]) == [((7, 0), (8, 0), 0.05)]


def test_transition_pairs_give_the_sum_over_orbitals():
    """The natural transition orbital pairs give sum_ia c phi_i phi_a (all the pairs kept)."""
    from quchemreport.data_enrichment import calc_orb
    m = model("gaussian/h2co/TD.log")
    t = m.results.excited_states.et_transitions[3]      # several amplitudes
    rho = calc_orb.transition_density(m, t, [], calc_orb.MO(m, [0], grid_step=0.2)[1], 0.2, 4)
    mos = sorted({x for (i, _), (a, _), _ in t for x in (i, a)})
    grids, *_ = calc_orb.MO(m, mos, spin="alpha", grid_step=0.2, nproc=4)
    phi = dict(zip(mos, grids))
    direct = np.sqrt(2) * sum(c * phi[i] * phi[a] for (i, _), (a, _), c in t)
    assert np.max(np.abs(rho - direct)) < 1e-6 * np.max(np.abs(direct))


def test_default_printing_gives_the_direction_but_not_the_norm():
    """X above 0.1 only: the dipole of the density is not the dipole of the log (Y missing),
    the deviation says so; the arrow is the dipole of the log, with the phase of the density."""
    from quchemreport.data_enrichment import compute
    m = model("gaussian/h2co/TD.log")
    es = m.results.excited_states
    states = [k for k, f in enumerate(es.et_oscs) if f > 0.01]
    out, transitions, _ = td_output(m, states)
    for j, k in enumerate(states):
        tdm = compute.transition_dipole(out[j], es.et_eldips[k], es.et_sym[k], transitions[j])
        assert tdm["source"] == "log"
        assert np.allclose(tdm["dipole"], es.et_eldips[k])
        grid, log = np.array(tdm["grid"]), np.array(es.et_eldips[k])
        assert grid @ log > 0                                  # phase of the log
        assert tdm["deviation"] > 0.05                         # Y missing: approximate


def test_all_amplitudes_give_the_dipole_of_the_log():
    from quchemreport.data_enrichment import compute
    m = model("gaussian/h2co/TD_full.log")
    es = m.results.excited_states
    if not any(es.et_deexcitations):
        pytest.skip("cclib without the de-excitation amplitudes (branch deexcitation-amplitudes)")
    states = list(range(len(es.et_oscs)))
    out, transitions, _ = td_output(m, states)
    for j, k in enumerate(states):
        tdm = compute.transition_dipole(out[j], es.et_eldips[k], es.et_sym[k], transitions[j])
        if np.linalg.norm(es.et_eldips[k]) > 0.05:
            assert tdm["deviation"] < 0.05, (k, tdm["deviation"])
        else:
            assert tdm["dipole"] is None                       # forbidden: no arrow
            assert np.linalg.norm(tdm["grid"]) < 0.01


def test_restricted_triplets_have_no_charge_transition_density():
    from quchemreport.data_enrichment import compute
    m = model("gaussian/h2co/TD_T.log")
    es = m.results.excited_states
    out, transitions, _ = td_output(m, [0])
    assert compute.transition_dipole(out[0], es.et_eldips[0], es.et_sym[0], transitions[0]) is None


def test_orca_transition_dipoles():
    """ORCA (TDA by default, amplitudes with weights above 0.01): dipoles read by cclib; the
    bright single-configuration states are reproduced by the amplitudes (a weak one, S3 with
    |mu| = 0.085 e.bohr, depends on the amplitudes below the printing threshold: 0.49)."""
    from quchemreport.data_enrichment import compute
    m = model("orca6/h2co/TD.out", "orca")
    es = m.results.excited_states
    if es.et_eldips is None:
        pytest.skip("cclib without the ORCA transition dipoles (branch transition-dipoles)")
    hartree = np.asarray(es.et_energies) / 219474.6313705
    assert np.allclose(2 / 3 * hartree * np.sum(np.asarray(es.et_eldips) ** 2, axis=1), es.et_oscs, atol=1e-4)
    single = [k for k, t in enumerate(es.et_transitions)
              if len(t) == 1 and np.linalg.norm(es.et_eldips[k]) > 0.3]
    assert single
    out, transitions, _ = td_output(m, single)
    for j, k in enumerate(single):
        tdm = compute.transition_dipole(out[j], es.et_eldips[k], es.et_sym[k], transitions[j])
        assert tdm["deviation"] < 0.1, (k, tdm["deviation"])


def test_report_warning_and_caption():
    from quchemreport.report_generator import generate_report as gr
    assert gr.transition_density_warning([]) is None
    text = gr.transition_density_warning(["S1 (1.4)", "S3 (0.9)"])
    assert "S1 (1.4), S3 (0.9)" in text and "IOp(9/40=4)" in text
    caption = gr.transition_density_caption(False, [], "TD.log")
    assert r"\frac{2}{3}" in caption and "TD.log" in caption
    assert "emitting states" in gr.transition_density_caption(False, [], "OPT_ES1.log", emission=True)


def test_exact_difference_density():
    """Attachment - detachment: the diagonal sum C^2 (phi_a^2 - phi_i^2) for one configuration,
    zero integral and a positive part of about one electron for several configurations."""
    from quchemreport.data_enrichment import calc_orb
    m = model("gaussian/h2co/TD.log")
    es = m.results.excited_states
    X = calc_orb.MO(m, [0], grid_step=0.2)[1]
    d3r = 0.2 ** 3
    single = next(t for t in es.et_transitions if len(t) == 1)
    (i, _), (a, _), c = single[0]
    grids, *_ = calc_orb.MO(m, [i, a], spin="alpha", grid_step=0.2, nproc=4)
    exact = calc_orb.difference_density(m, single, [], X, 0.2, 4)
    assert np.max(np.abs(exact - c * c * (grids[1] ** 2 - grids[0] ** 2))) < 1e-8 * np.max(np.abs(exact))
    several = max(es.et_transitions, key=len)
    rho = calc_orb.difference_density(m, several, [], X, 0.2, 4)
    assert abs(np.sum(rho) * d3r) < 0.02
    assert 0.3 < np.sum(rho[rho > 0]) * d3r < 1.0


def test_magnetic_dipole_for_chiral_molecules_only():
    """-m (Gaussian sign) is returned with the dipole of the log; is_chiral on the rotatory strengths."""
    from quchemreport.data_enrichment import compute
    assert not compute.is_chiral([0.0, 0.0]) and not compute.is_chiral(None)
    assert compute.is_chiral([0.0, -13.56])
    m = model("gaussian/h2co/TD.log")
    es = m.results.excited_states
    k = int(np.argmax(es.et_oscs))
    out, transitions, _ = td_output(m, [k])
    tdm = compute.transition_dipole(out[0], es.et_eldips[k], es.et_sym[k], transitions[0], [0.1, -0.2, 0.3])
    assert tdm["magnetic"] == [-0.1, 0.2, -0.3]
    out, transitions, _ = td_output(m, [k])
    assert compute.transition_dipole(out[0], es.et_eldips[k], es.et_sym[k], transitions[0])["magnetic"] is None
