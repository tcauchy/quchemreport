"""Tests on real Gaussian 09 / ORCA 6.1 calculations (tests/data), 23/11/2026.

Defects found on these calculations are marked xfail (strict) until they are fixed:
  1 ECP: MEP with the full nuclear charge     5 report with single points only
  2 mixed 6D/7F and 5D/10F basis sets          6 NTO skipped with ORCA (TD orbitals)
  3 TD without excitation amplitudes           7 state numbering per multiplicity
  4 TD without rotatory strengths              8 NTO names without the multiplicity
  9 dispersion correction not compared (B3LYP and B3LYP-D3BJ accepted together)

Run (see tests/README.md): tests/run_tests.sh all -k real_calculations
"""
import contextlib
import io
import re

import numpy as np
import pytest

from quchemreport.data_enrichment import calc_orb
from quchemreport.data_enrichment import qmckl_backend as qb
from tests.helpers import read_steps, run_report, without_lines


def defect(number, reason):
    return pytest.mark.xfail(strict=True, reason=f"defect {number} of 23/11/2026: {reason}")


# --------------------------------------------------------------------------- ECP (defect 1)

ECP_LOGS = [  # log, solver, core electrons, valence electrons
    ("gaussian/ch3i/OPT.log", "gaussian", [0, 46, 0, 0, 0], 16),
    ("gaussian/ch2bri/CH2BrI_pseudoread.log", "gaussian", [0, 28, 46, 0, 0], 22),
    ("gaussian/ch2bri/CH2BrI_iop.log", "gaussian", [0, 28, 46, 0, 0], 22),
    ("orca6/ch3i/OPT.out", "orca", [0, 28, 0, 0, 0], 34),
]


@pytest.mark.parametrize("path, solver, core, valence", ECP_LOGS)
def test_ecp_core_electrons_and_occupations(path, solver, core, valence):
    """Needs cclib with the patch of cclib/cclib#1972 (installed in /applis/cclib)."""
    model = read_steps(path, solver)[-1]
    assert list(model.comp_details.general.core_electrons_per_atoms) == core
    occupations = qb.occupations(model, 200)
    assert sum(o.sum() for o in occupations) == valence


@pytest.mark.parametrize("path, solver, core, valence", [e for e in ECP_LOGS if "ch3i" in e[0]])
def test_ecp_mep(path, solver, core, valence):
    """Defect 1 (fixed): nuclei seen with Z - core electrons, density of the valence electrons.
    Validated against Gaussian's cubegen potential (LANL2DZ): RMS 0.0006 a.u. on the 0.002 a.u.
    iso-density surface (see test_mep_against_gaussian)."""
    model = read_steps(path, solver)[-1]
    np.testing.assert_array_equal(calc_orb.nuclear_charges(model), np.array([6, 53, 1, 1, 1]) - core)
    with contextlib.redirect_stdout(io.StringIO()):
        rho, mep, X, Y, Z, _ = calc_orb.Potential(model, grid_step=0.25, nproc=4)
    assert np.sum(rho) * 0.25 ** 3 == pytest.approx(valence, abs=0.01)
    assert mep.min() < -0.005      # negative belt around iodine


@pytest.mark.parametrize("path, solver", [
    ("gaussian/h2o/H2O_OPT.log", "gaussian"), ("gaussian/small/O2_ROHF.log", "gaussian"),
    ("gaussian/h2o/H2O_OPT_unrestricted.log", "gaussian"), ("gaussian/h2co/SPplus.log", "gaussian"),
    ("orca6/h2co/SPminus.out", "orca"), ("gaussian/ch3i/OPT.log", "gaussian"), ("orca6/ch3i/OPT.out", "orca"),
])
def test_electron_count_is_consistent(path, solver):
    assert calc_orb.electron_count_error(read_steps(path, solver)[-1]) is None


def test_electron_count_detects_unread_core_electrons():
    """ECP core electrons not read (cclib without the patch of cclib/cclib#1972): MEP/Fukui skipped."""
    model = read_steps("gaussian/ch3i/OPT.log")[-1]
    model.comp_details.general.core_electrons_per_atoms = [0] * 5
    assert "16 electrons" in calc_orb.electron_count_error(model)


# ------------------------------------------------------------------- mixed basis (defect 2)

@pytest.mark.parametrize("path", ["gaussian/mixed/OPT_6D7F.log", "gaussian/mixed/SP_5D10F.log"])
def test_mixed_basis(path):
    """Defect 2 (fixed): spherical or Cartesian is decided shell by shell."""
    model = read_steps(path)[-1]
    wf = qb.Wavefunction(model, nproc=4)
    X, Y, Z = wf.grid(0.07, over_s=5)       # fine enough for the 1s orbitals of C and O
    d3r = 0.07 ** 3
    mos = wf.mo(list(range(10)), 0, X, Y, Z)
    overlap = np.array([[np.sum(a * b) * d3r for b in mos] for a in mos])
    np.testing.assert_allclose(overlap, np.eye(10), atol=2e-3)
    assert np.sum(wf.density(X, Y, Z)) * d3r == pytest.approx(16, abs=0.01)


# -------------------------------------------------------------- single points (defect 5)

def test_single_points_only_explain_that_an_optimization_is_needed(tmp_path):
    """Defect 5 (fixed): the stop stays (Thomas: an OPT reference is needed) with a clear message."""
    import yaml
    from quchemreport.config.config import Config
    from quchemreport.parser_engine.scanlog import process_logfile_list
    import quchemreport.quality_check_engine.conformity as conformity
    from tests.helpers import report_config
    cfg = report_config([{"path": "gaussian/h2co/SPplus.log", "type": "sp", "reference": True},
                         {"path": "gaussian/h2co/SPminus.log", "type": "sp"}])
    config = Config(**yaml.safe_load(yaml.safe_dump(cfg)))
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as stop:
        _, models = process_logfile_list(config, log_storage_path=str(tmp_path))
        conformity.tests(models, config)
    assert stop.value.code == 1
    assert "Contact admin" not in out.getvalue()
    assert "all the log files are single points" in out.getvalue() and "type opt" in out.getvalue()


# ------------------------------------------------------- state labels (defects 7 and 8)

def test_state_labels_per_multiplicity():
    """Defect 7 (fixed): states numbered within each multiplicity, in the order of the log."""
    from quchemreport.utility_services.naming import state_labels
    orca = read_steps("orca6/h2co/TD_T.out", "orca")[-1]       # singlets and triplets, energy order
    assert state_labels(orca.results.excited_states.et_sym) == \
        ["T1", "S1", "T2", "T3", "T4", "S2", "T5", "S3", "S4", "S5"]
    gaussian = read_steps("gaussian/h2co/TD_T.log")[-1]
    assert state_labels(gaussian.results.excited_states.et_sym) == [f"T{i}" for i in range(1, 6)]
    assert state_labels(["3.002-A", "3.010-B"]) == ["1", "2"]     # unrestricted: global number


# -------------------------------------------------- structure warnings (ORCA 6.1, Gaussian)

def _warnings(models):
    from quchemreport.data_enrichment import compute
    from quchemreport.report_generator.generate_report import quality_warnings
    from quchemreport.utility_services.computed_data import init_computed_data
    computed = init_computed_data(len(models))
    with contextlib.redirect_stdout(io.StringIO()):
        compute.extract_simple_data(models, models[0], computed)
    return quality_warnings(computed)


@pytest.mark.parametrize("path, solver, text", [
    ("gaussian/hcn_ts/OPT_FREQ.log", "gaussian", "1157.2i"),
    ("orca6/nh3_plan/OPT_FREQ.out", "orca", "830.2i"),
    ("orca6/maxiter/H2O_maxiter.out", "orca", "did not converge"),
])
def test_structure_warnings(path, solver, text):
    assert any(text in w for w in _warnings(read_steps(path, solver)))


# ---------------------------------------------------------------- dispersion (defect 9)

@pytest.mark.parametrize("path, solver, dispersion", [
    ("gaussian/dispersion/B3LYP.log", "gaussian", None),
    ("gaussian/dispersion/B3LYP_GD3.log", "gaussian", "d3zero"),
    ("gaussian/dispersion/B3LYP_GD3BJ.log", "gaussian", "d3bj"),
    ("gaussian/dispersion/wB97XD.log", "gaussian", "d2"),      # built into the functional
    ("gaussian/dispersion/B97D3.log", "gaussian", "d3bj"),     # built into the functional
    ("orca6/functionals/SP_PBE0.out", "orca", None),
    ("orca6/functionals/SP_B3LYP_D3BJ.out", "orca", "d3bj"),
    ("orca6/functionals/SP_B3LYP_D4.out", "orca", "d4"),
    ("orca6/functionals/SP_wB97X_D3.out", "orca", "d3zero"),
    ("orca6/functionals/SP_wB97X_V.out", "orca", "vv10"),
])
def test_dispersion_model(path, solver, dispersion):
    """Defect 9: the dispersion correction comes from cclib (metadata["dispersion"], cclib#1973)."""
    assert read_steps(path, solver)[0].comp_details.general.dispersion == dispersion


def test_functional_label():
    from quchemreport.utility_services.log_data import functional_label
    assert functional_label("B3LYP") == "B3LYP"
    assert functional_label("B3LYP", "d3bj") == "B3LYP + D3(BJ)"
    assert functional_label("wB97XD", "d2") == "wB97XD + D2"
    assert functional_label("B3LYP", "d5") == "B3LYP + d5"       # unknown model kept as is


def _theory_check(paths, tmp_path):
    """Output of the 'theory' conformity test on SP logs, and whether it stopped the run."""
    import yaml
    from quchemreport.config.config import Config
    from quchemreport.parser_engine.scanlog import process_logfile_list
    import quchemreport.quality_check_engine.conformity as conformity
    from tests.helpers import report_config
    cfg = report_config([{"path": p, "type": "sp", **({"reference": True} if i == 0 else {})}
                         for i, p in enumerate(paths)])
    config = Config(**yaml.safe_load(yaml.safe_dump(cfg)))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _, models = process_logfile_list(config, log_storage_path=str(tmp_path))
        try:
            conformity.test_theory(conformity.extract_data_models(models))
        except SystemExit:
            return out.getvalue(), True
    return out.getvalue(), False


def test_theory_rejects_mixed_dispersion(tmp_path):
    """Defect 9: cclib gives "B3LYP" with or without EmpiricalDispersion=GD3BJ."""
    output, stopped = _theory_check(["gaussian/dispersion/B3LYP.log",
                                     "gaussian/dispersion/B3LYP_GD3BJ.log"], tmp_path)
    assert stopped and "B3LYP + D3(BJ)" in output


def test_theory_accepts_same_dispersion(tmp_path):
    output, stopped = _theory_check(["gaussian/dispersion/B3LYP_GD3BJ.log",
                                     "gaussian/dispersion/B3LYP_GD3BJ.log"], tmp_path)
    assert not stopped and "Test OK" in output


# ------------------------------------------------------------------- full reports (slow)

H2CO = {
    "gaussian": [("gaussian/h2co/OPT.log", "opt"), ("gaussian/h2co/FREQ.log", "freq"),
                 ("gaussian/h2co/TD.log", "td"), ("gaussian/h2co/TD_T.log", "td"),
                 ("gaussian/h2co/OPT_ES1.log", "opt_es"), ("gaussian/h2co/OPT_ET1.log", "opt_es"),
                 ("gaussian/h2co/SPplus.log", "sp"), ("gaussian/h2co/SPminus.log", "sp")],
    "orca": [("orca6/h2co/OPT_FREQ.out", ["opt", "freq"]), ("orca6/h2co/TD.out", "td"),
             ("orca6/h2co/TD_T.out", "td"), ("orca6/h2co/OPT_ES1.out", "opt_es"),
             ("orca6/h2co/OPT_ET1.out", "opt_es"), ("orca6/h2co/SPplus.out", "sp"),
             ("orca6/h2co/SPminus.out", "sp")],
}


def _logfiles(entries):
    out = []
    for i, (path, kind) in enumerate(entries):
        entry = {"path": path, "type": kind}
        if i == 0:
            entry["reference"] = True
        if kind == "opt_es":
            entry["excited_state_number"] = 1
        out.append(entry)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["gaussian", "orca"])
def test_h2co_full_report(solver):
    report = run_report(_logfiles(H2CO[solver]), solver,
                        **{"output.include.wavefunction.MEP_maps": True,
                           "output.include.reactivity.CDFT_indices": True,
                           "output.include.reactivity.fukui_functions": True,
                           "output.include.optimized_excited_states.transitions": True,
                           "output.include.optimized_excited_states.electron_density_difference": True})
    assert report.returncode == 0, report.output[-3000:]
    for pattern in ("MO-homo", "MEP", "fukui-SP_plus", "fukui-SP_minus", "Fdual", "emi-EDD-S1", "emi-EDD-T1"):
        assert report.images(pattern), pattern
    assert len(report.images("-EDD-T")) >= 5
    # Transition densities (27/11/2026): singlets and the emitting singlet; none for the
    # spin-forbidden triplets of a restricted calculation
    assert report.images("-TDEN-S") and report.images("emi-TDEN-S1")
    assert not report.images("-TDEN-T")
    assert "Transition density (charge)" in report.tex
    # Planar molecule: a single view per picture (visualization_engine.camera), except the
    # transition densities and dipoles of the states whose dipole is perpendicular to the plane,
    # along the line of sight (visu_mayavi.dipole_views, 27/11/2026): then both pictures get the
    # second view
    second = [f for f in report.files if "_cam2" in f]
    assert all("-TDEN-" in f or "-TDIP-" in f for f in second)
    assert sorted(f.replace("-TDEN-", "-TDIP-") for f in second if "-TDEN-" in f) == \
        sorted(f for f in second if "-TDIP-" in f)
    assert "fukui functions" in report.tex.lower()


def _td_without_amplitudes():
    return without_lines("gaussian/h2co/TD.log", lambda l: re.match(r"^ +\d+[AB]? *(->|<-) *\d+[AB]? +-?\d+\.\d+", l),
                         "TD_without_amplitudes.log")


def _td_without_rotatory_strengths():
    state = {"skip": False}

    def drop(line):
        if "Rotatory Strengths" in line:
            state["skip"] = True
            return True
        if state["skip"] and re.match(r"^\s+(state|\d+)\s", line):
            return True
        state["skip"] = False
        return False
    return without_lines("gaussian/h2co/TD.log", drop, "TD_without_rotatory_strengths.log")


@pytest.mark.slow
def test_td_without_amplitudes():
    """Defect 3 (fixed): formerly a crash in qmckl with empty amplitude lists."""
    report = run_report([{"path": "gaussian/h2co/OPT.log", "type": "opt", "reference": True},
                         {"path": _td_without_amplitudes(), "type": "td"}])
    assert report.returncode == 0, report.output[-3000:]
    assert "amplitudes" in report.output
    assert not report.images("EDD")
    assert report.images("UV-Abso-Spectrum")


@pytest.mark.slow
def test_td_without_rotatory_strengths():
    """Defect 4 (fixed): formerly an IndexError in the transition table; no R column now."""
    report = run_report([{"path": "gaussian/h2co/OPT.log", "type": "opt", "reference": True},
                         {"path": _td_without_rotatory_strengths(), "type": "td"}])
    assert report.returncode == 0, report.output[-3000:]
    assert len(report.images("EDD-S")) == 5
    assert "R (" not in report.tex and "Rotatory" not in report.tex


@pytest.mark.slow
def test_orca_nto_from_the_td_orbitals():
    """Defect 6 (fixed): the EDD and NTO of a TD use its own orbitals at the reference geometry."""
    report = run_report([{"path": "orca6/h2co/OPT_FREQ.out", "type": ["opt", "freq"], "reference": True},
                         {"path": "orca6/h2co/TD.out", "type": "td"}], "orca",
                        **{"output.include.excited_states.natural_transition_orbitals.enabled": True})
    assert report.returncode == 0, report.output[-3000:]
    assert report.images("NTO-S1-pair1-MO-hole")


@pytest.mark.slow
def test_orca_singlet_triplet_edd_names():
    """Defect 7 (fixed): EDD of an ORCA singlet+triplet TD named S1-S5 and T1-T5."""
    report = run_report([{"path": "orca6/h2co/OPT_FREQ.out", "type": ["opt", "freq"], "reference": True},
                         {"path": "orca6/h2co/TD_T.out", "type": "td"}], "orca")
    assert report.returncode == 0, report.output[-3000:]
    names = {re.search(r"EDD-([ST]\d+)\.png", f).group(1) for f in report.images("EDD-")}
    assert names == {f"S{i}" for i in range(1, 6)} | {f"T{i}" for i in range(1, 6)}


@pytest.mark.slow
def test_triplet_nto_names():
    """Defect 8 (fixed): NTO named and captioned with the multiplicity (T1, not state 1)."""
    report = run_report([{"path": "gaussian/h2co/OPT.log", "type": "opt", "reference": True},
                         {"path": "gaussian/h2co/TD_T.log", "type": "td"}],
                        **{"output.include.excited_states.natural_transition_orbitals.enabled": True})
    assert report.returncode == 0, report.output[-3000:]
    assert report.images("NTO-T1-pair1-MO-hole")
    assert not report.images("NTO-S")
    # grid label of the pair ("T1: 100.0%"), never "state 1"
    assert re.search(r"T1: [0-9.]+\\%", report.tex) and "state 1" not in report.tex


@pytest.mark.slow
def test_mixed_basis_full_report():
    """Defect 2 (fixed): MO and EDD pictures with a 6D 7F basis set (formerly a crash)."""
    report = run_report([{"path": "gaussian/mixed/OPT_6D7F.log", "type": "opt", "reference": True},
                         {"path": "gaussian/mixed/TD_6D7F.log", "type": "td"}])
    assert report.returncode == 0, report.output[-3000:]
    assert report.images("MO-homo") and len(report.images("EDD-S")) == 3


@pytest.mark.slow
@pytest.mark.parametrize("path, solver", [("gaussian/ch3i/OPT.log", "gaussian"), ("orca6/ch3i/OPT.out", "orca")])
def test_ecp_full_report(path, solver):
    """Defect 1 (fixed): MEP of CH3I with an ECP on iodine."""
    report = run_report([{"path": path, "type": "opt", "reference": True}], solver,
                        **{"output.include.wavefunction.MEP_maps": True})
    assert report.returncode == 0, report.output[-3000:]
    assert report.images("MEP") and "MEP map skipped" not in report.output
    assert "Local minima and maxima of the MEP" in report.tex and "extrema skipped" not in report.output


# ------------------------------------------------- MEP against Gaussian (defects 1, 10, 11)

def _mep_error(path, reference, step=0.1):
    """RMS difference and correlation between our MEP and Gaussian's cubegen potential at the
    400 points of tests/data/mep_reference/<reference>.txt (0.002 a.u. iso-density surface)."""
    from scipy.interpolate import RegularGridInterpolator
    from tests.datafiles import DATA
    ref = np.loadtxt(f"{DATA}/mep_reference/{reference}.txt")
    model = read_steps(path)[-1]
    with contextlib.redirect_stdout(io.StringIO()):
        _, mep, X, Y, Z, _ = calc_orb.Potential(model, grid_step=step, nproc=4)
    ours = RegularGridInterpolator((X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]), mep)(ref[:, :3])
    return np.sqrt(np.mean((ours - ref[:, 3]) ** 2)), np.corrcoef(ours, ref[:, 3])[0, 1]


@pytest.mark.parametrize("path, reference", [
    ("gaussian/h2co/OPT.log", "h2co"),
    ("gaussian/ch3i_mep/SP_LANL2DZ.log", "ch3i_lanl2dz"),        # defect 1 (ECP), fixed
    ("gaussian/ch3i_mep/SP_DGDZVP.log", "ch3i_dgdzvp"),           # defect 11, fixed
    ("gaussian/ch3x_mep/SP_CH3Cl.log", "ch3cl"),                  # defect 11, fixed
    ("gaussian/ch3x_mep/SP_CH3Br.log", "ch3br"),                  # defect 11, fixed
])
def test_mep_against_gaussian(path, reference):
    rms, corr = _mep_error(path, reference)
    assert rms < 0.002 and corr > 0.999, (rms, corr)


@pytest.mark.parametrize("path, cores", [
    ("gaussian/h2co/OPT.log", []),
    ("gaussian/ch3i_mep/SP_LANL2DZ.log", []),     # ECP: no core density
    ("gaussian/ch3i_mep/SP_DGDZVP.log", [53]),    # all-electron iodine
    ("gaussian/ch3x_mep/SP_CH3Cl.log", [17]),
])
def test_mep_core_spheres(path, cores):
    """Defect 11 (fixed): only the cores that the grid samples badly are integrated apart, so
    that the MEP of light molecules is computed exactly as before. At step 0.075 the grid
    misses 0.09 of the 7.96 electrons within 0.5 bohr of Cl, 1.4 of the 28.6 of I."""
    model = read_steps(path)[-1]
    with contextlib.redirect_stdout(io.StringIO()):
        wf = qb.Wavefunction(model, nproc=4)
        X, Y, Z = wf.grid(0.075)
        axes = [X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]]
        rho = wf.density(X, Y, Z)
        found = calc_orb._badly_sampled_cores(wf, axes, rho, 0.075 ** 3)
    assert [model.molecule.atoms_Z[i] for i, _, _ in found] == cores
    for i, charge, mask in found:   # exact charge within CORE_RADIUS, close to the grid's
        assert 0.0 < charge < model.molecule.atoms_Z[i]
        assert abs(charge - rho[mask].sum() * 0.075 ** 3) < 2.0


def test_mo_energies_below_the_printed_format():
    """Defect 10 (fixed): the 1s energy of iodine (all-electron) is printed '**********' by
    Gaussian; the MO energies were lost (None) and the density crashed."""
    model = read_steps("gaussian/ch3i_mep/SP_DGDZVP.log")[-1]
    energies = model.results.wavefunction.MO_energies[0]
    assert energies[0] == -np.inf and np.isfinite(energies[1:]).all()
    assert qb.occupations(model, len(energies))[0].sum() == 62


# ------------------------------------------------------- MEP extrema on the molecular surface

def test_surface_extrema_on_a_sphere():
    """V = z on a sphere: one minimum at the south pole, one maximum at the north pole."""
    from quchemreport.data_enrichment import mep_surface
    X, Y, Z = np.mgrid[-4:4.01:0.1, -4:4.01:0.1, -4:4.01:0.1]
    rho = np.exp(-(X ** 2 + Y ** 2 + Z ** 2))
    extrema = mep_surface.mep_surface_extrema(rho, Z.copy(), X, Y, Z, [[0.0, 0.0, 0.0]], isovalue=np.exp(-4.0))
    assert [e["kind"] for e in extrema] == ["min", "max"]
    assert np.allclose(extrema[0]["position"], [0, 0, -2], atol=0.05)
    assert np.allclose(extrema[1]["position"], [0, 0, 2], atol=0.05)
    assert np.isclose(extrema[1]["value"], 2.0, atol=0.01) and np.isclose(extrema[1]["distance"], 2.0, atol=0.05)
    assert np.allclose(extrema[1]["direction"], [0, 0, 1], atol=0.03)


def test_mep_color_range():
    """Calibrated fixed scale: +-0.05 a.u. for neutral molecules, +-0.20 a.u. per unit charge."""
    from quchemreport.data_enrichment import mep_surface
    assert mep_surface.color_range(0) == mep_surface.color_range(None) == 0.05
    assert mep_surface.color_range(1) == mep_surface.color_range(-1) == 0.20
    assert mep_surface.color_range(-3) == 0.60


# Gaussian 09 cubegen (potential=SCF) at the extrema found at step 0.1 (kcal/mol): kind, nearest
# atom (index), value. Our values differ by less than 0.4 kcal/mol at this step.
SURFACE_EXTREMA = {
    "gaussian/h2co/OPT.log": [("min", 1, -29.71), ("min", 1, -29.71), ("max", 3, 24.01)],
    # sigma-hole on Br (max on the C-Br axis) inside a negative belt (3 minima, C3v)
    "gaussian/ch3x_mep/SP_CH3Br.log": [("min", 1, -15.09), ("min", 1, -15.09), ("min", 1, -15.08),
                                       ("max", None, 20.53), ("max", None, 20.53), ("max", None, 20.54),
                                       ("max", 1, 7.00)],
}


@pytest.mark.parametrize("path", SURFACE_EXTREMA)
def test_mep_surface_extrema(path):
    """MEP extrema on the 0.001 e-/bohr3 isosurface against Gaussian (cubegen at the same points)."""
    from quchemreport.data_enrichment import mep_surface
    from quchemreport.utility_services.units import a0_to_A, Eh_to_Kcal_mol
    model = read_steps(path)[-1]
    with contextlib.redirect_stdout(io.StringIO()):
        rho, mep, X, Y, Z, _ = calc_orb.Potential(model, grid_step=0.1, nproc=4)
    coords = np.asarray(model.results.geometry.elements_3D_coords).reshape(-1, 3) / a0_to_A
    extrema = mep_surface.mep_surface_extrema(rho, mep, X, Y, Z, coords)
    expected = SURFACE_EXTREMA[path]
    assert len(extrema) == len(expected)
    for ext, (kind, atom, value) in zip(extrema, expected):
        assert ext["kind"] == kind
        if atom is not None:
            assert ext["atom"] == atom
        assert abs(ext["value"] * Eh_to_Kcal_mol - value) < 0.5


# ------------------------------------------------------------- ORCA excited state energies

def test_orca_excited_state_energies_in_cm1():
    """The cclib JSON gives ORCA's excited state energies in hartree (dict of spectra); they were
    read as cm-1 (wavelengths of millions of nm in the ORCA reports). S1 of H2CO: ~33100 cm-1
    with both programs."""
    gaussian = read_steps("gaussian/h2co/TD.log")[-1].results.excited_states.et_energies
    orca = read_steps("orca6/h2co/TD.out", "orca")[-1].results.excited_states.et_energies
    assert isinstance(orca, list) and 30000 < orca[0] < 36000
    assert abs(orca[0] - gaussian[0]) / gaussian[0] < 0.02



# ------------------------------------------------------- spin-orbit coupling (ORCA dosoc)

def test_orca_soc_reader():
    """SOC of H2CO (ORCA 6.1, TD-DFT dosoc at the ground state geometry): El-Sayed rule,
    <T1|H_SO|S1> = 0 (both n-pi*) and <T2|H_SO|S1> about 42 cm-1; three T1 sublevels."""
    from quchemreport.parser_engine.orca_soc import read_orca_soc
    from tests.datafiles import log
    soc = read_orca_soc(log("orca6/h2co/TD_SOC.out"))
    assert len(soc["singlets"]) == 5 and len(soc["triplets"]) == 5
    assert abs(soc["socme"][(1, 0)] - 58.81) < 0.05
    assert soc["socme"][(1, 1)] < 0.05 and abs(soc["socme"][(2, 1)] - 41.78) < 0.05
    sublevels = [s for s in soc["states"][1:] if s["components"][0][1] == "T1"]
    assert len(sublevels) == 3 and all(abs(s["energy"] - 26849.5) < 1 for s in sublevels)
    assert soc["states"][0]["components"][0][1] == "S0"
    assert read_orca_soc(log("orca6/h2co/TD.out")) is None


def test_phosphorescence_from_optimized_triplet(tmp_path):
    """Acetone T1 optimized with dosoc (ORCA 6.1): emission at about 531 nm, one bright sublevel,
    mean radiative rate about 95 s-1 (radiative lifetime about 10 ms); Jablonski-Perrin picture."""
    from quchemreport.data_enrichment import compute
    from quchemreport.utility_services.computed_data import init_computed_data
    from quchemreport.visualization_engine import visu_plots
    model = read_steps("orca6/acetone_soc/OPT_ET1_SOC.out", "orca")[-1]
    computed = init_computed_data(1)
    with contextlib.redirect_stdout(io.StringIO()):
        compute.extract_simple_data([model], model, computed)
        compute.compute_phosphorescence(computed, 0, 1)
    phospho = computed.derived.optimized_excited_states.phosphorescence[0]
    assert phospho["state"] == "T1" and len(phospho["sublevels"]) == 3
    assert abs(1e7 / phospho["sublevels"][0]["energy"] - 531) < 2
    assert 80 < phospho["k_r"] < 110 and 8e-3 < phospho["lifetime"] < 13e-3
    visu_plots.jablonski(computed.results.excited_states.soc[0], file_name=str(tmp_path / "img"))
    assert (tmp_path / "img-Jablonski.png").is_file()


def test_edd_of_a_pure_beta_excitation():
    """The emitting state of triplet O2 (ORCA, UKS) is a pure beta excitation: it formerly got no
    EDD (calc_orb.TD only looked at the alpha orbitals) and an empty optimized-state table."""
    from quchemreport.data_enrichment import calc_orb
    model = read_steps("orca6/o2/OPT_ES1.out", "orca")[-1]
    state = model.results.excited_states.et_transitions[0]
    assert {sub[0][1] for sub in state} == {1} and {sub[1][1] for sub in state} == {1}
    out, X, Y, Z = calc_orb.TD(model, [state], grid_step=0.3, nproc=2)
    edd, tozer = out[0][0], out[0][1]
    d3r = (X[1, 0, 0] - X[0, 0, 0]) * (Y[0, 1, 0] - Y[0, 0, 0]) * (Z[0, 0, 1] - Z[0, 0, 0])
    assert abs(edd.sum() * d3r) < 0.02          # an electron moved, none created
    # pi_u(beta) -> pi_g(beta) between orbitals on the same atoms: 0.16 e moved (grid 0.3 bohr)
    assert 0.05 < edd[edd > 0].sum() * d3r < 1.0
    assert 0 < tozer < 1
