"""Changes after Thomas Cauchy's review of the acetone POV-Ray report (27/11/2026).

- views of the excited-state geometries: those of the reference, turned by the rotation that
  superimposes the reference on them (visu_mayavi.views_of, Kabsch);
- emission EDD drawn as ground minus excited state density (generate_visualization);
- white border of the pictures cut in the report, the same for all the cells of a grid
  (generate_report.trim_option);
- implicit solvent and CM5 charges read from cclib (scanlog);
- table of the log files grouped by kind of calculation (generate_report.logfile_rows).

Run (see tests/README.md): tests/run_tests.sh -k review_27
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from tests.helpers import read_steps


def geometry(path, solver="gaussian"):
    m = read_steps(path, solver)[-1]
    xyz = np.asarray(m.results.geometry.elements_3D_coords, dtype=float).reshape(-1, 3)
    return xyz, m.molecule.connectivity.atom_pairs, m.molecule.atoms_Z


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    a = np.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


@pytest.fixture
def reference_views():
    from quchemreport.visualization_engine import visu_mayavi
    yield visu_mayavi
    visu_mayavi.set_reference_geometry(None, None, None)


def test_kabsch_rotation_recovers_a_rotation():
    from quchemreport.visualization_engine.visu_mayavi import kabsch_rotation
    xyz, _, _ = geometry("gaussian/h2co/OPT.log")
    rot = rotation([1, 2, 3], 70)
    moved = (xyz - xyz.mean(axis=0)) @ rot.T + [3.0, -1.0, 2.0]
    assert np.allclose(kabsch_rotation(xyz, moved), rot, atol=1e-8)
    assert np.isclose(np.linalg.det(kabsch_rotation(xyz, moved)), 1.0)


def test_rotated_geometry_is_seen_from_the_same_side(reference_views):
    """A geometry of the reference in another orientation gets the reference views turned with it."""
    xyz, pairs, Z = geometry("gaussian/h2co/OPT.log")
    reference_views.set_reference_geometry(xyz, pairs, Z)
    ref = reference_views.views_of(xyz, pairs, Z)
    rot = rotation([0.3, -1, 0.5], 125)
    turned = reference_views.views_of(xyz @ rot.T + 1.5, pairs, Z)
    assert len(turned) == len(ref)
    for v, w in zip(ref, turned):
        assert np.allclose(rot @ v.direction, w.direction, atol=1e-8)
        assert np.allclose(rot @ v.up, w.up, atol=1e-8)


def test_excited_state_geometry_takes_the_reference_views(reference_views):
    """S1 of H2CO is pyramidal: its views are those of the reference turned by the Kabsch
    rotation (for acetone, its own views looked at it from another side)."""
    xyz, pairs, Z = geometry("gaussian/h2co/OPT.log")
    es, es_pairs, es_Z = geometry("gaussian/h2co/OPT_ES1.log")
    reference_views.set_reference_geometry(xyz, pairs, Z)
    ref = reference_views.views_of(xyz, pairs, Z)[0]
    moved = reference_views.views_of(es, es_pairs, es_Z)[0]
    rot = reference_views.kabsch_rotation(xyz, es)
    assert np.allclose(moved.direction, rot @ ref.direction)
    # the view of the S1 geometry in the reference frame is the reference view
    assert np.allclose(moved.up, rot @ ref.up)


def test_other_molecule_keeps_its_own_views(reference_views):
    xyz, pairs, Z = geometry("gaussian/h2co/OPT.log")
    other, o_pairs, o_Z = geometry("gaussian/ch3x_mep/SP_CH3Cl.log")
    own = reference_views.views_of(other, o_pairs, o_Z)
    reference_views.set_reference_geometry(xyz, pairs, Z)
    again = reference_views.views_of(other, o_pairs, o_Z)
    assert all(np.allclose(a.direction, b.direction) for a, b in zip(own, again))


def white_picture(path, box, size=(1200, 1200), dpi=None):
    from PIL import Image
    im = Image.new("RGB", size, "white")
    im.paste((0, 128, 128), box)
    im.save(path, dpi=dpi) if dpi else im.save(path)


def test_trim_option_cuts_the_common_white_border(tmp_path, monkeypatch):
    from quchemreport.report_generator import generate_report as gr
    monkeypatch.setattr(gr, "IMAGE_DIR", str(tmp_path))
    white_picture(tmp_path / "a.png", (100, 300, 1100, 900))
    white_picture(tmp_path / "b.png", (200, 250, 1000, 800))
    m = gr.TRIM_MARGIN * 1200
    # union 100-1100 x 250-900; trim = left bottom right top, in bp (72 dpi without pHYs)
    expected = [100 - m, 1200 - 900 - m, 1200 - 1100 - m, 250 - m]
    option = gr.trim_option(["a.png", "b.png"])
    assert option.startswith(",trim=") and option.endswith(",clip")
    values = [float(v[:-2]) for v in option[len(",trim="):-len(",clip")].split()]
    assert np.allclose(values, expected, atol=0.1)
    assert np.isclose(gr.trimmed_ratio(["a.png", "b.png"]), (650 + 2 * m) / (1000 + 2 * m))


def test_trim_option_uses_the_resolution_and_skips_unreadable(tmp_path, monkeypatch):
    from quchemreport.report_generator import generate_report as gr
    monkeypatch.setattr(gr, "IMAGE_DIR", str(tmp_path))
    white_picture(tmp_path / "hd.png", (0, 600, 1200, 1200), dpi=(144, 144))
    values = [float(v[:-2]) for v in gr.trim_option(["hd.png"])[len(",trim="):-len(",clip")].split()]
    assert np.allclose(values, [0, 0, 0, (600 - gr.TRIM_MARGIN * 1200) / 2], atol=0.1)
    assert gr.trim_option(["missing.png"]) == ""
    white_picture(tmp_path / "small.png", (0, 0, 10, 10), size=(100, 100))
    assert gr.trim_option(["hd.png", "small.png"]) == ""     # different sizes


def test_gas_phase_and_solvent_descriptions():
    from quchemreport.parser_engine.scanlog import solvent_description
    assert solvent_description({}) == ("none (gas phase)", None)
    text, model = solvent_description({"solvent_model": "SMD-IEFPCM", "solvent_name": "water",
                                       "solvent_params": {"epsilon": 78.3553}})
    assert model == "SMD-IEFPCM"
    assert text == "water (SMD-IEFPCM, dielectric constant 78.36)"
    m = read_steps("gaussian/h2co/OPT.log")[-1]
    assert m.comp_details.general.solvent == "none (gas phase)"


def test_gaussian_smd_solvent_and_cm5_charges():
    m = read_steps("gaussian/solvent/SP_SMD.log")[-1]
    general = m.comp_details.general
    assert general.solvent.startswith("water (SMD-IEFPCM, dielectric constant 78.3")
    assert general.solvent_reaction_field == "SMD-IEFPCM"
    w = m.results.wavefunction
    assert len(w.Hirshfeld_partial_charges) == 3 and len(w.CM5_partial_charges) == 3
    assert abs(sum(w.CM5_partial_charges)) < 1e-3
    assert w.CM5_partial_charges[0] < w.Hirshfeld_partial_charges[0] < 0     # CM5 more polar on O


def test_orca_cpcm_solvent_and_hirshfeld_charges():
    m = read_steps("orca6/solvent/SP_CPCM.out", "orca")[-1]
    assert m.comp_details.general.solvent_reaction_field.startswith("CPCM")
    assert "water" in m.comp_details.general.solvent or "unnamed" in m.comp_details.general.solvent
    assert len(m.results.wavefunction.Hirshfeld_partial_charges) == 3
    assert m.results.wavefunction.CM5_partial_charges is None      # not computed by ORCA


def fake_series():
    """ComputedData-like object of the acetone series: OPT, FREQ, TD, TD_T, SP+, SP-, S1, T1."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(4, 3))
    es = ref + 0.05
    logs = ["OPT.log", "FREQ.log", "TD.log", "TD_T.log", "SPplus.log", "SPminus.log", "OPT_ES1.log", "OPT_ET1.log"]
    jobs = [["opt"], ["freq"], ["td"], ["td"], ["sp"], ["sp"], ["opt_es"], ["opt_es"]]
    coords = [ref] * 6 + [es, es + 0.1]
    sym = [None, None, ["Singlet-A"] * 3, ["Triplet-A"] * 2, None, None, ["Singlet-A"], ["Triplet-A"]]
    return SimpleNamespace(
        metadata=SimpleNamespace(ref_log_file_idx=0, log_files=logs),
        comp_details=SimpleNamespace(general=SimpleNamespace(job_type=jobs)),
        molecule=SimpleNamespace(charges=[0, 0, 0, 0, 1, -1, 0, 0], multiplicity=[1, 1, 1, 1, 2, 2, 1, 1],
                                 atoms_Z=[[6, 8, 1, 1]] * 8),
        results=SimpleNamespace(geometry=SimpleNamespace(elements_3D_coords=[c.ravel().tolist() for c in coords]),
                                excited_states=SimpleNamespace(et_sym=sym)))


def test_logfile_rows_follow_the_sequence(monkeypatch):
    from quchemreport.report_generator import generate_report as gr
    monkeypatch.setattr(gr, "config_logfile", lambda config, cd, i: SimpleNamespace(excited_state_number=1))
    rows = gr.logfile_rows(fake_series(), config=None)
    assert [r[1] for r in rows] == ["OPT.log (reference)", "FREQ.log", "TD.log", "TD_T.log", "SPplus.log",
                                    "SPminus.log", "OPT_ES1.log", "OPT_ET1.log"]
    assert [r[0] for r in rows] == ["Ground-state optimization", "Frequencies", "Vertical excited states", "",
                                    "Single points", "", "Excited-state optimizations", ""]
    assert [r[2] for r in rows] == ["0, 1", "0, 1", "0, 1", "0, 1", "+1, 2", "-1, 2", "0, 1", "0, 1"]
    assert [r[3] for r in rows] == ["optimized"] + ["of OPT.log"] * 5 + ["optimized", "optimized"]
    assert [r[4] for r in rows] == ["", "", "3 singlets", "2 triplets", "", "", "S1", "T1"]


def test_orca_opt_freq_in_one_job(monkeypatch):
    from quchemreport.report_generator import generate_report as gr
    monkeypatch.setattr(gr, "config_logfile", lambda config, cd, i: SimpleNamespace(excited_state_number=1))
    cd = fake_series()
    cd.comp_details.general.job_type[0] = ["opt", "freq"]
    rows = gr.logfile_rows(cd, config=None)
    assert rows[0][0] == "Ground-state optimization and frequencies"


def test_last_geometry_xyz_for_open_babel():
    """The XYZ block given to Open Babel is built from atomnos and atomcoords: cclib master after
    1.8.1 raised IndexError in its XYZ writer for every Gaussian optimization (27/11/2026)."""
    import cclib
    from quchemreport.parser_engine.scanlog import last_geometry_xyz
    from tests.datafiles import log
    data = cclib.io.ccread(log("gaussian/h2co/OPT.log"))
    assert len(data.atomcoords) > 1
    lines = last_geometry_xyz(data).splitlines()
    assert lines[0] == str(len(data.atomnos)) and len(lines) == len(data.atomnos) + 2
    first = lines[2].split()
    assert first[0] == cclib.parser.utils.PeriodicTable().element[int(data.atomnos[0])]
    assert [float(x) for x in first[1:]] == pytest.approx(list(data.atomcoords[-1][0]), abs=1e-9)


def _spectrum_axes(energies, foscs, axis, tmp_path, user=None, top="auto", markers=False):
    """Main and top axes of a spectrum drawn by visu_plots.spectrum_figure (FWHM 3000 cm-1):
    limits of the main axis, top ticks (None without a top axis), and with markers=True the
    positions of the dark-state circles."""
    import matplotlib.pyplot as plt
    from quchemreport.data_enrichment import TD2UVvis
    from quchemreport.visualization_engine import visu_plots as vp
    vp.set_spectrum_axis(axis, top)
    TD2UVvis.set_user_range(user)
    try:
        start, end, n = TD2UVvis.spectrum_grid(energies, 3000.0)
        x, spec = TD2UVvis.GaussianSpectrum(start, end, n, energies, [foscs], 3000.0)
        close = vp.plt.close
        vp.plt.close = lambda *a, **k: None
        try:
            vp.spectrum_figure(x, spec[0], energies, foscs, "eps", "f", str(tmp_path / "s.png"))
            fig = plt.gcf()
        finally:
            vp.plt.close = close
        lo, hi = sorted(fig.axes[1].get_xlim())
        children = fig.axes[0].child_axes
        ticks = [float(t) for t in children[0].get_xticks()] if children else None
        circles = [float(v) for line in fig.axes[1].get_lines() if line.get_marker() == "o" for v in line.get_xdata()]
        plt.close(fig)
        return (lo, hi, ticks, circles) if markers else (lo, hi, ticks)
    finally:
        vp.set_spectrum_axis("wavenumber")
        TD2UVvis.set_user_range(None)


def test_spectrum_of_a_near_infrared_band(tmp_path):
    """A band at 9000 cm-1: the axis stays above the start of the grid (the margins of
    matplotlib put it below zero, and the top axis asked for 1e14 tick candidates)."""
    lo, hi, top = _spectrum_axes([9000, 15000, 30000], [1.2, 0.3, 0.5], "wavenumber", tmp_path)
    assert lo >= 1000 and hi > 30000 and 3 <= len(top) <= 12


@pytest.mark.parametrize("axis", ["wavenumber", "wavelength"])
def test_spectrum_range_is_where_something_is_visible(tmp_path, axis):
    """Both axes are cut to the visible curve and sticks with a 10 % margin: one band fills
    most of the axis (42 % on the wavenumber axis before)."""
    lo, hi, _ = _spectrum_axes([20000], [0.5], axis, tmp_path)
    band = (15000, 25000) if axis == "wavenumber" else (400, 667)
    assert lo > band[0] - 0.25 * (band[1] - band[0]) and hi < band[1] + 0.25 * (band[1] - band[0])
    assert lo < 20000 < hi if axis == "wavenumber" else lo < 500 < hi


def test_narrow_user_range_gets_finer_top_ticks(tmp_path):
    """400-420 nm on the wavelength axis: finer cm-1 ticks when the round ones are too few."""
    _, _, top = _spectrum_axes([24000, 25000], [0.5, 0.2], "wavelength", tmp_path, user=(1e7 / 420, 1e7 / 400))
    assert len(top) >= 3


@pytest.mark.parametrize("axis, top", [("eV", "nm"), ("nm", "eV"), ("cm-1", "eV"), ("energy", "none")])
def test_spectrum_axis_combinations(tmp_path, axis, top):
    """Main axis in cm-1, eV or nm, top axis in another unit or none (Thomas Cauchy, 27/11/2026)."""
    from quchemreport.visualization_engine import visu_plots as vp
    lo, hi, ticks = _spectrum_axes([20000, 30000], [0.5, 0.3], axis, tmp_path, top=top)
    band = sorted(vp.UNITS[vp.unit_name(axis)]["from_cm"](np.array([20000.0, 30000.0])))
    assert lo < band[0] and hi > band[1]
    assert ticks is None if top == "none" else len(ticks) >= 3


def test_low_dark_state_is_shown_high_one_is_not(tmp_path):
    """A dark state below the visible bands keeps the axis down to it (open circle); a dark state
    above them does not widen the axis (the states above the last computed one are unknown)."""
    lo, hi, _, circles = _spectrum_axes([22000, 36000, 40000, 70000], [0.0, 0.4, 0.2, 0.0], "wavenumber",
                                        tmp_path, markers=True)
    assert lo < 22000 < hi < 70000
    assert circles == [22000.0]


def test_spectrum_axis_names():
    from quchemreport.visualization_engine import visu_plots as vp
    assert [vp.unit_name(n) for n in ("cm-1", "eV", "NM", "energy")] == ["wavenumber", "energy", "wavelength", "energy"]
    with pytest.raises(ValueError):
        vp.unit_name("Hz")
    with pytest.raises(ValueError):
        vp.set_spectrum_axis("nm", "wavelength")
    try:
        vp.set_spectrum_axis("nm")
        assert vp.SECONDARY_AXIS == "wavenumber"
        vp.set_spectrum_axis("eV")
        assert vp.SECONDARY_AXIS == "wavelength"
    finally:
        vp.set_spectrum_axis("wavenumber")
    assert vp.to_cm("eV", [1.0])[0] == pytest.approx(8065.544, abs=1e-3)


def test_dark_states_note():
    from quchemreport.report_generator import generate_report as gr
    assert gr.dark_states_note([0.5, 0.2]) == ""
    assert "Open circles" in gr.dark_states_note([0.0, 0.5]) and gr.dark_states_note(None) == ""


def test_no_cd_spectrum_for_an_achiral_molecule():
    """The rotatory strengths of an achiral molecule are numerical noise (acetone: 4e-4): no CD or
    CPL spectrum, a sentence instead (Thomas Cauchy, 27/11/2026)."""
    from quchemreport.report_generator import generate_report as gr
    note = gr.achiral_note([0.0001, -0.0004], "circular dichroism", "TD.log")
    assert note.startswith("No circular dichroism spectrum") and "TD.log" in note and "4.0e-04" in note
    assert gr.achiral_note([0.0, -13.56], "circular dichroism", "TD.log") == ""
    assert gr.achiral_note(None, "circular dichroism", "TD.log") == ""
