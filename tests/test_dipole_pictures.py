"""Pictures of the transition dipoles and rotatory strengths (27/11/2026, Thomas Cauchy).

- Arrows of constant thickness (visu_mayavi._arrow): the glyph of mlab.quiver3d was scaled as a
  whole with the length, so long arrows were thick and short ones were threads.
- An arrow within 30 degrees of the line of sight is marked in the legend and a second view is
  added, turned by the smallest angle that shows every arrow (visu_mayavi.dipole_views).
- The values of the dipoles are written on the picture (write_legend), no minimum length.
- R(velocity) from the velocity and magnetic transition dipoles read by cclib
  (compute.rotatory_strengths_velocity), the dissymmetry factor g = 4R/D, both in the report.

Data: gaussian/hcfclbr/TD_full (chiral HCFClBr, PBE0/6-31G(d), 10 singlets, IOp(9/40=4));
orca6/hcfclbr/TD_tda (the same molecule, PBE0/def2-SVP, TDA, tprint 1e-6). cclib reads the
velocity and magnetic transition dipoles of ORCA with the branch orca-magnetic-dipoles (magnetic
ones in Bohr magnetons, as for Gaussian); without it the ORCA test is skipped.

Run (see tests/README.md): tests/run_tests.sh -k dipole_pictures
"""
import numpy as np
import pytest

from tests.datafiles import log


def hcfclbr():
    import cclib
    return cclib.io.ccread(log("gaussian/hcfclbr/TD_full.log"))


def printed_velocity_rotatory_strengths(path):
    """R(velocity) column printed by Gaussian."""
    text = open(path).read().split("R(velocity)    E-M Angle")[-1].splitlines()[1:]
    values = []
    for line in text:
        parts = line.split()
        if len(parts) != 6:
            break
        values.append(float(parts[4]))
    return values


def test_velocity_rotatory_strengths_match_gaussian():
    """R = 235.72 p.m / E from cclib's etveldips, etmagdips and etenergies gives the R(velocity)
    printed by Gaussian to the rounding of the vectors, with the same sign for every state."""
    from quchemreport.data_enrichment import compute
    d = hcfclbr()
    computed = compute.rotatory_strengths_velocity(list(d.etenergies), d.etveldips.tolist(), d.etmagdips.tolist())
    printed = printed_velocity_rotatory_strengths(log("gaussian/hcfclbr/TD_full.log"))
    assert len(computed) == len(printed) == 10
    assert np.max(np.abs(np.array(computed) - printed)) < 0.04
    assert all(np.sign(c) == np.sign(p) for c, p in zip(computed, printed))
    assert compute.rotatory_strengths_velocity([30000.0], None, [[0, 0, 1]]) is None


def test_rotatory_strength_and_dissymmetry_factor_from_the_vectors():
    """R(length) = 235.72 mu.(-m) with the sign of compute.transition_dipole; g = 4R/D; the
    angle between mu and -m is acute when R > 0."""
    from quchemreport.data_enrichment import compute
    d = hcfclbr()
    for k in range(10):
        mu, m = d.etdips[k], -d.etmagdips[k]
        assert abs(compute.ROTATORY_UNIT * mu @ m - d.etrotats[k]) < 0.05
        assert (compute.dipole_angle(mu, m) < 90) == (d.etrotats[k] > 0)
    g = compute.dissymmetry_factor(d.etrotats[9], d.etdips[9])
    assert g == pytest.approx(2.10e-3, rel=0.01)
    assert compute.dissymmetry_factor(1.0, None) is None and compute.dissymmetry_factor(1.0, [0, 0, 0]) is None


def test_line_of_sight_and_second_view():
    from quchemreport.visualization_engine import visu_mayavi as vm, camera
    view = camera.View(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), "best")
    assert vm.along_line_of_sight([0.1, 0.0, 1.0], view) == 1
    assert vm.along_line_of_sight([0.0, 0.2, -1.0], view) == -1
    assert vm.along_line_of_sight([1.0, 0.0, 1.0], view) == 0        # 45 degrees
    assert vm.along_line_of_sight([0.0, 0.0, 0.0], view) == 0
    # arrows readable in the first view: the views of the molecule are kept
    assert vm.dipole_views([view], [[1.0, 0.0, 0.0], None]) == [view]
    # an arrow along the line of sight: second view turned by 45 degrees, the arrow at 45 degrees
    views = vm.dipole_views([view], [[0.0, 0.0, 1.0]])
    assert len(views) == 2 and views[0] is views[0]
    d2 = views[1].direction
    assert np.degrees(np.arccos(abs(d2 @ [0, 0, 1]))) == pytest.approx(45.0, abs=1e-6)
    assert abs(np.linalg.norm(d2) - 1) < 1e-9 and abs(views[1].up @ d2) < 1e-9
    # two arrows: one along the line of sight, one along the first turn axes
    views = vm.dipole_views([view], [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    worst = min(np.degrees(np.arccos(abs(views[1].direction @ v))) for v in ([0, 0, 1], [1, 0, 0]))
    assert worst >= vm.SECOND_VIEW_ANGLE - 1e-6
    # a second view of the molecule that shows the arrow is kept
    side = camera.View(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), "globular")
    assert vm.dipole_views([view, side], [[0.0, 0.0, 1.0]]) == [view, side]


def test_dipole_legend():
    from quchemreport.visualization_engine import visu_mayavi as vm, camera
    view = camera.View(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), "best")
    lines = vm.dipole_legend([0.0, 0.239, 0.0], [0.0, 0.1, -1.0], view)
    assert lines[0][0] == "μ = 0.239 e·a₀" and lines[1][0].endswith(vm.AWAY_FROM_VIEWER)
    assert vm.dipole_legend(None, None, view)[0][0] == "μ = 0 (forbidden)"


def test_legend_is_written_on_the_picture(tmp_path):
    from PIL import Image
    from quchemreport.visualization_engine import visu_mayavi as vm
    png = str(tmp_path / "blank.png")
    Image.new("RGB", (300, 300), "white").save(png)
    vm.write_legend(png, [("μ = 0.086 e·a₀ ⊗", (0, 0.6, 1))])
    pixels = np.asarray(Image.open(png).convert("RGB"))
    corner, rest = pixels[:40, :200], pixels[100:, :]
    assert (corner < 250).any() and (rest == 255).all()


def test_arrows_have_a_constant_thickness():
    """The dimensions of the arrow glyph, fractions of its length, give the same radii (bohr)
    for every length; an arrow shorter than its tip is a cone scaled down."""
    from mayavi import mlab
    from quchemreport.visualization_engine import visu_mayavi as vm
    mlab.options.offscreen = True
    figure = mlab.figure()
    before = len(figure.children)
    for length in (0.3, 1.0, 4.0, 10.0):
        vm._arrow(figure, [0, 0, 0], [length, 0, 0], (0, 0, 1))
        source = figure.children[-1].children[0].children[0].glyph.glyph_source.glyph_source
        shrink = min(1.0, length / vm.ARROW_TIP_LENGTH)
        assert source.shaft_radius * length == pytest.approx(vm.ARROW_SHAFT_RADIUS * shrink)
        assert source.tip_radius * length == pytest.approx(vm.ARROW_TIP_RADIUS * shrink)
        assert source.tip_length * length == pytest.approx(min(length, vm.ARROW_TIP_LENGTH))
    vm._arrow(figure, [0, 0, 0], [0, 0, 0], (0, 0, 1))      # zero vector: nothing drawn
    assert len(figure.children) == before + 4
    mlab.close(figure)


def test_report_labels_and_caption():
    """Under the pictures: f and g only (the rotatory strengths are in the table)."""
    from quchemreport.report_generator import generate_report as gr
    label = gr.dissymmetry_label(33.84, 34.98, 2.1e-3)
    assert label == r"$g = 2.1\times10^{-3}$"
    assert "(sign uncertain)" in gr.dissymmetry_label(0.64, -0.55, 6.9e-4)
    assert gr.dissymmetry_label(-5.3, None, None) == ""
    caption = gr.transition_density_caption(False, [], "TD.log", chiral=True)
    assert "274" in caption and r"$\odot$" in caption and "velocity" in caption
    assert "274" not in gr.transition_density_caption(False, [], "TD.log")


def test_grid_cells_as_wide_as_their_pictures():
    """A state seen from two points of view (4 pictures) takes a row; two states with one view
    (2 pictures) share the next one, at the same picture size."""
    from quchemreport.report_generator import generate_report as gr

    class Doc(list):
        pass

    doc = Doc()
    cells = [(["a1.png", "a2.png", "a3.png", "a4.png"], "S1"), (["b1.png", "b2.png"], "S2"),
             (["c1.png", "c2.png"], "S3"), (["d1.png", "d2.png"], "S4")]
    gr.figure_grid(doc, cells, 1, "caption")
    text = "".join(str(x) for x in doc)
    rows = text.split(r"\\[2mm]")
    assert [row.count("minipage}[t]") for row in rows] == [1, 2, 1]
    assert r"\begin{minipage}[t]{1.000\textwidth}" in text
    assert text.count(r"\begin{minipage}[t]{0.495\textwidth}") == 3
    # uniform cells: unchanged, ncol per row
    doc = Doc()
    gr.figure_grid(doc, [([f"{k}.png"], str(k)) for k in range(5)], 3, "caption")
    rows = "".join(str(x) for x in doc).split(r"\\[2mm]")
    assert [row.count("minipage}[t]") for row in rows] == [3, 2]


def test_continued_caption_is_short():
    from quchemreport.report_generator import generate_report as gr
    caption = gr.transition_density_caption(True, [], "TD.log", chiral=True)
    short = gr.continued_caption(caption)
    assert short.startswith("Transition density (charge) of the excited states")
    assert short.endswith("(Source: TD.log) (continued)") and len(short) < 200
    assert gr.continued_caption("Orbitals. Blue: positive") == "Orbitals. (continued)"


def test_orca_velocity_rotatory_strengths():
    """With the magnetic dipoles of ORCA in Bohr magnetons (cclib branch orca-magnetic-dipoles),
    the same formula gives the R(velocity) printed by ORCA. With TDA the two gauges disagree
    strongly (S1: 0.05 in the length gauge, 9.32 in the velocity gauge)."""
    import cclib
    from quchemreport.data_enrichment import compute
    path = log("orca6/hcfclbr/TD_tda.out")
    d = cclib.io.ccread(path)
    if getattr(d, "etmagdips", None) is None or getattr(d, "etveldips", None) is None:
        pytest.skip("cclib without the ORCA velocity and magnetic dipoles (branch orca-magnetic-dipoles)")
    computed = compute.rotatory_strengths_velocity(list(d.etenergies), d.etveldips.tolist(), d.etmagdips.tolist())
    text = open(path).read().split("CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS")[1].splitlines()[5:15]
    printed = [float(line.split()[6]) for line in text]
    assert np.max(np.abs(np.array(computed) - printed)) < 0.01
    # R(length) of ORCA = -235.7 mu.m with m in Bohr magnetons, as for Gaussian
    lengths = -compute.ROTATORY_UNIT * np.einsum("ij,ij->i", d.etdips, d.etmagdips)
    assert np.max(np.abs(lengths - d.etrotats)) < 0.01
