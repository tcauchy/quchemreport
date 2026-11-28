"""Camera views of the 3D pictures (visualization_engine/camera.py), 24/11/2026.

Geometries: tests/data/geometries/mol01-10.xyz (real calculations, renamed by number: helicenes,
spiro compounds, folded adducts, a cobalt complex, a 408-atom cage...) and the logs of tests/data.
The former fixed views (mlab.view azimuth 90 / elevation 10, in the frame of the program) showed
H2CO edge-on and CH3X along their C3 axis.

Run (see tests/README.md): tests/run_tests.sh -k camera
"""
import os

import numpy as np
import pytest

from quchemreport.visualization_engine import camera
from tests.datafiles import DATA
from tests.helpers import read_steps

# Direction of the former first view (mlab.view(azimuth=90, elevation=10) in the program frame)
FORMER_VIEW = np.array([0.0, np.sin(np.radians(10)), np.cos(np.radians(10))])


def load(name):
    """Atomic numbers, coordinates (A) and bonds (as scanlog) of geometries/<name>.xyz."""
    from rdkit import Chem
    from quchemreport.utility_services.molecule_graph import bond_pairs
    with open(os.path.join(DATA, "geometries", name + ".xyz")) as fd:
        lines = fd.read().splitlines()
    atoms = [line.split() for line in lines[2:2 + int(lines[0])]]
    table = Chem.GetPeriodicTable()
    Z = [table.GetAtomicNumber(a[0]) for a in atoms]
    xyz = np.array([[float(v) for v in a[1:4]] for a in atoms])
    return Z, xyz, bond_pairs(Z, xyz)


def model_geometry(path, solver="gaussian"):
    m = read_steps(path, solver)[-1]
    xyz = np.asarray(m.results.geometry.elements_3D_coords).reshape(-1, 3)
    return m.molecule.atoms_Z, xyz, m.molecule.connectivity.atom_pairs


def angle(u, v):
    """Angle (degrees) between two lines."""
    return np.degrees(np.arccos(min(1.0, abs(np.dot(u, v)) / np.linalg.norm(u) / np.linalg.norm(v))))


def plane_normal(xyz):
    return camera.principal_axes(np.asarray(xyz))[1][:, 2]


# Second view only when one view is not enough (checked on contact sheets with Thomas)
EXPECTED_VIEWS = {
    "mol01": "facing rings",      # spiro-like
    "mol02": "facing rings",      # helicene
    "mol03": "rings edge-on",     # long adduct, rings edge-on in view 1
    "mol04": "facing rings",      # folded adduct
    "mol05": "facing rings",      # spiro
    "mol06": "globular",          # cobalt complex
    "mol07": None,                # planar
    "mol08": "facing rings",      # helicene
    "mol09": None,                # long, 102 atoms
    "mol10": "globular",          # cage, 408 atoms
}


@pytest.mark.parametrize("name", EXPECTED_VIEWS)
def test_second_view_only_when_needed(name):
    Z, xyz, pairs = load(name)
    views = camera.choose_views(xyz, Z, pairs)
    assert [v.reason for v in views[1:]] == ([EXPECTED_VIEWS[name]] if EXPECTED_VIEWS[name] else [])
    for v in views:
        assert np.isclose(np.linalg.norm(v.direction), 1) and np.isclose(np.linalg.norm(v.up), 1)
        assert abs(v.direction @ v.up) < 1e-6


def test_fewer_hidden_atoms_than_the_former_view():
    """Atom pairs hiding each other: 213 with the former view, 90 with the new one."""
    old = new = 0
    for name in EXPECTED_VIEWS:
        Z, xyz, pairs = load(name)
        view = camera.choose_views(xyz, Z, pairs)[0]
        old += camera.view_scores(xyz, [FORMER_VIEW])[1][0]
        new += camera.view_scores(xyz, [view.direction])[1][0]
    assert new < 0.5 * old


@pytest.mark.parametrize("name", ["mol02", "mol05", "mol07"])
def test_independent_of_the_program_orientation(name):
    """A rotated molecule gives the rotated view (within the 600-direction grid)."""
    from scipy.spatial.transform import Rotation
    Z, xyz, pairs = load(name)
    R = Rotation.from_euler("zyx", [37, -61, 112], degrees=True).as_matrix()
    first = camera.choose_views(xyz, Z, pairs)
    turned = camera.choose_views(xyz @ R.T, Z, pairs)
    assert [v.reason for v in first] == [v.reason for v in turned]
    assert angle(R @ first[0].direction, turned[0].direction) < 12


@pytest.mark.parametrize("path, solver", [
    ("gaussian/h2co/OPT.log", "gaussian"),       # formerly seen edge-on
    ("orca6/h2co/OPT_FREQ.out", "orca"),
    ("gaussian/h2o/H2O_OPT.log", "gaussian"),    # formerly seen edge-on
    ("orca6/nh3_plan/OPT_FREQ.out", "orca"),
])
def test_flat_molecules_seen_face_on(path, solver):
    Z, xyz, pairs = model_geometry(path, solver)
    views = camera.choose_views(xyz, Z, pairs)
    assert len(views) == 1
    assert angle(views[0].direction, plane_normal(xyz)) <= camera.PLANAR_TILT + 6


@pytest.mark.parametrize("path, solver, halogen", [
    ("gaussian/ch3i/OPT.log", "gaussian", 53),          # formerly seen along the C3 axis
    ("orca6/ch3i/OPT.out", "orca", 53),
    ("gaussian/ch3x_mep/SP_CH3Br.log", "gaussian", 35),
])
def test_symmetric_tops_seen_from_the_side(path, solver, halogen):
    Z, xyz, pairs = model_geometry(path, solver)
    view = camera.choose_views(xyz, Z, pairs)[0]
    axis = xyz[list(Z).index(halogen)] - xyz[list(Z).index(6)]
    assert angle(view.direction, axis) > 60


def test_heteroatoms_in_front():
    """Helicene mol02: its bithiophene is on the camera side."""
    Z, xyz, pairs = load("mol02")
    view = camera.choose_views(xyz, Z, pairs)[0]
    sulfur = np.asarray(Z) == 16
    assert (xyz[sulfur].mean(axis=0) - xyz.mean(axis=0)) @ view.direction > 0


def test_long_axis_horizontal():
    Z, xyz, pairs = load("mol07")
    view = camera.choose_views(xyz, Z, pairs)[0]
    long_axis = camera.principal_axes(xyz)[1][:, 0]
    assert angle(view.up, long_axis) > 75


@pytest.mark.parametrize("n", [1, 2, 3])
def test_tiny_molecules(n):
    xyz = [[0, 0, 0], [1.1, 0, 0], [2.2, 0, 0]][:n]
    views = camera.choose_views(xyz, [6, 8, 8][:n], [(0, 1), (1, 2)][:n - 1])
    assert len(views) == 1 and views[0].reason == "linear"


# ----------------------------------------------------------------- pictures (Mayavi, xvfb)

def _scene(name):
    from quchemreport.visualization_engine import visu_mayavi
    Z, xyz, pairs = load(name)
    pairs = pairs + [(j, i) for i, j in pairs]
    return visu_mayavi, xyz.ravel().tolist(), pairs, Z


def _drawn_box(path):
    """Bounding box (fractions of the picture) of the non-white pixels."""
    import matplotlib.image as mpimg
    img = mpimg.imread(path)[..., :3]
    ink = np.argwhere(img.min(axis=2) < 0.95)
    (y0, x0), (y1, x1) = ink.min(axis=0), ink.max(axis=0)
    h, w = img.shape[:2]
    return x0 / w, y0 / h, (x1 + 1) / w, (y1 + 1) / h


@pytest.mark.parametrize("name, second", [("mol07", False), ("mol06", True)])
def test_second_picture_only_when_needed(name, second, tmp_path):
    visu, coords, pairs, Z = _scene(name)
    figure = visu._init_scene(coords, pairs, Z)
    visu._save_views(figure, str(tmp_path / "img.png"), coords, pairs, Z, size=(200, 200))
    visu.mlab.close(all=True)
    assert (tmp_path / "img.png").is_file()
    assert (tmp_path / "img_cam2.png").is_file() == second


@pytest.mark.parametrize("name", ["mol07", "mol09"])
def test_picture_framing(name, tmp_path):
    """The molecule fills the picture without touching its border."""
    visu, coords, pairs, Z = _scene(name)
    visu.viz_topology(coords, pairs, Z, file_name=str(tmp_path / "t"), size=(200, 200))
    x0, y0, x1, y1 = _drawn_box(tmp_path / "t-TOPOLOGY.png")
    assert min(x0, y0, 1 - x1, 1 - y1) > 0.005
    assert max(x1 - x0, y1 - y0) > 0.75


@pytest.mark.parametrize("name", ["mol07", "mol08", "mol09"])
def test_forced_number_of_views(name):
    """output.camera_views: 1 drops view 2, 2 adds one at 60-90 degrees when auto gives none,
    and keeps the automatic view 2 when there is one."""
    Z, xyz, pairs = load(name)
    auto = camera.choose_views(xyz, Z, pairs)
    one = camera.choose_views(xyz, Z, pairs, n_views=1)
    two = camera.choose_views(xyz, Z, pairs, n_views=2)
    assert len(one) == 1 and np.allclose(one[0].direction, auto[0].direction)
    assert len(two) == 2 and np.allclose(two[0].direction, auto[0].direction)
    if len(auto) == 2:
        assert two[1].reason == auto[1].reason
    else:
        assert two[1].reason == "requested" and abs(two[0].direction @ two[1].direction) < 0.5
    with pytest.raises(ValueError):
        camera.choose_views(xyz, Z, pairs, n_views=3)


def test_forced_views_of_a_linear_molecule():
    """Two views of a linear molecule would be the same picture: one view."""
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]])
    assert [v.reason for v in camera.choose_views(xyz, [8, 8], [(0, 1)], n_views=2)] == ["linear"]


def test_camera_views_option():
    """output.camera_views: auto (any case), 1 or 2, also written as strings."""
    from quchemreport.config.config import Output
    for given, read in [("auto", "auto"), ("AUTO", "auto"), (1, 1), ("2", 2)]:
        assert Output.validate_camera_views(given) == read
    for wrong in (0, 3, True, "both"):
        with pytest.raises(ValueError):
            Output.validate_camera_views(wrong)
