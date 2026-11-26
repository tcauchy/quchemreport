"""POV-Ray rendering (output.renderer: povray) and specular reflections (output.specular)."""

import locale
import os
import re

import numpy as np
import pytest

from quchemreport.visualization_engine import povray, visu_mayavi

# POV-Ray of the test environment (conda-forge povray 3.7.0.10), or one in the PATH
POVRAY = povray.find_executable(os.environ.get("QUCHEMREPORT_POVRAY")) or \
    povray.find_executable("/home/mcadmin/povray_env_20261126/bin/povray")
needs_povray = pytest.mark.skipif(POVRAY is None, reason="POV-Ray not installed")

# H2O, Angstrom
GEOM = [0.0, 0.0, 0.117, 0.0, 0.757, -0.467, 0.0, -0.757, -0.467]
PAIRS, Z = [(0, 1), (0, 2), (1, 0), (2, 0)], [8, 1, 1]


@pytest.fixture(autouse=True)
def reset():
    yield
    povray.configure(None)
    visu_mayavi.set_specular(0.0)


def test_unmix_recovers_the_diffuse_colour():
    """vtkPOVExporter writes (Kd Cd + Ks white) / (Kd + Ks): the Tyrian purple with the
    highlights 0.5 comes back as (0.6, 0.333, 0.490) and must be restored."""
    text = ("pigment {\n\t\t\tcolor rgbf <0.600000, 0.333333, 0.490196 0.000000>\n\t\t}\n\t\tfinish {\n"
            "\t\t\tambient 0.000000  diffuse 1.000000  phong 0.500000  phong_size 40.000000  \n\t\t}")
    out = povray._TEXTURE.sub(povray._unmix_safe, text)
    rgb = [float(x) for x in re.search(r"rgbf <([^>]*)>", out).group(1).split(",")[:3]]
    assert np.allclose(rgb, (0.4, 0.0, 0.235), atol=2e-3)
    assert "phong 0.500" in out
    matte = text.replace("  phong 0.500000  phong_size 40.000000", "").replace("0.600000, 0.333333, 0.490196", "0.4, 0.0, 0.235")
    out = povray._TEXTURE.sub(povray._unmix_safe, matte)
    assert "rgbf <0.4, 0.0, 0.235, 0.000000>" in out and "phong" not in out


def test_renderer_option():
    from quchemreport.config.config import Output
    assert Output.validate_renderer("POVRAY") == "povray"
    with pytest.raises(ValueError):
        Output.validate_renderer("blender")
    assert visu_mayavi.set_renderer("vtk") == "vtk" and not povray.enabled()
    # asked for but missing: VTK pictures, no error
    assert visu_mayavi.set_renderer("povray", "/nonexistent/povray") == "vtk"
    assert not povray.enabled()


def test_specular_zero_leaves_the_properties():
    """output.specular 0: the actors keep VTK's defaults (former pictures pixel for pixel)."""
    from mayavi import mlab
    visu_mayavi.set_specular(0.0)
    fig = visu_mayavi._init_scene(GEOM, PAIRS, Z)
    from tvtk.api import tvtk
    ren = tvtk.to_vtk(fig.scene.renderer)
    acts = ren.GetActors(); acts.InitTraversal()
    specs = [acts.GetNextActor().GetProperty().GetSpecular() for _ in range(acts.GetNumberOfItems())]
    mlab.close(fig)
    visu_mayavi.set_specular(0.5)
    fig = visu_mayavi._init_scene(GEOM, PAIRS, Z)
    ren = tvtk.to_vtk(fig.scene.renderer)
    acts = ren.GetActors(); acts.InitTraversal()
    shiny = [acts.GetNextActor().GetProperty().GetSpecular() for _ in range(acts.GetNumberOfItems())]
    mlab.close(fig)
    assert all(s == 0 for s in specs) and all(s == 0.5 for s in shiny)


def test_export_in_a_french_locale(tmp_path):
    """The exported file uses dots, the camera of fit_camera and the world lights, whatever
    the locale of the session (a French session wrote "1,000000")."""
    from mayavi import mlab
    try:
        old = locale.setlocale(locale.LC_NUMERIC)
        locale.setlocale(locale.LC_NUMERIC, "fr_FR.UTF-8")
    except locale.Error:
        pytest.skip("no French locale")
    try:
        fig = visu_mayavi._init_scene(GEOM, PAIRS, Z)
        visu_mayavi.fit_camera(fig, visu_mayavi.views_of(GEOM, PAIRS, Z)[0])
        mlab.savefig(str(tmp_path / "w.png"), figure=fig, size=(200, 200))
        pov = tmp_path / "w.pov"
        povray.export(fig, str(pov))
        cam = fig.scene.camera
        text = pov.read_text()
        mlab.close(fig)
    finally:
        locale.setlocale(locale.LC_NUMERIC, old)
    assert not re.search(r"\d,\d{6}", text)
    assert "camera { perspective location <%.6f, %.6f, %.6f>" % tuple(cam.position) in text
    assert text.count("light_source") == 3 and text.count("area_light") == 1


@needs_povray
def test_povray_picture_same_framing(tmp_path):
    """POV-Ray replaces the VTK picture, with the molecule at the same place (bounding box
    of the drawn pixels within 2 % of the side)."""
    from PIL import Image
    from mayavi import mlab
    povray.configure(POVRAY, threads=2)
    fig = visu_mayavi._init_scene(GEOM, PAIRS, Z)
    visu_mayavi.fit_camera(fig, visu_mayavi.views_of(GEOM, PAIRS, Z)[0])
    vtk_png, pov_png = str(tmp_path / "vtk.png"), str(tmp_path / "pov.png")
    mlab.savefig(vtk_png, figure=fig, size=(300, 300))
    assert visu_mayavi._savefig(fig, pov_png, (300, 300)) is True
    mlab.close(fig)
    boxes = []
    for f in (vtk_png, pov_png):
        a = np.asarray(Image.open(f).convert("RGB"), int)
        ys, xs = np.nonzero(a.min(axis=2) < 230)
        boxes.append(np.array([xs.min(), xs.max(), ys.min(), ys.max()]))
    assert np.abs(boxes[0] - boxes[1]).max() <= 6
    assert not os.path.exists(str(tmp_path / "pov.pov"))


@needs_povray
def test_transparent_background(tmp_path):
    """alpha=True (skeleton overlay of the MEP): transparent background, opaque molecule.
    Without "rgbt ... 1" POV-Ray kept the white background opaque and the overlay paled the
    whole MEP picture."""
    from PIL import Image
    from mayavi import mlab
    povray.configure(POVRAY, threads=2)
    fig = visu_mayavi._init_scene(GEOM, PAIRS, Z)
    visu_mayavi.fit_camera(fig, visu_mayavi.views_of(GEOM, PAIRS, Z)[0])
    png = str(tmp_path / "skeleton.png")
    assert visu_mayavi._savefig(fig, png, (200, 200), alpha=True)
    mlab.close(fig)
    a = np.asarray(Image.open(png))
    assert a.shape[-1] == 4 and a[0, 0, 3] == 0 and a[..., 3].max() == 255
