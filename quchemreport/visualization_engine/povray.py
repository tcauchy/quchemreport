"""POV-Ray rendering of the 3D pictures (output.renderer: povray).

The Mayavi scene, already framed by visu_mayavi.fit_camera and saved by VTK, is exported with
VTK's POV-Ray exporter (tvtk scene.save_povray, vtkPOVExporter), which writes the meshes with
their normals and materials. Its header (camera, lights) is replaced, the colours are corrected,
and POV-Ray renders the picture under the same file name: the report uses the POV-Ray pictures
without any other change, and the VTK picture stays when POV-Ray fails.

Why (Thomas, 26/11/2026): a ray-traced look, like POV-Ray pictures of molecules, which VTK
cannot give in the off-screen rendering (Xvfb, Mesa llvmpipe): its ambient occlusion
(vtkSSAOPass) cut the lobes and drew stray lines, its shadow maps (vtkShadowMapPass) gave
striped and black faces, and its anti-aliasing (FXAA, MSAA) gave black or unchanged pictures
(19/11/2026). POV-Ray adds anti-aliased edges, soft shadows and ambient occlusion (radiosity).

What was measured on the acetone HOMO (26/11/2026, scripts in validation/couleurs_20261126):
  - the camera of fit_camera, written as is, gives the same framing as VTK to the pixel, so
    that the labels placed from the VTK picture (atom numbers, MEP extrema) stay in place;
  - the exporter writes numbers with the locale of the process: "1,000000" in a French
    session, which POV-Ray cannot read; the export is done with LC_NUMERIC = C;
  - Mayavi's lights follow the camera; the exporter writes them as fixed lights aimed at the
    origin. They are rewritten here from their transformed (world) positions after rendering;
  - vtkProperty.GetColor, used by the exporter, averages the diffuse and specular colours:
    with the white highlights (visu_mayavi.SPECULAR = 0.5) the Tyrian purple #66003C was
    written as (0.60, 0.33, 0.49). The diffuse colour is recovered from the finish (_unmix);
  - VTK lights the colours as they are (no gamma); with POV-Ray's physical default
    (assumed_gamma 1, sRGB colours) the pictures were much paler. assumed_gamma 2.2 with the
    same light intensities gives the VTK brightness (mean purple (83, 1, 46) against (87, 4, 53)
    for VTK; mean lightness of the molecule 122 for both);
  - presets compared: lights only, + soft shadow of the key light (area light), + radiosity.
    Chosen: soft shadow and radiosity (brightness 0.3, lights x 0.8: same lightness as VTK, 126
    against 122). 1200 px, 16 threads: 4.5 s per picture (2.7 s without radiosity, 1.2 s for VTK);
  - the two other Mayavi lights cast no shadow: their shadows doubled the dark areas.
"""

import locale
import os
import re
import shutil
import subprocess
import sys

import numpy as np

# Rendering settings (see the module docstring for the measurements)
ASSUMED_GAMMA = 2.2          # colours lit as VTK does
LIGHT_SCALE = 0.8            # Mayavi light intensities x LIGHT_SCALE (radiosity adds light)
RADIOSITY_BRIGHTNESS = 0.3   # white sky gathered by radiosity: ambient occlusion
AREA_LIGHT_SIZE = 40.0       # side (bohr, at 1000 bohr) of the key light: soft shadows
AREA_LIGHT_SAMPLES = 6
ANTIALIAS = ["+A0.1", "+AM2", "+R3"]

_executable = None
_threads = 4
_warned = set()


def find_executable(path=None):
    """POV-Ray executable: the given path, else ``povray`` in the PATH, else next to the Python
    interpreter (conda environment). None when not found."""
    candidates = [path] if path else [shutil.which("povray"),
                                      os.path.join(os.path.dirname(sys.executable), "povray")]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def configure(executable, threads=4):
    """Executable used by render (None: POV-Ray disabled) and number of threads."""
    global _executable, _threads
    _executable = executable
    _threads = max(1, int(threads or 1))


def enabled():
    return _executable is not None


def _warn_once(message):
    if message not in _warned:
        _warned.add(message)
        print("Warning (POV-Ray): " + message + " The VTK pictures are kept.")


def _vec(v):
    return "<%.6f, %.6f, %.6f>" % tuple(v)


def _unmix(match):
    """Diffuse colour of a texture written by vtkPOVExporter: GetColor returns
    (Ka Ca + Kd Cd + Ks Cs) / (Ka + Kd + Ks), with Ca = Cd (mlab color=) and Cs white for the
    glossy actors (visu_mayavi._shine)."""
    rgb = np.array([float(x) for x in match.group("rgb").split(",")])
    ka, kd, ks = (float(match.group(k)) for k in ("ka", "kd", "ks"))
    if ks > 0 and ka + kd > 0:
        rgb = np.clip((rgb * (ka + kd + ks) - ks) / (ka + kd), 0.0, 1.0)
    finish = "ambient 0 diffuse 1" + (" phong %.3f phong_size %s" % (ks, match.group("ksp")) if ks > 0 else "")
    return ("pigment {\n\t\t\tcolor rgbf <%.6f, %.6f, %.6f, %s>\n\t\t}\n\t\tfinish {\n\t\t\t%s\n\t\t}"
            % (*rgb, match.group("f"), finish))


_TEXTURE = re.compile(
    r"pigment \{\s*color rgbf <(?P<rgb>[^,]+,[^,]+, \S+) (?P<f>\S+)>\s*\}\s*finish \{\s*"
    r"ambient (?P<ka>\S+)\s+diffuse (?P<kd>\S+)(?:\s+phong (?P<ks>\S+)\s+phong_size (?P<ksp>\S+))?\s*\}")


def _unmix_safe(match):
    if match.group("ks") is None:
        rgb = match.group("rgb")
        return ("pigment {\n\t\t\tcolor rgbf <%s, %s>\n\t\t}\n\t\tfinish {\n\t\t\tambient 0 diffuse 1\n\t\t}"
                % (rgb, match.group("f")))
    return _unmix(match)


def export(figure, pov_path, background=(1.0, 1.0, 1.0), alpha=False, shadows=True):
    """Writes the scene of the figure (rendered once, camera set) as a POV-Ray file; alpha:
    transparent background (with +UA, POV-Ray keeps an opaque background unless it transmits:
    the first MEP pictures were washed out by the opaque white of the skeleton overlay);
    shadows: soft shadow of the key light (False for the MEP: the shadows of the extrema
    markers made dark spots on the colour scale of the surface)."""
    from tvtk.api import tvtk
    renderer = tvtk.to_vtk(figure.scene.renderer)
    figure.scene.render()
    old = locale.setlocale(locale.LC_NUMERIC)
    locale.setlocale(locale.LC_NUMERIC, "C")
    try:
        figure.scene.save_povray(pov_path)
    finally:
        locale.setlocale(locale.LC_NUMERIC, old)
    with open(pov_path) as f:
        body = f.read()
    if "mesh2" not in body:
        raise RuntimeError("nothing exported")
    body = _TEXTURE.sub(_unmix_safe, body[body.index("mesh2"):])
    cam = renderer.GetActiveCamera()
    focal = np.array(cam.GetFocalPoint())
    head = [
        "#version 3.7;",
        "global_settings { assumed_gamma %.1f ambient_light rgb <1, 1, 1>" % ASSUMED_GAMMA,
        "  radiosity { pretrace_start 0.08 pretrace_end 0.005 count 200 nearest_count 10 error_bound 0.4"
        " recursion_limit 1 low_error_factor 0.5 gray_threshold 0 brightness %.2f always_sample off } }"
        % RADIOSITY_BRIGHTNESS,
        # the background is also the light gathered by radiosity (no sky_sphere: it would be
        # drawn instead of the transparent background of the skeleton overlay, +UA)
        "background { color rgbt <%.6f, %.6f, %.6f, %d> }" % (*background, 1 if alpha else 0),
        # finish of the per-vertex colours (texture_list, the MEP surface), which the exporter
        # writes without one: POV-Ray's default (ambient 0.1, diffuse 0.6) paled them
        "#default { finish { ambient 0 diffuse 1 } }",
        # VTK's view angle is vertical, POV-Ray's horizontal: equal for the square pictures;
        # right <-1, 0, 0>: VTK is right-handed, POV-Ray left-handed
        "camera { perspective location %s sky %s up <0, 1, 0> right <-1, 0, 0> angle %.6f look_at %s }"
        % (_vec(cam.GetPosition()), _vec(cam.GetViewUp()), cam.GetViewAngle(), _vec(focal)),
    ]
    lights = renderer.GetLights()
    lights.InitTraversal()
    key = True
    for _ in range(lights.GetNumberOfItems()):
        light = lights.GetNextItem()
        if not light.GetSwitch() or light.GetIntensity() == 0:
            continue
        d = np.array(light.GetTransformedPosition()) - np.array(light.GetTransformedFocalPoint())
        d /= np.linalg.norm(d)
        k = light.GetIntensity() * LIGHT_SCALE
        if key and shadows:     # first light of Mayavi's light kit (intensity 1): soft shadows
            u = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
            u /= np.linalg.norm(u)
            v = np.cross(d, u)
            v /= np.linalg.norm(v)
            extra = "area_light %s, %s, %d, %d adaptive 1 jitter circular" % (
                _vec(AREA_LIGHT_SIZE * u), _vec(AREA_LIGHT_SIZE * v), AREA_LIGHT_SAMPLES, AREA_LIGHT_SAMPLES)
            key = False
        else:
            extra = "parallel point_at %s shadowless" % _vec(focal)
            key = False
        head.append("light_source { %s color rgb <%.3f, %.3f, %.3f> %s }" % (_vec(focal + 1000 * d), k, k, k, extra))
    with open(pov_path, "w") as f:
        f.write("\n".join(head) + "\n\n" + body)


def render(pov_path, png_path, size, alpha=False):
    """Renders a POV-Ray file to png_path (width, height); alpha: transparent background."""
    cmd = [_executable, "+I" + pov_path, "+O" + png_path, "+W%d" % size[0], "+H%d" % size[1],
           *ANTIALIAS, "-D", "+FN", "-GA", "+WT%d" % _threads] + (["+UA"] if alpha else [])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(png_path):
        raise RuntimeError(result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no picture")


def save(figure, png_path, size, alpha=False, background=(1.0, 1.0, 1.0), shadows=True):
    """Replaces png_path (the VTK picture) by the POV-Ray rendering of the figure. False, with
    one warning, when POV-Ray fails (the VTK picture is then kept)."""
    if not enabled():
        return False
    root = os.path.splitext(png_path)[0]
    pov, out = root + ".pov", root + "-povray.png"
    try:
        export(figure, pov, background, alpha, shadows)
        render(pov, out, size, alpha)
        os.replace(out, png_path)
        return True
    except Exception as error:  # the VTK picture is a valid fallback
        _warn_once(f"{os.path.basename(png_path)}: {error}.")
        return False
    finally:
        for f in (pov, out):
            if os.path.exists(f):
                os.remove(f)
