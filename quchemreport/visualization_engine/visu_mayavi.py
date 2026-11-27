
## -*- encoding: utf-8 -*-

"""3D pictures of the molecule and of the grids (Mayavi/VTK, off-screen rendering).

Every picture starts from the same scene (_init_scene: atoms as spheres, bonds as tubes, in bohr),
adds iso-surfaces or a coloured surface, and is saved from the camera views chosen from the
geometry (visualization_engine.camera, _save_views): one picture, plus ``_cam2`` for the
molecules that need a second view. The iso-values are chosen so that the surfaces enclose a
fixed share of the integrated values (CalcCutOff). Grids are drawn on every k-th point
(output.image_grid_reduction, _reduced_grid); the numerical results keep the full grid.

The choices of the MEP pictures (opaque surface with the skeleton drawn over it, fixed colour
scale, turbo colour table) and of the camera are explained where they are made
(viz_Potential, _save_views_through, fit_camera).
"""

from mayavi import mlab
import numpy as np
import functools
import inspect
from sys import exit
import os
from math import sqrt, acos, atan2

from quchemreport.utility_services.log_data import LogData

#from utils import a0_to_A
from quchemreport.utility_services import units
from quchemreport.data_enrichment import mep_surface
from quchemreport.utility_services.naming import spin_letter

## Import mayavi defaults parameters
from quchemreport.visualization_engine import povray
from quchemreport.utility_services.parameters import img_width, img_height, surf_opacity, colors, surface_colors
from quchemreport.visualization_engine import camera
width = img_width
height = img_height


def set_image_size(pixels):
    """Side in pixels of the square images saved by the viz_* functions (config output.image_size)."""
    global width, height
    width = height = int(pixels)


# Specular reflections of the atoms, bonds and iso-surfaces (output.specular, set_specular):
# intensity 0-1 of white highlights, Phong interpolation, exponent SPECULAR_POWER. Why (Thomas,
# 26/11/2026): a POV-Ray-like glossy look; with diffuse light only, the dark surfaces (Tyrian
# purple) looked flat and nearly black. Tested on the acetone HOMO under Xvfb/llvmpipe: about
# the same rendering time; the VTK ambient occlusion (vtkSSAOPass) cut the lobes and drew
# stray lines, the shadow maps (vtkShadowMapPass) gave striped and black faces: not used.
# 0 keeps the former pictures pixel for pixel (the properties are then left untouched).
# The MEP surface is not made glossy: a highlight would hide its colour scale.
SPECULAR = 0.0
SPECULAR_POWER = 40


def set_specular(intensity):
    """Intensity 0-1 of the specular reflections (config output.specular)."""
    global SPECULAR
    SPECULAR = min(1.0, max(0.0, float(intensity)))


def _shine(module):
    """Specular reflections (SPECULAR) on the actor of a Mayavi module or glyph."""
    if SPECULAR > 0:
        prop = module.actor.property
        prop.interpolation = "phong"
        prop.specular = SPECULAR
        prop.specular_power = SPECULAR_POWER
        prop.specular_color = (1.0, 1.0, 1.0)
    return module


def _size(size):
    """Size (width, height) in pixels of a saved picture: the given one, or the default."""
    return size if size is not None else (width, height)


def set_renderer(renderer="vtk", executable=None, threads=4):
    """Renderer of the 3D pictures (config output.renderer): "vtk", or "povray" (the VTK
    picture is then replaced by its POV-Ray rendering, see povray.py). Returns the renderer
    used: "vtk" with a warning when POV-Ray is asked for but not found."""
    if renderer != "povray":
        povray.configure(None)
        return "vtk"
    found = povray.find_executable(executable)
    if found is None:
        print("Warning: output.renderer is povray but no POV-Ray executable was found "
              f"({executable or 'povray in the PATH or the conda environment'}); the pictures are rendered by VTK.")
        povray.configure(None)
        return "vtk"
    povray.configure(found, threads)
    return "povray"


def _savefig(figure, name, size=None, alpha=False, shadows=True):
    """Saves the figure as name with VTK, then, when output.renderer is povray, replaces it by
    the POV-Ray rendering of the same scene and camera (same framing to the pixel, so that the
    labels projected with the VTK camera stay in place). The VTK picture is always made first:
    the render sets the lights exported to POV-Ray, and it stays if POV-Ray fails.
    Returns True when the picture is the POV-Ray one (alpha: transparent background)."""
    mlab.savefig(name, figure=figure, size=_size(size))
    return povray.save(figure, name, _size(size), alpha=alpha, shadows=shadows)


## Images are drawn on every k-th point of the computation grid (config
## output.image_grid_reduction). The values are those already computed: no
## interpolation, the numerical results keep the full grid.
## "smooth": MO, MEP (and NTO, reduced when computed, see
## generate_visualization.output_nto_images); "sensitive": EDD, Oif, Fukui, Fdual.
image_grid_reduction = {"smooth": 1, "sensitive": 1}


def set_image_grid_reduction(smooth, sensitive):
    """Set the grid reductions of the pictures (config output.image_grid_reduction and
    image_grid_reduction_sensitive): 1 keeps every point, k every k-th point.
    """
    image_grid_reduction["smooth"] = max(1, int(smooth))
    image_grid_reduction["sensitive"] = max(1, int(sensitive))


def _reduce_values(values, s):
    """Every k-th point (slices s) of voxel data: one array, or a list of arrays."""
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        return [np.asarray(v)[s] for v in values]
    values = np.asarray(values)
    return values[(slice(None),) + s] if values.ndim == 4 else values[s]


def _reduced_grid(category):
    """Decorator: take every k-th grid point of X, Y, Z and of the voxel data
    (arguments data, r_data, V_data) before drawing, k being the factor of the
    image category. Keyword reduction=k overrides it (1 keeps the full grid)."""
    return functools.partial(_reduced_grid_wrap, category)


def _reduced_grid_wrap(category, func):
    """Implementation of the _reduced_grid decorator for func."""
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, reduction=None, **kwargs):
        """Call func on the reduced grid (keyword reduction=k overrides the category factor)."""
        k = image_grid_reduction[category] if reduction is None else max(1, int(reduction))
        bound = signature.bind(*args, **kwargs)
        a = bound.arguments
        if k > 1 and all(a.get(n) is not None for n in ("X", "Y", "Z")):
            s = (slice(None, None, k),) * 3
            for n in ("X", "Y", "Z"):
                a[n] = a[n][s]
            for n in ("data", "r_data", "V_data"):
                if n in a:
                    a[n] = _reduce_values(a[n], s)
        return func(*bound.args, **bound.kwargs)
    return wrapper

## No screen
mlab.options.offscreen = True
#mlab.options.offscreen = False

## Initialize visualization details common to all jobs
with open('%s/../utility_services/Atoms_properties.csv' % os.path.dirname(__file__), "r") as f:
    tab = [line.split() for line in f]


# Radius of the bonds (bohr); the atoms are spheres of diameter 0.5 bohr.
# Why tubes of constant radius (24/11/2026): the bonds were cylinder glyphs of mlab.quiver3d,
# which scales the whole glyph, radius included, with the length of the bond vector. Long bonds
# were thick: in CH3I the C-I bond (2.1 A) was twice as thick as the C-H bonds and wider than
# the iodine sphere. The tubes keep one VTK pipeline per colour, as the glyphs did.
BOND_RADIUS = 0.12
ATOM_DIAMETER = 0.5
# Thinner skeleton of the pictures of dipole arrows (_init_scene(thin=True)).
# Why (27/11/2026, Thomas Cauchy): with the usual skeleton, an arrow lying along a bond (the
# transition dipole of S5 of acetone, along C=O) or a short arrow was hidden in the bonds. Tested
# on acetone and HCFClBr: the molecule stays readable with half the bond radius.
THIN_BOND_RADIUS = 0.06
THIN_ATOM_DIAMETER = 0.30


def _init_scene(elements_3D_coords, atom_pairs, atoms_Z, thin=False):
    """New off-screen scene with the molecule: atoms as spheres coloured by element, bonds as
    half tubes coloured like their atom (BOND_RADIUS). Coordinates are converted to bohr,
    the unit of the grids. thin: thinner skeleton (THIN_BOND_RADIUS, THIN_ATOM_DIAMETER), for
    the pictures of dipole arrows.

    Parameters
    ----------
    elements_3D_coords : list of float
        Coordinates (Angstrom), flat or (n, 3).
    atom_pairs : list of (int, int)
        Bonds, both directions (LogData connectivity).
    atoms_Z : list of int

    Returns
    -------
    mayavi.core.scene.Scene
        The figure, with rendering disabled until it is saved.

    Raises
    ------
    ValueError
        A geometry argument is missing.
    """
      
    if (
      elements_3D_coords == None
      or atom_pairs == None
      or atoms_Z == None
    ):
      raise ValueError(
        "Missing required data to initialize scene: "
        "elements_3D_coords, atom_pairs, and atoms_Z must all be provided."
      )
    
    figure = mlab.figure(bgcolor=(1,1,1), fgcolor=(0,0,0))
    figure.scene.disable_render= True
    geom = np.array(elements_3D_coords).reshape((-1,3))/units.a0_to_A
    atom_nums = np.asarray(atoms_Z)
    color_of = lambda z: tuple(float(x)/255.0 for x in tab[z][3:6])
    ## Draw atoms and bonds. Each mlab call builds its own VTK pipeline (~50 ms), so the
    ## atoms are drawn with one call per element and the half-bonds with one call per
    ## colour instead of one call per atom/bond (same glyphs, identical images).
    ## Requires >=MayaVi-4.6.0
    for z in dict.fromkeys(atom_nums.tolist()):
        p = geom[atom_nums == z]
        _shine(mlab.points3d(p[:, 0], p[:, 1], p[:, 2],
                             figure=figure, mode='sphere', color=color_of(z), resolution=15,
                             scale_factor=THIN_ATOM_DIAMETER if thin else ATOM_DIAMETER))
    ## Each bond (i, j) and (j, i) is a half tube from atom i to the middle of the bond, coloured
    ## like atom i. Tubes have the same radius for all bonds (cylinder glyphs scaled with the bond
    ## length made the long bonds, e.g. C-I, thicker than the atoms).
    half_bonds = {}
    for i, j in atom_pairs:
        half_bonds.setdefault(color_of(atom_nums[i]), []).append((i, j))
    for color, pairs in half_bonds.items():
        p1 = geom[[i for i, _ in pairs]]
        p2 = (p1 + geom[[j for _, j in pairs]]) / 2
        points = np.vstack([p1, p2])
        n = len(p1)
        src = mlab.pipeline.scalar_scatter(points[:, 0], points[:, 1], points[:, 2], figure=figure)
        src.mlab_source.dataset.lines = np.column_stack([np.arange(n), np.arange(n) + n])
        src.update()
        tube = mlab.pipeline.tube(src, tube_radius=THIN_BOND_RADIUS if thin else BOND_RADIUS, tube_sides=15)
        _shine(mlab.pipeline.surface(tube, color=color, figure=figure))

    return figure

# Number of views of the 3D pictures: "auto" (camera.choose_views decides), 1 or 2
# (output.camera_views, set_camera_views)
CAMERA_VIEWS = "auto"


def set_camera_views(n_views):
    """Number of views of the 3D pictures: "auto", 1 or 2 (camera.choose_views)."""
    global CAMERA_VIEWS
    if n_views not in ("auto", 1, 2):
        raise ValueError(f"camera views must be 'auto', 1 or 2, not {n_views!r}")
    CAMERA_VIEWS = n_views


@functools.lru_cache(maxsize=16)
def _cached_views(coords, atoms_Z, atom_pairs, n_views):
    """camera.choose_views on hashable arguments, cached (views_of)."""
    return camera.choose_views(np.array(coords), atoms_Z, atom_pairs, n_views)


# Geometry of the reference log (set_reference_geometry): the views of every other geometry of
# the same molecule are those of the reference, turned into the frame of that geometry.
# Why (27/11/2026, Thomas Cauchy): the views were chosen on each geometry. The S1 and T1
# geometries of acetone are pyramidal at the carbonyl carbon and in another Gaussian standard
# orientation, so their EDD pictures were seen from another side than the topology, the
# orbitals and the vertical EDD of the same report.
REFERENCE_GEOMETRY = None


def set_reference_geometry(elements_3D_coords, atom_pairs, atoms_Z):
    """Geometry whose views are used for all the pictures of the same molecule (None: each
    geometry gets its own views)."""
    global REFERENCE_GEOMETRY
    if elements_3D_coords is None or atoms_Z is None:
        REFERENCE_GEOMETRY = None
        return
    REFERENCE_GEOMETRY = (np.asarray(elements_3D_coords, dtype=float).reshape(-1, 3),
                          atom_pairs, [int(z) for z in atoms_Z])


def kabsch_rotation(source, target):
    """Rotation R (det +1) that best superimposes source on target, both (n, 3) and centred on
    their means: target - mean ~ (source - mean) @ R.T."""
    p = source - source.mean(axis=0)
    q = target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def _own_views(elements_3D_coords, atom_pairs, atoms_Z):
    coords = tuple(np.round(np.asarray(elements_3D_coords, dtype=float).ravel(), 6))
    pairs = tuple(tuple(int(k) for k in p) for p in atom_pairs) if atom_pairs is not None else None
    return _cached_views(coords, tuple(int(z) for z in atoms_Z), pairs, CAMERA_VIEWS)


def views_of(elements_3D_coords, atom_pairs, atoms_Z):
    """Camera views of the molecule (visualization_engine.camera), computed once per geometry.
    For a geometry of the reference molecule (same atoms in the same order), the views of the
    reference geometry turned by the rotation that superimposes the reference on it (Kabsch):
    an excited-state or redox geometry is then seen from the same side as the reference."""
    xyz = np.asarray(elements_3D_coords, dtype=float).reshape(-1, 3)
    if REFERENCE_GEOMETRY is not None:
        ref_xyz, ref_pairs, ref_Z = REFERENCE_GEOMETRY
        if ref_xyz.shape == xyz.shape and ref_Z == [int(z) for z in atoms_Z]:
            rot = kabsch_rotation(ref_xyz, xyz)
            return [camera.View(rot @ v.direction, rot @ v.up, v.reason)
                    for v in _own_views(ref_xyz, ref_pairs, ref_Z)]
    return _own_views(xyz, atom_pairs, atoms_Z)


# Margin around the drawn objects in the pictures (1 = touching the border). 1.08 left the
# labels of the topology picture touching the border: their orientation (vtkFollower) is only
# fixed at rendering time, so their bounds are approximate when the camera is placed.
FRAME_MARGIN = 1.12


def fit_camera(figure, view, margin=FRAME_MARGIN):
    """Puts the camera of the figure along view.direction, with view.up vertical, and frames all
    the objects drawn (atoms, iso-surfaces, markers, labels).

    Why the VTK camera is set directly (24/11/2026): the former code called
    mlab.view(azimuth, elevation) with fixed angles. While the rendering of the scene is disabled
    (_init_scene), mlab.view() does not put the camera in the requested direction (measured: the
    direction of projection was (-0.32, -0.27, -0.91) instead of (0.00, -0.45, -0.89)), and
    mlab's automatic zoom fits a bounding sphere, which leaves elongated molecules small. The
    distance is computed here with the perspective, so that the farthest point from the axis
    touches the margin, whatever the depth of the scene."""
    from tvtk.api import tvtk
    from vtk.util.numpy_support import vtk_to_numpy
    renderer = tvtk.to_vtk(figure.scene.renderer)
    points = []
    actors = renderer.GetActors()
    actors.InitTraversal()
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        mapper = actor.GetMapper()
        if mapper is None:
            continue
        mapper.Update()
        if actor.IsA("vtkFollower"):
            # Labels (mlab.text3d): their points are in a local frame, use their world bounds
            b = np.array(actor.GetBounds()).reshape(3, 2)
            points.append(np.array(np.meshgrid(*b, indexing="ij")).reshape(3, -1).T)
            continue
        data = mapper.GetInputAsDataSet()
        if data is not None and data.GetNumberOfPoints():
            points.append(vtk_to_numpy(data.GetPoints().GetData()))
    d = np.asarray(view.direction, dtype=float)
    u = np.asarray(view.up, dtype=float)
    u = (u - (u @ d) * d) / np.linalg.norm(u - (u @ d) * d)
    r = np.cross(u, d)
    if points:
        p = np.concatenate(points)
        x, y, z = p @ r, p @ u, p @ d
    else:
        x = y = z = np.zeros(1)
    xc, yc, zc = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2, (z.min() + z.max()) / 2
    centre = r * xc + u * yc + d * zc
    cam = figure.scene.camera
    # Perspective: the camera is at distance D from the centre along d; a point at height h
    # from the axis and z towards the camera is inside the frame when h <= (D - z) tan(angle/2)
    # / margin. The smallest D that keeps every point inside:
    t = np.tan(np.radians(cam.view_angle) / 2)
    h = np.maximum(np.abs(x - xc), np.abs(y - yc))
    distance = max(np.max((z - zc) + h * margin / t), 1.0)
    cam.focal_point = tuple(centre)
    cam.position = tuple(centre + distance * d)
    cam.view_up = tuple(u)
    renderer.ResetCameraClippingRange()
    return figure


# Labels written on the saved pictures (atom numbers, MEP extrema), in 2D.
# Why (26/11/2026): the former 3D labels (mlab.text3d) had a size that changed with the depth,
# were cut at the border (O2: "O2" half out of the picture, their bounds being taken before they
# turn to the camera), were hidden by the MEP surface or drawn over the atoms. They are now
# written on the picture at the projection of their point, with a fixed size and a white halo,
# on the free side among 8 positions around the point, inside the picture.
LABEL_FONT = 0.032          # height of the labels, fraction of the picture width
LABEL_HALO = 0.15           # width of the white halo, fraction of the label height


def project(points, camera, width, height):
    """Pixel coordinates (x right, y down) of points (n, 3) seen by the VTK camera, and their
    distance in front of it (perspective projection, view_angle = vertical angle)."""
    pos = np.asarray(camera.position, dtype=float)
    f = np.asarray(camera.focal_point, dtype=float) - pos
    f /= np.linalg.norm(f)
    up = np.asarray(camera.view_up, dtype=float)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    v = np.asarray(points, dtype=float).reshape(-1, 3) - pos
    z = v @ f
    t = np.tan(np.radians(camera.view_angle) / 2)
    x_ndc = (v @ r) / (z * t) * height / width
    y_ndc = (v @ u) / (z * t)
    return np.column_stack([(x_ndc + 1) / 2 * width, (1 - y_ndc) / 2 * height]), z


def write_labels(png, labels, avoid=()):
    """Write labels on the picture png: [(x, y, text, (dx, dy), r)], (x, y) the pixel of the
    point, (dx, dy) the preferred side (y up), r the radius (pixels) of the drawn point, which
    the label must clear. avoid: [(x, y, radius)] discs (pixels) the labels should not cover
    (atoms, bonds, markers). Each label takes the free place nearest to its point, preferring
    its side; overlapping another label costs more than covering a bond."""
    if not labels:
        return
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.font_manager as fm
    image = Image.open(png).convert("RGB")
    W, H = image.size
    size = max(10, int(LABEL_FONT * W))
    font = ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    halo = max(1, int(LABEL_HALO * size))
    draw = ImageDraw.Draw(image)
    taken = []
    avoid = np.asarray(avoid, dtype=float).reshape(-1, 3)

    def cost(box):
        x0, y0, x1, y1 = box
        if x0 < 2 or y0 < 2 or x1 > W - 2 or y1 > H - 2:
            return np.inf
        c = 3.0 * sum(1 for b in taken if not (x1 < b[0] or b[2] < x0 or y1 < b[1] or b[3] < y0))
        if len(avoid):
            cx = np.clip(avoid[:, 0], x0, x1)
            cy = np.clip(avoid[:, 1], y0, y1)
            c += np.count_nonzero((cx - avoid[:, 0]) ** 2 + (cy - avoid[:, 1]) ** 2 < avoid[:, 2] ** 2)
        return c

    angles = np.radians(np.arange(0, 360, 30))
    for x, y, text, (dx, dy), r in labels:
        l, t_, r_, b_ = draw.textbbox((0, 0), text, font=font, stroke_width=halo)
        w, h = r_ - l, b_ - t_
        prefer = np.arctan2(dy, dx)
        order = sorted(angles, key=lambda a: abs(np.angle(np.exp(1j * (a - prefer)))))
        best = None
        for ring in range(5):
            for rank, a in enumerate(order):
                # distance that clears the drawn point, plus rings further out
                reach = r + 0.15 * size + abs(np.cos(a)) * w / 2 + abs(np.sin(a)) * h / 2 + ring * 0.5 * size
                cx, cy = x + np.cos(a) * reach, y - np.sin(a) * reach
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                # small penalties: far from the point, far from the preferred side
                score = cost(box) + 0.2 * ring + 0.02 * rank
                if best is None or score < best[0]:
                    best = (score, box)
            if best[0] < 0.2 * (ring + 1):
                break
        box = best[1]
        taken.append(box)
        draw.text((box[0] - l, box[1] - t_), text, font=font, fill=(0, 0, 0),
                  stroke_width=halo, stroke_fill=(255, 255, 255))
    image.save(png)


def _point_labels(figure, size, points, texts, radius, centre=None, bonds=None, bond_radius=None):
    """Labels for write_labels and discs to avoid, for points (bohr) drawn as spheres of radius
    (bohr) and seen by the camera of the figure: each label prefers the side away from centre
    (default: the middle of the points). bonds: index pairs, whose tubes are also avoided."""
    W, H = _size(size)
    cam = figure.scene.camera
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    px, _ = project(points, cam, W, H)
    up = np.asarray(cam.view_up, dtype=float)
    edge, _ = project(points + radius * up, cam, W, H)
    radii = np.linalg.norm(edge - px, axis=1)
    c = px.mean(axis=0) if centre is None else project(np.atleast_2d(centre), cam, W, H)[0][0]
    labels = []
    for (x, y), text, r in zip(px, texts, radii):
        d = np.array([x - c[0], c[1] - y])
        side = tuple(d / np.linalg.norm(d)) if np.linalg.norm(d) > 1 else (0.7, 0.7)
        labels.append((x, y, text, side, r))
    avoid = [(x, y, r) for (x, y), r in zip(px, radii)]
    if bonds:
        tube = bond_radius or 0.12
        # discs along each bond, about one tube radius apart
        for i, j in {tuple(sorted(b)) for b in bonds}:
            n = max(2, int(np.linalg.norm(points[i] - points[j]) / tube))
            seg = points[i] + np.linspace(0, 1, n)[:, None] * (points[j] - points[i])
            sp, _ = project(seg, cam, W, H)
            e, _ = project(seg + tube * up, cam, W, H)
            avoid += [(x, y, rr) for (x, y), rr in zip(sp, np.linalg.norm(e - sp, axis=1))]
    return labels, avoid


# Opacity of the skeleton drawn over the opaque MEP surface (see _save_views_through).
# 0.5 and 0.7 were compared on H2CO and CH3I; 0.6 keeps the colours of the surface readable.
SKELETON_THROUGH = 0.6


def _save_views_through(figure, file_name, elements_3D_coords, atom_pairs, atoms_Z, size=None,
                        alpha=SKELETON_THROUGH, extrema=None):
    """As _save_views, for an opaque surface hiding the molecule: the skeleton (atoms and bonds),
    rendered alone with the same camera, is drawn over the picture with the opacity alpha, so
    that the positions of the atoms stay visible through the surface ("X-ray" view).

    Why not a translucent surface (tests of 24/11/2026, MEP of H2CO, turbo colours):
      - the off-screen rendering (xvfb, software OpenGL) washes out translucent surfaces: mean
        colour saturation of the surface 0.64 when opaque, but only 0.16-0.20 with an opacity of
        0.75, where about 0.48 is expected;
      - neither the colour map (the former pictures, with Mayavi's default map, were as pale),
        nor the lighting (ambient, diffuse, no lighting), nor back-face culling change this;
      - with depth peeling (the VTK method for translucency) the surface disappears entirely.
    Compositing the two pictures gives the expected translucency over the atoms, with the true
    colours elsewhere. If the pictures are ever rendered on a GPU or on screen, translucency may
    be tried again, comparing the colour saturation of the surface with the opaque picture.

    The skeleton is rendered on a magenta background: on a white one, the white hydrogens were
    taken for the background and drawn as crescents. Its atoms are all drawn over the surface,
    including those behind it (the price of the X-ray view)."""
    views = views_of(elements_3D_coords, atom_pairs, atoms_Z)
    root, ext = os.path.splitext(file_name)
    skeleton = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)
    # Magenta background: the white hydrogens must not be taken for the background
    skeleton.scene.background = (1.0, 0.0, 1.0)
    for k, view in enumerate(views[:2]):
        name = file_name if k == 0 else f"{root}_cam2{ext}"
        fit_camera(figure, view)
        # no cast shadow: those of the extrema markers looked like dark MEP values
        _savefig(figure, name, size, shadows=False)
        _composite_overlay(figure, skeleton, name, size, alpha)
        labels, avoid = _extrema_labels(figure, size, extrema, name)
        write_labels(name, labels, avoid)
    mlab.close(skeleton)
    return figure


def _composite_overlay(figure, overlay, name, size, alpha):
    """Draws the scene overlay over the saved picture name of figure, with the camera of figure
    and opacity alpha: what overlay holds stays visible through the surfaces of figure.

    The overlay is rendered on a magenta background with VTK (on a white one, white hydrogens
    were taken for the background), or with a transparent background with POV-Ray (+UA: the
    anti-aliased edges of a colour key would leave a magenta fringe)."""
    import matplotlib.image as mpimg
    root, ext = os.path.splitext(name)
    cam, cam2 = figure.scene.camera, overlay.scene.camera
    cam2.position, cam2.focal_point = cam.position, cam.focal_point
    cam2.view_up, cam2.view_angle = cam.view_up, cam.view_angle
    overlay.scene.renderer.reset_camera_clipping_range()
    overlay_file = f"{root}_overlay{ext}"
    through_povray = _savefig(overlay, overlay_file, size, alpha=True, shadows=False)
    picture = mpimg.imread(name)[..., :3]
    overlay_rgba = mpimg.imread(overlay_file)
    drawn_rgb = overlay_rgba[..., :3]
    os.remove(overlay_file)
    if through_povray and overlay_rgba.shape[-1] == 4 and drawn_rgb.shape == picture.shape:
        drawn = alpha * overlay_rgba[..., 3:4]
        mpimg.imsave(name, np.clip(drawn * drawn_rgb + (1 - drawn) * picture, 0, 1))
    elif drawn_rgb.shape == picture.shape:
        background = (drawn_rgb[..., 0] > 0.9) & (drawn_rgb[..., 1] < 0.1) & (drawn_rgb[..., 2] > 0.9)
        drawn = (~background)[..., None]
        mpimg.imsave(name, np.clip(np.where(drawn, alpha * drawn_rgb + (1 - alpha) * picture, picture), 0, 1))


def _save_views(figure, file_name, elements_3D_coords, atom_pairs, atoms_Z, size=None, views=None):
    """Saves the first view as file_name and, when the molecule needs one (camera.choose_views),
    the second view as file_name with "_cam2" before the extension. views: the views to use
    instead of those of the molecule (views_of)."""
    if views is None:
        views = views_of(elements_3D_coords, atom_pairs, atoms_Z)
    root, ext = os.path.splitext(file_name)
    for k, view in enumerate(views[:2]):
        fit_camera(figure, view)
        _savefig(figure, file_name if k == 0 else f"{root}_cam2{ext}", size)
    return figure

## Set the Iso contour value for mayavi from a percent
# Choose the % (between 0 to 100) of the positive values to show in picture.
def CalcCutOff(data,IsoContourPercent=30):
    """Positive and negative iso-values that enclose IsoContourPercent % of the integrated
    positive (and negative) values of the data.

    The values of each sign are sorted and accumulated; the iso-value is the one above which
    the largest values hold the requested share. A sign absent from the data (e.g. a 1s
    orbital, the sigma_g orbital of H2) gives None: no surface is drawn for it.

    Parameters
    ----------
    data : numpy.ndarray
    IsoContourPercent : float, optional

    Returns
    -------
    cut_p : float or None
    cut_n : float or None
        Negative iso-value.
    """
    #data are the function values for each voxels. Transforma as a list, sort and select the positive values
    datar = np.asarray(data).ravel()

    def cutoff(values):
        """Iso-value of one sign (absolute values), or None for no values."""
        if values.size == 0:
            return None
        datasort = np.sort(values)
        #cumulative sum of function values, normalize
        cumdatasort = np.cumsum(datasort)
        cumdatasortnorm = cumdatasort/cumdatasort[-1]*100.
        return datasort[cumdatasortnorm>=(100.-IsoContourPercent)][0]

    cut_p = cutoff(datar[datar>0])
    cut_n = cutoff(np.abs(datar[datar<0]))
    return cut_p, (-cut_n if cut_n is not None else None)


def _iso_surfaces(source, figure, cutoffs_and_colors):
    """Iso-surfaces of source at each (cutoff, color), skipping the None cutoffs (see CalcCutOff)."""
    return [_shine(mlab.pipeline.iso_surface(source, figure=figure, contours=[cut], color=color, opacity=surf_opacity))
            for cut, color in cutoffs_and_colors if cut is not None]

## Visualize

# Margin of the molecule picture (FRAME_MARGIN elsewhere): room for the atom labels
TOPOLOGY_MARGIN = 1.3
# Radius of the atom spheres of _init_scene (bohr; scale_factor 0.5 is the diameter)
ATOM_RADIUS = 0.25


def viz_topology(elements_3D_coords, atom_pairs, atoms_Z, file_name=None, size=None):
    """Picture of the molecule with the symbol and number of each atom (``-TOPOLOGY.png``),
    from the first view of the other pictures, so that the numbers can be followed.

    Parameters
    ----------
    elements_3D_coords, atom_pairs, atoms_Z :
        Geometry (Angstrom), bonds and atomic numbers (see _init_scene).
    file_name : str, optional
        Path prefix of the picture.
    size : tuple of int, optional
    """
    geom = np.array(elements_3D_coords).reshape((-1,3))/units.a0_to_A
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)
    texts = [tab[z][1] + str(i + 1) for i, z in enumerate(atoms_Z)]
    # Second picture when the other pictures have one (camera.choose_views, output.camera_views),
    # so that the atom numbers can be followed on both
    for k, view in enumerate(views_of(elements_3D_coords, atom_pairs, atoms_Z)[:2]):
        # wider margin: the labels are written outside the atoms
        fit_camera(figure, view, margin=TOPOLOGY_MARGIN)
        if file_name is None:
            continue
        name = "{}-TOPOLOGY{}.png".format(file_name, "" if k == 0 else "_cam2")
        _savefig(figure, name, size)
        labels, avoid = _point_labels(figure, size, geom, texts, ATOM_RADIUS,
                                      bonds=atom_pairs, bond_radius=BOND_RADIUS)
        write_labels(name, labels, avoid)
    mlab.close()

@_reduced_grid("smooth")
def viz_MO(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, file_name=None, labels=None, size=None, kind="MO"):
    """Pictures of molecular orbitals (``-MO-<label>.png``), positive and negative lobes at
    the iso-values enclosing 20 % of each sign (CalcCutOff).

    Parameters
    ----------
    data : numpy.ndarray
        One grid per orbital (calc_orb.MO).
    X, Y, Z : numpy.ndarray
        Grid (bohr).
    elements_3D_coords, atom_pairs, atoms_Z :
        Geometry, bonds and atomic numbers (see _init_scene).
    file_name : str, optional
        Path prefix of the pictures.
    labels : list of str, optional
        Labels of the orbitals in the file names (index when omitted).
    size : tuple of int, optional
    kind : str, optional
        Colours of the lobes: surface_colors["MO"] (default) or ["NTO"] (output.colors).
    """
    
    # Skip if missing data
    if (
      data is None
      or X is None
      or Y is None
      or Z is None
      or elements_3D_coords is None
      or atom_pairs is None
      or atoms_Z is None
    ):
      return

    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)
    for i, series in enumerate(data):
        Cutoffp, Cutoffn = CalcCutOff(series,IsoContourPercent=20) 
        #print(Cutoffp, Cutoffn)
        MO_data = mlab.pipeline.scalar_field(X, Y, Z, series, figure=figure)
        surfaces = _iso_surfaces(MO_data, figure, [(Cutoffp, surface_colors[kind]["positive"]),
                                                   (Cutoffn, surface_colors[kind]["negative"])])
        if file_name is not None:
            _save_views(figure, "{}-MO-{}.png".format(file_name, labels[i] if labels is not None else i), elements_3D_coords, atom_pairs, atoms_Z, size)
        for module in [MO_data] + surfaces:
            module.remove()
    mlab.close()

@_reduced_grid("sensitive")
def viz_EDD(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, et_sym, file_name=None, labels=None, size=None):
    """Pictures of electron density differences of excited states
    (``-EDD-<S|T><n>.png``): electron gain and loss at 30 % (CalcCutOff).

    Parameters
    ----------
    data : list of numpy.ndarray
        Density differences (calc_orb.TD).
    X, Y, Z : numpy.ndarray
    elements_3D_coords, atom_pairs, atoms_Z :
        See _init_scene.
    et_sym : str
        Symmetry string of the states (S or T in the file names).
    file_name : str, optional
    labels : list, optional
        State numbers in the file names.
    size : tuple of int, optional
    """

    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)
    for i, series in enumerate(data):
        Cutoffp, Cutoffn = CalcCutOff(series,IsoContourPercent=30) 
        D_data = mlab.pipeline.scalar_field(X, Y, Z, series, figure=figure)
        surfaces = _iso_surfaces(D_data, figure, [(Cutoffp, surface_colors["EDD"]["positive"]),
                                                  (Cutoffn, surface_colors["EDD"]["negative"])])
        if file_name is not None:
            # EDD-S<n> (singlet), EDD-T<n> (triplet), EDD-<n> (other, e.g. unrestricted)
            name = "{}-EDD-{}{}".format(file_name, spin_letter(et_sym), labels[i] if labels is not None else i)
            _save_views(figure, name + ".png", elements_3D_coords, atom_pairs, atoms_Z, size)
        for module in [D_data] + surfaces:
            module.remove()

    mlab.close()

# Length of the dipole arrows: 1 Debye = 1 bohr (0.53 A), for the ground state dipole (GSDIP)
# and the transition dipoles (ELDIP, given in e.bohr: 1 e.bohr = 2.54 D = 2.54 bohr), so that
# their lengths can be compared on the same picture.
BOHR_PER_DEBYE = 1.0
EBOHR_TO_DEBYE = 1.0 / units.ea0_to_D
# Transition dipole pictures (viz_TDIP): 1 e.bohr (2.54 D) = 4 bohr. With the scale of the
# dipole pictures (2.54 bohr per e.bohr), the arrows of HCFClBr (0.06 to 1 e.bohr) were a
# few pixels long (27/11/2026).
TRANSITION_DIPOLE_BOHR_PER_EBOHR = 4.0


# Arrows of constant thickness (bohr): only their length depends on the vector.
# Why (27/11/2026, Thomas Cauchy): the arrow glyph of mlab.quiver3d is scaled as a whole with the
# length of the vector (shaft radius 3 % of the length, tip radius 10 %, tip 35 % of the length):
# a 4 bohr arrow had the thickness of a bond, a 1 bohr arrow was a thread hidden in the skeleton,
# and the magnetic dipole of S1 of HCFClBr (4 bohr, almost along the line of sight) was a wide cone
# hiding the electric one. Same defect as the bond cylinders before 24/11/2026 (BOND_RADIUS).
# A minimum length was rejected (Thomas): the lengths must stay proportional to the vectors; the
# values are written on the picture instead (write_legend).
ARROW_SHAFT_RADIUS = 0.10
ARROW_TIP_RADIUS = 0.25
ARROW_TIP_LENGTH = 0.6


def _arrow(figure, start, vector, color, label=None):
    """Arrow from start along vector (bohr), of constant thickness (ARROW_*), with its label
    just beyond the tip. An arrow shorter than its tip is a cone scaled down with its length;
    a zero vector draws nothing."""
    start, vector = np.asarray(start, dtype=float), np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length < 1e-6:
        return
    glyph = mlab.quiver3d(*start, *vector, figure=figure, mode='arrow', scale_factor=1.0, color=color)
    # The glyph is scaled by the length: its dimensions are given as fractions of the length
    source = glyph.glyph.glyph_source.glyph_source
    shrink = min(1.0, length / ARROW_TIP_LENGTH)
    source.tip_length = min(1.0, ARROW_TIP_LENGTH / length)
    source.tip_radius = ARROW_TIP_RADIUS * shrink / length
    source.shaft_radius = ARROW_SHAFT_RADIUS * shrink / length
    source.tip_resolution = source.shaft_resolution = 24
    if label:
        tip = start + vector * (1.0 + 0.3 / max(np.linalg.norm(vector), 1e-6))
        mlab.text3d(*tip, label, color=color, scale=0.3, figure=figure)


@_reduced_grid("sensitive")
def viz_TDEN(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, et_sym, file_name=None, labels=None,
             size=None, vectors=None):
    """Pictures of the charge transition density of excited states (``-TDEN-<S|T><n>.png``):
    its positive and negative regions at 30 % (CalcCutOff). The transition dipole is drawn
    beside it (viz_TDIP).

    Parameters
    ----------
    data : list of numpy.ndarray
        Charge transition densities (compute.transition_dipole).
    X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, et_sym, file_name, labels, size :
        As viz_EDD.
    vectors : list, optional
        Arrows of the dipole picture drawn beside it (viz_TDIP): the density is seen from the
        same views (dipole_views).

    Why (27/11/2026, Thomas Cauchy): replaces the overlap pictures (Oif, sum C^2 phi_i phi_a),
    which lost the signs of the amplitudes (see calc_orb.TD). The transition density is what
    gives the transition dipole, hence the oscillator strength: a forbidden transition shows
    positive and negative regions whose dipoles cancel. The arrow was first drawn on the same
    picture, over the lobes: unreadable, hence the separate picture of viz_TDIP.
    """
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)
    for i, series in enumerate(data):
        Cutoffp, Cutoffn = CalcCutOff(series, IsoContourPercent=30)
        T_data = mlab.pipeline.scalar_field(X, Y, Z, series, figure=figure)
        surfaces = _iso_surfaces(T_data, figure, [(Cutoffp, surface_colors["transition_density"]["positive"]),
                                                  (Cutoffn, surface_colors["transition_density"]["negative"])])
        if file_name is not None:
            name = "{}-TDEN-{}{}".format(file_name, spin_letter(et_sym), labels[i] if labels is not None else i)
            _save_views(figure, name + ".png", elements_3D_coords, atom_pairs, atoms_Z, size,
                        views=dipole_views(views_of(elements_3D_coords, atom_pairs, atoms_Z), vectors))
        for module in [T_data] + surfaces:
            module.remove()
    mlab.close()


# Radius (bohr) of the sphere drawn at the centre of the transition density of a forbidden
# transition (no transition dipole) in viz_TDIP
FORBIDDEN_SPHERE_RADIUS = 0.45
# An arrow within this angle (degrees) of the line of sight cannot be read on the picture: it is
# marked in the legend (towards or away from the viewer) and a second view is added (dipole_views).
# Why (27/11/2026, Thomas Cauchy): the magnetic dipole of S1 of HCFClBr (18 degrees from the line
# of sight) and the electric dipole of S4 of acetone (17 degrees) were seen end-on.
LINE_OF_SIGHT_ANGLE = 30.0


def along_line_of_sight(vector, view):
    """+1 when vector points towards the camera of view within LINE_OF_SIGHT_ANGLE, -1 when it
    points away from it, 0 otherwise (or for a zero vector)."""
    v = np.asarray(vector, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 0
    c = float(v @ np.asarray(view.direction, dtype=float)) / n
    if abs(c) < np.cos(np.radians(LINE_OF_SIGHT_ANGLE)):
        return 0
    return 1 if c > 0 else -1


# Smallest angle (degrees) between an arrow and the line of sight in the second view added by
# dipole_views; the view is turned from the first one by the smallest of SECOND_VIEW_TURNS that
# reaches it. Why not 90 degrees (27/11/2026): the out-of-plane dipole of S4 of acetone was then
# seen along the plane of the molecule, which was reduced to a line; at 45 degrees the molecule is
# recognized and the arrow keeps 71 % of its length.
SECOND_VIEW_ANGLE = 45.0
SECOND_VIEW_TURNS = (45.0, 60.0, 75.0, 90.0)


def dipole_views(views, vectors):
    """Views of the pictures of dipole arrows (vectors: arrows, None or zero ones ignored).
    The views of the molecule when its first view shows every arrow, or when its second view
    does; otherwise the first view and a second one turned about the vertical or the horizontal
    of the first view by the smallest angle (SECOND_VIEW_TURNS) that puts every arrow at least
    SECOND_VIEW_ANGLE from the line of sight (the most turned one when none does)."""
    vectors = [np.asarray(v, dtype=float) / np.linalg.norm(v) for v in (vectors or []) if v is not None
               and np.linalg.norm(v) > 1e-12]
    if not vectors or not views:
        return views
    hidden = lambda view: any(along_line_of_sight(v, view) for v in vectors)
    if not hidden(views[0]) or (len(views) > 1 and not hidden(views[1])):
        return views
    d = np.asarray(views[0].direction, dtype=float)
    up = np.asarray(views[0].up, dtype=float)
    up = (up - (up @ d) * d) / np.linalg.norm(up - (up @ d) * d)
    right = np.cross(up, d)

    def worst(view):            # smallest angle (degrees) between an arrow and the line of sight
        return min(np.degrees(np.arccos(min(1.0, abs(v @ view.direction)))) for v in vectors)

    best = None
    for turn in SECOND_VIEW_TURNS:
        candidates = [camera.View(camera._rotate(d, axis, sign * turn), camera._rotate(up, axis, sign * turn),
                                  "dipole") for axis in (up, right) for sign in (1, -1)]
        best = max(candidates, key=worst)
        if worst(best) >= SECOND_VIEW_ANGLE:
            break
    return [views[0], best]


# Symbols of the legend for an arrow along the line of sight (physics convention: tip / tail)
TOWARDS_VIEWER, AWAY_FROM_VIEWER = "\u2299", "\u2297"


# Height of the legend (write_legend), fraction of the picture width. Larger than the atom labels
# (LABEL_FONT): the dipole pictures are printed four per row (about 40 mm wide), where 0.032 gave
# 3.6 pt letters.
LEGEND_FONT = 0.06


def write_legend(png, lines):
    """Writes lines [(text, (r, g, b) in 0-1)] in the upper left corner of the picture png, one
    per line, in the font of the labels (write_labels) with a white halo."""
    if not lines:
        return
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.font_manager as fm
    image = Image.open(png).convert("RGB")
    W = image.size[0]
    size = max(10, int(LEGEND_FONT * W))
    font = ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    halo = max(1, int(LABEL_HALO * size))
    draw = ImageDraw.Draw(image)
    x = y = int(0.02 * W)
    for text, color in lines:
        fill = tuple(int(round(255 * c)) for c in color)
        draw.text((x, y), text, font=font, fill=fill, stroke_width=halo, stroke_fill=(255, 255, 255))
        y += int(1.3 * size)
    image.save(png)


def dipole_legend(dipole, magnetic, view):
    """Legend of a picture of viz_TDIP: |mu| (e.bohr) and |m| (Bohr magnetons), with the
    symbol of an arrow along the line of sight of view (TOWARDS_VIEWER, AWAY_FROM_VIEWER)."""
    def mark(vector):
        side = along_line_of_sight(vector, view)
        return {1: " " + TOWARDS_VIEWER, -1: " " + AWAY_FROM_VIEWER}.get(side, "")
    if dipole is None:
        lines = [("\u03bc = 0 (forbidden)", colors["ELDIP"])]
    else:
        lines = [(f"\u03bc = {np.linalg.norm(dipole):.3f} e\u00b7a\u2080" + mark(dipole), colors["ELDIP"])]
    if magnetic is not None:
        lines.append((f"m = {np.linalg.norm(magnetic):.3f} \u03bcB" + mark(magnetic), colors["MAGDIP"]))
    return lines


def viz_TDIP(elements_3D_coords, atom_pairs, atoms_Z, et_sym, dipole, centre, magnetic=None, file_name=None,
             label=None, size=None):
    """Picture of the transition dipoles of one excited state (``-TDIP-<S|T><n>.png``), drawn
    beside its transition density (viz_TDEN, same views: dipole_views), on a thin skeleton:
    the electric transition dipole as a blue arrow centred on the density
    (1 e.bohr = TRANSITION_DIPOLE_BOHR_PER_EBOHR bohr), or a blue sphere for a forbidden
    transition; for a chiral molecule, the magnetic transition dipole as a magenta arrow, with
    the sign that makes the rotatory strength positive for an acute angle between the two
    arrows (compute.transition_dipole). The values are written in the upper left corner
    (dipole_legend), with the arrows along the line of sight marked.

    The magnetic dipole (Bohr magnetons) is drawn with the same number of bohr per unit as the
    electric one (e.bohr): in common units (cgs, or SI with m/c), 1 Bohr magneton = alpha/2 e.bohr,
    so the magnetic arrow is magnified 2/alpha = 274 times relative to the electric one. The
    picture is then quantitative: the dissymmetry factor is g = 4 R / D = (4 / 274) times the
    projection of the magenta arrow on the blue one divided by the length of the blue one.

    Parameters
    ----------
    elements_3D_coords, atom_pairs, atoms_Z, et_sym, file_name, size :
        As viz_EDD.
    dipole : list or None
        Electric transition dipole (e.bohr), None for a forbidden transition.
    centre : array
        Centre of the transition density (bohr).
    magnetic : list, optional
        Magnetic transition dipole (Bohr magnetons, as printed by Gaussian in "a.u."), drawn
        when given.
    label : int, optional
        State number in the file name.
    """
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z, thin=True)
    centre = np.asarray(centre, dtype=float)
    if dipole is None:
        mlab.points3d(*centre, scale_factor=2 * FORBIDDEN_SPHERE_RADIUS, resolution=24,
                      color=colors["ELDIP"], figure=figure)
    else:
        vector = np.asarray(dipole, dtype=float) * TRANSITION_DIPOLE_BOHR_PER_EBOHR
        _arrow(figure, centre - vector / 2, vector, colors["ELDIP"])
    if magnetic is not None:
        vector = np.asarray(magnetic, dtype=float) * TRANSITION_DIPOLE_BOHR_PER_EBOHR
        _arrow(figure, centre - vector / 2, vector, colors["MAGDIP"])
    if file_name is not None:
        name = "{}-TDIP-{}{}.png".format(file_name, spin_letter(et_sym), label)
        views = dipole_views(views_of(elements_3D_coords, atom_pairs, atoms_Z), [dipole, magnetic])
        root, ext = os.path.splitext(name)
        for k, view in enumerate(views[:2]):
            view_name = name if k == 0 else f"{root}_cam2{ext}"
            fit_camera(figure, view)
            _savefig(figure, view_name, size)
            write_legend(view_name, dipole_legend(dipole, magnetic, view))
    mlab.close(figure)


def viz_dip(data, elements_3D_coords, atom_pairs, atoms_Z, et_sym, file_name=None, labels=None, size=None):
    """Picture of dipole arrows (``-DIP-<S|T><n>.png``): ground state dipole (GSDIP, from the
    origin of the program), transition dipole (ELDIP, from the centre of the transition
    density) with the same scale (BOHR_PER_DEBYE), and the charge-transfer arrow (CTDIP, from
    the barycentre of the density lost to that of the density gained: its length is d_CT).

    Parameters
    ----------
    data : dict
        Arrow per type: (end, start) coordinates (bohr); GSDIP in Debye from its origin,
        ELDIP in e.bohr from the centre, CTDIP in bohr.
    elements_3D_coords, atom_pairs, atoms_Z :
        See _init_scene.
    et_sym : str
    file_name : str, optional
    labels : list, optional
        State number (first element).
    size : tuple of int, optional

    Until 27/11/2026: an overlap arrow (OVDIP, see calc_orb.TD) was drawn too, the transition
    dipole started at the origin of the program and was scaled differently from the ground state
    dipole, and the labels were written at end x scale instead of at the tip of the arrows.
    """
    lengths = {"GSDIP": BOHR_PER_DEBYE, "ELDIP": EBOHR_TO_DEBYE * BOHR_PER_DEBYE, "CTDIP": 1.0}
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z, thin=True)
    for dip_type, (end, start) in data.items():
        if dip_type not in lengths:
            continue
        start = np.asarray(start, dtype=float)
        vector = (np.asarray(end, dtype=float) - start) * lengths[dip_type]
        _arrow(figure, start, vector, colors[dip_type], dip_type)

    if file_name is not None:
        name = "{}-DIP-{}{}".format(file_name, spin_letter(et_sym), labels[0])
        _save_views(figure, name + ".png", elements_3D_coords, atom_pairs, atoms_Z, size)

    mlab.clf(figure=figure)
    mlab.close()


# Turbo colour map (Google, 2019), 256 entries (r, g, b in 0-1).
TURBO = np.array([[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]])
# Colour table given to VTK: it must be uint8 with 4 components (r, g, b, alpha in 0-255).
# Until 24/11/2026 the turbo map was given as floats with 3 components: VTK refused it
# ("Number of components in given table (3) is incorrect, it should have 4") and silently kept
# Mayavi's default colour map, so the MEP pictures never used the turbo map.
TURBO_TABLE = np.column_stack([np.round(TURBO * 255), np.full(len(TURBO), 255)]).astype(np.uint8)
# Default half-width of the fixed colour scale of the MEP pictures (hartree/e), used when no
# range is given (neutral molecule); the reports use mep_surface.color_range(charge).
MEP_COLOR_RANGE = mep_surface.COLOR_RANGE_NEUTRAL


def mep_colorbar(file_name, color_range, width=6.0):
    """Horizontal colour bar of the MEP pictures, in kcal/mol (red negative, blue positive)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    kcal = color_range * units.Eh_to_Kcal_mol
    fig, ax = plt.subplots(figsize=(width, 0.85))
    ax.imshow(np.linspace(-kcal, kcal, 512)[None, :], aspect="auto", cmap=ListedColormap(TURBO[::-1]),
              extent=(-kcal, kcal, 0, 1), vmin=-kcal, vmax=kcal)
    ax.set_yticks([])
    ticks = np.linspace(-kcal, kcal, 5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:+.0f}" if abs(t) > 1e-9 else "0" for t in ticks])
    ax.set_xlabel("MEP on the surface (kcal/mol); values beyond the scale take the end colours", fontsize=8)
    fig.tight_layout()
    fig.savefig(file_name, dpi=200)
    plt.close(fig)


# Markers of the MEP surface extrema: offset outwards from the surface so that the opaque
# surface does not hide them (bohr), sphere diameter and label size (bohr). 0.35 bohr was
# enough with the former translucent surface; with the opaque one, drawn on a coarser grid
# (output.image_grid_reduction), markers and numbers were half hidden.
EXTREMA_OFFSET = 0.7
EXTREMA_MARKER_SIZE = 0.45


def _marker_position(ext):
    """Position (bohr) of the marker of an extremum, outside the surface."""
    return np.asarray(ext["position"], dtype=float) + EXTREMA_OFFSET * np.asarray(ext["direction"], dtype=float)


# A marker is numbered when it is seen: the picture shows its colour (black, white or grey,
# spread of the RGB components below this), not the saturated colours of the surface.
# Why: a test on the orientation of the surface (outward direction towards the camera) left
# the markers standing out at the edge of the surface without a number (HCFClBr, 26/11/2026).
EXTREMA_SEEN_SPREAD = 0.2


def _extrema_labels(figure, size, extrema, picture):
    """Labels (write_labels) of the markers seen in the picture (file): the numbers of the
    markers hidden by the surface are not written."""
    if not extrema:
        return [], []
    import matplotlib.image as mpimg
    image = mpimg.imread(picture)[..., :3]
    H, W = image.shape[:2]
    points = np.array([_marker_position(e) for e in extrema])
    px, _ = project(points, figure.scene.camera, W, H)
    seen = []
    for number, (x, y) in enumerate(px, start=1):
        i, j = int(round(y)), int(round(x))
        if not (1 <= i < H - 1 and 1 <= j < W - 1):
            continue
        rgb = image[i - 1:i + 2, j - 1:j + 2].reshape(-1, 3).mean(axis=0)
        if rgb.max() - rgb.min() < EXTREMA_SEEN_SPREAD:
            seen.append(number - 1)
    if not seen:
        return [], []
    # equivalent extrema (mep_surface) share the number of their group, that of the table
    texts = [str(extrema[k].get("group", k + 1)) for k in seen]
    return _point_labels(figure, size, points[seen], texts, EXTREMA_MARKER_SIZE / 2)


def _draw_extrema(figure, extrema):
    """Markers of the MEP extrema: white spheres for the minima, black spheres for the maxima,
    one per member of a group of equivalent extrema; their numbers (those of the groups in the
    report table) are written by _extrema_labels."""
    if not extrema:
        return
    for number, ext in enumerate(extrema, start=1):
        p = _marker_position(ext)
        color = (1.0, 1.0, 1.0) if ext["kind"] == "min" else (0.0, 0.0, 0.0)
        mlab.points3d(p[0], p[1], p[2], figure=figure, mode="sphere", color=color,
                      resolution=12, scale_factor=EXTREMA_MARKER_SIZE)


@_reduced_grid("smooth")
def viz_Potential(r_data, V_data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, file_name=None, size=None,
                  isovalue=0.002, extrema=None, color_range=None):
    """Picture of the molecular electrostatic potential mapped on a density isosurface
    (``-MEP.png``) and its colour bar (``-MEP_colorbar.png``).

    Parameters
    ----------
    r_data : numpy.ndarray
        Electron density on the grid (calc_orb.Potential).
    V_data : numpy.ndarray
        MEP on the grid (hartree/e).
    X, Y, Z : numpy.ndarray
        Grid (bohr).
    elements_3D_coords, atom_pairs, atoms_Z :
        See _init_scene.
    file_name : str, optional
        Path prefix of the pictures.
    size : tuple of int, optional
    isovalue : float, optional
        Density of the isosurface (e-/bohr3); the reports use
        mep_surface.MEP_SURFACE_ISOVALUE (0.001).
    extrema : list of dict, optional
        Local extrema of the MEP on that surface (mep_surface), drawn as numbered markers:
        white spheres for the minima, black ones for the maxima.
    color_range : float, optional
        Half-width of the symmetric colour scale (hartree/e); MEP_COLOR_RANGE by default.
    """
    
    # Skip if missing data
    if (
      r_data is None
      or V_data is None
      or X is None
      or Y is None
      or Z is None
      or elements_3D_coords is None
      or atom_pairs is None
      or atoms_Z is None
    ):
      return

    # Choices of 24/11/2026 (with Thomas Cauchy):
    #  - one picture with a fixed, symmetric colour scale, the same for all the compounds of the
    #    same charge (mep_surface.color_range), so that pictures can be compared; the former
    #    second picture with an automatic scale was dropped, since the exact values of the
    #    extrema are now given by the numbered markers and the table of the report;
    #  - red for the negative MEP, green around zero, blue for the positive MEP (usual
    #    convention): the turbo map is applied to -V;
    #  - surface 0.001 e-/bohr3 (Bader's van der Waals surface, as Murray-Politzer and Multiwfn);
    #  - opaque surface with the skeleton composited over it (_save_views_through: translucent
    #    surfaces are washed out by the off-screen rendering);
    #  - smooth normals: without them the triangles of the opaque surface show.
    if color_range is None:
        color_range = MEP_COLOR_RANGE
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z)
    src = mlab.pipeline.scalar_field(X, Y, Z, r_data, figure=figure)
    src.image_data.point_data.add_array((-V_data).T.ravel())
    src.image_data.point_data.get_array(1).name = "potential"
    src.update()
    density = mlab.pipeline.set_active_attribute(src, figure=figure, point_scalars="scalar")
    contour = mlab.pipeline.contour(density, figure=figure)
    contour.filter.contours = [isovalue]
    potential = mlab.pipeline.set_active_attribute(contour, figure=figure, point_scalars="potential")
    # Smooth shading of the opaque surface (otherwise its triangles show)
    smooth = mlab.pipeline.poly_data_normals(potential, figure=figure)
    smooth.filter.splitting = False
    surf = mlab.pipeline.surface(smooth, figure=figure, opacity=1.0, vmin=-color_range, vmax=color_range)
    surf.module_manager.scalar_lut_manager.lut.table = TURBO_TABLE
    _draw_extrema(figure, extrema)
    if file_name is not None:
        _save_views_through(figure, "{}-MEP.png".format(file_name), elements_3D_coords, atom_pairs, atoms_Z, size,
                            extrema=extrema)
        mep_colorbar("{}-MEP_colorbar.png".format(file_name), color_range)
    mlab.close(figure)

@_reduced_grid("sensitive")
def viz_Fukui(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, file_name=None, labels=None, size=None):
    """Picture of a Fukui function (``-fukui-<label>.png``) at 30 % (CalcCutOff).

    Parameters
    ----------
    data : numpy.ndarray
        Fukui function on the grid (calc_orb.Fukui).
    X, Y, Z : numpy.ndarray
    elements_3D_coords, atom_pairs, atoms_Z :
        See _init_scene.
    file_name : str, optional
    labels : str, optional
        ``SP_plus`` or ``SP_minus`` in the file name.
    size : tuple of int, optional
    """

    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z) 
    Cutoffp, Cutoffn = CalcCutOff(data,IsoContourPercent=30) 
    F_data = mlab.pipeline.scalar_field(X, Y, Z, data, figure=figure)
    surfaces = _iso_surfaces(F_data, figure, [(Cutoffp, surface_colors["fukui"]["positive"]),
                                              (Cutoffn, surface_colors["fukui"]["negative"])])
    
    if file_name is not None:
        _save_views(figure, "{}-fukui-{}.png".format(file_name, labels), elements_3D_coords, atom_pairs, atoms_Z, size)
    for module in surfaces:
        module.remove()


@_reduced_grid("sensitive")
def viz_Fdual(data, X, Y, Z, elements_3D_coords, atom_pairs, atoms_Z, file_name=None, size=None):
    """Picture of the dual descriptor (``-Fdual.png``) at 30 % (CalcCutOff); parameters as
    viz_Fukui, without labels.
    """
    figure = _init_scene(elements_3D_coords, atom_pairs, atoms_Z) 
    Cutoffp, Cutoffn = CalcCutOff(data,IsoContourPercent=30) 

    F_data = mlab.pipeline.scalar_field(X, Y, Z, data, figure=figure)
    #print("positive Cutoff :", Cutoffp)
    surfaces = _iso_surfaces(F_data, figure, [(Cutoffp, surface_colors["fukui"]["positive"]),
                                              (Cutoffn, surface_colors["fukui"]["negative"])])

    if file_name is not None:
        _save_views(figure, "{}-Fdual.png".format(file_name), elements_3D_coords, atom_pairs, atoms_Z, size)
        # np.save("{}-Fdual.npy".format(file_name), data)
    for module in surfaces:
        module.remove()

