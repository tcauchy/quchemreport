"""Extrema of the molecular electrostatic potential (MEP) on the molecular surface.

The surface is the 0.001 e-/bohr3 isosurface of the electron density (Bader's van der Waals
surface, used by Murray and Politzer and by Multiwfn), extracted with VTK's flying edges, the
same contouring as the MEP pictures. The MEP is interpolated (trilinear) at its vertices from
the grid of calc_orb.Potential. A vertex is a local minimum (maximum) when no vertex closer
than EXTREMA_RADIUS has a lower (higher) value: the most negative sites (lone pairs, pi
systems) and the most positive ones (acidic hydrogens, sigma-holes of halogens).

Checked against Gaussian 09 (cubegen potential=SCF at the same points) for H2CO, CH3Cl,
CH3Br and CH3I: differences below 0.75 kcal/mol with the default 0.075 bohr grid.

Why these choices (24/11/2026, with Thomas Cauchy): the 0.001 surface is the convention of the
literature on molecular surface potentials (the pictures used 0.002 before), so the values can
be compared with published V_S,min / V_S,max; all local extrema are kept, since secondary sites
matter (a sigma-hole can be weaker than an acidic hydrogen); the extrema are found on the same
VTK contour as the pictures, so that the markers lie on the drawn surface. The radius of 1 bohr
separates the two lone pairs of a carbonyl oxygen and the three minima of the negative belt of
a halogen, and is checked on azulene (no spurious extrema on a large pi system).
"""
import numpy as np

# Isovalue of the electron density (e-/bohr3) for the surface of the MEP maps and extrema
MEP_SURFACE_ISOVALUE = 0.001
# Two extrema of the same kind are at least this far apart (bohr)
EXTREMA_RADIUS = 1.0
# Half-width (hartree/e) of the fixed, symmetric colour scale of the MEP pictures, by charge of
# the molecule (decision of Thomas Cauchy, 24/11/2026). No standard range exists: the usual
# practice is a symmetric scale around zero, the same for a series of compounds, given with a
# colour bar. The former scale (+-0.01 a.u.) had been set for the 0.002 surface and saturated
# nearly all the 0.001 surface (H2O: -40 / +43 kcal/mol).
# Calibration (24/11/2026, 141 Gaussian calculations of the laboratory, 88 neutral, 45 ions
# +-1; MEP on the 0.001 surface, percentiles weighted by area):
# - neutral: 1st / 99th percentiles -28 / +25 kcal/mol (medians). +-0.05 a.u. (+-31 kcal/mol)
#   leaves 1.6 % of the area saturated for the median molecule, 4.9 % at the 90th percentile;
#   only the extrema saturate, and their values are in the table. +-0.06 and +-0.07 (0 % / 2.8 %
#   and 0 % / 1.0 %) were rejected: less contrast for the many weakly polar molecules.
# - ions: the surface has the sign of the charge everywhere (mean about +72 kcal/mol for +1,
#   -74 for -1; 1st-99th percentile width about 64). +-0.20 a.u. per unit charge covers the
#   2.5-97.5 percentiles of 85 % (anions) to 92 % (cations) of them and keeps red = negative,
#   blue = positive, at the cost of using only one half of the palette. Rejected: a range
#   shifted per charge (centre +-70 kcal/mol: more contrast, but red would be a positive
#   potential for a cation) and a range per molecule (no comparison between compounds).
#   +2 and +3 (6 calculations) stay within 0.20 a.u. per unit charge.
COLOR_RANGE_NEUTRAL = 0.05
COLOR_RANGE_PER_CHARGE = 0.20


def color_range(charge):
    """Half-width of the fixed colour scale of the MEP pictures (hartree/e) for this charge.

    Parameters
    ----------
    charge : int
        Charge of the molecule.

    Returns
    -------
    float
        0.05 a.u. for a neutral molecule, 0.20 a.u. per unit of charge for an ion (see the
        calibration above).
    """
    charge = abs(int(round(charge or 0)))
    return COLOR_RANGE_NEUTRAL if charge == 0 else round(COLOR_RANGE_PER_CHARGE * charge, 6)


def isosurface(rho, X, Y, Z, value=MEP_SURFACE_ISOVALUE):
    """Vertices (n, 3, bohr) and triangles (m, 3) of the isosurface rho = value."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    axes = X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]
    image = vtk.vtkImageData()
    image.SetDimensions(*rho.shape)
    image.SetOrigin(axes[0][0], axes[1][0], axes[2][0])
    image.SetSpacing(*(a[1] - a[0] for a in axes))
    # VTK point order: x fastest
    image.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(rho.ravel(order="F")), deep=True))
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, value)
    contour.ComputeNormalsOff()
    contour.ComputeGradientsOff()
    contour.Update()
    surface = contour.GetOutput()
    if surface.GetNumberOfPoints() == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return points, triangles


def local_extrema(points, triangles, values, radius=EXTREMA_RADIUS):
    """[(kind, vertex index)] of the local minima ("min") and maxima ("max") of values."""
    from scipy.spatial import cKDTree
    # Candidates: vertices not exceeded by their neighbours on the mesh
    first = np.concatenate([triangles.ravel(), triangles[:, [1, 2, 0]].ravel()])
    second = np.concatenate([triangles[:, [1, 2, 0]].ravel(), triangles.ravel()])
    lowest = np.full(len(values), np.inf)
    np.minimum.at(lowest, first, values[second])
    highest = np.full(len(values), -np.inf)
    np.maximum.at(highest, first, values[second])
    tree = cKDTree(points)
    found = []
    for kind, candidates, sign in (("min", np.flatnonzero(values <= lowest), 1.0),
                                   ("max", np.flatnonzero(values >= highest), -1.0)):
        kept = []
        # Most extreme first, so that a plateau keeps a single vertex
        for c in candidates[np.argsort(sign * values[candidates], kind="stable")]:
            ball = tree.query_ball_point(points[c], radius)
            if sign * values[c] > np.min(sign * values[ball]):
                continue
            if any(np.linalg.norm(points[c] - points[k]) <= radius for k in kept):
                continue
            kept.append(c)
        found += [(kind, c) for c in kept]
    return found


def mep_surface_extrema(rho, MEP, X, Y, Z, coords, isovalue=MEP_SURFACE_ISOVALUE,
                        radius=EXTREMA_RADIUS):
    """Local extrema of the MEP on the density isosurface.

    rho, MEP, X, Y, Z: grids of calc_orb.Potential (bohr, hartree/e); coords: nuclei (n, 3, bohr).
    Returns a list of dicts, minima from the most negative then maxima from the most positive:
    kind ("min"/"max"), value (hartree/e), position (bohr), atom (index of the nearest nucleus),
    distance (to that nucleus, bohr) and direction (unit vector from that nucleus, to place
    markers outside the surface). Empty list when the isosurface is empty.
    """
    from scipy.interpolate import RegularGridInterpolator
    points, triangles = isosurface(rho, X, Y, Z, isovalue)
    if len(points) == 0:
        return []
    values = RegularGridInterpolator((X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]), MEP,
                                     bounds_error=False, fill_value=None)(points)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    extrema = []
    for kind, i in local_extrema(points, triangles, values, radius):
        offsets = points[i] - coords
        distances = np.linalg.norm(offsets, axis=1)
        atom = int(np.argmin(distances))
        extrema.append({"kind": kind, "value": float(values[i]), "position": points[i].tolist(),
                        "atom": atom, "distance": float(distances[atom]),
                        "direction": (offsets[atom] / distances[atom]).tolist()})
    minima = sorted((e for e in extrema if e["kind"] == "min"), key=lambda e: e["value"])
    maxima = sorted((e for e in extrema if e["kind"] == "max"), key=lambda e: -e["value"])
    return minima + maxima
