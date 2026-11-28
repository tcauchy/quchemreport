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

Equivalent extrema are grouped and the weak ones left out by their topological prominence
(decision of Thomas Cauchy, 27/11/2026, see PROMINENCE_MIN and EQUIVALENT_TOL): the members of a
group share one number in the picture and one line in the table.
"""
import numpy as np

from quchemreport.utility_services.molecule_graph import symmetry_classes

# Isovalue of the electron density (e-/bohr3) for the surface of the MEP maps and extrema
MEP_SURFACE_ISOVALUE = 0.001
# Two extrema of the same kind are at least this far apart (bohr)
EXTREMA_RADIUS = 1.0
# Topological prominence of an extremum (0-dimensional persistence on the surface mesh): how deep
# the saddle is that leads from it to a stronger extremum of the same kind, |V_ext - V_saddle|;
# the strongest extremum of the surface gets the whole range of V. Equivalent extrema (see
# EQUIVALENT_TOL) form a group: the saddle between two members does not end either of them, so
# that a group is ended only by a stronger, non-equivalent extremum, and all its members share the
# prominence of the group. Groups less prominent than PROMINENCE_MIN (hartree/e; 0.5 kcal/mol)
# are not reported.
# Why (decision of Thomas Cauchy, 27/11/2026; study validation/calculs_20261127/mep_topology on
# the 146 calculations of the laboratory used below, 4661 extrema):
# - the former criterion (26/11/2026: V_ext - median of V on a shell 1.5-2.5 bohr around, at least
#   0.5 kcal/mol) depended on the radius of the shell, and kept "shoulders": a weaker extremum on
#   the flank of a stronger one (the row of maxima along the H atoms of a helicene, saddles of
#   0.05-0.5 kcal/mol; a second minimum on the same oxygen at -15.8 kcal/mol next to -23.0, shell
#   prominence 9.5 but saddle 0.4), the small positive minima between the H atoms of a methyl
#   group (acetone +4.9 kcal/mol: shell 2.0, saddle 0.13) and minima above zero or maxima below
#   zero in the regions of the opposite sign; the topological prominence needs no radius;
# - alone, it is too severe: two equivalent extrema (the lone pairs of a carbonyl, the H atoms
#   of a methyl group) are joined by a shallow saddle, and the weaker one would be dropped; hence
#   the groups;
# - with the groups and the same 0.5 kcal/mol threshold: 3909 extrema reported instead of 4638
#   (16 % fewer, a single one gained), in 3120 groups; no real chemical site lost in the cases
#   examined. The threshold keeps its meaning: below it, a site is not separated from a stronger
#   one;
# - cost: about 1 s for a surface of 180 000 vertices, 2 s for 410 000 (Python lists: 4 to 5
#   times faster than the same loop reading numpy arrays element by element).
PROMINENCE_MIN = 0.5 / 627.509
# Equivalent extrema: same kind, nearest atoms in the same symmetry class of the molecular graph
# (molecule_graph.symmetry_classes; Open Babel OBGraphSym until 28/11/2026) and values within EQUIVALENT_TOL (hartree/e; 0.3 kcal/mol), chained.
# Why (same study): the class alone is not enough, since atoms equivalent in the graph can differ
# by several kcal/mol through the conformation (median 2.1 kcal/mol between extrema of the same
# class); the differences between symmetry equivalents come from the grid (21 % of the pairs of
# the same class within 0.05 kcal/mol, 25 % within 0.1, then a slow rise without valley). 0.1 was
# too strict (two equivalent oxygens of DAdd_133 differ by 0.10 and 0.13 kcal/mol); 1.0 also
# groups sites that are only similar.
EQUIVALENT_TOL = 0.3 / 627.509
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


def mesh_neighbours(n, triangles):
    """Neighbours of the n vertices of the mesh, as CSR lists (indptr, indices)."""
    from scipy.sparse import coo_matrix
    t = np.asarray(triangles)
    i = np.concatenate([t[:, 0], t[:, 1], t[:, 2], t[:, 1], t[:, 2], t[:, 0]])
    j = np.concatenate([t[:, 1], t[:, 2], t[:, 0], t[:, 0], t[:, 1], t[:, 2]])
    m = coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n)).tocsr()
    return m.indptr.tolist(), m.indices.tolist()


def group_prominences(values, neighbours, kind, group_of_vertex):
    """Topological prominence (see PROMINENCE_MIN) of the vertices of group_of_vertex, extrema of
    the given kind ("min" or "max"): {vertex: prominence of its group}.

    Vertices are added from the most extreme value (union-find); when two components meet at a
    vertex (a saddle), the one born at the less extreme value ends there, with prominence
    |V_birth - V_saddle|, unless both were born at members of the same group. The loop reads
    Python lists, not numpy arrays (see PROMINENCE_MIN for the cost).

    values: V at the vertices; neighbours: mesh_neighbours; group_of_vertex: {vertex: group}."""
    sign = -1.0 if kind == "min" else 1.0
    signed = (sign * np.asarray(values, dtype=float)).tolist()      # larger: more extreme
    order = np.argsort(-sign * np.asarray(values, dtype=float), kind="stable").tolist()
    indptr, indices = neighbours
    parent = [-1] * len(signed)
    birth = [0] * len(signed)
    ended = {}

    def root(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    for v in order:
        parent[v] = v
        birth[v] = v
        for k in range(indptr[v], indptr[v + 1]):
            u = indices[k]
            if parent[u] < 0:
                continue
            ru, rv = root(u), root(v)
            if ru == rv:
                continue
            older, younger = (ru, rv) if signed[birth[ru]] >= signed[birth[rv]] else (rv, ru)
            dying = birth[younger]
            same = group_of_vertex.get(dying)
            if dying != v and not (same is not None and same == group_of_vertex.get(birth[older])):
                ended[dying] = signed[dying] - signed[v]
            parent[younger] = older
    last = signed[order[-1]]
    for r in {root(v) for v in range(len(signed))}:
        ended.setdefault(birth[r], signed[birth[r]] - last)
    best = {}
    for vertex, group in group_of_vertex.items():
        best[group] = max(best.get(group, 0.0), ended.get(vertex, 0.0))
    return {vertex: best[group] for vertex, group in group_of_vertex.items()}


def number_groups(extrema):
    """Number (1, 2, ...) of the group of each extremum, in the order of the list, and the number
    of members of its group: keys "group" and "multiplicity". extrema: sorted by group."""
    numbers = {}
    for e in extrema:
        e["group"] = numbers.setdefault(e.get("_group", e.get("group")), len(numbers) + 1)
    counts = {}
    for e in extrema:
        counts[e["group"]] = counts.get(e["group"], 0) + 1
    for e in extrema:
        e["multiplicity"] = counts[e["group"]]
        e.pop("_group", None)
    return extrema


def mep_surface_extrema(rho, MEP, X, Y, Z, coords, isovalue=MEP_SURFACE_ISOVALUE,
                        radius=EXTREMA_RADIUS, atoms_Z=None):
    """Local extrema of the MEP on the density isosurface, grouped by equivalence.

    rho, MEP, X, Y, Z: grids of calc_orb.Potential (bohr, hartree/e); coords: nuclei (n, 3, bohr);
    atoms_Z: atomic numbers, for the symmetry classes of the equivalent extrema (EQUIVALENT_TOL;
    without them, only extrema near the same atom can be grouped). Groups less prominent than
    PROMINENCE_MIN (topological prominence, group_prominences) are left out.
    Returns a list of dicts, the groups of minima from the most negative then those of maxima
    from the most positive, each group from its most extreme member: kind ("min"/"max"), value
    (hartree/e), position (bohr), atom (index of the nearest nucleus), distance (to that nucleus,
    bohr), direction (unit vector from that nucleus, to place markers outside the surface),
    prominence (of the group, hartree/e), group (its number, shared by its members) and
    multiplicity (number of members). Empty list when the isosurface is empty.
    """
    from scipy.interpolate import RegularGridInterpolator
    points, triangles = isosurface(rho, X, Y, Z, isovalue)
    if len(points) == 0:
        return []
    values = RegularGridInterpolator((X[:, 0, 0], Y[0, :, 0], Z[0, 0, :]), MEP,
                                     bounds_error=False, fill_value=None)(points)
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    classes = symmetry_classes(atoms_Z, coords)
    extrema, vertices = [], []
    for kind, i in local_extrema(points, triangles, values, radius):
        offsets = points[i] - coords
        distances = np.linalg.norm(offsets, axis=1)
        atom = int(np.argmin(distances))
        extrema.append({"kind": kind, "value": float(values[i]), "position": points[i].tolist(),
                        "atom": atom, "distance": float(distances[atom]),
                        "direction": (offsets[atom] / distances[atom]).tolist()})
        vertices.append(int(i))
    # Groups of equivalent extrema (chained: a ~ b and b ~ c put a, b and c together)
    group = list(range(len(extrema)))

    def root(a):
        while group[a] != a:
            a = group[a]
        return a

    for i, a in enumerate(extrema):
        for j in range(i):
            b = extrema[j]
            if (a["kind"] == b["kind"] and classes[a["atom"]] == classes[b["atom"]]
                    and abs(a["value"] - b["value"]) <= EQUIVALENT_TOL):
                group[root(i)] = root(j)
    neighbours = mesh_neighbours(len(points), triangles)
    for kind in ("min", "max"):
        members = {vertices[k]: root(k) for k, e in enumerate(extrema) if e["kind"] == kind}
        if members:
            prominence = group_prominences(values, neighbours, kind, members)
            for k, e in enumerate(extrema):
                if e["kind"] == kind:
                    e["prominence"] = prominence[vertices[k]]
    for k, e in enumerate(extrema):
        e["_group"] = root(k)
    extrema = [e for e in extrema if e["prominence"] >= PROMINENCE_MIN]
    sign = {"min": 1.0, "max": -1.0}
    strongest = {}
    for e in extrema:
        strongest[e["_group"]] = min(strongest.get(e["_group"], np.inf), sign[e["kind"]] * e["value"])
    extrema.sort(key=lambda e: (e["kind"] != "min", strongest[e["_group"]], e["_group"], sign[e["kind"]] * e["value"]))
    return number_groups(extrema)


def merge_rings(extrema, coords, up, tol=1.0e-4):
    """Extrema of a linear molecule: one entry per ring around the axis.

    The MEP of a linear molecule has cylindrical symmetry: an extremum off the axis is a whole
    ring (the positive belt of O2), of which the mesh keeps one or more arbitrary points (two
    for O2, both on the hidden side, 26/11/2026). Points of the same kind, value (within tol,
    hartree/e), axial position and radius (0.3 bohr) are merged into one entry with
    ring = True, put on the side of up (the vertical of the picture: the marker stands out at
    the edge of the surface instead of hiding behind it). Other molecules: extrema unchanged.

    extrema: mep_surface_extrema; coords: nuclei (n, 3, bohr); up: unit vector."""
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if len(coords) < 2 or not extrema:
        return extrema
    origin = coords[0]
    axis = coords[np.argmax(np.linalg.norm(coords - origin, axis=1))] - origin
    axis /= np.linalg.norm(axis)
    off = coords - origin - np.outer((coords - origin) @ axis, axis)
    if np.max(np.linalg.norm(off, axis=1)) > 1.0e-3:
        return extrema
    side = np.asarray(up, dtype=float) - (np.asarray(up, dtype=float) @ axis) * axis
    side /= np.linalg.norm(side)
    merged = []
    for e in extrema:
        p = np.asarray(e["position"], dtype=float)
        t = (p - origin) @ axis
        radius = np.linalg.norm(p - origin - t * axis)
        if radius < 0.3:
            merged.append(e)
            continue
        if any(m.get("ring") and m["kind"] == e["kind"] and abs(m["value"] - e["value"]) < tol
               and abs(m["_t"] - t) < 0.3 and abs(m["_r"] - radius) < 0.3 for m in merged):
            continue
        q = origin + t * axis + radius * side
        offsets = q - coords
        distances = np.linalg.norm(offsets, axis=1)
        atom = int(np.argmin(distances))
        merged.append(dict(e, position=q.tolist(), atom=atom, distance=float(distances[atom]),
                           direction=(offsets[atom] / distances[atom]).tolist(), ring=True, _t=t, _r=radius))
    for m in merged:
        m.pop("_t", None)
        m.pop("_r", None)
    # a ring keeps the number of its group, now with a single entry
    return number_groups(merged) if all("group" in m for m in merged) else merged
