"""Choice of the camera views of the 3D pictures, from the geometry only.

The views do not depend on the orientation chosen by the quantum chemistry program (Gaussian
standard orientation, ORCA input orientation): the same molecule always gives the same pictures.

View 1 is the direction, among 600 directions spread over a half sphere, that best separates the
atoms on the picture: largest projected spread of the atoms, fewest atoms hiding each other.
Heteroatoms weigh more than carbons and hydrogens less, and the camera is put on the side of the
heteroatoms. Flat molecules are tilted by a few degrees to give some depth, and the picture is
turned so that the long axis of the molecule is horizontal.

View 2 is only drawn when one view is not enough:
  - "facing rings": two planar rings face each other (pi-stacking, helicenes, spiro compounds,
    folded adducts, stacked complexes); view 2 looks across them, to show the gap;
  - "rings edge-on": view 1 shows most rings edge-on; view 2, at 60-90 degrees, faces them;
  - "globular": the smallest principal axis is at least 0.4 times the largest one.
The YAML option output.camera_views overrides this choice: 1 never draws view 2, 2 always draws
it ("requested": the best view at 60-90 degrees from view 1, as for a globular molecule), except
for linear molecules, whose side views are all the same.
Why the override (26/11/2026, Thomas Cauchy): the threshold cannot suit every use. HCFClBr
(lambda3/lambda1 = 0.398 after optimization, g09 and ORCA) falls just below 0.4 and gets one
view, whereas showing a stereocentre from two sides is often wanted; lowering the threshold
would give a second view to most CH3X (0.35-0.40) as well.

Why (24/11/2026, with Thomas Cauchy): the former views were fixed angles
(mlab.view(azimuth=90, elevation=10) and (10, 80)) in the frame of the program, so the pictures
depended on the orientation chosen by Gaussian or ORCA: H2CO and H2O were seen edge-on, CH3I
along its C-I axis, and the second view of planar molecules was edge-on. Half of the pictures
were second views, often useless.

Tuned on 255 geometries of real calculations (helicenes, spiro compounds, cages, complexes...),
checked on contact sheets: the atom pairs hidden in the pictures went from 1674 with the former
fixed views to 899, and 34 molecules get a second view. The test geometries
tests/data/geometries/mol01-10 come from that set (tests/test_camera.py).
"""
from dataclasses import dataclass

import numpy as np

N_DIRECTIONS = 600
# Heteroatoms carry the chemistry (the bithiophene of a thiahelicene was put behind the helix
# with equal weights); hydrogens matter less than the skeleton.
WEIGHT_H, WEIGHT_C, WEIGHT_HETERO = 0.5, 1.0, 3.0
HIDDEN_PROJECTED = 0.5     # A: two atoms closer than this on the picture...
HIDDEN_REAL = 1.0          # ...but farther apart in space hide each other
HIDDEN_PENALTY = 2.0       # weight of the hidden atoms against the spread in the score
PLANAR = 0.1               # smallest / largest principal variance of a flat molecule
PLANAR_TILT = 10.0         # degrees of tilt of the face-on view of flat molecules (depth cue,
                           # as the former elevation of 10 degrees)
GLOBULAR = 0.4             # smallest / largest principal variance of a globular molecule
                           # (0.5 missed the spiro compound PhT4, at 0.47, which needs two views)
RING_FLATNESS = 0.2        # A: rms distance of the ring atoms to their mean plane
EDGE_ON = 0.5              # mean |cos(ring normal, view)| below which rings are seen edge-on
# Strict pi-stacking (parallel rings 3.3-4 A apart) missed the cases that need a side view: the
# aryl rings of folded Diels-Alder adducts face each other about 5 A apart, inclined by 40 degrees,
# and the terminal rings of a [7]helicene are 4.2 A apart but inclined by 57 degrees.
FACING_DISTANCE = (3.2, 6.0)   # A: centroid distance of two rings facing each other...
FACING_ANGLE = 60.0            # ...planes inclined by at most this (degrees)...
FACING_OFFSET = 3.0            # ...lateral offset of the centroids below this (A)...
FACING_BONDS = 3               # ...and at least this many bonds between the two rings (fused or
                               # directly bonded rings, e.g. bithiophene, do not face each other)


@dataclass
class View:
    """A camera view: direction from the molecule to the camera, vertical of the picture, and why it was chosen."""
    direction: np.ndarray   # unit vector from the molecule to the camera
    up: np.ndarray          # unit vector, vertical of the picture
    reason: str             # "best", "linear", "facing rings", "rings edge-on", "globular" or "requested"


def choose_views(coords, atoms_Z=None, atom_pairs=None, n_views="auto"):
    """Views (1 or 2) of the molecule. coords: (n, 3) in Angstrom (any frame); atoms_Z: atomic
    numbers (weights); atom_pairs: bonds as index pairs (rings), both optional. n_views: "auto"
    (second view only when needed), 1 or 2 (forced, except for linear molecules: one view)."""
    if n_views not in ("auto", 1, 2):
        raise ValueError(f"n_views must be 'auto', 1 or 2, not {n_views!r}")
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    w, v = principal_axes(xyz)
    if len(xyz) < 3 or w[1] <= 1e-6 * max(w[0], 1e-12):
        # Atom, diatomic or linear molecule: any side view
        d = v[:, 2] if len(xyz) > 1 else np.array([0.0, 0.0, 1.0])
        up = v[:, 1] if len(xyz) > 1 else np.array([0.0, 1.0, 0.0])
        return [View(d, up, "linear")]
    weights = atom_weights(atoms_Z) if atoms_Z is not None else np.ones(len(xyz))
    directions = half_sphere(N_DIRECTIONS)
    d = _best(xyz, directions, weights)
    untilted = d
    if w[2] < PLANAR * w[0] and PLANAR_TILT:
        # The sign of the long axis is arbitrary: orient it by the skewness of the atoms along it,
        # so that the tilt does not depend on the orientation of the molecule
        axis = v[:, 0]
        if np.sum(weights * ((xyz - xyz.mean(axis=0)) @ axis) ** 3) < 0:
            axis = -axis
        d = _rotate(d, axis, PLANAR_TILT)
    d = _front(xyz, d, atoms_Z)
    views = [View(d, _up(d, v), "best")]
    if n_views == 1:
        return views

    rings = small_rings(len(xyz), atom_pairs) if atom_pairs is not None else []
    facing = facing_rings_normal(xyz, rings, atom_pairs) if rings else None
    normals = ring_normals(xyz, rings)
    if facing is not None:
        side = directions[np.abs(directions @ facing) < 0.17]
        d2 = _front(xyz, _best(xyz, side, weights), atoms_Z)
        views.append(View(d2, _unit(facing - (facing @ d2) * d2), "facing rings"))
    elif len(normals) >= 2 and np.mean(np.abs(normals @ untilted)) < EDGE_ON:
        across = directions[np.abs(directions @ d) < 0.5]
        d2 = across[int(np.argmax(np.mean(np.abs(across @ normals.T), axis=1)))]
        d2 = _front(xyz, d2, atoms_Z)
        views.append(View(d2, _up(d2, v), "rings edge-on"))
    elif w[2] >= GLOBULAR * w[0]:
        across = directions[np.abs(directions @ d) < 0.5]
        d2 = _front(xyz, _best(xyz, across, weights), atoms_Z)
        views.append(View(d2, _up(d2, v), "globular"))
    elif n_views == 2:
        across = directions[np.abs(directions @ d) < 0.5]
        d2 = _front(xyz, _best(xyz, across, weights), atoms_Z)
        views.append(View(d2, _up(d2, v), "requested"))
    return views


def atom_weights(atoms_Z):
    """Weights of the atoms in the scores: WEIGHT_H for H, WEIGHT_C for C, WEIGHT_HETERO for the other elements."""
    Z = np.asarray(atoms_Z)
    return np.where(Z == 1, WEIGHT_H, np.where(Z == 6, WEIGHT_C, WEIGHT_HETERO))


def half_sphere(n):
    """n directions evenly spread over the upper half sphere (Fibonacci lattice); opposite
    directions give mirror pictures and the same scores."""
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - k / n)
    theta = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def principal_axes(xyz):
    """Variances and axes (columns) of the atom positions, largest first."""
    c = xyz - xyz.mean(axis=0)
    w, v = np.linalg.eigh(c.T @ c / len(c))
    return w[::-1], v[:, ::-1]


def view_scores(xyz, directions, weights=None):
    """For each direction: weighted spread of the projected atoms (sqrt of the determinant of
    their 2D covariance, A^2) and weighted number of atom pairs hiding each other."""
    xyz = np.asarray(xyz, dtype=float)
    weights = np.ones(len(xyz)) if weights is None else np.asarray(weights, dtype=float)
    c = xyz - weights @ xyz / weights.sum()
    cov = (c * weights[:, None]).T @ c / weights.sum()
    i, j = np.triu_indices(len(xyz), 1)
    diff = xyz[i] - xyz[j]
    real2 = np.einsum("ij,ij->i", diff, diff)
    far = real2 > HIDDEN_REAL ** 2
    diff, real2, pair_w = diff[far], real2[far], (weights[i] * weights[j])[far]
    spread = np.empty(len(directions))
    hidden = np.empty(len(directions))
    for k, d in enumerate(np.asarray(directions, dtype=float)):
        P = np.eye(3) - np.outer(d, d)
        e = np.linalg.eigvalsh(P @ cov @ P)
        spread[k] = np.sqrt(max(e[1], 0.0) * max(e[2], 0.0))
        hidden[k] = pair_w[real2 - (diff @ d) ** 2 < HIDDEN_PROJECTED ** 2].sum()
    return spread, hidden


def _best(xyz, directions, weights):
    """Direction of directions with the best score: normalized spread minus HIDDEN_PENALTY times the weighted hidden pairs per unit weight."""
    spread, hidden = view_scores(xyz, directions, weights)
    score = spread / max(spread.max(), 1e-12) - HIDDEN_PENALTY * hidden / weights.sum()
    return directions[int(np.argmax(score))]


def _front(xyz, d, atoms_Z):
    """Opposite views have the same scores: put the heteroatoms in front."""
    if atoms_Z is None:
        return d
    Z = np.asarray(atoms_Z)
    hetero = (Z != 1) & (Z != 6)
    if not hetero.any() or hetero.all():
        return d
    return d if (xyz[hetero].mean(axis=0) - xyz.mean(axis=0)) @ d >= 0 else -d


def _up(d, axes):
    """Vertical of the picture: the second (or third) principal axis seen from d, so that the
    long axis of the molecule is horizontal."""
    for k in (1, 2, 0):
        u = axes[:, k] - (axes[:, k] @ d) * d
        if np.linalg.norm(u) > 0.3:
            return _unit(u)
    return _unit(np.cross(d, [1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.cross(d, [0.0, 1.0, 0.0]))


def _rotate(d, axis, degrees):
    """Rotation of d about the unit vector axis by an angle in degrees (Rodrigues)."""
    t = np.radians(degrees)
    return d * np.cos(t) + np.cross(axis, d) * np.sin(t) + axis * (axis @ d) * (1 - np.cos(t))


def _unit(u):
    """Unit vector along u."""
    return np.asarray(u, dtype=float) / np.linalg.norm(u)


def small_rings(n, atom_pairs, sizes=(5, 6)):
    """Rings of 5 and 6 atoms of the bond graph (lists of atom indices)."""
    neighbours = [set() for _ in range(n)]
    for i, j in atom_pairs:
        if i != j:
            neighbours[i].add(j)
            neighbours[j].add(i)
    rings = set()

    def walk(path):
        """Extend the path by depth-first search; a path back to its first atom closes a ring."""
        for k in neighbours[path[-1]]:
            if k == path[0] and len(path) in sizes:
                rings.add(tuple(sorted(path)))
            elif k > path[0] and k not in path and len(path) < max(sizes):
                walk(path + [k])

    for a in range(n):
        walk([a])
    return [list(r) for r in sorted(rings)]


def _plane(xyz, ring):
    """Centre, unit normal and flatness (rms distance to the mean plane below RING_FLATNESS) of a ring."""
    p = xyz[ring]
    centre = p.mean(axis=0)
    _, sv, vt = np.linalg.svd(p - centre)
    return centre, vt[2], sv[2] / np.sqrt(len(ring)) < RING_FLATNESS


def ring_normals(xyz, rings):
    """Unit normals of the planar rings, (m, 3)."""
    normals = [n for centre, n, flat in (_plane(xyz, r) for r in rings) if flat]
    return np.array(normals).reshape(-1, 3)


def bond_distances(n, atom_pairs, sources):
    """Number of bonds from the atoms of sources to every atom (inf if not connected)."""
    neighbours = [[] for _ in range(n)]
    for i, j in atom_pairs:
        neighbours[i].append(j)
        neighbours[j].append(i)
    dist = np.full(n, np.inf)
    front = list(sources)
    dist[front] = 0
    while front:
        following = []
        for a in front:
            for b in neighbours[a]:
                if dist[b] == np.inf:
                    dist[b] = dist[a] + 1
                    following.append(b)
        front = following
    return dist


def facing_rings_normal(xyz, rings, atom_pairs):
    """Mean normal of the planar rings facing each other (None if there are none)."""
    planes = [(r, *_plane(xyz, r)) for r in rings]
    planes = [(set(r), c, n) for r, c, n, flat in planes if flat]
    normals = []
    for a in range(len(planes)):
        ra, ca, na = planes[a]
        bonds = None
        for b in range(a + 1, len(planes)):
            rb, cb, nb = planes[b]
            delta = cb - ca
            if ra & rb or not FACING_DISTANCE[0] <= np.linalg.norm(delta) <= FACING_DISTANCE[1]:
                continue
            if abs(na @ nb) < np.cos(np.radians(FACING_ANGLE)):
                continue
            n = _unit((na if na @ nb > 0 else -na) + nb)
            if np.linalg.norm(delta - (delta @ n) * n) > FACING_OFFSET:
                continue
            if atom_pairs is not None:
                if bonds is None:
                    bonds = bond_distances(len(xyz), atom_pairs, sorted(ra))
                if bonds[sorted(rb)].min() < FACING_BONDS:
                    continue
            normals.append(n if not normals or n @ normals[0] >= 0 else -n)
    return _unit(np.sum(normals, axis=0)) if normals else None
