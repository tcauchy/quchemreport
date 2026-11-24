"""Exact MEP reference from Gaussian: python make_mep_reference.py JOB.fchk OUT.txt "label"

Runs cubegen (density and potential=SCF on the same "fine" grid -3, 6 points/bohr), takes 400
random points of the 0.002 a.u. iso-density surface (linear interpolation along grid edges)
and interpolates the potential there. Crossings closer than MIN_DISTANCE to a nucleus are
dropped: g09 D.01 cubegen prints spurious near-zero densities about 0.55 bohr from the Cl
nucleus of CH3Cl/6-31G(d) (qmckl gives 2.8 there), which look like 0.002 crossings.
Existing cubes are reused. Needs g09 in the environment (g09.profile).
Output: same format as tests/data/mep_reference/*.txt (x y z in bohr, potential in hartree/e).
"""
import os
import subprocess
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ISO, NPOINTS, SEED, MIN_DISTANCE = 0.002, 400, 0, 1.5


def read_cube(path):
    with open(path) as f:
        f.readline(); f.readline()
        natoms, *origin = f.readline().split()
        natoms = abs(int(natoms)); origin = np.array(origin[:3], float)
        axes = [f.readline().split() for _ in range(3)]
        n = [int(a[0]) for a in axes]
        steps = [float(a[i + 1]) for i, a in enumerate(axes)]
        assert all(float(a[j + 1]) == 0 for i, a in enumerate(axes) for j in range(3) if j != i), "non orthogonal cube"
        nuclei = np.array([f.readline().split()[2:5] for _ in range(natoms)], float)
        values = np.array(f.read().split(), float)
    grid = [origin[i] + steps[i] * np.arange(n[i]) for i in range(3)]
    return grid, values.reshape(n), nuclei


def surface_points(grid, rho):
    points = []
    for axis in range(3):
        a = np.moveaxis(rho, axis, 0)
        lo, hi = a[:-1], a[1:]
        idx = np.argwhere((lo - ISO) * (hi - ISO) < 0)
        t = (lo[tuple(idx.T)] - ISO) / (lo[tuple(idx.T)] - hi[tuple(idx.T)])
        coords = np.empty((len(idx), 3))
        order = [axis] + [i for i in range(3) if i != axis]
        for k, ax in enumerate(order):
            g = grid[ax]
            coords[:, ax] = g[idx[:, k]] + (t * (g[1] - g[0]) if k == 0 else 0)
        points.append(coords)
    return np.concatenate(points)


def main(fchk, out, label):
    cubes = {}
    for kind in ("density", "potential"):
        cube = fchk.replace(".fchk", f"_{kind[:3]}.cube")
        if not os.path.exists(cube):
            subprocess.run(["cubegen", "8", f"{kind}=SCF", fchk, cube, "-3", "h"], check=True)
        cubes[kind] = read_cube(cube)
    grid, rho, nuclei = cubes["density"]
    grid_v, pot, _ = cubes["potential"]
    assert all(np.allclose(a, b) for a, b in zip(grid, grid_v))
    pts = surface_points(grid, rho)
    far = np.linalg.norm(pts[:, None] - nuclei[None], axis=-1).min(axis=1) > MIN_DISTANCE
    print(f"{len(pts)} crossings, {np.sum(~far)} dropped (closer than {MIN_DISTANCE} bohr to a nucleus)")
    pts = pts[far]
    pts = pts[np.random.default_rng(SEED).choice(len(pts), NPOINTS, replace=False)]
    v = RegularGridInterpolator(grid, pot, method="cubic")(pts)
    header = (f"Molecular electrostatic potential computed by Gaussian 09 (cubegen 8 potential=SCF, fine grid)\n"
              f"at {NPOINTS} random points of its {ISO} a.u. iso-density surface, for {label}.\n"
              "Columns: x y z (bohr, frame of the log) and potential (hartree/e).")
    np.savetxt(out, np.column_stack([pts, v]), fmt="%.6f %.6f %.6f %.8f", header=header)


if __name__ == "__main__":
    main(*sys.argv[1:4])
