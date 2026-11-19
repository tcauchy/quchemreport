"""Evaluation of orbitals and densities on grids with QMCkl (replacement for orbkit).

The basis set, the MO coefficients and the geometry are taken from the LogData
model (cclib parsing). QMCkl only evaluates Cartesian Gaussian AOs, so the
coefficients expressed in a spherical (pure) basis are converted once into the
Cartesian AO basis: phi = C_sph . chi_sph = (C_sph . T) . chi_cart.

All Cartesian AOs of a shell share the normalization of the x^l component
(ao_factor = 1); T contains the exact normalized real solid harmonics in that
representation. Coordinates are in bohr, as in orbkit.
"""
import re
from math import comb, sqrt
from pickle import loads

import numpy as np
import scipy.sparse
import qmckl
from threadpoolctl import threadpool_limits

# Same conversion factor as orbkit.units.aa_to_a0, so that the grid is identical.
ANGSTROM_TO_BOHR = 1.8897261259693365
L_OF = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}
POINT_CHUNK = 20000


def _dfact(n):
    """Double factorial with (-1)!! = 1."""
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def cartesian_powers(l):
    """Cartesian monomials of a shell in TREXIO/QMCkl order (xx, xy, xz, yy, yz, zz...)."""
    return [(a, l - a - c2, c2) for a in range(l, -1, -1) for c2 in range(0, l - a + 1)]


def _monomial_overlap(p, q):
    """Angular overlap of two monomials, relative to the one of x^l with itself."""
    e = [p[i] + q[i] for i in range(3)]
    if any(v % 2 for v in e):
        return 0.0
    l = sum(p)
    return _dfact(e[0] - 1) * _dfact(e[1] - 1) * _dfact(e[2] - 1) / _dfact(2 * l - 1)


def _normalize_row(row, powers):
    norm2 = sum(row[i] * row[j] * _monomial_overlap(powers[i], powers[j])
                for i in range(len(powers)) for j in range(len(powers)))
    return row / sqrt(norm2)


def solid_harmonic_row(l, m):
    """Normalized real solid harmonic S_lm on Cartesian monomials (Helgaker 6.4.47).

    Signs follow the usual convention also used by Gaussian: S_1,1 = x,
    S_2,2 ~ x^2 - y^2, S_2,-2 ~ xy, S_3,-3 ~ 3x^2y - y^3, ...
    """
    powers = cartesian_powers(l)
    index = {p: i for i, p in enumerate(powers)}
    row = np.zeros(len(powers))
    am = abs(m)
    twice_vm = 0 if m >= 0 else 1
    for t in range((l - am) // 2 + 1):
        for u in range(t + 1):
            # v runs over integers (m >= 0) or half-integers (m < 0); work with 2v.
            for twice_v in range(twice_vm, am + 1, 2):
                coef = ((-1) ** (t + (twice_v - twice_vm) // 2) * 0.25 ** t * comb(l, t)
                        * comb(l - t, am + t) * comb(t, u) * comb(am, twice_v))
                py = 2 * u + twice_v
                px = 2 * t + am - py
                pz = l - 2 * t - am
                row[index[(px, py, pz)]] += coef
    return _normalize_row(row, powers)


def cartesian_row(l, power):
    """Individually normalized Cartesian function (Cartesian 6D/10F basis sets)."""
    powers = cartesian_powers(l)
    row = np.zeros(len(powers))
    row[powers.index(tuple(power))] = 1.0
    return _normalize_row(row, powers)


def _parse_m(label, l):
    """Magnetic number from an AO label: Gaussian 'C1_3D+1', 'C1_10F 0', ORCA 'O1_1F-3'."""
    name = label.split("_")[-1].upper()
    found = re.search(r"[DFGHI]\s*([+-]?\d+)$", name)
    if found is None:
        raise ValueError(f"Cannot read the magnetic number of AO label {label!r}")
    return int(found.group(1))


# ORCA's real spherical harmonics differ in sign from the standard (Gaussian) ones
# for these (l, |m|) components (same correction as in ORCA's molden export).
ORCA_SIGN_FLIPS = {(3, 3), (4, 3), (4, 4)}


def _contracted_norm(l, exponents, coefficients):
    """Factors making each primitive and the contraction normalized for x^l."""
    exponents = np.asarray(exponents, dtype=float)
    prim = (2 * exponents / np.pi) ** 0.75 * (4 * exponents) ** (l / 2) / sqrt(_dfact(2 * l - 1))
    c = np.asarray(coefficients, dtype=float)
    a = exponents[:, None] + exponents[None, :]
    overlap = (2 * np.sqrt(np.outer(exponents, exponents)) / a) ** (l + 1.5)
    shell = 1.0 / sqrt(float(c @ overlap @ c))
    return prim, shell


def mo_coefficients(data_model):
    """MO coefficient matrices (one per spin, MO x basis), dense."""
    wf = data_model.results.wavefunction
    shape = (wf.MO_number_kept, data_model.comp_details.general.basis_set_size)
    pre = wf.MO_coefs
    if type(pre[0]) is tuple:
        return [np.asarray(scipy.sparse.csr_matrix(tuple(np.asarray(d) for d in sp), shape=shape).todense())
                for sp in pre]
    return [np.asarray(dense, dtype=float) for dense in pre]


def occupations(data_model, n_mo):
    """Occupation numbers per spin, with the same rules as json2orbkit."""
    wf = data_model.results.wavefunction
    restricted = len(wf.MO_energies) == 1
    if not restricted:
        return [np.array([1.0 if i <= wf.homo_indexes[s] else 0.0 for i in range(n_mo)]) for s in range(2)]
    general = data_model.comp_details.general
    ele_num = (np.sum(data_model.molecule.atoms_Z) - np.sum(general.core_electrons_per_atoms)
               - data_model.molecule.charge)
    ue = data_model.molecule.multiplicity - 1
    occ = np.zeros(n_mo)
    for i in range(n_mo):
        if ele_num > ue:
            occ[i] = 2.0
            ele_num -= 2.0
        elif 0.0 < ele_num <= ue:
            occ[i] = 1.0
            ele_num -= 1.0
            ue -= 1.0
    return [occ]


class Wavefunction:
    """QMCkl context for one LogData model."""

    def __init__(self, data_model, orca_signs=True, nproc=None):
        # nproc limits the OpenMP (qmckl) and BLAS threads during the evaluations;
        # None keeps the library defaults (OMP_NUM_THREADS, or all the cores).
        self.nproc = nproc
        general = data_model.comp_details.general
        self.data_model = data_model
        self.geometry = (np.array(data_model.results.geometry.elements_3D_coords, dtype=float)
                         .reshape((-1, 3)) * ANGSTROM_TO_BOHR)
        charges = np.array(data_model.molecule.atoms_Z, dtype=float)
        gbasis = loads(bytes(general.basis_set, "utf-8"))
        labels = list(general.ao_names) if general.ao_names is not None else None
        if labels is None:
            raise ValueError("AO labels (ao_names) are required to identify the spherical components")
        if general.package == "orca":
            for old, new in (("DZ2", "D 0"), ("DXZ", "D+1"), ("DYZ", "D-1"), ("DX2Y2", "D+2"), ("DXY", "D-2")):
                labels = [lab.replace(old, new) for lab in labels]
        self.orca_signs = general.package == "orca" and orca_signs
        spherical = any("+" in lab or "-" in lab for lab in labels)

        shell_l, shell_nucleus, prim_num, exps, coefs, prim_f, shell_f = [], [], [], [], [], [], []
        blocks = []          # (rows of T for this shell, number of Cartesian AOs)
        position = 0
        for atom, shells in enumerate(gbasis):
            for shell_type, primitives in shells:
                l = L_OF[str(shell_type).lower()]
                e = [p[0] for p in primitives]
                c = [p[1] for p in primitives]
                pf, sf = _contracted_norm(l, e, c)
                shell_l.append(l)
                shell_nucleus.append(atom)
                prim_num.append(len(primitives))
                exps += e
                coefs += c
                prim_f += list(pf)
                shell_f.append(sf)
                n_sph = 2 * l + 1 if (spherical and l > 1) else (l + 1) * (l + 2) // 2
                rows = []
                for label in labels[position:position + n_sph]:
                    if l == 0:
                        rows.append(np.ones(1))
                    elif l == 1:
                        rows.append(cartesian_row(1, _p_power(label)))
                    elif spherical:
                        m = _parse_m(label, l)
                        sign = -1.0 if (self.orca_signs and (l, abs(m)) in ORCA_SIGN_FLIPS) else 1.0
                        rows.append(sign * solid_harmonic_row(l, m))
                    else:
                        name = label.lower().split("_")[-1]
                        rows.append(cartesian_row(l, (name.count("x"), name.count("y"), name.count("z"))))
                blocks.append(np.array(rows))
                position += n_sph
        if position != general.basis_set_size:
            raise ValueError("Basis set description does not match the number of basis functions")

        n_cart = sum(b.shape[1] for b in blocks)
        self.transform = np.zeros((position, n_cart))   # spherical (file) AO -> Cartesian AO
        r, col = 0, 0
        for b in blocks:
            self.transform[r:r + b.shape[0], col:col + b.shape[1]] = b
            r += b.shape[0]
            col += b.shape[1]

        nucl_num = len(charges)
        shell_nucleus = np.array(shell_nucleus)
        nucleus_shell_num = np.array([np.sum(shell_nucleus == a) for a in range(nucl_num)], dtype=np.int64)
        self._basis = dict(
            charges=charges, nucl_num=nucl_num, shell_num=len(shell_l), prim_num=len(exps),
            nucleus_shell_num=nucleus_shell_num,
            nucleus_index=np.concatenate([[0], np.cumsum(nucleus_shell_num)[:-1]]).astype(np.int64),
            shell_ang_mom=np.array(shell_l, dtype=np.int32),
            shell_prim_num=np.array(prim_num, dtype=np.int64),
            shell_prim_index=np.concatenate([[0], np.cumsum(prim_num)[:-1]]).astype(np.int64),
            shell_factor=np.array(shell_f), exponent=np.array(exps), coefficient=np.array(coefs),
            prim_factor=np.array(prim_f))
        self.n_cart = n_cart

        self.coefficients = mo_coefficients(data_model)
        self.occupations = occupations(data_model, self.coefficients[0].shape[0])

    # ------------------------------------------------------------------ grids
    def grid(self, grid_step, over_s=5):
        """Same regular grid as calc_orb._init_ORB_grid (bohr)."""
        gmax = np.around(np.amax(self.geometry, axis=0) + over_s)
        gmin = np.around(np.amin(self.geometry, axis=0) - over_s)
        return np.mgrid[gmin[0]:gmax[0] + grid_step:grid_step,
                        gmin[1]:gmax[1] + grid_step:grid_step,
                        gmin[2]:gmax[2] + grid_step:grid_step]

    def _context(self, cartesian_coefficients):
        """New QMCkl context whose MO basis is the given rows (Cartesian AO coefficients)."""
        p = self._basis
        ctx = qmckl.context_create()
        qmckl.set_nucleus_num(ctx, p["nucl_num"])
        qmckl.set_nucleus_charge(ctx, p["charges"])
        qmckl.set_nucleus_coord(ctx, "N", np.ascontiguousarray(self.geometry).ravel())
        qmckl.set_ao_basis_type(ctx, "G")
        qmckl.set_ao_basis_shell_num(ctx, p["shell_num"])
        qmckl.set_ao_basis_prim_num(ctx, p["prim_num"])
        qmckl.set_ao_basis_nucleus_index(ctx, p["nucleus_index"])
        qmckl.set_ao_basis_nucleus_shell_num(ctx, p["nucleus_shell_num"])
        qmckl.set_ao_basis_shell_ang_mom(ctx, p["shell_ang_mom"])
        qmckl.set_ao_basis_shell_prim_num(ctx, p["shell_prim_num"])
        qmckl.set_ao_basis_shell_prim_index(ctx, p["shell_prim_index"])
        qmckl.set_ao_basis_shell_factor(ctx, p["shell_factor"])
        qmckl.set_ao_basis_exponent(ctx, p["exponent"])
        qmckl.set_ao_basis_coefficient(ctx, p["coefficient"])
        qmckl.set_ao_basis_prim_factor(ctx, p["prim_factor"])
        qmckl.set_ao_basis_ao_num(ctx, self.n_cart)
        # ao_cartesian is true by default; qmckl 1.1.0 set_ao_basis_cartesian shares the
        # "type" mask and always fails with QMCKL_ALREADY_SET, so it is not called.
        qmckl.set_ao_basis_ao_factor(ctx, np.ones(self.n_cart))
        qmckl.set_mo_basis_mo_num(ctx, cartesian_coefficients.shape[0])
        qmckl.set_mo_basis_coefficient(ctx, np.ascontiguousarray(cartesian_coefficients).ravel())
        return ctx

    def _evaluate(self, coefficients, X, Y, Z, reduce=None):
        """Orbital values on the grid, chunk by chunk; reduce(values) may sum them on the fly."""
        cart = np.atleast_2d(np.asarray(coefficients, dtype=float)) @ self.transform
        n = cart.shape[0]
        ctx = self._context(cart)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        out = np.empty(len(points)) if reduce else np.empty((n, len(points)))
        limits = threadpool_limits(limits=self.nproc) if self.nproc else None
        try:
            for start in range(0, len(points), POINT_CHUNK):
                p = np.ascontiguousarray(points[start:start + POINT_CHUNK])
                qmckl.set_point(ctx, "N", len(p), p.ravel())
                values = qmckl.get_mo_basis_mo_value(ctx, len(p) * n).reshape(len(p), n)
                if reduce:
                    out[start:start + len(p)] = reduce(values)
                else:
                    out[:, start:start + len(p)] = values.T
        finally:
            qmckl.context_destroy(ctx)
            if limits is not None:
                limits.restore_original_limits()
        return out.reshape(X.shape) if reduce else out.reshape((n,) + X.shape)

    def orbitals(self, coefficients, X, Y, Z):
        """Values of orbitals given by rows of coefficients in the file AO basis (e.g. NTOs)."""
        return self._evaluate(coefficients, X, Y, Z)

    def mo(self, indices, spin, X, Y, Z):
        """MO values; indices are zero-based in the spin block (0 = alpha/restricted, 1 = beta)."""
        return self._evaluate(self.coefficients[spin][list(indices)], X, Y, Z)

    def density(self, X, Y, Z):
        """Total electron density sum_s sum_i n_i phi_i^2 (orbitals scaled by sqrt(n_i))."""
        rows = [coefs[occ > 1e-7] * np.sqrt(occ[occ > 1e-7])[:, None]
                for coefs, occ in zip(self.coefficients, self.occupations)]
        return self._evaluate(np.vstack(rows), X, Y, Z, reduce=lambda v: np.einsum("ij,ij->i", v, v))


def _p_power(label):
    axis = label.lower().split("_")[-1][-1]
    return {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
