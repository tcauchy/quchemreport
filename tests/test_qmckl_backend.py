import os
import unittest
from pickle import dumps
from types import SimpleNamespace as NS

import numpy as np

from quchemreport.data_enrichment import qmckl_backend as qb

from tests.datafiles import log


def model_from_log(path, package="gaussian"):
    """Minimal LogData-like object built with cclib (only the fields used by the backend)."""
    import cclib
    d = cclib.io.ccread(path)
    general = NS(basis_set=dumps(d.gbasis, protocol=0).decode("latin-1"), basis_set_size=d.nbasis,
                 ao_names=list(d.aonames), package=package, core_electrons_per_atoms=list(d.coreelectrons))
    wfn = NS(MO_number_kept=d.nmo, MO_coefs=[np.asarray(m) for m in d.mocoeffs], MO_energies=d.moenergies,
             homo_indexes=list(d.homos))
    return NS(comp_details=NS(general=general),
              results=NS(geometry=NS(elements_3D_coords=d.atomcoords[-1].ravel()), wavefunction=wfn),
              molecule=NS(atoms_Z=list(d.atomnos), charge=d.charge, multiplicity=d.mult))


class HarmonicsTests(unittest.TestCase):
    def test_solid_harmonics_are_orthonormal(self):
        for l in range(2, 6):
            powers = qb.cartesian_powers(l)
            rows = np.array([qb.solid_harmonic_row(l, m) for m in range(-l, l + 1)])
            metric = np.array([[qb._monomial_overlap(p, q) for q in powers] for p in powers])
            np.testing.assert_allclose(rows @ metric @ rows.T, np.eye(2 * l + 1), atol=1e-12)

    def test_known_components(self):
        powers = qb.cartesian_powers(2)            # xx xy xz yy yz zz
        d0 = qb.solid_harmonic_row(2, 0)
        self.assertAlmostEqual(d0[powers.index((2, 0, 0))] / d0[powers.index((0, 0, 2))], -0.5)
        f_m3 = qb.solid_harmonic_row(3, -3)        # 3x^2y - y^3
        powers = qb.cartesian_powers(3)
        self.assertAlmostEqual(f_m3[powers.index((2, 1, 0))] / f_m3[powers.index((0, 3, 0))], -3.0)

    def test_label_parsing(self):
        self.assertEqual(qb._parse_m("C1_3D 0", 2), 0)
        self.assertEqual(qb._parse_m("C1_3D+1", 2), 1)
        self.assertEqual(qb._parse_m("O1_10F-3", 3), -3)
        self.assertEqual(qb._parse_m("O1_1F0", 3), 0)          # ORCA style
        self.assertEqual(qb._parse_m("O1_15G+4", 4), 4)


class WavefunctionTests(unittest.TestCase):
    def check_log(self, name, electrons):
        model = model_from_log(log("gaussian/h2o/" + name))
        wf = qb.Wavefunction(model)
        X, Y, Z = wf.grid(0.1, over_s=5)
        d3r = 0.1 ** 3
        homo = model.results.wavefunction.homo_indexes[0]
        mos = wf.mo([homo, homo + 1], 0, X, Y, Z)
        for mo in mos:
            self.assertAlmostEqual(np.sum(mo * mo) * d3r, 1.0, delta=5e-3)
        self.assertAlmostEqual(np.sum(mos[0] * mos[1]) * d3r, 0.0, delta=1e-3)
        self.assertAlmostEqual(np.sum(wf.density(X, Y, Z)) * d3r, electrons, delta=0.1)

    def test_restricted_spherical(self):
        self.check_log("H2O_OPT_basis2.log", 10)

    def test_unrestricted(self):
        self.check_log("H2O_OPT_unrestricted.log", 10)



class ReducedGridTests(unittest.TestCase):
    def test_nto_pair_reduction_takes_every_kth_point(self):
        from quchemreport.data_enrichment import calc_orb
        model = model_from_log(log("gaussian/h2o/H2O_OPT_basis2.log"))
        n = model.comp_details.general.basis_set_size
        rng = np.random.default_rng(0)
        hole, electron = rng.normal(size=n), rng.normal(size=n)
        full, X, Y, Z = calc_orb.nto_pair(model, hole, electron, grid_step=0.2, nproc=2)
        red, Xr, Yr, Zr = calc_orb.nto_pair(model, hole, electron, grid_step=0.2, nproc=2, reduction=2)
        s = (slice(None, None, 2),) * 3
        np.testing.assert_array_equal(Xr, X[s])
        np.testing.assert_array_equal(Zr, Z[s])
        np.testing.assert_allclose(red, full[(slice(None),) + s], rtol=0, atol=1e-14)

    def test_viz_decorator_reduces_grid_and_data(self):
        from quchemreport.visualization_engine import visu_mayavi as vm
        seen = {}

        @vm._reduced_grid("sensitive")
        def fake(data, X, Y, Z, file_name=None):
            seen.update(shape=np.shape(data), grid=X.shape)

        X, Y, Z = np.mgrid[0:1:11j, 0:1:7j, 0:1:5j]
        vm.set_image_grid_reduction(1, 3)
        fake(np.zeros((2,) + X.shape), X, Y, Z)
        self.assertEqual(seen, {"shape": (2, 4, 3, 2), "grid": (4, 3, 2)})
        fake(np.zeros((2,) + X.shape), X, Y, Z, reduction=1)
        self.assertEqual(seen["grid"], X.shape)
        vm.set_image_grid_reduction(1, 1)


if __name__ == "__main__":
    unittest.main()
