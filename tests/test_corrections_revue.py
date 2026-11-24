"""Regression tests for the corrections of the code review of 20/11/2026 (H2O examples).

Run (see tests/README.md): xvfb-run -a $ENV -m pytest tests/test_corrections_revue.py
"""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from quchemreport.config.config import Config
from quchemreport.parser_engine.scanlog import process_logfile_list
import quchemreport.quality_check_engine.conformity as conformity
from quchemreport.data_enrichment import calc_orb, compute, qmckl_backend
from quchemreport.utility_services.computed_data import init_computed_data
from tests.datafiles import log

def parse(files):
    """files: list of (file name, type[, reference]) -> (config, models, validated data)."""
    with open(os.path.join(ROOT, "example.yaml")) as fd:
        cfg = yaml.safe_load(fd)
    cfg["logfiles"] = [dict(path=log("gaussian/h2o/" + f), type=t, **({"reference": True} if ref else {}))
                       for f, t, ref in files]
    config = Config(**cfg)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        _, models = process_logfile_list(config, log_storage_path=tmp)
        validated = conformity.tests(models, config)
    return config, models, validated


class CorrectionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # SP listed before the reference (formerly IndexError) and anion first (formerly ignored)
        cls.config, cls.models, cls.validated = parse([
            ("H2O_SPminus.log", "sp", False), ("H2O_SPplus.log", "sp", False), ("H2O_OPT.log", "opt", True)])
        cls.ani, cls.cat, cls.opt = cls.models

    def test_reference_after_single_points(self):
        self.assertEqual(self.validated.ref_log_file_idx, 2)
        self.assertIs(self.validated.data_for_discretization, self.opt)

    def test_cdft_in_eV_and_mulliken_fplus(self):
        E = lambda m: m.results.wavefunction.total_molecular_energy
        r = calc_orb.CDFT_Indices(self.opt, self.ani, self.cat)
        self.assertAlmostEqual(r[0], (E(self.opt) - E(self.ani)) * 27.21138505, places=8)
        self.assertAlmostEqual(r[1], (E(self.cat) - E(self.opt)) * 27.21138505, places=8)
        q = lambda m: np.array(m.results.wavefunction.Mulliken_partial_charges)
        np.testing.assert_allclose(r[6], q(self.opt) - q(self.ani))

    def test_sp_found_at_index_zero(self):
        cd = init_computed_data(3)
        compute.extract_simple_data(self.models, self.opt, cd)
        flags = compute.ComputeFlags(**{k: False for k in compute.ComputeFlags.__dataclass_fields__})
        flags.compute_CDFT_global_indices = True
        with contextlib.redirect_stdout(io.StringIO()):
            compute.compute_reactivity(cd, self.validated, self.opt, self.models, flags, 0.3)
        self.assertIsNotNone(cd.results.wavefunction.A)
        self.assertIsNotNone(cd.results.wavefunction.I)

    def test_fukui_cation_only_and_invalid_precomputed_density(self):
        cd = init_computed_data(3)
        with contextlib.redirect_stdout(io.StringIO()):
            compute.compute_fukui_functions(cd, self.opt, None, self.cat, 0.3)
            f_plus, *_ = calc_orb.Fukui(self.opt, self.ani, precomputed_rho_opt=False, grid_step=0.3, nproc=2)
        f_minus = cd.derived.reactivity.fukui_functions.SP_minus.rho_voxels
        self.assertAlmostEqual(f_minus.sum() * 0.3**3, 1.0, delta=0.02)
        self.assertAlmostEqual(f_plus.sum() * 0.3**3, 1.0, delta=0.02)

    def test_mep_matches_direct_sum(self):
        step = 0.4
        # At this coarse step every core is badly sampled: check the convolution alone,
        # without the core spheres (tested in test_real_calculations.test_mep_core_spheres).
        with contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(calc_orb, "CORE_CHARGE_TOLERANCE", np.inf):
            rho, mep, X, Y, Z, raw = calc_orb.Potential(self.opt, grid_step=step, nproc=2)
        geo = qmckl_backend.Wavefunction(self.opt).geometry
        charges = np.array(self.opt.molecule.atoms_Z, float)
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        r, d3r = rho.ravel(), step**3
        for k in np.random.default_rng(1).choice(len(points), 50, replace=False):
            d = np.linalg.norm(points - points[k], axis=1)
            d[k] = np.inf
            dn = np.linalg.norm(geo - points[k], axis=1)
            dn[dn < 5e-4] = np.inf
            direct = np.sum(charges / dn) - d3r * (r[k] + np.sum(r / d))
            self.assertAlmostEqual(mep.ravel()[k], direct, places=10)
        # raw density returned for the Fukui functions, renormalized one for the MEP
        self.assertAlmostEqual(rho.sum() * d3r, 10.0, places=8)
        self.assertFalse(np.allclose(raw, rho))

    def test_reference_without_mo_coefficients(self):
        _, models, validated = parse([("H2O_OPT_NOpop.log", "opt", True)])
        self.assertIsNone(validated.data_for_discretization)
        cd = init_computed_data(1)
        compute.extract_simple_data(models, models[validated.ref_log_file_idx], cd)
        with contextlib.redirect_stdout(io.StringIO()):
            compute.compute_MO_diagrams(cd, None, 2, 0.3)
            self.assertIsNone(compute.compute_MEP_maps(cd, None, 2, 0.3))

    def test_unreadable_log_gives_clear_error(self):
        from quchemreport.parser_engine.scanlog import ScanlogException
        with self.assertRaises(ScanlogException) as ctx:
            parse([("H2O_OPT_ES.log", "opt", True)])
        self.assertIn("H2O_OPT_ES.log", str(ctx.exception))

    def test_zero_point_and_thermal_energies(self):
        _, models, _ = parse([("H2O_FREQ.log", "freq", True)])
        freq = models[0].results.freq
        self.assertAlmostEqual(freq.zero_point_energy, 0.021167)
        self.assertAlmostEqual(freq.electronic_thermal_energy, -76.384952)


if __name__ == "__main__":
    unittest.main()
