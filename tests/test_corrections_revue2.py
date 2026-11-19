"""Regression tests for the corrections of the second code review (parser, quality checks,
visualization, report), 19/11/2026.

Small Gaussian/ORCA logs made for these tests: validation/revue2/logs (He, H2, ROHF O2,
planar NH3 saddle point, ORCA optimization stopped at MaxIter).

Run: OMP_NUM_THREADS=4 PYTHONPATH=.:/applis/cclib/lib/python3.12/site-packages \
     xvfb-run -a $ENV -m unittest tests.test_corrections_revue2
"""
import contextlib
import io
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples")
# small test logs: validation/revue2/logs of the test copy (validation/ is not installed in /applis)
LOGS = next((d for d in (os.environ.get("QUCHEMREPORT_TEST_LOGS"),
                         os.path.join(ROOT, "validation", "revue2", "logs"),
                         "/home/mcadmin/quchemreport_qmckl_20261119/validation/revue2/logs")
             if d and os.path.isdir(d)), None)
if LOGS is None:
    raise unittest.SkipTest("test logs of validation/revue2/logs not found (set QUCHEMREPORT_TEST_LOGS)")
ORCA_TD = "/home/mcadmin/quchemreport_nto_20261117/validation_orca/TD.out"

from quchemreport.config.config import Config
from quchemreport.parser_engine.scanlog import (logfile_to_obj, split_logfile, nuclear_repulsion_energy,
                                                _orca_functional)
import quchemreport.quality_check_engine.conformity as conformity
from quchemreport.data_enrichment import compute
from quchemreport.utility_services.naming import spin_letter
from quchemreport.utility_services.computed_data import init_computed_data
from quchemreport.visualization_engine.visu_mayavi import CalcCutOff
from quchemreport.report_generator.generate_report import quality_warnings


def read_steps(path, solver="gaussian"):
    """Models of all the steps of a log."""
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        return [logfile_to_obj(step) for step in split_logfile(path, solver, tmp)]


def config_for(paths_types, solver="gaussian"):
    with open(os.path.join(ROOT, "example.yaml")) as fd:
        cfg = yaml.safe_load(fd)
    cfg["common_solver"] = solver
    cfg["logfiles"] = [dict(path=p, type=t, **({"reference": True} if i == 0 else {}))
                       for i, (p, t) in enumerate(paths_types)]
    return Config(**cfg)


class CutOffTest(unittest.TestCase):
    def test_single_sign(self):
        # 1s orbital of an atom / sigma_g of H2: no negative values, no negative iso-surface
        x = np.linspace(-3, 3, 30)
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        data = np.exp(-(X**2 + Y**2 + Z**2))
        cut_p, cut_n = CalcCutOff(data, 20)
        self.assertIsNone(cut_n)
        self.assertGreater(cut_p, 0)
        self.assertIsNone(CalcCutOff(-data, 20)[0])

    def test_two_signs_unchanged(self):
        # same values as the former code for the usual case
        data = np.random.default_rng(1).standard_normal(5000) ** 3
        p, n = np.sort(data[data > 0]), np.sort(-data[data < 0])
        cp, cn = np.cumsum(p), np.cumsum(n)
        old = (p[cp / cp[-1] * 100. >= 70.][0], -n[cn / cn[-1] * 100. >= 70.][0])
        self.assertEqual(CalcCutOff(data, 30), old)


class NamingTest(unittest.TestCase):
    def test_spin_letter(self):
        self.assertEqual(spin_letter("Singlet-B1"), "S")
        self.assertEqual(spin_letter("Singlet"), "S")          # ORCA
        self.assertEqual(spin_letter("Triplet-A1"), "T")
        self.assertEqual(spin_letter("3.002-?Sym"), "")        # unrestricted Gaussian


class BasisTest(unittest.TestCase):
    def reference(self, ao_names, package="gaussian"):
        return conformity.ReferenceData(charge_ref=0, multiplicity_ref=1, MO_coeff_ref=[[[1.0]]],
                                        basis_sets_ref="basis", ao_names_ref=ao_names,
                                        package_ref=package, data_for_discretization=None)

    def test_s_functions_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            discret, _ = conformity.test_basis_and_mo_coeff(self.reference(["He1_1S", "He1_2S"]))
        self.assertTrue(discret)

    def test_s_and_p_still_cartesian(self):
        with contextlib.redirect_stdout(io.StringIO()):
            discret, _ = conformity.test_basis_and_mo_coeff(self.reference(["H1_1S", "H1_2PX", "H1_2PY"]))
        self.assertTrue(discret)


class ConformityExitTest(unittest.TestCase):
    def test_exit_code_is_1(self):
        data = SimpleNamespace(nb_files=2, formulas_nocharges=["H2O", "NH3"])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ctx:
            conformity.test_formula(data)
        self.assertEqual(ctx.exception.code, 1)


class NuclearRepulsionTest(unittest.TestCase):
    def test_orientation_and_small_change(self):
        model = read_steps(os.path.join(EX, "phenantren_OPT.log"))[0]
        x = np.array(model.results.geometry.elements_3D_coords).reshape(-1, 3)
        Z = np.array(model.molecule.atoms_Z)

        def nre(coords):
            return nuclear_repulsion_energy(SimpleNamespace(natom=len(Z), atomnos=Z, atomcoords=np.array([coords])))

        ref = nre(x)
        for seed in range(50):   # same geometry, other orientations (printed with 6 decimals)
            y = np.round(Rotation.random(random_state=seed).apply(x - x.mean(0)), 6)
            self.assertTrue(conformity.same_nre(ref, nre(y)))
        z = x.copy()             # one bond 0.001 A longer: another geometry
        z[0] += (z[0] - z[1]) / np.linalg.norm(z[0] - z[1]) * 0.001
        self.assertFalse(conformity.same_nre(ref, nre(z)))

    def test_same_geometry_needs_same_orientation(self):
        a = read_steps(os.path.join(EX, "phenantren_OPT.log"))[0]
        b = read_steps(os.path.join(EX, "phenantren_TD.log"))[0]      # same geometry (geom=allcheck)
        es = read_steps(os.path.join(EX, "phenantren_OPT_ES1.log"))[0]
        self.assertTrue(conformity.same_geometry(a, b))
        self.assertFalse(conformity.same_geometry(a, es))


class OpenShellTest(unittest.TestCase):
    def test_rohf_is_not_closed_shell(self):
        rohf = read_steps(os.path.join(LOGS, "O2_ROHF.log"))[0]
        self.assertEqual(str(rohf.comp_details.general.is_closed_shell), "False")
        h2o = read_steps(os.path.join(EX, "H2O_OPT.log"))[0]
        self.assertEqual(str(h2o.comp_details.general.is_closed_shell), "True")


class MOLabelsTest(unittest.TestCase):
    def test_lumo_plus_one_beyond_the_last_mo(self):
        # He/6-31G: 2 MOs, lumo+1 does not exist (it passed the check and crashed in the grids)
        he = read_steps(os.path.join(LOGS, "He.log"))[0]
        config = config_for([(os.path.join(LOGS, "He.log"), "opt")])
        with contextlib.redirect_stdout(io.StringIO()):
            labels, indexes = compute.compute_MO_labels_and_indexes(he, config, "He.log")
        self.assertEqual(indexes, [[0, 1]])
        self.assertEqual(labels, [["homo", "lumo"]])


class StructureWarningsTest(unittest.TestCase):
    def warnings_for(self, models):
        computed = init_computed_data(len(models))
        with contextlib.redirect_stdout(io.StringIO()):
            compute.extract_simple_data(models, models[0], computed)
        return quality_warnings(computed)

    def test_imaginary_frequency(self):
        warnings = self.warnings_for(read_steps(os.path.join(LOGS, "NH3_plan.log")))
        self.assertEqual(len(warnings), 1)
        self.assertIn("889.8i", warnings[0])

    def test_minimum_has_no_warning(self):
        self.assertEqual(self.warnings_for(read_steps(os.path.join(EX, "H2O_FREQ.log"))), [])

    def test_orca_optimization_not_converged(self):
        models = read_steps(os.path.join(LOGS, "orca_maxiter", "H2O_maxiter.out"), "orca")
        warnings = self.warnings_for(models)
        self.assertTrue(any("did not converge" in w for w in warnings))


class OrcaFunctionalTest(unittest.TestCase):
    def test_functional_definition(self):
        b3lyp = _orca_functional(os.path.join(LOGS, "orca_maxiter", "H2O_maxiter.out"), ["B3LYP", "def2-SVP", "Opt"])
        self.assertEqual(b3lyp, "B3LYP (Exchange B88, Correlation LYP, HF exchange 0.2, DF-GGA-X 0.72, DF-GGA-C 0.81)")
        self.assertIsNone(_orca_functional(os.path.join(EX, "H2O_OPT.log")))

    @unittest.skipUnless(os.path.isfile(ORCA_TD), "ORCA TD log of the NTO validation not found")
    def test_parsed_model(self):
        model = read_steps(ORCA_TD, "orca")[0]
        self.assertTrue(model.comp_details.general.functional.startswith("PBE0 (Exchange PBE, Correlation PBE"))


class EmissionOrbitalsTest(unittest.TestCase):
    def test_opt_es_log_has_its_orbitals(self):
        es = read_steps(os.path.join(EX, "phenantren_OPT_ES1.log"))[0]
        self.assertTrue(compute._has_orbitals(es))


if __name__ == "__main__":
    unittest.main()
